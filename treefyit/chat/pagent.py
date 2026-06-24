from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from types import SimpleNamespace
from uuid import uuid4

from fastapi import FastAPI

from treefyit.config import get_settings
from treefyit.model.tree import Tree
from treefyit.query.query import (
    content_to_search_text,
    build_tree_index,
    score_nodes_bm25,
)

try:
    from pagent import Agent, DeepSeek, LLM, Session, tool
except ImportError:  # pragma: no cover - handled at runtime by error event.
    Agent = DeepSeek = LLM = Session = tool = None  # type: ignore[assignment]


path_pattern = re.compile(r"^\d+(?:\.\d+)*$")


async def build_pagent_events(
    app: FastAPI,
    *,
    tree_id: str,
    question: str,
    session_id: str | None,
) -> AsyncIterator[dict]:
    if Agent is None or Session is None or tool is None:
        yield {
            "type": "error",
            "message": "pagent is not installed; install pagent>=0.2.0.",
        }
        return

    tree = app.state.tree_registry.get(tree_id)
    if tree is None:
        yield {"type": "error", "message": f"unknown tree_id: {tree_id}"}
        return

    sid, history = ensure_chat_session(
        app,
        session_id=session_id,
        tree_id=tree_id,
        title=question[:120],
    )
    session = Session(build_system_prompt(app, tree_id, tree))
    replay_history(session, history)

    try:
        llm = resolve_llm()
        agent = Agent(
            llm=llm,
            session=session,
            tools=[*forest_tools(app, tree_id), *tree_tools(app, tree_id)],
            max_turns=8,
        )
    except Exception as exc:  # noqa: BLE001
        yield {
            "type": "error",
            "message": f"failed to initialize pagent compatible API: {exc}",
        }
        return

    yield {
        "type": "start",
        "bid": tree_id,
        "tree_id": tree_id,
        "filename": tree.title,
        "session_id": sid,
    }

    assistant_parts: list[str] = []
    tool_calls: list[dict] = []
    tool_results: list[dict] = []
    tool_call_ids: dict[str, str] = {}

    try:
        async for event in agent.arun_events(question):
            event_name = type(event).__name__
            if event_name == "TextDelta":
                text = getattr(event, "text", "")
                assistant_parts.append(text)
                yield {"type": "text", "text": text}
                continue

            if event_name == "ReasoningDelta":
                yield {"type": "reasoning", "text": getattr(event, "text", "")}
                continue

            if event_name == "ToolCallBegin":
                raw_id = getattr(event, "id", "") or ""
                synthetic_id = f"tc-{len(tool_calls)}"
                if raw_id:
                    tool_call_ids[raw_id] = synthetic_id
                item = {
                    "id": synthetic_id,
                    "name": getattr(event, "name", ""),
                    "arguments": getattr(event, "arguments", ""),
                }
                tool_calls.append(item)
                yield {"type": "tool_call", **item}
                continue

            if event_name == "ToolResult":
                content = getattr(event, "content", "")
                if len(content) > 8000:
                    content = content[:8000] + "\n... (truncated)"
                raw_id = getattr(event, "tool_call_id", getattr(event, "id", "")) or ""
                if raw_id in tool_call_ids:
                    result_id = tool_call_ids[raw_id]
                elif len(tool_results) < len(tool_calls):
                    result_id = tool_calls[len(tool_results)]["id"]
                    if raw_id:
                        tool_call_ids[raw_id] = result_id
                else:
                    result_id = raw_id or f"tc-{len(tool_results)}"
                item = {
                    "id": result_id,
                    "name": getattr(event, "name", ""),
                    "ok": bool(getattr(event, "ok", True)),
                    "content": content,
                }
                tool_results.append(item)
                yield {"type": "tool_result", **item}
                continue

            if event_name == "RunEnd":
                answer = getattr(event, "content", "") or "".join(assistant_parts)
                usage = getattr(event, "usage", None) or SimpleNamespace()
                append_chat_turn(
                    app,
                    sid,
                    question=question,
                    answer=answer,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                )
                yield {
                    "type": "done",
                    "answer": answer,
                    "turns": len(app.state.sessions[sid]["turns"]),
                    "prompt_tokens": getattr(usage, "prompt_tokens", None),
                    "completion_tokens": getattr(usage, "completion_tokens", None),
                    "total_tokens": getattr(usage, "total_tokens", None),
                }
                return
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "message": str(exc)}


def resolve_llm():
    settings = get_settings().llm
    requested = (settings.model or "gpt-4o-mini").strip()
    if requested.startswith("deepseek/") or requested in {
        "deepseek-chat",
        "deepseek-reasoner",
    }:
        return DeepSeek(requested.removeprefix("deepseek/"))

    if requested.startswith("ollama/"):
        model_id = requested.removeprefix("ollama/")
        base_url = (settings.base_url or "http://127.0.0.1:11434").rstrip("/")
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        return LLM(
            model_id,
            base_url=base_url,
            apikey=settings.api_key or "ollama",
        )

    return LLM(
        requested,
        base_url=settings.base_url,
        apikey=settings.api_key,
    )


def tree_tools(app: FastAPI, tree_id: str) -> list:
    @tool()
    def document_overview() -> str:
        """Return the current document's top-level structure."""
        tree = app.state.tree_registry.get(tree_id)
        if tree is None:
            return f"error: unknown tree_id: {tree_id}"
        lines = [
            f"document: {tree.title}",
            f"nodes: {tree.subtree_size or 1}, max_depth: {max_tree_depth(tree)}",
        ]
        for index, child in enumerate(tree.children):
            lines.append(
                f"- [{index}] {child.title} (sub-sections: {len(child.children)})"
            )
            if child.summary:
                lines.append(f"  summary: {child.summary[:200]}")
        return "\n".join(lines)

    @tool()
    def node_children(path: str) -> str:
        """List sub-sections of a current-document section by numeric path."""
        if not valid_path(path):
            return path_error(path)
        tree = app.state.tree_registry.get(tree_id)
        if tree is None:
            return f"error: unknown tree_id: {tree_id}"
        node = resolve_tree_path(tree, path)
        if node is None:
            return f"error: invalid path: {path}"
        lines = [f"children of [{path}] {node.title}: {len(node.children)}"]
        for index, child in enumerate(node.children):
            child_path = f"{path}.{index}"
            lines.append(
                f"- [{child_path}] {child.title} (sub-sections: {len(child.children)})"
            )
            if child.summary:
                lines.append(f"  summary: {child.summary[:200]}")
        return "\n".join(lines)

    @tool()
    def node_content(path: str) -> str:
        """Return text and summary for a current-document section by numeric path."""
        if not valid_path(path):
            return path_error(path)
        tree = app.state.tree_registry.get(tree_id)
        if tree is None:
            return f"error: unknown tree_id: {tree_id}"
        node = resolve_tree_path(tree, path)
        if node is None:
            return f"error: invalid path: {path}"
        lines = [f"# [{path}] {node.title}"]
        if node.summary:
            lines.extend(["## summary", node.summary[:2000]])
        text = content_to_search_text(node.content)
        if text:
            lines.extend(["## text", text[:8000]])
        if len(lines) == 1:
            lines.append("(no content)")
        return "\n\n".join(lines)

    return [document_overview, node_children, node_content]


def forest_tools(app: FastAPI, current_tree_id: str) -> list:
    @tool()
    def forest_catalog() -> str:
        """List all documents in the forest."""
        trees = sorted(app.state.tree_registry.values(), key=lambda tree: tree.node_id)
        lines = [f"forest: {len(trees)} trees"]
        for tree in trees:
            marker = " (current chat)" if tree.node_id == current_tree_id else ""
            lines.append(
                f"- [{tree.node_id}] {tree.title}{marker} ({tree.subtree_size or 1} nodes)"
            )
            for index, child in enumerate(tree.children[:6]):
                lines.append(f"  · [{index}] {child.title}")
        return "\n".join(lines)

    @tool()
    def find_sections(query: str, limit: int = 8) -> str:
        """Search all documents for sections matching a topic."""
        hits = []
        for tree in app.state.tree_registry.values():
            index = app.state.index_registry.get(tree.node_id)
            if index is None:
                index = build_tree_index(tree)
                app.state.index_registry[tree.node_id] = index
            hits.extend(score_nodes_bm25(index, query, limit=limit))
        hits.sort(key=lambda hit: hit.score, reverse=True)
        if not hits:
            return f"no sections matched: {query!r}"
        lines = [f"sections matching {query!r}:"]
        for hit in hits[:limit]:
            lines.append(
                f"- tree={hit.tree_id} path={hit.path} {hit.title}: {hit.snippet[:160]}"
            )
        return "\n".join(lines)

    @tool()
    def tree_node_content(tree_id: str, path: str) -> str:
        """Return text and summary for a section in any tree."""
        if not valid_path(path):
            return path_error(path)
        tree = app.state.tree_registry.get(tree_id)
        if tree is None:
            return f"error: unknown tree_id: {tree_id}"
        node = resolve_tree_path(tree, path)
        if node is None:
            return f"error: invalid path: {path}"
        text = content_to_search_text(node.content)
        return f"# {node.title}\n\n{node.summary or ''}\n\n{text[:8000]}".strip()

    return [forest_catalog, find_sections, tree_node_content]


def build_system_prompt(app: FastAPI, tree_id: str, tree: Tree) -> str:
    forest_count = len(app.state.tree_registry)
    return (
        f"You are a document assistant for `{tree.title}` (tree_id={tree_id}). "
        f"The server currently has {forest_count} document tree(s). "
        "Use tools for document-content questions. Use numeric dot paths like `0` "
        "or `0.1`, never section titles, when navigating. Answer in the user's language."
    )


def replay_history(session, turns: list[dict]) -> None:
    for turn in turns:
        role = turn.get("role")
        content = turn.get("content") or turn.get("text") or ""
        if role in {"user", "assistant"} and content:
            session.messages.append({"role": role, "content": content})


def ensure_chat_session(
    app: FastAPI,
    *,
    session_id: str | None,
    tree_id: str,
    title: str,
) -> tuple[str, list[dict]]:
    sid = session_id or uuid4().hex
    if sid not in app.state.sessions:
        now = now_iso()
        app.state.sessions[sid] = {
            "session_id": sid,
            "tree_id": tree_id,
            "title": title,
            "created_at": now,
            "updated_at": now,
            "turns": [],
        }
    return sid, list(app.state.sessions[sid]["turns"])


def append_chat_turn(
    app: FastAPI,
    session_id: str,
    *,
    question: str,
    answer: str,
    tool_calls: list[dict],
    tool_results: list[dict],
) -> None:
    session = app.state.sessions[session_id]
    session["updated_at"] = now_iso()
    session["turns"].append(
        {
            "role": "user",
            "content": question,
            "created_at": now_iso(),
        }
    )
    session["turns"].append(
        {
            "role": "assistant",
            "content": answer,
            "tool_calls": tool_calls,
            "tool_results": tool_results,
            "created_at": now_iso(),
        }
    )
    store = app.state.registry_store
    if store is not None:
        store.save_session(session)


def resolve_tree_path(tree: Tree, path: str) -> Tree | None:
    current = tree
    for segment in path.split("."):
        if not segment.isdigit():
            return None
        index = int(segment)
        if index >= len(current.children):
            return None
        current = current.children[index]
    return current


def max_tree_depth(tree: Tree) -> int:
    if not tree.children:
        return tree.depth or 0
    return max(max_tree_depth(child) for child in tree.children)


def valid_path(path: str) -> bool:
    return bool(path_pattern.fullmatch(path or ""))


def path_error(path: str) -> str:
    return (
        f"error: argument must be a numeric dot-path like '0' or '0.1' (got {path!r})"
    )


def now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def event_to_ndjson(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


__all__ = ["build_pagent_events", "event_to_ndjson", "resolve_llm"]
