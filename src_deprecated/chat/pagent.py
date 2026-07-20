from __future__ import annotations

import json
import re
from collections.abc import AsyncIterator
from types import SimpleNamespace

from fastapi import FastAPI

from src.chat.session import ChatSessionService
from src.config import get_settings
from src.model.tree import Tree
from src.query.query import (
    build_tree_index,
    content_to_search_text,
    score_nodes_bm25,
)

from pagent import (
    Agent,
    DeepSeek,
    LLM,
    Ollama,
    ReasoningDelta,
    RunEnd,
    Session,
    TextDelta,
    ToolCallBegin,
    ToolResult,
    TurnEnd,
    tool,
)


path_pattern = re.compile(r"^\d+(?:\.\d+)*$")
document_intent_keywords = (
    "知识库",
    "资料",
    "文档",
    "文件",
    "原文",
    "上传",
    "tree",
    "forest",
    "节点",
    "章节",
    "段落",
    "内容",
    "这份",
    "这个库",
    "这篇",
    "里面",
    "根据",
    "引用",
    "检索",
    "搜索",
    "查找",
    "总结",
    "概括",
    "讲了什么",
    "说了什么",
    "提到",
    "提及",
    "section",
    "document",
    "knowledge base",
    "uploaded",
    "source",
    "reference",
    "summarize",
    "summary",
    "find",
    "search",
    "content",
)
document_intent_patterns = (
    re.compile(r"\b\d+(?:\.\d+)+\b"),
    re.compile(r"(第[一二三四五六七八九十\d]+[章节])"),
)


def optional_setting_text(value: str | None) -> str | None:
    if value is None:
        return None
    cleaned = value.strip()
    return cleaned or None


def compact_tool_content(content: str, limit: int = 240) -> str:
    text = " ".join(content.split()).strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit].rstrip()}..."


def fallback_answer_from_tools(tool_results: list[dict]) -> str:
    snippets: list[str] = []
    for item in tool_results:
        if item.get("ok") is False:
            continue
        content = compact_tool_content(str(item.get("content") or ""))
        if not content:
            continue
        name = str(item.get("name") or "tool")
        snippets.append(f"- {name}: {content}")
        if len(snippets) >= 3:
            break

    if not snippets:
        return ""

    return (
        "已完成检索，但模型没有返回最终文本答复。以下是本轮工具结果摘要：\n"
        + "\n".join(snippets)
    )


async def build_pagent_events(
    app: FastAPI,
    *,
    tree_id: str | None,
    question: str,
    session_id: str | None,
) -> AsyncIterator[dict]:
    current_tree = app.state.tree_registry.get(tree_id) if tree_id else None
    if tree_id and current_tree is None:
        yield {"type": "error", "message": f"unknown tree_id: {tree_id}"}
        return

    chat_sessions = get_chat_sessions(app)
    chat_session = chat_sessions.get_or_create(
        session_id=session_id,
        title=question[:120],
    )
    tools_enabled = question_needs_document_tools(
        question,
        tree_id=tree_id,
        forest_count=len(app.state.tree_registry),
    )
    session = Session(build_system_prompt(app, tree_id, current_tree, tools_enabled))
    replay_history(session, chat_session.turns)

    try:
        agent = build_agent(
            app,
            tree_id=tree_id,
            current_tree=current_tree,
            session=session,
            tools_enabled=tools_enabled,
        )
    except Exception as exc:  # noqa: BLE001
        yield {
            "type": "error",
            "message": f"failed to initialize pagent agent: {exc}",
        }
        return

    yield {
        "type": "start",
        "bid": tree_id,
        "tree_id": tree_id,
        "filename": current_tree.title if current_tree is not None else None,
        "session_id": chat_session.session_id,
        "tools_enabled": tools_enabled,
    }

    assistant_parts: list[str] = []
    tool_calls: list[dict] = []
    tool_results: list[dict] = []
    tool_call_ids: dict[str, str] = {}
    assistant_events: list[dict] = []

    try:
        async for event in agent.arun_events(question):
            match event:
                case TextDelta(text=text):
                    assistant_parts.append(text)
                    assistant_events.append({"type": "text", "text": text})
                    yield {"type": "text", "text": text}
                case ReasoningDelta(text=text):
                    assistant_events.append({"type": "reasoning", "text": text})
                    yield {"type": "reasoning", "text": text}
                case ToolCallBegin(tool_call_id=raw_id, name=name, arguments=arguments):
                    item = start_tool_call(
                        tool_calls,
                        tool_call_ids,
                        raw_id=raw_id,
                        name=name,
                        arguments=arguments,
                    )
                    assistant_events.append({"type": "tool_call", **item})
                    yield {"type": "tool_call", **item}
                case ToolResult(
                    tool_call_id=raw_id,
                    name=name,
                    content=content,
                    ok=ok,
                ):
                    item = finish_tool_call(
                        tool_calls,
                        tool_results,
                        tool_call_ids,
                        raw_id=raw_id,
                        name=name,
                        content=content,
                        ok=ok,
                    )
                    assistant_events.append({"type": "tool_result", **item})
                    yield {"type": "tool_result", **item}
                case TurnEnd(turn=turn, stopped=stopped):
                    yield {"type": "turn_end", "turn": turn, "stopped": stopped}
                case RunEnd(content=content, usage=usage):
                    answer = (
                        content
                        or "".join(assistant_parts)
                        or fallback_answer_from_tools(tool_results)
                    )
                    token_usage = usage or SimpleNamespace()
                    chat_session = chat_sessions.append_turn(
                        chat_session.session_id,
                        question=question,
                        answer=answer,
                        tool_calls=tool_calls,
                        tool_results=tool_results,
                        assistant_events=assistant_events,
                    )
                    yield {
                        "type": "done",
                        "answer": answer,
                        "turns": len(chat_session.turns),
                        "prompt_tokens": getattr(token_usage, "prompt_tokens", None),
                        "completion_tokens": getattr(
                            token_usage, "completion_tokens", None
                        ),
                        "total_tokens": getattr(token_usage, "total_tokens", None),
                    }
                    return
                case _:
                    continue
    except Exception as exc:  # noqa: BLE001
        yield {"type": "error", "message": str(exc)}


def build_agent(
    app: FastAPI,
    *,
    tree_id: str | None,
    current_tree: Tree | None,
    session: Session,
    tools_enabled: bool,
) -> Agent:
    return Agent(
        resolve_llm(),
        session,
        tools=build_tools(
            app,
            tree_id=tree_id,
            current_tree=current_tree,
            tools_enabled=tools_enabled,
        ),
        max_turns=8,
    )


def build_tools(
    app: FastAPI,
    *,
    tree_id: str | None,
    current_tree: Tree | None,
    tools_enabled: bool,
) -> list:
    if not tools_enabled or not app.state.tree_registry:
        return []

    tools = list(forest_tools(app, tree_id))
    if tree_id and current_tree is not None:
        tools.extend(tree_tools(app, tree_id))
    return tools


def question_needs_document_tools(
    question: str,
    *,
    tree_id: str | None,
    forest_count: int,
) -> bool:
    if forest_count == 0:
        return False

    text = question.strip().lower()
    if not text:
        return False

    if any(pattern.search(text) for pattern in document_intent_patterns):
        return True

    if any(keyword in text for keyword in document_intent_keywords):
        return True

    return bool(
        tree_id and len(text) <= 12 and text in {"总结", "总结一下", "概括", "讲讲"}
    )


def start_tool_call(
    tool_calls: list[dict],
    tool_call_ids: dict[str, str],
    *,
    raw_id: str,
    name: str,
    arguments: str,
) -> dict:
    synthetic_id = f"tc-{len(tool_calls)}"
    if raw_id:
        tool_call_ids[raw_id] = synthetic_id
    item = {
        "id": synthetic_id,
        "name": name,
        "arguments": arguments,
    }
    tool_calls.append(item)
    return item


def finish_tool_call(
    tool_calls: list[dict],
    tool_results: list[dict],
    tool_call_ids: dict[str, str],
    *,
    raw_id: str,
    name: str,
    content: str,
    ok: bool,
) -> dict:
    if len(content) > 8000:
        content = content[:8000] + "\n... (truncated)"

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
        "name": name,
        "ok": bool(ok),
        "content": content,
    }
    tool_results.append(item)
    return item


def resolve_llm():
    settings = get_settings().llm
    requested = (settings.model or "gpt-4o-mini").strip()
    if requested.startswith("deepseek/") or requested in {
        "deepseek-chat",
        "deepseek-reasoner",
    }:
        return DeepSeek(
            requested.removeprefix("deepseek/"),
            apikey=optional_setting_text(settings.api_key),
            base_url=optional_setting_text(settings.base_url),
        )

    if requested.startswith("ollama/"):
        base_url = optional_setting_text(settings.base_url)
        if base_url is not None:
            base_url = base_url.rstrip("/")
            if not base_url.endswith("/v1"):
                base_url = f"{base_url}/v1"
        return Ollama(
            requested.removeprefix("ollama/"),
            base_url=base_url,
            apikey=optional_setting_text(settings.api_key) or "ollama",
        )

    return LLM(
        requested,
        base_url=optional_setting_text(settings.base_url),
        apikey=optional_setting_text(settings.api_key),
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


def forest_tools(app: FastAPI, current_tree_id: str | None) -> list:
    @tool()
    def forest_catalog() -> str:
        """List all documents in the forest."""
        trees = sorted(app.state.tree_registry.values(), key=lambda tree: tree.node_id)
        lines = [f"forest: {len(trees)} trees"]
        for tree in trees:
            marker = (
                " (current chat)"
                if current_tree_id and tree.node_id == current_tree_id
                else ""
            )
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


def build_system_prompt(
    app: FastAPI,
    tree_id: str | None,
    tree: Tree | None,
    tools_enabled: bool,
) -> str:
    forest_count = len(app.state.tree_registry)
    if forest_count == 0:
        return (
            "You are a general-purpose assistant. "
            "No document knowledge base is attached to this chat. "
            "Answer in the user's language."
        )
    if not tools_enabled:
        attached_text = (
            f"A document knowledge base `{tree.title}` is attached to this chat. "
            if tree_id and tree is not None
            else f"The server currently has {forest_count} document tree(s), but none is selected for this turn. "
        )
        return (
            "You are a general-purpose assistant. "
            + attached_text
            + "The current question does not require document retrieval. "
            "Answer directly in the user's language. "
            "Only discuss document contents when the user clearly asks about the attached knowledge base, uploaded files, sections, nodes, or cited material."
        )
    if tree_id and tree is not None:
        return (
            f"You are a document assistant for `{tree.title}` (tree_id={tree_id}). "
            f"The server currently has {forest_count} document tree(s). "
            "Use tools only when the answer depends on document content. Use numeric dot paths like `0` "
            "or `0.1`, never section titles, when navigating. Answer in the user's language."
        )
    return (
        "You are a document assistant over a forest of document trees. "
        f"The server currently has {forest_count} document tree(s). "
        "Use tools only when the answer depends on document content. "
        "For document-content questions without a selected knowledge base, inspect the forest first, "
        "find relevant sections across trees, and ground the answer in retrieved content. "
        "Answer in the user's language."
    )


def replay_history(session, turns: list) -> None:
    for turn in turns:
        role = getattr(turn, "role", None)
        content = getattr(turn, "content", "")
        if role in {"user", "assistant"} and content:
            session += {"role": role, "content": content}


def get_chat_sessions(app: FastAPI) -> ChatSessionService:
    return app.state.chat_sessions


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


def event_to_ndjson(event: dict) -> str:
    return json.dumps(event, ensure_ascii=False) + "\n"


__all__ = ["build_pagent_events", "event_to_ndjson", "resolve_llm"]
