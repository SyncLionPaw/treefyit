"""Chat: expose a pagent.Agent whose tools inspect the tree of a build.

The agent runs a multi-turn tool loop: it receives the user's question,
decides by itself whether to inspect a node / list children / check the
document summary, and returns a final answer.  Events (tool calls, tool
results, text deltas, thinking) are streamed line-by-line as JSON.

This module is intentionally small — it wires the existing
``src.tools.overview/inspect/get_children`` with ``pagent.Agent``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import AsyncIterator
from types import SimpleNamespace

from src.llm import client as _llm_client  # noqa: F401 — side-effect: loads .env

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Tools bound to a specific build
# ---------------------------------------------------------------------------


def _tree_tools(build: dict):
    """Return three ``pagent.tool``-wrapped functions that read from *build*.

    Tools accept numeric dot-paths (``0``, ``0.1``, …) and return section
    titles in their output for the model to reference in answers.
    """

    from pagent import tool
    from src.tools import (
        inspect as _inspect,
        get_children as _get_children,
        overview as _overview,
    )

    tree_id = build["id"]
    bid = tree_id

    @tool()
    def document_overview() -> str:
        """Return the document's top-level structure.

        Produces one line per top-level section with its numeric dot-path
        (e.g. ``'0'``, ``'1'``), its title, the number of sub-sections,
        and a short summary when available. Use this first to locate the
        right section — then call ``node_children`` / ``node_content``
        with the numeric path as the argument.
        """
        logger.info("[agent] tool=document_overview bid=%s", bid)
        data = _overview(tree_id)
        if "error" in data:
            logger.warning("[agent] document_overview error: %s", data["error"])
            return f"error: {data['error']}"

        lines: list[str] = [
            f"document: {build.get('filename') or tree_id}",
            f"nodes: {data.get('node_count')}, max_depth: {data.get('max_depth')}",
        ]
        for i, node in enumerate(data.get("roots", [])):
            path = str(i)
            title = (node.get("title") or "").strip()
            cc = node.get("children_count", 0)
            lines.append(f"- [{path}] {title}  (sub-sections: {cc})")
            s = (node.get("summary") or "").strip()
            if s:
                snippet = s if len(s) <= 200 else s[:200] + "…"
                lines.append(f"    summary: {snippet}")
        logger.info("[agent] document_overview ok roots=%d", len(data.get("roots", [])))
        return "\n".join(lines)

    @tool()
    def node_children(path: str) -> str:
        """List the sub-sections of a section.

        Pass the numeric dot-path (e.g. ``'0'`` or ``'0.1'``) of the
        section you want to expand. Returns one line per sub-section
        with its own numeric dot-path, its title, its sub-section count,
        and a short summary when available.
        """
        logger.info("[agent] tool=node_children bid=%s path=%s", bid, path)
        if not re.fullmatch(r"\d+(?:\.\d+)*", path or ""):
            logger.warning("[agent] node_children invalid path: %r", path)
            return f"error: argument must be a numeric dot-path like '0' or '0.1' (got {path!r})"

        data = _get_children(tree_id, path)
        if "error" in data:
            logger.warning("[agent] node_children error: %s", data["error"])
            return f"error: {data['error']}"

        parent_title = (data.get("title") or "").strip() or path
        children = data.get("children", [])
        lines = [
            f"children of [{path}] {parent_title}  ({len(children)} sub-sections):"
        ]
        for i, child in enumerate(children):
            title = (child.get("title") or "").strip()
            child_path = f"{path}.{i}"
            lines.append(
                f"- [{child_path}] {title}  (sub-sections: {child.get('children_count', 0)})"
            )
            s = (child.get("summary") or "").strip()
            if s:
                snippet = s if len(s) <= 200 else s[:200] + "…"
                lines.append(f"    summary: {snippet}")
        logger.info("[agent] node_children ok path=%s children=%d", path, len(children))
        return "\n".join(lines)

    @tool()
    def node_content(path: str) -> str:
        """Return the full text and summary of a section.

        Pass the numeric dot-path (e.g. ``'0'`` or ``'0.1'``) of the
        section. Prefer calling ``document_overview`` and
        ``node_children`` first to narrow down the path before reading
        a section's content.
        """
        logger.info("[agent] tool=node_content bid=%s path=%s", bid, path)
        if not re.fullmatch(r"\d+(?:\.\d+)*", path or ""):
            logger.warning("[agent] node_content invalid path: %r", path)
            return f"error: argument must be a numeric dot-path like '0' or '0.1' (got {path!r})"

        data = _inspect(tree_id, path)
        if "error" in data:
            logger.warning("[agent] node_content error: %s", data["error"])
            return f"error: {data['error']}"

        title = (data.get("title") or "").strip() or path
        summary = (data.get("summary") or "").strip()
        text = (data.get("text") or "").strip()
        cc = data.get("children_count", 0)

        lines = [f"# [{path}] {title}  (sub-sections: {cc})"]
        if summary:
            lines.append("## summary")
            lines.append(summary if len(summary) <= 2000 else summary[:2000] + "…")
        if text:
            lines.append("## text")
            lines.append(
                text if len(text) <= 8000 else text[:8000] + "\n...(truncated)"
            )
        if not (summary or text):
            lines.append("(no content)")
        logger.info("[agent] node_content ok path=%s text=%d chars", path, len(text))
        return "\n\n".join(lines)

    return document_overview, node_children, node_content


# ---------------------------------------------------------------------------
# Agent builder + streamer
# ---------------------------------------------------------------------------


async def build_events(
    bid: str,
    question: str,
    model: str = "deepseek-chat",
    session_id: str | None = None,
    history: list[dict] | None = None,
):
    """Run a pagent agent loop and yield structured events.

    Args:
        bid: Build id (document to chat about).
        question: User message for this turn.
        model: LLM model identifier.
        session_id: Optional persistent session id. When provided, the turn
            is appended to the session after the agent finishes.
        history: Previous turns loaded from storage. Each turn is a dict
            with ``role``, ``text``, ``tool_calls``, ``tool_results``.

    Yields a sequence of ``dict`` with a ``type`` key:

    * ``{"type": "start", "bid", "filename", "model", "session_id"}``
    * ``{"type": "text", "text": "..."}`` (a chunk of the final answer)
    * ``{"type": "reasoning", "text": "..."}`` (model thinking, if supported)
    * ``{"type": "tool_call", "id", "name", "arguments"}``
    * ``{"type": "tool_result", "id", "name", "ok", "content"}``
    * ``{"type": "done", "answer", "turns", "prompt_tokens", "completion_tokens", "total_tokens"}``
    * ``{"type": "error", "message"}`` — on hard failure; stops the stream
    """

    from src import store as _store

    build = _store.history.get(bid) or _store.load_build(bid)
    if not build:
        yield {"type": "error", "message": f"unknown build id: {bid}"}
        return

    has_tree = bool(build.get("tree"))
    if has_tree:
        try:
            from src.tools import register

            register(bid, build["tree"])
        except Exception as e:  # noqa: BLE001
            logger.info("register failed for %s: %s", bid, e)
            has_tree = False

    # --- wire up pagent ---
    try:
        from pagent import Agent, DeepSeek, LLM, Session
    except ImportError as e:
        yield {
            "type": "error",
            "message": (
                "pagent is not installed; run `uv add pagent` and restart the server "
                f"(got {e})."
            ),
        }
        return

    # Resolve the pagent provider from the requested model name:
    #   prefix "deepseek/"  OR exact "deepseek-chat"/"deepseek-reasoner"
    #       -> pagent.DeepSeek, reads DEEPSEEK_API_KEY / DEEPSEEK_API_BASE
    #   otherwise
    #       -> pagent.LLM, reads OPENAI_API_KEY
    trimmed = (model or "").strip()
    normalized = trimmed.removeprefix("deepseek/")
    want_deepseek = trimmed.startswith("deepseek/") or trimmed in {
        "deepseek-chat",
        "deepseek-reasoner",
    }
    if want_deepseek:
        env_key = "DEEPSEEK_API_KEY"
        if not os.environ.get(env_key):
            yield {
                "type": "error",
                "message": (
                    f"{env_key} is not set. Add `{env_key}=sk-...` to your `.env` "
                    "file (in the project root) and restart the server."
                ),
            }
            return
        try:
            llm = DeepSeek(normalized)
        except Exception as e:  # noqa: BLE001
            yield {
                "type": "error",
                "message": f"failed to initialize DeepSeek({normalized!r}): {e}",
            }
            return
    else:
        env_key = "OPENAI_API_KEY"
        if not os.environ.get(env_key):
            yield {
                "type": "error",
                "message": (
                    f"{env_key} is not set. Add `{env_key}=sk-...` to your `.env` file, "
                    "or switch to a `deepseek/...` model."
                ),
            }
            return
        try:
            llm = LLM(normalized or "gpt-4o-mini")
        except Exception as e:  # noqa: BLE001
            yield {
                "type": "error",
                "message": f"failed to initialize LLM({normalized!r}): {e}",
            }
            return

    tools = list(_tree_tools(build)) if has_tree else []
    filename = build.get("filename", bid)

    system_prompt = (
        f"You are a helpful assistant chatting with a user who has uploaded the "
        f"document '{filename}'. "
    )
    if has_tree:
        system_prompt += (
            "The document is available to you; you also have three tools to browse it. "
            "IMPORTANT — when you CALL these tools, pass a numeric dot-path (like "
            "'0', '0.1', '0.2.1') as the argument — never a title string. "
            "The tools return titles back to you in their output, so you can still "
            "refer to sections by title inside your final answer to the user.\n"
            "1. document_overview() — list of top-level sections with summaries and "
            "their numeric paths ('0', '1', ...). Use it first.\n"
            "2. node_children(path) — given a numeric dot-path like '0' or '0.1', "
            "list its direct sub-sections with their numeric paths (e.g. '0.0', "
            "'0.1'). Use it to narrow down.\n"
            "3. node_content(path) — given a numeric dot-path, return the full "
            "text/summary of that section. Use it once you have pinned down the "
            "relevant section.\n\n"
        )
    else:
        system_prompt += (
            "The document was uploaded but could not be parsed — the document tools "
            "are NOT available for this build. Answer the user as a general-purpose "
            "assistant instead of trying to reference the document.\n\n"
        )

    system_prompt += (
        "How to decide whether to use the tools:\n"
        "- Use the tools when the question is about the document's content, "
        "structure, or claims (e.g. 'what does section X say?', 'summarize the "
        "API', 'list the conclusions').\n"
        "- You MAY answer directly without calling any tool when the question is "
        "general knowledge, a greeting, small talk, or unrelated to the document.\n"
        "- When in doubt but the document IS available, call document_overview "
        "once and then decide.\n\n"
    )

    if has_tree:
        system_prompt += (
            "Rules for answers that use the document:\n"
            "- Prefer narrow sections over reading the whole document.\n"
            "- Quote short passages inside your answer so the user can locate them.\n"
            "- Do NOT repeat the full content of a section; summarize.\n"
            "- Do NOT mention section indices or path-like numbers (e.g. '0.1') in "
            "your answer.  If a section has a numbered title, only quote the number "
            "that appears in the title itself; do not invent new index numbers.\n"
            "- If a section is not relevant, try another or tell the user.\n"
        )

    system_prompt += "- Answer in the user's language (usually Chinese)."

    session = Session(system_prompt)

    # Replay previous turns into the session so the agent remembers context.
    if history:
        for turn in history:
            role = turn.get("role")
            text = turn.get("text") or ""
            if role == "user":
                session.messages.append({"role": "user", "content": text})
            elif role == "assistant":
                msg: dict = {"role": "assistant", "content": text}
                tc = turn.get("tool_calls")
                if tc:
                    try:
                        raw_tc = json.loads(tc) if isinstance(tc, str) else tc
                        # Our stored tool_calls use a compact schema
                        #   {"id": "...", "name": "...", "arguments": "..."}
                        # OpenAI (and pagent) expect the standard schema
                        #   {"id": "...", "type": "function",
                        #    "function": {"name": "...", "arguments": "..."}}
                        msg["tool_calls"] = [
                            {
                                "id": t.get("id", ""),
                                "type": "function",
                                "function": {
                                    "name": t.get("name", ""),
                                    "arguments": t.get("arguments", ""),
                                },
                            }
                            for t in raw_tc
                        ]
                    except Exception:
                        pass
                session.messages.append(msg)
                tr = turn.get("tool_results")
                if tr:
                    try:
                        tool_results = json.loads(tr) if isinstance(tr, str) else tr
                        tc_ids = [t.get("id", "") for t in raw_tc] if tc else []
                        for idx, item in enumerate(tool_results):
                            tr_id = item.get("id", "")
                            # Defensive: if the stored result id does not match any
                            # tool_call id (e.g. old bug where raw DeepSeek id was
                            # saved instead of synthetic id), fix it by position.
                            if tc_ids and tr_id not in tc_ids:
                                tr_id = tc_ids[idx] if idx < len(tc_ids) else tr_id
                            session.messages.append(
                                {
                                    "role": "tool",
                                    "tool_call_id": tr_id,
                                    "content": item.get("content", ""),
                                }
                            )
                    except Exception:
                        pass

    try:
        agent = Agent(
            llm=llm,
            session=session,
            tools=tools,
            max_turns=8,
        )
    except Exception as e:  # noqa: BLE001
        yield {"type": "error", "message": f"failed to initialize agent: {e}"}
        return

    yield {
        "type": "start",
        "bid": bid,
        "filename": filename,
        "model": model,
        "session_id": session_id,
    }

    # Collectables for this turn (saved to DB at the end).
    turn_tool_calls: list[dict] = []
    turn_tool_results: list[dict] = []
    assistant_text_parts: list[str] = []
    turn_ok = False
    # DeepSeek may return empty tool_call ids in ToolCallBegin but non-empty ids
    # in ToolResult.  We keep a map so the two sides stay consistent.
    _tc_id_map: dict[str, str] = {}  # original_id -> synthetic_id

    try:
        async for event in agent.arun_events(question):
            event_name = type(event).__name__
            # Map pagent event classes to our own wire format.
            if event_name == "TextDelta":
                chunk = getattr(event, "text", "")
                assistant_text_parts.append(chunk)
                yield {"type": "text", "text": chunk}
            elif event_name == "ReasoningDelta":
                yield {"type": "reasoning", "text": getattr(event, "text", "")}
            elif event_name == "ToolCallBegin":
                raw_id = getattr(event, "id", "") or ""
                # Always synthesise a deterministic id so that ToolResult can
                # unambiguously match us even when the provider sends empty ids.
                synthetic_id = f"tc-{len(turn_tool_calls)}"
                if raw_id:
                    _tc_id_map[raw_id] = synthetic_id
                turn_tool_calls.append(
                    {
                        "id": synthetic_id,
                        "name": getattr(event, "name", ""),
                        "arguments": getattr(event, "arguments", ""),
                    }
                )
                yield {
                    "type": "tool_call",
                    "id": synthetic_id,
                    "name": getattr(event, "name", ""),
                    "arguments": getattr(event, "arguments", ""),
                }
            elif event_name == "ToolResult":
                content = getattr(event, "content", "")
                # Truncate very large tool outputs to keep prompt budgets sane.
                if content and len(content) > 8000:
                    content = content[:8000] + "\n... (truncated)"
                raw_tr_id = (
                    getattr(event, "tool_call_id", getattr(event, "id", "")) or ""
                )
                # Map back to the synthetic id we created in ToolCallBegin.
                # DeepSeek sometimes sends an empty id in ToolCallBegin but a real
                # id in ToolResult; when that happens we fall back to positional
                # matching so the two sides stay consistent.
                if raw_tr_id in _tc_id_map:
                    tr_id = _tc_id_map[raw_tr_id]
                elif len(turn_tool_results) < len(turn_tool_calls):
                    tr_id = turn_tool_calls[len(turn_tool_results)]["id"]
                    if raw_tr_id:
                        _tc_id_map[raw_tr_id] = tr_id
                else:
                    tr_id = raw_tr_id or f"tc-{len(turn_tool_results)}"
                turn_tool_results.append(
                    {
                        "id": tr_id,
                        "name": getattr(event, "name", ""),
                        "ok": bool(getattr(event, "ok", True)),
                        "content": content,
                    }
                )
                yield {
                    "type": "tool_result",
                    "id": tr_id,
                    "name": getattr(event, "name", ""),
                    "ok": bool(getattr(event, "ok", True)),
                    "content": content,
                }
            elif event_name == "RunEnd":
                turn_ok = True
                usage = getattr(event, "usage", None) or SimpleNamespace(
                    prompt_tokens=0,
                    completion_tokens=0,
                    total_tokens=0,
                )
                yield {
                    "type": "done",
                    "answer": getattr(event, "content", "") or "",
                    "turns": int(getattr(agent.stats, "turns", 0)),
                    "prompt_tokens": int(getattr(usage, "prompt_tokens", 0)),
                    "completion_tokens": int(getattr(usage, "completion_tokens", 0)),
                    "total_tokens": int(getattr(usage, "total_tokens", 0)),
                }
            else:
                # Fall through silently for TurnBegin/TurnEnd/RunBegin/...
                yield {"type": "debug", "event": event_name}
    except Exception as e:  # noqa: BLE001
        # Surface OpenAI / DeepSeek tool-call format errors so the client
        # can decide whether to retry without the session history.
        msg = str(e)
        if "BadRequestError" in type(e).__name__ or "tool_call" in msg.lower():
            logger.error("chat agent failed (tool-call format): %s", msg)
            yield {
                "type": "error",
                "code": "TOOL_CALL_FORMAT",
                "message": (
                    "The LLM returned an malformed tool-call sequence. "
                    "Please start a new chat session."
                ),
            }
        else:
            logger.exception("chat agent failed")
            yield {"type": "error", "message": f"agent failure: {e}"}

    # Persist the turn if a session_id was provided and the agent ran to
    # completion (we got a RunEnd event).
    if session_id and turn_ok:
        try:
            _store.append_turn(session_id, "user", text=question)
            _store.append_turn(
                session_id,
                "assistant",
                text="".join(assistant_text_parts),
                tool_calls=turn_tool_calls or None,
                tool_results=turn_tool_results or None,
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("failed to persist chat turn for %s: %s", session_id, e)


async def build_streamer(
    bid: str,
    question: str,
    model: str = "deepseek-chat",
    session_id: str | None = None,
    history: list[dict] | None = None,
) -> AsyncIterator[bytes]:
    """Yield NDJSON (newline-delimited JSON) bytes from :func:`build_events`.

    Each yielded line is ``b'{"type": "...", ...}\n'``, ready to be sent
    over an SSE stream or a raw HTTP response body.
    """
    async for event in build_events(
        bid=bid, question=question, model=model, session_id=session_id, history=history
    ):
        yield (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")
