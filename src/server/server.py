"""treefyit backend API — build and query endpoints.

Usage:
    uv run python src/server/server.py
    # Open http://localhost:8765
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
from pathlib import Path
import time
from urllib.parse import quote

import uvicorn
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse

from src import store as _store
from src.storage.local import storage as _storage
from src.server.build_helpers import (
    build_stats,
    cached_build_result,
    error_build_result,
    success_build_result,
)
from src.tree.builder import flatten_tree
from src.tree.pipeline import build_tree_from_upload, classify_upload, text_from_upload

logger = logging.getLogger("treefyit")

# Configure logging so that INFO-level messages from our code are visible.
# Uvicorn already has its own handler; we just need to ensure the treefyit
# logger (and module-level loggers) are not silenced by the default WARNING.
_logging_handler = logging.StreamHandler()
_logging_handler.setLevel(logging.INFO)
_logging_handler.setFormatter(logging.Formatter("%(levelname)s:%(name)s:%(message)s"))
for _log_name in (
    "treefyit",
    "src.parser.pdf",
    "src.parser.md",
    "src.tree.semantic",
    "src.tree.pipeline",
    "src.chat.agent",
):
    _log = logging.getLogger(_log_name)
    _log.setLevel(logging.INFO)
    if not _log.handlers:
        _log.addHandler(_logging_handler)

app = FastAPI(title="treefyit")

_cors_origins = os.getenv("TREEFYIT_CORS_ORIGINS", "*")
_allow_origins = [o.strip() for o in _cors_origins.split(",") if o.strip()] or ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=_allow_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---- Helpers (defined before module-level init) ----------------------------


def _register_or_none(
    tree_id: str,
    tree: list,
    *,
    filename: str = "",
    doc_kind: str = "",
) -> None:
    """Register a tree with the agent-tools module; don't crash on error."""
    try:
        from src.tools import register

        if not filename or not doc_kind:
            build = _store.history.get(tree_id) or _store.load_build(tree_id)
            if build:
                filename = filename or build.get("filename", "")
                stats = build.get("stats") or {}
                doc_kind = doc_kind or stats.get("doc_kind", "")
        register(tree_id, tree, filename=filename, doc_kind=doc_kind)
    except Exception:  # noqa: BLE001 — best-effort
        logger.exception("failed to register tree %s", tree_id)


# ---- Storage init -----------------------------------------------------------

_store.init()
_store.rebuild_history(_store.history, _register_or_none)


# ---------------------------------------------------------------------------
# Build API
# ---------------------------------------------------------------------------


@app.post("/api/build")
async def api_build(
    file: UploadFile = File(...),
    model: str = Form("deepseek/deepseek-chat"),
    mode: str = Form("auto"),
    summarize: bool = Form(True),
):
    t0 = time.time()
    bid = f"{int(t0 * 1000):x}"
    filename = file.filename or "unknown"

    logger.info(
        "[build] bid=%s file=%s mode=%s summarize=%s", bid, filename, mode, summarize
    )

    raw = await file.read()
    kind = classify_upload(filename)

    file_meta: dict | None = None
    try:
        file_meta = _storage.save_original(bid, filename, raw)
        logger.info("[build] bid=%s original saved key=%s", bid, file_meta["key"])
    except Exception as e:
        logger.error(
            "[build] bid=%s failed to save original: %s", bid, e, exc_info=True
        )

    if kind.is_pdf:
        logger.info("[build] bid=%s parsing PDF (%d bytes)", bid, len(raw))
    try:
        text = text_from_upload(raw, filename)
    except Exception as e:
        elapsed = time.time() - t0
        logger.error("[build] bid=%s parse failed: %s", bid, e, exc_info=True)
        result = error_build_result(
            bid,
            filename,
            str(e),
            elapsed_sec=elapsed,
            model=model,
            mode=mode,
            file_meta=file_meta,
        )
        _store.history[bid] = result
        _store.save_build(bid, result, None)
        return result
    logger.info("[build] bid=%s document text ready (%d chars)", bid, len(text))

    ck = _store.cache_key_for(text, model, mode, summarize)
    cached = _store.cache_get(ck)
    if cached:
        logger.info("[build] bid=%s cache hit", bid)
        result = cached_build_result(bid, filename, cached, file_meta=file_meta)
        _store.history[bid] = result
        _register_or_none(bid, result["tree"])
        _store.save_build(bid, result, ck)
        return result

    logger.info("[build] bid=%s cache miss, starting build", bid)

    try:
        output = await build_tree_from_upload(
            raw=raw,
            text=text,
            filename=filename,
            bid=bid,
            model=model,
            mode=mode,
            summarize=summarize,
        )
        elapsed = time.time() - t0
        node_count = len(flatten_tree(output.tree))
        stats = build_stats(
            input_tokens=output.input_tokens,
            output_tokens=output.output_tokens,
            elapsed_sec=elapsed,
            model=model,
            mode=mode,
            node_count=node_count,
            doc_kind=output.doc_kind,
            verify_result=output.verify_result,
        )
        if output.verify_result:
            logger.info(
                "[build] bid=%s verify ok=%s score=%.2f issues=%s",
                bid,
                output.verify_result["ok"],
                output.verify_result["score"],
                output.verify_result["issues"],
            )
        result = success_build_result(
            bid,
            filename,
            text,
            output.tree,
            stats=stats,
            file_meta=file_meta,
        )
        logger.info(
            "[build] bid=%s done nodes=%d elapsed=%.1fs tokens=%d/%d",
            bid,
            node_count,
            elapsed,
            output.input_tokens,
            output.output_tokens,
        )
        _store.cache_put(
            ck,
            {
                "tree": output.tree,
                "mermaid": result["mermaid"],
                "stats": stats,
                "raw_text": text,
            },
        )
        _register_or_none(bid, output.tree)
        _store.history[bid] = result
        _store.save_build(bid, result, ck)
        return result

    except Exception as e:
        elapsed = time.time() - t0
        logger.error("[build] bid=%s failed: %s", bid, e, exc_info=True)
        result = error_build_result(
            bid,
            filename,
            str(e),
            elapsed_sec=elapsed,
            model=model,
            mode=mode,
            file_meta=file_meta,
        )
        _store.history[bid] = result
        _store.save_build(bid, result, None)
        return result


@app.post("/api/build/stream")
async def api_build_stream(
    file: UploadFile = File(...),
    model: str = Form("deepseek/deepseek-chat"),
    mode: str = Form("auto"),
    summarize: bool = Form(True),
):
    """Build a document and stream real progress as NDJSON events.

    This is additive: the existing ``POST /api/build`` endpoint remains the
    stable one-shot API.  Each yielded line is a JSON object with ``type``.
    """
    raw = await file.read()
    filename = file.filename or "unknown"
    return StreamingResponse(
        _build_stream_events(
            raw=raw,
            filename=filename,
            model=model,
            mode=mode,
            summarize=summarize,
        ),
        media_type="application/x-ndjson",
    )


async def _build_stream_events(
    *,
    raw: bytes,
    filename: str,
    model: str,
    mode: str,
    summarize: bool,
):
    t0 = time.time()
    bid = f"{int(t0 * 1000):x}"
    file_meta: dict | None = None
    progress_queue: asyncio.Queue[dict] = asyncio.Queue()

    def line(event: dict) -> bytes:
        event.setdefault("bid", bid)
        event.setdefault("elapsed_sec", round(time.time() - t0, 1))
        return (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")

    async def on_progress(payload: dict) -> None:
        await progress_queue.put(payload)

    logger.info(
        "[build-stream] bid=%s file=%s mode=%s summarize=%s",
        bid,
        filename,
        mode,
        summarize,
    )
    yield line(
        {
            "type": "start",
            "stage": "start",
            "filename": filename,
            "model": model,
            "mode": mode,
            "summarize": summarize,
            "file_size": len(raw),
        }
    )

    try:
        yield line(
            {
                "type": "progress",
                "stage": "save_original",
                "message": "saving original file",
            }
        )
        try:
            file_meta = _storage.save_original(bid, filename, raw)
            yield line(
                {
                    "type": "progress",
                    "stage": "save_original_done",
                    "storage_key": file_meta["key"],
                    "file_size": file_meta["size"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "[build-stream] bid=%s save original failed: %s",
                bid,
                exc,
                exc_info=True,
            )
            yield line(
                {
                    "type": "warning",
                    "stage": "save_original_failed",
                    "message": str(exc),
                }
            )

        yield line(
            {"type": "progress", "stage": "parse", "message": "reading document"}
        )
        text = text_from_upload(raw, filename)
        yield line({"type": "progress", "stage": "parse_done", "chars": len(text)})

        ck = _store.cache_key_for(text, model, mode, summarize)
        cached = _store.cache_get(ck)
        if cached:
            result = cached_build_result(bid, filename, cached, file_meta=file_meta)
            _store.history[bid] = result
            _register_or_none(bid, result["tree"])
            _store.save_build(bid, result, ck)
            yield line(
                {
                    "type": "done",
                    "stage": "done",
                    "cached": True,
                    "result": result,
                }
            )
            return

        yield line(
            {"type": "progress", "stage": "structure", "message": "building structure"}
        )

        build_task = asyncio.create_task(
            build_tree_from_upload(
                raw=raw,
                text=text,
                filename=filename,
                bid=bid,
                model=model,
                mode=mode,
                summarize=summarize,
                progress=on_progress,
            )
        )

        while not build_task.done() or not progress_queue.empty():
            try:
                payload = progress_queue.get_nowait()
            except asyncio.QueueEmpty:
                if build_task.done():
                    break
                await asyncio.sleep(0.05)
                continue

            stage = payload.pop("stage", "progress")
            event_type = "warning" if stage == "verify_failed" else "progress"
            yield line({"type": event_type, "stage": stage, **payload})

        output = await build_task
        elapsed = time.time() - t0
        node_count = len(flatten_tree(output.tree))
        stats = build_stats(
            input_tokens=output.input_tokens,
            output_tokens=output.output_tokens,
            elapsed_sec=elapsed,
            model=model,
            mode=mode,
            node_count=node_count,
            doc_kind=output.doc_kind,
            verify_result=output.verify_result,
        )
        result = success_build_result(
            bid,
            filename,
            text,
            output.tree,
            stats=stats,
            file_meta=file_meta,
        )
        _store.cache_put(
            ck,
            {
                "tree": output.tree,
                "mermaid": result["mermaid"],
                "stats": stats,
                "raw_text": text,
            },
        )
        _register_or_none(bid, output.tree)
        _store.history[bid] = result
        _store.save_build(bid, result, ck)
        yield line({"type": "done", "stage": "done", "cached": False, "result": result})

    except Exception as exc:  # noqa: BLE001
        logger.error("[build-stream] bid=%s failed: %s", bid, exc, exc_info=True)
        elapsed = time.time() - t0
        result = error_build_result(
            bid,
            filename,
            str(exc),
            elapsed_sec=elapsed,
            model=model,
            mode=mode,
            file_meta=file_meta,
        )
        _store.history[bid] = result
        _store.save_build(bid, result, None)
        yield line(
            {"type": "error", "stage": "error", "message": str(exc), "result": result}
        )


@app.get("/api/history")
def api_history():
    """List recent builds (lightweight — no raw_text / tree)."""
    items = list(_store.history.values())
    items.sort(key=lambda x: x["id"], reverse=True)
    out = []
    for b in items:
        entry = {
            "id": b["id"],
            "filename": b["filename"],
            "content_type": b.get("content_type"),
            "file_size": b.get("file_size"),
            "has_original_file": b.get("has_original_file", False),
            "original_file_url": b.get("original_file_url"),
            "stats": b.get("stats", {}),
            "created_at": b["created_at"],
            "cached": b.get("cached", False),
            "error": b.get("error"),
        }
        out.append(entry)
    return out


@app.get("/api/build/{bid}")
def api_get_build(bid: str):
    """Get a specific build result."""
    if bid not in _store.history:
        return JSONResponse({"error": "not found"}, status_code=404)
    return _store.history[bid]


@app.get("/api/build/{bid}/file")
def api_get_build_file(bid: str):
    """Return the original uploaded file for a build."""
    build = _store.history.get(bid)
    if not build:
        return JSONResponse(
            {"error": {"code": "BUILD_NOT_FOUND", "message": "Build not found."}},
            status_code=404,
        )

    if not build.get("has_original_file") or not build.get("storage_key"):
        return JSONResponse(
            {
                "error": {
                    "code": "ORIGINAL_FILE_NOT_FOUND",
                    "message": "Original file is not available for this build.",
                }
            },
            status_code=404,
        )

    try:
        data, meta = _storage.open_original(build["storage_key"])
    except FileNotFoundError:
        return JSONResponse(
            {
                "error": {
                    "code": "ORIGINAL_FILE_NOT_FOUND",
                    "message": "Original file is not available for this build.",
                }
            },
            status_code=404,
        )

    ct = build.get("content_type") or meta.get(
        "content_type", "application/octet-stream"
    )
    filename = build.get("filename", "file")
    encoded_filename = quote(filename)
    headers = {
        "Content-Disposition": (
            f"inline; filename*=UTF-8''{encoded_filename}; filename=\"file\""
        ),
        "Cache-Control": "private, max-age=300",
        "ETag": f'"sha256-{build.get("sha256", "")}"',
    }
    return Response(content=data, media_type=ct, headers=headers)


@app.delete("/api/build/{bid}")
def api_delete_build(bid: str):
    """Delete a build from memory + SQLite + disk."""
    build = _store.history.pop(bid, None)
    if not build:
        build = _store.load_build(bid)
    if build and build.get("storage_key"):
        try:
            _storage.delete_original(build["storage_key"])
        except Exception:
            pass
    try:
        from src.tools import unregister

        unregister(bid)
    except Exception:  # noqa: BLE001
        pass
    _store.delete_build(bid)
    return {"ok": True}


# ---------------------------------------------------------------------------
# Chat API
# ---------------------------------------------------------------------------


@app.post("/api/chat")
async def api_chat(payload: dict):
    """Run a pagent agent over a previously-built document and stream events.

    Request body: ``{"bid": "<build id>", "question": "...", "model": "deepseek-chat"}``

    Response body: ``text/event-stream`` of NDJSON lines, each with a
    ``{"type": ...}`` envelope. See :mod:`src.chat.agent` for the full
    event schema (``start``, ``text``, ``reasoning``, ``tool_call``,
    ``tool_result``, ``done``, ``error``).
    """

    from src.chat import build_streamer

    bid = (payload.get("bid") or "").strip()
    question = (payload.get("question") or "").strip()
    model = (payload.get("model") or "deepseek-chat").strip()
    session_id = (payload.get("session_id") or "").strip() or None

    if not bid:
        return JSONResponse({"error": "field 'bid' is required"}, status_code=400)
    if not question:
        return JSONResponse({"error": "field 'question' is required"}, status_code=400)

    # Ensure the session exists (create new or validate existing).
    sid = _store.ensure_session(session_id, bid, model, title=question[:120])
    history = _store.session_turns(sid)

    return StreamingResponse(
        build_streamer(
            bid=bid, question=question, model=model, session_id=sid, history=history
        ),
        media_type="text/event-stream",
    )


# ---------------------------------------------------------------------------
# Agent tools API
# ---------------------------------------------------------------------------


@app.get("/api/trees")
def api_list_trees():
    from src.tools import list_trees

    return list_trees()


@app.get("/api/forest")
def api_forest_catalog():
    from src.tools import forest_catalog

    return forest_catalog()


@app.get("/api/forest/search")
def api_forest_search(q: str, limit: int = 8):
    from src.tools import find_sections, find_trees

    return {
        "trees": find_trees(q, limit=min(limit, 20)),
        "sections": find_sections(q, limit=min(limit, 20)),
    }


@app.get("/api/trees/{tree_id}")
def api_overview(tree_id: str):
    from src.tools import overview

    result = overview(tree_id)
    _store.log_query("overview", tree_id, "", result)
    return result


@app.get("/api/trees/{tree_id}/nodes/{path:path}")
def api_inspect(tree_id: str, path: str):
    from src.tools import inspect

    result = inspect(tree_id, path)
    _store.log_query("inspect", tree_id, path, result)
    return result


@app.get("/api/trees/{tree_id}/children/{path:path}")
def api_get_children(tree_id: str, path: str):
    from src.tools import get_children

    result = get_children(tree_id, path)
    _store.log_query("get_children", tree_id, path, result)
    return result


@app.get("/api/queries")
def api_query_history():
    return _store.recent_queries(200)


@app.get("/api/queries/stats")
def api_query_stats():
    from collections import Counter

    items = _store.recent_queries(200)
    tools = Counter(q["tool"] for q in items)
    trees = Counter(q["tree_id"] for q in items)
    return {
        "total": len(items),
        "by_tool": dict(tools.most_common()),
        "by_tree": dict(trees.most_common()),
        "recent": items[:20],
    }


# ---------------------------------------------------------------------------
# Chat sessions
# ---------------------------------------------------------------------------


@app.get("/api/sessions")
def api_list_sessions(bid: str | None = None, limit: int = 100):
    """List chat sessions, optionally filtered by ``bid``."""
    return {"sessions": _store.list_sessions(bid=bid, limit=limit)}


@app.get("/api/sessions/{sid}/turns")
def api_session_turns(sid: str, limit: int = 200):
    """Return the turns of a chat session in chronological order."""
    return {"session_id": sid, "turns": _store.session_turns(sid, limit=limit)}


@app.delete("/api/sessions/{sid}")
def api_delete_session(sid: str):
    """Delete a chat session and all its turns."""
    ok = _store.delete_session(sid)
    return {"deleted": ok, "session_id": sid}


# ---------------------------------------------------------------------------
# OpenAPI spec + Swagger UI / Redoc
# ---------------------------------------------------------------------------

_OPENAPI_PATH = Path(__file__).resolve().parent.parent.parent / "openapi.yaml"
_openapi_yaml_cache: bytes | None = None


def _openapi_yaml() -> bytes:
    """Load the OpenAPI spec from the project root (cached in memory)."""
    global _openapi_yaml_cache
    if _openapi_yaml_cache is None:
        _openapi_yaml_cache = _OPENAPI_PATH.read_bytes()
    return _openapi_yaml_cache


@app.get("/openapi.yaml", include_in_schema=False)
def openapi_yaml():
    return Response(
        content=_openapi_yaml(),
        media_type="application/x-yaml",
    )


@app.get("/docs", response_class=HTMLResponse, include_in_schema=False)
def swagger_ui():
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>treefyit — Swagger UI</title>
<link
  rel="stylesheet"
  href="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/swagger-ui.css"
/>
</head>
<body style="margin:0">
<div id="swagger-ui"></div>
<script
  src="https://cdn.jsdelivr.net/npm/swagger-ui-dist@5.17.14/swagger-ui-bundle.js"
></script>
<script>
  window.onload = function() {{
    SwaggerUIBundle({{
      url: "/openapi.yaml",
      dom_id: "#swagger-ui",
      deepLinking: true,
      presets: [SwaggerUIBundle.presets.apis],
    }});
  }};
</script>
</body>
</html>"""
    return HTMLResponse(content=html)


@app.get("/redoc", response_class=HTMLResponse, include_in_schema=False)
def redoc():
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>treefyit — Redoc</title>
</head>
<body style="margin:0">
<redoc spec-url="/openapi.yaml"></redoc>
<script src="https://cdn.jsdelivr.net/npm/redoc@2.1.5/bundles/redoc.standalone.js"></script>
</body>
</html>"""
    return HTMLResponse(content=html)


# ---------------------------------------------------------------------------
# Start
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8765)
