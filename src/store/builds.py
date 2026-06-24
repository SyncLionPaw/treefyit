"""Build persistence: SQLite index rows + per-build JSON payloads on disk."""

from __future__ import annotations

import json
import time
from pathlib import Path

from src.tree.model import from_wire_tree, to_wire_tree

from . import _ROOT
from .sqlite import connect


# ---- builds table -----------------------------------------------------------

def list_builds(limit: int = 200) -> list[dict]:
    """Return recent builds as plain dicts (from SQLite; no ``tree`` payload)."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT id, filename, content_type, file_size, sha256, storage_key, "
            "has_original_file, cache_key, stats_json, created_at, is_cached, error "
            "FROM builds ORDER BY created_at DESC LIMIT ?",
            (limit,),
        ).fetchall()
    out: list[dict] = []
    for row in rows:
        item = dict(row)
        # stats_json was stored as a JSON string; parse for the caller
        if item.get("stats_json"):
            try:
                item["stats"] = json.loads(item["stats_json"])
            except json.JSONDecodeError:
                item["stats"] = {}
        item.pop("stats_json", None)
        item["cached"] = bool(item.pop("is_cached", 0))
        item["has_original_file"] = bool(item.pop("has_original_file", 0))
        if item.get("file_size") is not None:
            item["file_size"] = int(item["file_size"])
        out.append(item)
    return out


def save_build(bid: str, result: dict, cache_key: str | None = None) -> None:
    """Persist a build: SQLite metadata row + JSON payload file.

    ``result`` is the same dict the server hands back to the UI.  It should
    contain at least ``filename`` and ``created_at``; ``tree``/``raw_text``/
    ``mermaid``/``stats`` are optional (the JSON file keeps the full payload).
    """
    stats = json.dumps(result.get("stats", {}), ensure_ascii=False)
    err = result.get("error")
    is_cached = 1 if result.get("cached") else 0
    has_original = 1 if result.get("has_original_file") else 0

    with connect() as conn:
        conn.execute(
            """
            INSERT INTO builds (
                id, filename, content_type, file_size, sha256, storage_key,
                has_original_file, cache_key, stats_json, created_at, is_cached, error
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                filename          = excluded.filename,
                content_type      = excluded.content_type,
                file_size         = excluded.file_size,
                sha256            = excluded.sha256,
                storage_key       = excluded.storage_key,
                has_original_file = excluded.has_original_file,
                cache_key         = excluded.cache_key,
                stats_json        = excluded.stats_json,
                is_cached         = excluded.is_cached,
                error             = excluded.error
            """,
            (
                bid,
                result["filename"],
                result.get("content_type"),
                result.get("file_size"),
                result.get("sha256"),
                result.get("storage_key"),
                has_original,
                cache_key,
                stats,
                result["created_at"],
                is_cached,
                err,
            ),
        )
        conn.commit()

    # Full payload on disk — includes tree, raw_text, mermaid etc.
    full = {
        "id": bid,
        "filename": result["filename"],
        "raw_text": result.get("raw_text", ""),
        "mermaid": result.get("mermaid", ""),
        "tree": to_wire_tree(result.get("tree", [])),
        "stats": result.get("stats", {}),
        "created_at": result["created_at"],
        "cached": result.get("cached", False),
    }
    # Mirror original-file metadata into the JSON payload too
    for k in ("content_type", "file_size", "sha256", "storage_key", "has_original_file", "original_file_url"):
        if k in result:
            full[k] = result[k]
    if err:
        full["error"] = err

    _ROOT.mkdir(parents=True, exist_ok=True)
    (_ROOT / f"build_{bid}.json").write_text(
        json.dumps(full, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_build(bid: str) -> dict | None:
    """Read a build's full payload from its JSON file."""
    p = _ROOT / f"build_{bid}.json"
    if not p.exists():
        return None
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None
    if "tree" in data:
        data["tree"] = from_wire_tree(data.get("tree", []))
    return data


def delete_build(bid: str) -> None:
    """Remove a build from SQLite and delete its JSON payload file."""
    with connect() as conn:
        conn.execute("DELETE FROM builds WHERE id = ?", (bid,))
        conn.commit()
    p = _ROOT / f"build_{bid}.json"
    if p.exists():
        try:
            p.unlink()
        except OSError:
            pass


# ---- Start-up hydration -----------------------------------------------------

def rebuild_history(history: dict[str, dict], register_fn) -> int:
    """Load persisted builds into ``history`` and call ``register_fn(id, tree)``.

    Returns the number of builds rehydrated.  ``register_fn`` is the tool
    registry hook (``src.tools.register``) so queries work after a restart.
    """
    rows = list_builds(200)
    for row in rows:
        data = load_build(row["id"])
        if not data:
            continue
        history[data["id"]] = data
        if data.get("tree"):
            try:
                stats = data.get("stats") or {}
                register_fn(
                    data["id"],
                    data["tree"],
                    filename=data.get("filename", ""),
                    doc_kind=stats.get("doc_kind", ""),
                )
            except Exception:  # noqa: BLE001 — best-effort; don't crash startup
                pass
    return len(history)


# ---- queries table ----------------------------------------------------------

def _summarize(result: dict) -> str:
    if result.get("error"):
        return f"error: {result['error']}"
    if "title" in result:
        return f"node: {result['title']} ({result.get('children_count', 0)} children)"
    if "roots" in result:
        return f"tree: {result.get('node_count', 0)} nodes, depth {result.get('max_depth', 0)}"
    return "ok"


def log_query(tool: str, tree_id: str, path: str, result: dict) -> None:
    """Append a query-log row.  The table is pruned to the most recent 200."""
    summary = _summarize(result)
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    snippet = json.dumps(result, ensure_ascii=False)[:2000]
    with connect() as conn:
        conn.execute(
            "INSERT INTO queries (tool, tree_id, path, summary, result_json, timestamp) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (tool, tree_id, path, summary, snippet, ts),
        )
        conn.execute(
            "DELETE FROM queries WHERE id NOT IN "
            "(SELECT id FROM queries ORDER BY id DESC LIMIT 200)"
        )
        conn.commit()


def recent_queries(limit: int = 200) -> list[dict]:
    """Return recent query rows as plain dicts (no heavy result payload)."""
    with connect() as conn:
        rows = conn.execute(
            "SELECT tool, tree_id, path, summary, timestamp "
            "FROM queries ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
    return [dict(r) for r in rows]
