"""Persistent storage for the TreefyIt server.

Layout:
    results/data.sqlite    →  SQLite index (builds + queries metadata)
    results/*.json         →  Per-build full payloads (tree, raw_text, mermaid)
    results/cache_*.json   →  Input-content cache (deduplicate identical parses)

Public API (see individual docstrings for details):
    init()                  – one-time init (idempotent)
    connect()               – fresh SQLite connection
    history                 – in-memory mirror (dict[str, dict])
    rebuild_history(reg_fn) – hydrate ``history`` + call register(tree_id, tree)
    list_builds(limit)      – recent build rows (SQLite)
    save_build(bid, result, cache_key) – persist build (SQLite + JSON file)
    load_build(bid)         – read full build payload from JSON file
    delete_build(bid)       – remove from SQLite + JSON file
    cache_get(key)          – content-hash cache lookup
    cache_put(key, data)    – content-hash cache write
    log_query(tool, tree_id, path, result) – SQLite append (pruned to 200)
    recent_queries(limit)   – read recent query rows
"""

from __future__ import annotations

from pathlib import Path
import shutil

# ---- Resolve results/ directory once at package import ---------------------
# treefyit/src/store/__init__.py → three parents up = project root → /results
_ROOT = Path(__file__).resolve().parent.parent.parent / "results"

from . import builds as _builds  # noqa: E402  (needs _ROOT)
from . import cache as _cache  # noqa: E402
from . import chat as _chat  # noqa: E402
from . import sqlite as _sqlite  # noqa: E402

# ---- Stable, public API -----------------------------------------------------

init = _sqlite.init
connect = _sqlite.connect

rebuild_history = _builds.rebuild_history
list_builds = _builds.list_builds
save_build = _builds.save_build
load_build = _builds.load_build
delete_build = _builds.delete_build

cache_get = _cache.cache_get
cache_put = _cache.cache_put
cache_key_for = _cache.key_for

log_query = _builds.log_query
recent_queries = _builds.recent_queries

create_session = _chat.create_session
get_session = _chat.get_session
list_sessions = _chat.list_sessions
session_turns = _chat.session_turns
append_turn = _chat.append_turn
delete_session = _chat.delete_session
ensure_session = _chat.ensure_session

# In-memory mirror — server reads/writes this; call save_build() to persist.
history: dict[str, dict] = {}


def root_dir() -> Path:
    """Return the ``results/`` directory (created lazily if missing)."""
    _ROOT.mkdir(parents=True, exist_ok=True)
    return _ROOT


def clear_all() -> Path:
    """Clear all generated server state and return the cleared results path.

    This removes the SQLite database, WAL/SHM side files, build payload JSON,
    input caches, uploaded originals, and any other generated files under
    ``results/``.  The directory is recreated empty so normal startup can
    initialize a fresh database immediately after this call.
    """
    history.clear()
    if _ROOT.exists():
        shutil.rmtree(_ROOT)
    _ROOT.mkdir(parents=True, exist_ok=True)
    # If clear_all() is called in-process after init(), allow sqlite.init() to
    # recreate the schema on the next server import/use.
    try:
        _sqlite._initialized = False  # noqa: SLF001
    except Exception:  # noqa: BLE001
        pass
    return _ROOT


__all__ = [
    "init",
    "connect",
    "history",
    "root_dir",
    "clear_all",
    "rebuild_history",
    "list_builds",
    "save_build",
    "load_build",
    "delete_build",
    "cache_get",
    "cache_put",
    "cache_key_for",
    "log_query",
    "recent_queries",
]
