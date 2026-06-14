"""SQLite connection management and schema.

Kept deliberately small — all business queries live in :mod:`src.store.builds`.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from . import _ROOT  # results/ directory, resolved at package import

_DB_PATH = _ROOT / "data.sqlite"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS builds (
    id          TEXT PRIMARY KEY,
    filename    TEXT NOT NULL,
    content_type TEXT,
    file_size   INTEGER,
    sha256      TEXT,
    storage_key TEXT,
    has_original_file INTEGER DEFAULT 0,
    cache_key   TEXT,
    stats_json  TEXT,
    created_at  TEXT NOT NULL,
    is_cached   INTEGER DEFAULT 0,
    error       TEXT
);

CREATE TABLE IF NOT EXISTS queries (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    tool        TEXT NOT NULL,
    tree_id     TEXT NOT NULL,
    path        TEXT,
    summary     TEXT,
    result_json TEXT,
    timestamp   TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_builds_created ON builds (created_at DESC);
CREATE INDEX IF NOT EXISTS idx_queries_time  ON queries (timestamp DESC);
CREATE INDEX IF NOT EXISTS idx_queries_tree  ON queries (tree_id);

CREATE TABLE IF NOT EXISTS chat_sessions (
    id          TEXT PRIMARY KEY,
    bid         TEXT NOT NULL,
    model       TEXT NOT NULL,
    title       TEXT,
    turn_count  INTEGER DEFAULT 0,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_turns (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL,
    turn_idx    INTEGER NOT NULL,
    role        TEXT NOT NULL,          -- 'user' or 'assistant'
    text        TEXT,
    tool_calls  TEXT,                   -- JSON array of {name, arguments, id}
    tool_results TEXT,                  -- JSON array of {id, name, content, ok}
    created_at  TEXT NOT NULL,
    FOREIGN KEY (session_id) REFERENCES chat_sessions(id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_chat_sessions_updated ON chat_sessions (updated_at DESC);
CREATE INDEX IF NOT EXISTS idx_chat_sessions_bid     ON chat_sessions (bid);
CREATE INDEX IF NOT EXISTS idx_chat_turns_session    ON chat_turns (session_id, turn_idx);
"""

_initialized = False


def db_path() -> Path:
    return _DB_PATH


def connect() -> sqlite3.Connection:
    """Return a fresh SQLite connection (row-factory enabled, WAL mode)."""
    conn = sqlite3.connect(str(_DB_PATH))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


def init() -> None:
    """Create tables on first call; no-op thereafter (thread-safe enough for dev).

    Also migrates the ``builds`` table by adding any missing columns so that
    existing databases keep working after schema changes.
    """
    global _initialized
    if _initialized:
        return
    _ROOT.mkdir(parents=True, exist_ok=True)
    with connect() as conn:
        conn.executescript(_SCHEMA)
        # ---- lightweight migration: add missing columns to builds ----
        existing = {
            row[1]
            for row in conn.execute("PRAGMA table_info(builds)").fetchall()
        }
        _BUILD_MIGRATIONS = [
            ("content_type", "TEXT"),
            ("file_size", "INTEGER"),
            ("sha256", "TEXT"),
            ("storage_key", "TEXT"),
            ("has_original_file", "INTEGER DEFAULT 0"),
        ]
        for col, dtype in _BUILD_MIGRATIONS:
            if col not in existing:
                conn.execute(f"ALTER TABLE builds ADD COLUMN {col} {dtype}")
        conn.commit()
    _initialized = True
