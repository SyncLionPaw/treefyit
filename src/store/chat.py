"""Chat session / turn persistence.

Schema (see :mod:`src.store.sqlite`):
    ``chat_sessions(id, bid, model, title, turn_count, created_at, updated_at)``
    ``chat_turns(id, session_id, turn_idx, role, text, tool_calls, tool_results, created_at)``

The expected usage is:

1. A chat request comes in — call :func:`ensure_session` to either create a
   new session row or touch an existing one.
2. Read previous turns via :func:`session_turns` to replay them into the
   agent's ``Session``.
3. After the agent finishes, append the new user/assistant turns with
   :func:`append_turn` (``role='user'`` then ``role='assistant'``).

Turn text fields are stored verbatim — the caller decides how much to
truncate before passing it in.
"""

from __future__ import annotations

import json
import secrets
import time
from datetime import datetime

from .sqlite import connect, init as _sqlite_init


# ---- Public API -------------------------------------------------------------


def create_session(bid: str, model: str, title: str | None = None) -> str:
    """Create a new chat session and return its id."""
    _sqlite_init()
    now = _now()
    sid = "s_" + secrets.token_urlsafe(12)
    with connect() as conn:
        conn.execute(
            "INSERT INTO chat_sessions (id, bid, model, title, turn_count, created_at, updated_at) VALUES (?, ?, ?, ?, 0, ?, ?)",
            (sid, bid, model, title or now, now, now),
        )
        conn.commit()
    return sid


def get_session(sid: str) -> dict | None:
    """Return a session row as a dict (or ``None`` if missing)."""
    _sqlite_init()
    with connect() as conn:
        row = conn.execute(
            "SELECT id, bid, model, title, turn_count, created_at, updated_at "
            "FROM chat_sessions WHERE id = ?",
            (sid,),
        ).fetchone()
    return dict(row) if row else None


def list_sessions(bid: str | None = None, limit: int = 100) -> list[dict]:
    """Return recent sessions, optionally filtered by ``bid``."""
    _sqlite_init()
    if bid:
        with connect() as conn:
            rows = conn.execute(
                "SELECT id, bid, model, title, turn_count, created_at, updated_at "
                "FROM chat_sessions WHERE bid = ? ORDER BY updated_at DESC LIMIT ?",
                (bid, limit),
            ).fetchall()
    else:
        with connect() as conn:
            rows = conn.execute(
                "SELECT id, bid, model, title, turn_count, created_at, updated_at "
                "FROM chat_sessions ORDER BY updated_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
    return [dict(r) for r in rows]


def session_turns(sid: str, limit: int = 200) -> list[dict]:
    """Return the turns of a session in chronological order."""
    _sqlite_init()
    with connect() as conn:
        rows = conn.execute(
            "SELECT session_id, turn_idx, role, text, tool_calls, tool_results, created_at "
            "FROM chat_turns WHERE session_id = ? ORDER BY turn_idx ASC LIMIT ?",
            (sid, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def append_turn(
    sid: str,
    role: str,
    text: str | None = None,
    tool_calls: list[dict] | None = None,
    tool_results: list[dict] | None = None,
) -> int:
    """Append a turn to a session and bump its ``turn_count``/``updated_at``.

    Returns the new ``turn_idx``. Caller is responsible for alternating
    ``role='user'`` and ``role='assistant'`` — no enforcement here.
    """
    if role not in ("user", "assistant"):
        raise ValueError(f"role must be 'user' or 'assistant', got {role!r}")

    _sqlite_init()
    now = _now()
    tool_calls_json = json.dumps(tool_calls, ensure_ascii=False) if tool_calls else None
    tool_results_json = (
        json.dumps(tool_results, ensure_ascii=False) if tool_results else None
    )

    with connect() as conn:
        row = conn.execute(
            "SELECT COALESCE(MAX(turn_idx), -1) AS last_idx FROM chat_turns WHERE session_id = ?",
            (sid,),
        ).fetchone()
        turn_idx = (row["last_idx"] if row else -1) + 1

        conn.execute(
            "INSERT INTO chat_turns (session_id, turn_idx, role, text, tool_calls, tool_results, created_at) VALUES (?, ?, ?, ?, ?, ?, ?)",
            (sid, turn_idx, role, text, tool_calls_json, tool_results_json, now),
        )
        conn.execute(
            "UPDATE chat_sessions SET turn_count = turn_count + 1, updated_at = ? WHERE id = ?",
            (now, sid),
        )
        conn.commit()
    return turn_idx


def delete_session(sid: str) -> bool:
    """Delete a session and all its turns (cascade handled by FK)."""
    _sqlite_init()
    with connect() as conn:
        cursor = conn.execute("DELETE FROM chat_sessions WHERE id = ?", (sid,))
        conn.commit()
        return cursor.rowcount > 0


def ensure_session(
    sid: str | None, bid: str, model: str, title: str | None = None
) -> str:
    """Return an existing session id, or create a new one when ``sid`` is falsy.

    If ``sid`` is provided but the row doesn't exist, a new session is
    created with a fresh id — the caller should use the returned id in
    subsequent requests.
    """
    if sid:
        row = get_session(sid)
        if row is not None:
            # touch updated_at + model (model may have changed on client)
            now = _now()
            with connect() as conn:
                conn.execute(
                    "UPDATE chat_sessions SET model = ?, updated_at = ? WHERE id = ?",
                    (model, now, sid),
                )
                conn.commit()
            return sid
    return create_session(bid, model, title)


# ---- helpers ---------------------------------------------------------------


def _now() -> str:
    return datetime.utcfromtimestamp(time.time()).isoformat(timespec="seconds") + "Z"
