from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol
from uuid import uuid4

from pydantic import BaseModel, Field


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


class ChatTurn(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    created_at: str = Field(default_factory=now_iso)
    tool_calls: list[dict] = Field(default_factory=list)
    tool_results: list[dict] = Field(default_factory=list)
    assistant_events: list[dict] = Field(default_factory=list)


class ChatSession(BaseModel):
    session_id: str
    title: str
    created_at: str = Field(default_factory=now_iso)
    updated_at: str = Field(default_factory=now_iso)
    turns: list[ChatTurn] = Field(default_factory=list)

    def summary(self) -> dict:
        data = self.model_dump()
        data["turn_count"] = count_conversation_turns(self.turns)
        data.pop("turns", None)
        return data


class ChatSessionStorage(Protocol):
    def save(self, session: ChatSession) -> None: ...

    def get(self, session_id: str) -> ChatSession | None: ...

    def list(self) -> list[ChatSession]: ...

    def delete(self, session_id: str) -> bool: ...


def count_conversation_turns(turns: list[ChatTurn]) -> int:
    return sum(1 for turn in turns if turn.role == "assistant")


class InMemoryChatSessionStorage:
    def __init__(self, sessions: Iterable[ChatSession] | None = None) -> None:
        self.sessions = {session.session_id: session for session in sessions or []}

    def save(self, session: ChatSession) -> None:
        self.sessions[session.session_id] = session

    def get(self, session_id: str) -> ChatSession | None:
        return self.sessions.get(session_id)

    def list(self) -> list[ChatSession]:
        return list(self.sessions.values())

    def delete(self, session_id: str) -> bool:
        return self.sessions.pop(session_id, None) is not None


class JsonChatSessionStorage:
    def __init__(self, directory: str | Path) -> None:
        self.directory = Path(directory)

    def session_path(self, session_id: str) -> Path:
        return self.directory / f"{session_id}.json"

    def save(self, session: ChatSession) -> None:
        self.directory.mkdir(parents=True, exist_ok=True)
        write_json_atomically(
            self.session_path(session.session_id), session.model_dump()
        )

    def get(self, session_id: str) -> ChatSession | None:
        path = self.session_path(session_id)
        if not path.exists():
            return None
        return ChatSession.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[ChatSession]:
        if not self.directory.exists():
            return []
        return [
            ChatSession.model_validate_json(path.read_text(encoding="utf-8"))
            for path in sorted(self.directory.glob("*.json"))
        ]

    def delete(self, session_id: str) -> bool:
        path = self.session_path(session_id)
        if not path.exists():
            return False
        path.unlink()
        return True


class SqliteChatSessionStorage:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        return sqlite3.connect(self.database_path)

    def initialize(self) -> None:
        with self.connect() as conn:
            conn.execute(
                """
                create table if not exists chat_sessions (
                    session_id text primary key,
                    tree_id text,
                    title text not null,
                    created_at text not null,
                    updated_at text not null,
                    payload text not null
                )
                """
            )

    def save(self, session: ChatSession) -> None:
        payload = session.model_dump_json()
        with self.connect() as conn:
            conn.execute(
                """
                insert into chat_sessions (
                    session_id, title, created_at, updated_at, payload
                ) values (?, ?, ?, ?, ?)
                on conflict(session_id) do update set
                    title = excluded.title,
                    updated_at = excluded.updated_at,
                    payload = excluded.payload
                """,
                (
                    session.session_id,
                    session.title,
                    session.created_at,
                    session.updated_at,
                    payload,
                ),
            )

    def get(self, session_id: str) -> ChatSession | None:
        with self.connect() as conn:
            row = conn.execute(
                "select payload from chat_sessions where session_id = ?",
                (session_id,),
            ).fetchone()
        if row is None:
            return None
        return ChatSession.model_validate_json(row[0])

    def list(self) -> list[ChatSession]:
        with self.connect() as conn:
            rows = conn.execute(
                "select payload from chat_sessions order by updated_at desc"
            ).fetchall()
        return [ChatSession.model_validate_json(row[0]) for row in rows]

    def delete(self, session_id: str) -> bool:
        with self.connect() as conn:
            cursor = conn.execute(
                "delete from chat_sessions where session_id = ?",
                (session_id,),
            )
        return cursor.rowcount > 0


class ChatSessionService:
    def __init__(self, storage: ChatSessionStorage) -> None:
        self.storage = storage

    def get_or_create(
        self,
        *,
        session_id: str | None,
        title: str,
    ) -> ChatSession:
        if session_id is not None:
            existing = self.storage.get(session_id)
            if existing is not None:
                return existing

        session = ChatSession(
            session_id=session_id or uuid4().hex,
            title=title,
        )
        self.storage.save(session)
        return session

    def append_turn(
        self,
        session_id: str,
        *,
        question: str,
        answer: str,
        tool_calls: list[dict],
        tool_results: list[dict],
        assistant_events: list[dict],
    ) -> ChatSession:
        session = self.storage.get(session_id)
        if session is None:
            raise KeyError(f"unknown session_id: {session_id}")

        normalized_events = normalize_assistant_events(assistant_events)
        timestamp = now_iso()
        session.updated_at = timestamp
        session.turns.append(
            ChatTurn(role="user", content=question, created_at=timestamp)
        )
        session.turns.append(
            ChatTurn(
                role="assistant",
                content=answer,
                tool_calls=tool_calls,
                tool_results=tool_results,
                assistant_events=normalized_events,
                created_at=timestamp,
            )
        )
        self.storage.save(session)
        return session

    def list(self, *, limit: int = 100) -> list[ChatSession]:
        sessions = self.storage.list()
        sessions.sort(key=lambda session: session.updated_at, reverse=True)
        return sessions[:limit]

    def get_turns(self, session_id: str, *, limit: int = 200) -> list[ChatTurn] | None:
        session = self.storage.get(session_id)
        if session is None:
            return None
        return session.turns[-limit:]

    def delete(self, session_id: str) -> bool:
        return self.storage.delete(session_id)


def write_json_atomically(path: Path, data: dict) -> None:
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    tmp_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    tmp_path.replace(path)


def normalize_assistant_events(events: list[dict]) -> list[dict]:
    normalized: list[dict] = []

    for event in events:
        event_type = str(event.get("type") or "").strip()
        if event_type in {"text", "reasoning"}:
            text = str(event.get("text") or "")
            if not text:
                continue
            if normalized and normalized[-1]["type"] == event_type:
                normalized[-1]["text"] += text
                continue
            normalized.append({"type": event_type, "text": text})
            continue

        if event_type == "tool_call":
            normalized.append(
                {
                    "type": "tool_call",
                    "id": event.get("id"),
                    "name": event.get("name"),
                    "arguments": event.get("arguments"),
                }
            )
            continue

        if event_type == "tool_result":
            normalized.append(
                {
                    "type": "tool_result",
                    "id": event.get("id"),
                    "name": event.get("name"),
                    "ok": event.get("ok"),
                    "content": event.get("content"),
                }
            )

    for index, event in enumerate(normalized, start=1):
        event["seq"] = index

    return normalized


__all__ = [
    "ChatSession",
    "ChatSessionService",
    "ChatSessionStorage",
    "ChatTurn",
    "InMemoryChatSessionStorage",
    "JsonChatSessionStorage",
    "normalize_assistant_events",
    "SqliteChatSessionStorage",
]
