from treefyit.chat.session import (
    ChatSessionService,
    InMemoryChatSessionStorage,
    JsonChatSessionStorage,
    SqliteChatSessionStorage,
    normalize_assistant_events,
)


def test_chat_session_service_appends_turns_with_memory_storage():
    service = ChatSessionService(InMemoryChatSessionStorage())

    session = service.get_or_create(
        session_id=None,
        title="hello",
    )
    updated = service.append_turn(
        session.session_id,
        question="question",
        answer="answer",
        tool_calls=[{"name": "document_overview"}],
        tool_results=[{"ok": True, "content": "result"}],
        assistant_events=[
            {"type": "tool_call", "name": "document_overview"},
            {"type": "tool_result", "name": "document_overview", "ok": True},
            {"type": "text", "text": "answer"},
        ],
    )

    assert updated.session_id == session.session_id
    assert [turn.role for turn in updated.turns] == ["user", "assistant"]
    assert updated.turns[0].content == "question"
    assert updated.turns[1].content == "answer"
    assert updated.turns[1].tool_calls == [{"name": "document_overview"}]
    assert updated.turns[1].assistant_events[0]["type"] == "tool_call"


def test_json_chat_session_storage_round_trips(tmp_path):
    service = ChatSessionService(JsonChatSessionStorage(tmp_path / "sessions"))
    session = service.get_or_create(
        session_id="sid-json",
        title="json",
    )
    service.append_turn(
        session.session_id,
        question="q",
        answer="a",
        tool_calls=[],
        tool_results=[],
        assistant_events=[{"type": "text", "text": "a"}],
    )

    reloaded = ChatSessionService(JsonChatSessionStorage(tmp_path / "sessions"))
    sessions = reloaded.list()

    assert len(sessions) == 1
    assert sessions[0].session_id == "sid-json"
    assert [turn.content for turn in sessions[0].turns] == ["q", "a"]
    assert sessions[0].summary()["turn_count"] == 1
    assert "tree_id" not in sessions[0].summary()


def test_sqlite_chat_session_storage_round_trips(tmp_path):
    service = ChatSessionService(SqliteChatSessionStorage(tmp_path / "chat.sqlite3"))
    session = service.get_or_create(
        session_id="sid-sqlite",
        title="sqlite",
    )
    service.append_turn(
        session.session_id,
        question="q",
        answer="a",
        tool_calls=[],
        tool_results=[],
        assistant_events=[{"type": "text", "text": "a"}],
    )

    reloaded = ChatSessionService(SqliteChatSessionStorage(tmp_path / "chat.sqlite3"))
    turns = reloaded.get_turns("sid-sqlite")

    assert turns is not None
    assert [turn.content for turn in turns] == ["q", "a"]
    assert reloaded.delete("sid-sqlite") is True
    assert reloaded.get_turns("sid-sqlite") is None


def test_normalize_assistant_events_aggregates_chunks_and_assigns_seq():
    events = normalize_assistant_events(
        [
            {"type": "reasoning", "text": "A"},
            {"type": "reasoning", "text": "B"},
            {
                "type": "tool_call",
                "id": "tc-0",
                "name": "forest_catalog",
                "arguments": "{}",
            },
            {
                "type": "tool_result",
                "id": "tc-0",
                "name": "forest_catalog",
                "ok": True,
                "content": "ok",
            },
            {"type": "text", "text": "Hello"},
            {"type": "text", "text": " world"},
        ]
    )

    assert events == [
        {"type": "reasoning", "text": "AB", "seq": 1},
        {
            "type": "tool_call",
            "id": "tc-0",
            "name": "forest_catalog",
            "arguments": "{}",
            "seq": 2,
        },
        {
            "type": "tool_result",
            "id": "tc-0",
            "name": "forest_catalog",
            "ok": True,
            "content": "ok",
            "seq": 3,
        },
        {"type": "text", "text": "Hello world", "seq": 4},
    ]
