from __future__ import annotations

from fastapi.testclient import TestClient

from treefyit.server import create_app


def test_health_endpoint_returns_ok():
    client = TestClient(create_app())

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "treefyit",
    }


def test_post_trees_builds_and_registers_tree():
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/api/trees",
        json={
            "text": "# Intro\n\nHello world.\n\n## Detail\n\nMore text.",
            "filename": "sample.md",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "sample.md"
    assert payload["title"] == "sample.md"
    assert payload["node_count"] == 3
    assert payload["max_depth"] == 2
    assert payload["root_count"] == 1
    assert payload["roots"][0]["path"] == "0"
    assert payload["roots"][0]["title"] == "Intro"

    tree_id = payload["tree_id"]
    assert tree_id in app.state.tree_registry
    assert app.state.tree_registry[tree_id].title == "sample.md"
    assert app.state.tree_registry[tree_id].node_id == tree_id


def test_post_build_keeps_old_path_compatible():
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/api/build",
        json={
            "text": "# Overview\n\nBody",
            "filename": "legacy.md",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "legacy.md"
    assert payload["title"] == "legacy.md"
    assert payload["root_count"] == 1
    assert payload["tree_id"] in app.state.tree_registry


def test_post_trees_from_file_builds_and_registers_tree():
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/api/trees/from-file",
        files={
            "file": (
                "upload.md",
                b"# Upload\n\nFile body\n\n## Section\n\nMore text.",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "upload.md"
    assert payload["title"] == "upload.md"
    assert payload["node_count"] == 3
    assert payload["max_depth"] == 2
    assert payload["tree_id"] in app.state.tree_registry


def test_post_build_accepts_multipart_for_legacy_compatibility():
    app = create_app()
    client = TestClient(app)

    response = client.post(
        "/api/build",
        files={
            "file": (
                "legacy-upload.md",
                b"# Legacy Upload\n\nBody",
                "text/markdown",
            )
        },
        data={
            "summarize": "false",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["filename"] == "legacy-upload.md"
    assert payload["title"] == "legacy-upload.md"
    assert payload["root_count"] == 1
    assert payload["tree_id"] in app.state.tree_registry


def test_post_tree_index_builds_and_registers_index():
    app = create_app()
    client = TestClient(app)
    build_response = client.post(
        "/api/trees",
        json={
            "text": "# Intro\n\nHello world.\n\n## Detail\n\nMore text.",
            "filename": "sample.md",
        },
    )
    tree_id = build_response.json()["tree_id"]

    response = client.post(f"/api/trees/{tree_id}/index")

    assert response.status_code == 200
    payload = response.json()
    assert payload["tree_id"] == tree_id
    assert payload["tree_title"] == "sample.md"
    assert payload["document_count"] == 3
    assert payload["term_count"] > 0
    assert tree_id in app.state.index_registry


def test_get_tree_index_meta_returns_existing_index_summary():
    app = create_app()
    client = TestClient(app)
    build_response = client.post(
        "/api/trees",
        json={
            "text": "# Intro\n\nHello world.",
            "filename": "meta.md",
        },
    )
    tree_id = build_response.json()["tree_id"]
    client.post(f"/api/trees/{tree_id}/index")

    response = client.get(f"/api/trees/{tree_id}/index/meta")

    assert response.status_code == 200
    payload = response.json()
    assert payload["tree_id"] == tree_id
    assert payload["tree_title"] == "meta.md"


def test_post_tree_index_supports_legacy_alias_and_overwrites_existing_index():
    app = create_app()
    client = TestClient(app)
    build_response = client.post(
        "/api/trees",
        json={
            "text": "# Intro\n\nHello world.",
            "filename": "legacy-index.md",
        },
    )
    tree_id = build_response.json()["tree_id"]

    first = client.post(f"/api/tree/{tree_id}/index")
    first_average = first.json()["average_document_length"]

    app.state.tree_registry[tree_id].children.append(
        app.state.tree_registry[tree_id].children[0].model_copy(deep=True)
    )
    second = client.post(f"/api/tree/{tree_id}/index")

    assert first.status_code == 200
    assert second.status_code == 200
    assert second.json()["document_count"] >= first.json()["document_count"]
    assert (
        app.state.index_registry[tree_id].corpus.average_document_length
        >= first_average
    )


def test_tree_index_endpoints_return_404_for_unknown_tree():
    client = TestClient(create_app())

    create_response = client.post("/api/trees/missing/index")
    meta_response = client.get("/api/trees/missing/index/meta")

    assert create_response.status_code == 404
    assert meta_response.status_code == 404


def test_search_nodes_returns_ranked_hits_from_existing_index():
    app = create_app()
    client = TestClient(app)
    build_response = client.post(
        "/api/trees",
        json={
            "text": "# Intro\n\nHello world.\n\n## Detail\n\nBM25 reranking section.",
            "filename": "search.md",
        },
    )
    tree_id = build_response.json()["tree_id"]
    client.post(f"/api/trees/{tree_id}/index")

    response = client.get(
        f"/api/trees/{tree_id}/search/nodes",
        params={"q": "reranking", "limit": 1},
    )

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 1
    assert payload[0]["tree_id"] == tree_id
    assert payload[0]["path"] == "0.0"
    assert payload[0]["score"] > 0


def test_search_nodes_returns_explicit_error_when_index_is_missing():
    app = create_app()
    client = TestClient(app)
    build_response = client.post(
        "/api/trees",
        json={
            "text": "# Intro\n\nHello world.",
            "filename": "no-index.md",
        },
    )
    tree_id = build_response.json()["tree_id"]

    response = client.get(
        f"/api/tree/{tree_id}/search/nodes",
        params={"q": "hello"},
    )

    assert response.status_code == 404
    assert response.json()["detail"] == f"index not found for tree_id: {tree_id}"


def test_get_tree_overview_returns_registered_tree_summary():
    app = create_app()
    client = TestClient(app)
    build_response = client.post(
        "/api/trees",
        json={
            "text": "# Intro\n\nHello world.\n\n## Detail\n\nMore text.",
            "filename": "browse.md",
        },
    )
    tree_id = build_response.json()["tree_id"]

    response = client.get(f"/api/tree/{tree_id}")

    assert response.status_code == 200
    payload = response.json()
    assert payload["tree_id"] == tree_id
    assert payload["title"] == "browse.md"
    assert payload["root_count"] == 1
    assert payload["roots"][0]["path"] == "0"


def test_get_tree_node_and_children_support_path_navigation():
    app = create_app()
    client = TestClient(app)
    build_response = client.post(
        "/api/trees",
        json={
            "text": "# Intro\n\nHello world.\n\n## Detail\n\nMore text.",
            "filename": "nodes.md",
        },
    )
    tree_id = build_response.json()["tree_id"]

    node_response = client.get(f"/api/trees/{tree_id}/nodes/0")
    children_response = client.get(f"/api/tree/{tree_id}/children/0")

    assert node_response.status_code == 200
    assert node_response.json()["title"] == "Intro"
    assert node_response.json()["children"] == ["0.0"]

    assert children_response.status_code == 200
    assert children_response.json()["path"] == "0"
    assert children_response.json()["children"][0]["path"] == "0.0"
    assert children_response.json()["children"][0]["title"] == "Detail"


def test_get_tree_node_returns_404_for_invalid_path():
    app = create_app()
    client = TestClient(app)
    build_response = client.post(
        "/api/trees",
        json={
            "text": "# Intro\n\nHello world.",
            "filename": "invalid-path.md",
        },
    )
    tree_id = build_response.json()["tree_id"]

    response = client.get(f"/api/trees/{tree_id}/nodes/9")

    assert response.status_code == 404
    assert response.json()["detail"] == "invalid path: 9"


def test_list_trees_returns_registered_tree_summaries():
    app = create_app()
    client = TestClient(app)
    client.post(
        "/api/trees",
        json={
            "text": "# First\n\nBody",
            "filename": "first.md",
        },
    )
    client.post(
        "/api/trees",
        json={
            "text": "# Second\n\nBody",
            "filename": "second.md",
        },
    )

    response = client.get("/api/trees")

    assert response.status_code == 200
    payload = response.json()
    assert len(payload) == 2
    assert {item["title"] for item in payload} == {"first.md", "second.md"}


def test_delete_tree_removes_tree_and_index():
    app = create_app()
    client = TestClient(app)
    build_response = client.post(
        "/api/trees",
        json={
            "text": "# Intro\n\nHello world.",
            "filename": "delete.md",
        },
    )
    tree_id = build_response.json()["tree_id"]
    client.post(f"/api/trees/{tree_id}/index")

    response = client.delete(f"/api/tree/{tree_id}")

    assert response.status_code == 200
    assert response.json() == {"ok": True, "tree_id": tree_id}
    assert tree_id not in app.state.tree_registry
    assert tree_id not in app.state.index_registry


def test_delete_tree_returns_404_for_missing_tree():
    client = TestClient(create_app())

    response = client.delete("/api/trees/missing")

    assert response.status_code == 404


def test_get_forest_returns_registered_tree_summary():
    app = create_app()
    client = TestClient(app)
    client.post(
        "/api/trees",
        json={
            "text": "# First\n\nBody",
            "filename": "first.md",
        },
    )
    client.post(
        "/api/trees",
        json={
            "text": "# Second\n\nBody",
            "filename": "second.md",
        },
    )

    response = client.get("/api/forest")

    assert response.status_code == 200
    payload = response.json()
    assert payload["forest_id"] == "default"
    assert payload["tree_count"] == 2
    assert {item["title"] for item in payload["trees"]} == {"first.md", "second.md"}


def test_search_forest_trees_returns_matching_tree():
    app = create_app()
    client = TestClient(app)
    client.post(
        "/api/trees",
        json={
            "text": "# Retrieval\n\nGeneral introduction.\n\n## Dense Retrieval\n\nThis section covers reranking and recall.",
            "filename": "retrieval.md",
        },
    )
    client.post(
        "/api/trees",
        json={
            "text": "# Vision\n\nImage parsing and table extraction.",
            "filename": "vision.md",
        },
    )

    response = client.get("/api/forest/search/trees", params={"q": "reranking recall"})

    assert response.status_code == 200
    payload = response.json()
    assert payload
    assert payload[0]["tree_title"] == "retrieval.md"


def test_search_forest_nodes_returns_matching_node():
    app = create_app()
    client = TestClient(app)
    client.post(
        "/api/trees",
        json={
            "text": "# Retrieval\n\nGeneral introduction.\n\n## Dense Retrieval\n\nThis section covers reranking and recall.",
            "filename": "retrieval.md",
        },
    )
    client.post(
        "/api/trees",
        json={
            "text": "# Vision\n\nImage parsing and table extraction.",
            "filename": "vision.md",
        },
    )

    response = client.get("/api/forest/search/nodes", params={"q": "table"})

    assert response.status_code == 200
    payload = response.json()
    assert payload
    assert payload[0]["tree_title"] == "vision.md"


def test_default_forest_syncs_after_tree_delete():
    app = create_app()
    client = TestClient(app)
    first = client.post(
        "/api/trees",
        json={
            "text": "# First\n\nBody",
            "filename": "first.md",
        },
    )
    client.post(
        "/api/trees",
        json={
            "text": "# Second\n\nBody",
            "filename": "second.md",
        },
    )

    tree_id = first.json()["tree_id"]
    client.delete(f"/api/trees/{tree_id}")
    response = client.get("/api/forest")

    assert response.status_code == 200
    payload = response.json()
    assert payload["tree_count"] == 1
    assert payload["trees"][0]["title"] == "second.md"


def test_history_and_build_detail_are_recorded_for_text_build():
    app = create_app()
    client = TestClient(app)
    build_response = client.post(
        "/api/trees",
        json={
            "text": "# History\n\nBody",
            "filename": "history.md",
        },
    )
    tree_id = build_response.json()["tree_id"]

    history_response = client.get("/api/history")
    detail_response = client.get(f"/api/build/{tree_id}")

    assert history_response.status_code == 200
    assert history_response.json()[0]["id"] == tree_id
    assert history_response.json()[0]["filename"] == "history.md"
    assert "raw_text" not in history_response.json()[0]

    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["id"] == tree_id
    assert detail["raw_text"] == "# History\n\nBody"
    assert detail["tree"]["title"] == "history.md"


def test_file_build_saves_original_file_and_build_delete_removes_tree():
    app = create_app()
    client = TestClient(app)
    build_response = client.post(
        "/api/trees/from-file",
        files={
            "file": (
                "original.md",
                b"# Original\n\nFile body",
                "text/markdown",
            )
        },
    )
    tree_id = build_response.json()["tree_id"]

    file_response = client.get(f"/api/build/{tree_id}/file")
    delete_response = client.delete(f"/api/build/{tree_id}")

    assert file_response.status_code == 200
    assert file_response.content == b"# Original\n\nFile body"
    assert file_response.headers["content-type"].startswith("text/markdown")

    assert delete_response.status_code == 200
    assert delete_response.json() == {"ok": True, "id": tree_id}
    assert tree_id not in app.state.tree_registry


def test_stream_build_returns_ndjson_events():
    client = TestClient(create_app())

    response = client.post(
        "/api/build/stream",
        files={
            "file": (
                "stream.md",
                b"# Stream\n\nBody",
                "text/markdown",
            )
        },
    )

    assert response.status_code == 200
    events = [line for line in response.text.splitlines() if line]
    assert '"type": "start"' in events[0]
    assert any('"type": "done"' in event for event in events)


def test_query_history_and_stats_are_recorded():
    app = create_app()
    client = TestClient(app)
    build_response = client.post(
        "/api/trees",
        json={
            "text": "# Query\n\nBody",
            "filename": "query.md",
        },
    )
    tree_id = build_response.json()["tree_id"]

    client.get(f"/api/trees/{tree_id}")
    client.get(f"/api/trees/{tree_id}/nodes/0")
    queries_response = client.get("/api/queries")
    stats_response = client.get("/api/queries/stats")

    assert queries_response.status_code == 200
    assert {item["tool"] for item in queries_response.json()} >= {"overview", "inspect"}

    assert stats_response.status_code == 200
    assert stats_response.json()["total"] >= 2
    assert stats_response.json()["by_tool"]["overview"] == 1


def test_chat_creates_session_and_streams_events(monkeypatch):
    from treefyit.chat import pagent

    class TextDelta:
        def __init__(self, text):
            self.text = text

    class RunEnd:
        def __init__(self, content):
            self.content = content
            self.usage = None

    class FakeAgent:
        initialized = False
        tool_count = 0

        def __init__(self, llm, session, tools=None, max_turns=8):
            self.llm = llm
            self.session = session
            self.tools = tools or []
            self.max_turns = max_turns
            FakeAgent.initialized = True
            FakeAgent.tool_count = len(self.tools)

        async def arun_events(self, question):
            yield TextDelta(f"answer from pagent: {question}")
            yield RunEnd(f"answer from pagent: {question}")

    monkeypatch.setattr(pagent, "Agent", FakeAgent)
    monkeypatch.setattr(pagent, "resolve_llm", lambda: object())

    app = create_app()
    client = TestClient(app)
    build_response = client.post(
        "/api/trees",
        json={
            "text": "# Chat\n\nAnswerable content",
            "filename": "chat.md",
        },
    )
    tree_id = build_response.json()["tree_id"]

    chat_response = client.post(
        "/api/chat",
        json={
            "bid": tree_id,
            "question": "content",
        },
    )

    assert chat_response.status_code == 200
    events = [line for line in chat_response.text.splitlines() if line]
    assert FakeAgent.initialized is True
    assert FakeAgent.tool_count > 0
    assert any('"type": "start"' in event for event in events)
    assert any("answer from pagent" in event for event in events)
    assert any('"type": "done"' in event for event in events)

    sessions_response = client.get("/api/sessions", params={"bid": tree_id})
    session_id = sessions_response.json()["sessions"][0]["session_id"]
    turns_response = client.get(f"/api/sessions/{session_id}/turns")
    delete_response = client.delete(f"/api/sessions/{session_id}")

    assert turns_response.status_code == 200
    assert len(turns_response.json()["turns"]) == 2
    assert delete_response.json() == {"deleted": True, "session_id": session_id}
