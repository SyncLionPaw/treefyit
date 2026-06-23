"""HTTP API smoke tests — no external LLM calls."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

TESTFILE = Path(__file__).resolve().parent / "testfile" / "short.md"


@pytest.fixture
def verify_ok():
    with patch(
        "src.tree.pipeline.verify_tree",
        new_callable=AsyncMock,
        return_value={
            "ok": True,
            "score": 1.0,
            "issues": [],
            "suspicious_nodes": [],
        },
    ):
        yield


def test_health_openapi(api_client):
    resp = api_client.get("/openapi.yaml")
    assert resp.status_code == 200
    assert "openapi:" in resp.text


def test_list_trees_empty(api_client):
    resp = api_client.get("/api/trees")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_forest_catalog_empty(api_client):
    resp = api_client.get("/api/forest")
    assert resp.status_code == 200
    data = resp.json()
    assert "tree_count" in data
    assert "trees" in data


def test_forest_search_empty_query(api_client):
    resp = api_client.get("/api/forest/search", params={"q": ""})
    assert resp.status_code == 200
    data = resp.json()
    assert data["trees"]["error"] == "empty query"
    assert data["sections"]["error"] == "empty query"


def test_build_md_no_llm(api_client, verify_ok):
    with open(TESTFILE, "rb") as fh:
        resp = api_client.post(
            "/api/build",
            files={"file": ("short.md", fh, "text/markdown")},
            data={"mode": "md", "summarize": "false"},
        )
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("error") is None
    assert len(body.get("tree", [])) > 0
    assert body["id"] in {t["tree_id"] for t in api_client.get("/api/trees").json()}


def test_build_parse_error_returns_json(api_client):
    resp = api_client.post(
        "/api/build",
        files={"file": ("broken.pdf", b"not-a-pdf", "application/pdf")},
        data={"mode": "md", "summarize": "false"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body.get("error")
    assert body.get("tree") == []


def test_get_build_not_found(api_client):
    resp = api_client.get("/api/build/deadbeef")
    assert resp.status_code == 404


def test_chat_missing_fields(api_client):
    resp = api_client.post("/api/chat", json={"bid": "x"})
    assert resp.status_code == 400
