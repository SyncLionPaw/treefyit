from __future__ import annotations

from pathlib import Path

import pytest

from treefyit.tools import TreefyitClient, TreefyitToolError


class FakeResponse:
    def __init__(self, payload, *, ok: bool = True, status_code: int = 200, reason: str = "OK"):
        self.payload = payload
        self.ok = ok
        self.status_code = status_code
        self.reason = reason
        self.text = str(payload)

    def json(self):
        return self.payload


def test_build_knowledge_posts_file_and_summarize(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    captured: dict = {}

    def fake_post(url, files=None, data=None, timeout=None):
        captured["url"] = url
        captured["data"] = data
        captured["timeout"] = timeout
        captured["filename"] = files["file"][0]
        captured["content"] = files["file"][1].read()
        return FakeResponse({"tree_id": "tree-1"})

    monkeypatch.setattr("treefyit.tools.requests.post", fake_post)
    file_path = tmp_path / "paper.md"
    file_path.write_text("# Intro\n\nHello", encoding="utf-8")

    client = TreefyitClient(base_url="http://localhost:8765/")
    result = client.build_knowledge(file_path, summarize=False)

    assert result == {"tree_id": "tree-1"}
    assert captured["url"] == "http://localhost:8765/api/build"
    assert captured["data"] == {"summarize": "false"}
    assert captured["filename"] == "paper.md"
    assert captured["content"] == b"# Intro\n\nHello"


def test_build_knowledge_rejects_directory(tmp_path: Path):
    client = TreefyitClient()

    with pytest.raises(TreefyitToolError):
        client.build_knowledge(tmp_path)


def test_overview_forest_calls_expected_endpoint(monkeypatch: pytest.MonkeyPatch):
    captured: dict = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        captured["timeout"] = timeout
        return FakeResponse({"forest_id": "default"})

    monkeypatch.setattr("treefyit.tools.requests.get", fake_get)

    client = TreefyitClient()
    result = client.overview_forest()

    assert result == {"forest_id": "default"}
    assert captured["url"] == "http://127.0.0.1:8765/api/forest"
    assert captured["params"] is None


def test_client_uses_env_base_url_when_not_explicit(
    monkeypatch: pytest.MonkeyPatch,
):
    captured: dict = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        return FakeResponse({"forest_id": "remote"})

    monkeypatch.setenv("TREEFYIT_BASE_URL", "https://treefyit.example.com/")
    monkeypatch.setattr("treefyit.tools.requests.get", fake_get)

    client = TreefyitClient()
    result = client.overview_forest()

    assert result == {"forest_id": "remote"}
    assert captured["url"] == "https://treefyit.example.com/api/forest"


def test_explicit_base_url_overrides_env(monkeypatch: pytest.MonkeyPatch):
    captured: dict = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        return FakeResponse({"forest_id": "explicit"})

    monkeypatch.setenv("TREEFYIT_BASE_URL", "https://env.example.com")
    monkeypatch.setattr("treefyit.tools.requests.get", fake_get)

    client = TreefyitClient(base_url="https://explicit.example.com/")
    result = client.overview_forest()

    assert result == {"forest_id": "explicit"}
    assert captured["url"] == "https://explicit.example.com/api/forest"


def test_search_trees_calls_expected_endpoint(monkeypatch: pytest.MonkeyPatch):
    captured: dict = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        captured["params"] = params
        return FakeResponse([{"tree_id": "tree-1"}])

    monkeypatch.setattr("treefyit.tools.requests.get", fake_get)

    client = TreefyitClient()
    result = client.search_trees("agent memory", limit=3)

    assert result == [{"tree_id": "tree-1"}]
    assert captured["url"] == "http://127.0.0.1:8765/api/forest/search/trees"
    assert captured["params"] == {"q": "agent memory", "limit": 3}


def test_search_trees_returns_empty_list_for_blank_query():
    client = TreefyitClient()

    assert client.search_trees("   ") == []


@pytest.mark.parametrize(
    ("method_name", "expected_url"),
    [
        ("overview", "http://127.0.0.1:8765/api/trees/tree-1"),
        ("children", "http://127.0.0.1:8765/api/trees/tree-1/children/0.2"),
        ("inspect", "http://127.0.0.1:8765/api/trees/tree-1/nodes/0.2"),
    ],
)
def test_tree_navigation_tools_call_expected_endpoints(
    monkeypatch: pytest.MonkeyPatch,
    method_name: str,
    expected_url: str,
):
    captured: dict = {}

    def fake_get(url, params=None, timeout=None):
        captured["url"] = url
        return FakeResponse({"ok": True})

    monkeypatch.setattr("treefyit.tools.requests.get", fake_get)

    client = TreefyitClient()
    if method_name == "overview":
        result = client.overview("tree-1")
    else:
        result = getattr(client, method_name)("tree-1", "0.2")

    assert result == {"ok": True}
    assert captured["url"] == expected_url


def test_non_ok_response_raises_tool_error(monkeypatch: pytest.MonkeyPatch):
    def fake_get(url, params=None, timeout=None):
        return FakeResponse({"detail": "boom"}, ok=False, status_code=500, reason="Server Error")

    monkeypatch.setattr("treefyit.tools.requests.get", fake_get)

    client = TreefyitClient()
    with pytest.raises(TreefyitToolError):
        client.overview_forest()
