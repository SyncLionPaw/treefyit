"""Store persistence tests."""

from __future__ import annotations


def test_cache_roundtrip(isolated_store):
    store = isolated_store
    key = store.cache_key_for("hello", "gpt-4o", "md", False)
    payload = {"tree": [{"title": "A", "children": []}], "stats": {"node_count": 1}}
    store.cache_put(key, payload)
    assert store.cache_get(key) == payload


def test_save_and_load_build(isolated_store):
    store = isolated_store
    bid = "abc123"
    result = {
        "id": bid,
        "filename": "t.md",
        "raw_text": "# Hi",
        "mermaid": "",
        "tree": [{"title": "Hi", "text": "", "summary": "", "children": []}],
        "stats": {"node_count": 1},
        "created_at": "12:00:00",
    }
    store.save_build(bid, result, cache_key=None)
    loaded = store.load_build(bid)
    assert loaded is not None
    assert loaded["filename"] == "t.md"
    assert loaded["tree"][0]["title"] == "Hi"


def test_list_builds_after_save(isolated_store):
    store = isolated_store
    bid = "def456"
    result = {
        "id": bid,
        "filename": "x.md",
        "tree": [],
        "stats": {},
        "created_at": "12:01:00",
    }
    store.save_build(bid, result, cache_key=None)
    ids = {row["id"] for row in store.list_builds()}
    assert bid in ids
