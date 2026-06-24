from __future__ import annotations

from fastapi.testclient import TestClient

from treefyit.builder import build_tree_from_text
from treefyit.query.query import build_tree_index
from treefyit.server import create_app
from treefyit.store import RegistryStore


def test_registry_store_saves_and_loads_tree_and_index(tmp_path):
    store = RegistryStore(tmp_path)
    tree = build_tree_from_text("# Intro\n\nHello world.", filename="saved.md")
    tree.node_id = "tree-1"
    index = build_tree_index(tree)

    tree_path = store.save_tree(tree)
    index_path = store.save_index(index)

    assert tree_path.exists()
    assert index_path.exists()

    loaded_trees = store.load_trees()
    loaded_indexes = store.load_indexes(tree_ids={"tree-1"})

    assert loaded_trees["tree-1"].title == "saved.md"
    assert loaded_indexes["tree-1"].tree_title == "saved.md"


def test_registry_store_delete_bundle_removes_tree_and_index_files(tmp_path):
    store = RegistryStore(tmp_path)
    tree = build_tree_from_text("# Intro\n\nHello world.", filename="delete.md")
    tree.node_id = "tree-2"
    index = build_tree_index(tree)

    store.save_tree(tree)
    store.save_index(index)
    store.delete_bundle("tree-2")

    assert not store.tree_path("tree-2").exists()
    assert not store.index_path("tree-2").exists()


def test_create_app_restores_tree_and_index_registry_from_store(tmp_path):
    store = RegistryStore(tmp_path)
    first_app = create_app(store=store)
    first_client = TestClient(first_app)

    build_response = first_client.post(
        "/api/trees",
        json={
            "text": "# Intro\n\nHello world.\n\n## Detail\n\nMore text.",
            "filename": "persisted.md",
        },
    )
    tree_id = build_response.json()["tree_id"]
    first_client.post(f"/api/trees/{tree_id}/index")

    second_app = create_app(store=RegistryStore(tmp_path))
    second_client = TestClient(second_app)

    list_response = second_client.get("/api/trees")
    meta_response = second_client.get(f"/api/trees/{tree_id}/index/meta")

    assert list_response.status_code == 200
    assert list_response.json()[0]["tree_id"] == tree_id
    assert meta_response.status_code == 200
    assert meta_response.json()["tree_id"] == tree_id
