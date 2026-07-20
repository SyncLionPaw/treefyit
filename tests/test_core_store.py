from __future__ import annotations

from pathlib import Path

import pytest

from core import TreeStore, create_node, markdown_to_tree, mount_node


def test_tree_store_save_load_list_delete(tmp_path: Path):
    store = TreeStore(tmp_path / "lib")
    root = create_node("root", "Doc")
    root = mount_node(root, "root", create_node("a", "A", content="hello"))

    record = store.save(root, tree_id="doc-1", title="Doc One")
    assert record.tree_id == "doc-1"
    assert record.node_count == 2
    assert store.exists("doc-1")

    loaded = store.load("doc-1")
    assert loaded.root == root
    assert loaded.title == "Doc One"

    listed = store.list()
    assert listed == [
        {
            "tree_id": "doc-1",
            "title": "Doc One",
            "source_path": None,
            "updated_at": record.updated_at,
            "node_count": 2,
        }
    ]

    assert store.delete("doc-1") is True
    assert store.exists("doc-1") is False
    with pytest.raises(KeyError):
        store.load("doc-1")


def test_tree_store_records_source_hash(tmp_path: Path):
    md = tmp_path / "paper.md"
    md.write_text("# Hi\n\nBody\n", encoding="utf-8")
    store = TreeStore(tmp_path / "lib")
    root = markdown_to_tree(md.read_text(encoding="utf-8"), root_title="paper")
    record = store.save(root, source_path=md)
    assert record.source_path == str(md)
    assert record.source_sha256
    assert len(record.source_sha256) == 64
