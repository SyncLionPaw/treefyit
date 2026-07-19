from __future__ import annotations

import pytest
from pydantic import ValidationError

from core import (
    NodeKind,
    TreeNode,
    collect_ids,
    create_node,
    get_by_path,
    get_node,
    mount_node,
    move_node,
    remove_node,
    update_node,
    view_node,
)


def sample_tree() -> TreeNode:
    root = create_node("r", "Root", summary="root")
    root = mount_node(root, "r", create_node("a", "A", kind=NodeKind.TEXT, content="aaa"))
    root = mount_node(root, "r", create_node("b", "B"))
    root = mount_node(root, "a", create_node("a1", "A1", content="nested"))
    return root


def test_create_get_view_and_path():
    root = sample_tree()
    assert get_node(root, "a1").title == "A1"
    assert get_by_path(root, "0.0").id == "a1"
    assert get_by_path(root, "root").id == "r"
    outline = view_node(root, "r", depth=2)
    assert 'id="a1"' in outline
    assert "path=0.0" in outline


def test_mount_rejects_duplicate_id():
    root = sample_tree()
    with pytest.raises(ValueError, match="duplicate node id"):
        mount_node(root, "b", create_node("a", "dup"))


def test_update_node_fields_and_clear():
    root = sample_tree()
    original = root
    root = update_node(root, "a", title="A2", content="bbb", kind=NodeKind.TABLE)
    assert original.children[0].title == "A"
    assert get_node(root, "a").title == "A2"
    assert get_node(root, "a").content == "bbb"
    assert get_node(root, "a").kind == NodeKind.TABLE

    root = update_node(root, "a", content=None, summary=None, kind=None)
    node = get_node(root, "a")
    assert node.content is None
    assert node.summary is None
    assert node.kind is None


def test_remove_and_cannot_remove_root():
    root = sample_tree()
    root = remove_node(root, "a")
    assert collect_ids(root) == {"r", "b"}
    with pytest.raises(KeyError):
        get_node(root, "a1")
    with pytest.raises(ValueError, match="cannot remove root"):
        remove_node(root, "r")


def test_move_node_and_guardrails():
    root = sample_tree()
    with pytest.raises(ValueError, match="descendant"):
        move_node(root, "a", "a1")

    root = move_node(root, "a1", "b")
    assert [c.id for c in get_node(root, "a").children] == []
    assert [c.id for c in get_node(root, "b").children] == ["a1"]

    with pytest.raises(ValueError, match="cannot move root"):
        move_node(root, "r", "b")


def test_move_with_index():
    root = sample_tree()
    root = mount_node(root, "r", create_node("c", "C"))
    root = move_node(root, "c", "r", index=0)
    assert [c.id for c in root.children] == ["c", "a", "b"]


def test_validation_rejects_empty_id():
    with pytest.raises(ValidationError):
        create_node("", "x")
