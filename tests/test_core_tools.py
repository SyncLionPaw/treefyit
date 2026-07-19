from __future__ import annotations

from core import TreeSession, build_tree_tools, create_node, get_node


def _text(tool, **kwargs) -> str:
    result = tool.call(kwargs)
    assert result.ok, result.content
    return result.content


def _tool_map(tools: list) -> dict[str, object]:
    return {tool.name: tool for tool in tools}


def test_build_tree_tools_crud_flow():
    root = create_node("r", "Root")
    session, tools = build_tree_tools(root)
    by_name = _tool_map(tools)

    assert set(by_name) == {
        "view_outline",
        "view_detail",
        "create_child",
        "update_fields",
        "delete_node",
        "relocate_node",
    }

    created = _text(
        by_name["create_child"],
        parent_id="r",
        id="c1",
        title="Child",
        kind="text",
        content="hello",
    )
    assert 'id="c1"' in created
    assert get_node(session.root, "c1").content == "hello"

    updated = _text(
        by_name["update_fields"],
        node_id="c1",
        title="Child2",
        content="world",
    )
    assert 'title="Child2"' in updated
    assert get_node(session.root, "c1").content == "world"

    _text(by_name["create_child"], parent_id="r", id="c2", title="Other")
    moved = _text(by_name["relocate_node"], node_id="c1", new_parent_id="c2")
    assert 'id="c1"' in moved
    assert [c.id for c in get_node(session.root, "c2").children] == ["c1"]

    deleted = _text(by_name["delete_node"], node_id="c1")
    assert "deleted c1" in deleted
    assert "c1" not in {n.id for n in session.root.children[0].children}


def test_tools_return_error_strings_instead_of_raising():
    session = TreeSession(root=create_node("r", "Root"))
    by_name = _tool_map(session.build_tools())

    err = by_name["view_outline"].call({"node_id": "missing"})
    assert err.ok
    assert err.content.startswith("error:")

    err = by_name["create_child"].call(
        {"parent_id": "r", "id": "r", "title": "dup"}
    )
    assert err.ok
    assert "duplicate" in err.content

    err = by_name["delete_node"].call({"node_id": "r"})
    assert err.ok
    assert "cannot remove root" in err.content


def test_session_undo_restores_previous_root():
    session = TreeSession(root=create_node("r", "Root"))
    by_name = _tool_map(session.build_tools())

    _text(by_name["create_child"], parent_id="r", id="c1", title="Child")
    assert len(session.root.children) == 1
    assert session.undo() is True
    assert session.root.children == []
    assert session.undo() is False
