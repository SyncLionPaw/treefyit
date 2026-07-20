from __future__ import annotations

from pathlib import Path

from core import TreeSession, TreeStore, build_tree_tools, create_node, get_node
from pagentv4 import FunctionTool, ToolOutput, to_openai_tools


def _call(tool: FunctionTool, **kwargs) -> ToolOutput:
    return tool.call(kwargs)


def _text(tool: FunctionTool, **kwargs) -> str:
    result = _call(tool, **kwargs)
    assert result.ok, result.content
    return result.content


def _tool_map(tools: list[FunctionTool]) -> dict[str, FunctionTool]:
    return {tool.name: tool for tool in tools}


def test_build_tree_tools_are_pagentv4_function_tools(tmp_path: Path):
    store = TreeStore(tmp_path / "lib")
    session, tools = build_tree_tools(create_node("r", "Root"), store=store)
    assert all(isinstance(tool, FunctionTool) for tool in tools)
    names = {item["function"]["name"] for item in to_openai_tools(tools)}
    assert {
        "view_outline",
        "create_child",
        "seed_from_markdown",
        "save_tree",
        "load_tree",
        "list_saved_trees",
        "search_working_tree",
        "search_library",
    }.issubset(names)
    assert session.store is store


def test_build_tree_tools_crud_flow():
    root = create_node("r", "Root")
    session, tools = build_tree_tools(root)
    by_name = _tool_map(tools)

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


def test_seed_save_load_search_library_roundtrip(tmp_path: Path):
    md = tmp_path / "tea.md"
    md.write_text(
        "# 白茶\n\n简介。\n\n## 冲泡\n\n水温适宜。\n",
        encoding="utf-8",
    )
    store = TreeStore(tmp_path / "lib")
    session, tools = build_tree_tools(
        create_node("root", "tea"),
        store=store,
        source_md_path=md,
    )
    by_name = _tool_map(tools)

    seeded = _text(by_name["seed_from_markdown"])
    assert "白茶" in seeded
    assert session.root.children[0].title == "白茶"

    saved = _text(by_name["save_tree"], tree_id="tea-1")
    assert "tree_id=tea-1" in saved
    assert store.exists("tea-1")

    # mutate away then reload
    session.root = create_node("root", "empty")
    loaded = _text(by_name["load_tree"], tree_id="tea-1")
    assert "白茶" in loaded
    assert session.root.children[0].title == "白茶"

    listed = _text(by_name["list_saved_trees"])
    assert "tea-1" in listed

    hits = _text(by_name["search_library"], query="冲泡")
    assert "tea-1" in hits
    assert "冲泡" in hits


def test_tools_return_tooloutput_fail_instead_of_raising():
    session = TreeSession(root=create_node("r", "Root"))
    by_name = _tool_map(session.build_tools())

    err = by_name["view_outline"].call({"node_id": "missing"})
    assert isinstance(err, ToolOutput)
    assert err.ok is False
    assert "node not found" in err.content

    err = by_name["save_tree"].call({})
    assert err.ok is False
    assert "no tree store" in err.content


def test_session_undo_restores_previous_root():
    session = TreeSession(root=create_node("r", "Root"))
    by_name = _tool_map(session.build_tools())

    _text(by_name["create_child"], parent_id="r", id="c1", title="Child")
    assert len(session.root.children) == 1
    assert session.undo() is True
    assert session.root.children == []
    assert session.undo() is False
