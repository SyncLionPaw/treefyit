from __future__ import annotations

import asyncio
from types import SimpleNamespace

from core import create_node, open_runner
from pagentv4 import FunctionTool
from pagentv4.tools import HARNESS_WEB_TOOLS


def test_open_runner_uses_local_backend_and_merges_tree_tools(monkeypatch):
    captured: dict = {}

    async def fake_create(thread_id, provider, **kwargs):
        captured["thread_id"] = thread_id
        captured["provider"] = provider
        captured["kwargs"] = kwargs
        return SimpleNamespace(thread_id=thread_id)

    monkeypatch.setattr("core.harness.Runner.create", fake_create)

    provider = object()
    root = create_node("r", "Root")
    session, runner = asyncio.run(
        open_runner(
            "tree-demo",
            provider,
            root=root,
            max_turns=4,
            tools=[],
        )
    )

    assert runner.thread_id == "tree-demo"
    assert session.root.id == "r"
    assert captured["thread_id"] == "tree-demo"
    assert captured["provider"] is provider
    assert captured["kwargs"]["overrides"] == {"backend": "local"}
    assert captured["kwargs"]["max_turns"] == 4
    tools = captured["kwargs"]["tools"]
    assert all(isinstance(tool, FunctionTool) for tool in tools)
    names = {tool.name for tool in tools}
    assert {
        "view_outline",
        "view_detail",
        "create_child",
        "update_fields",
        "delete_node",
        "relocate_node",
    }.issubset(names)


def test_open_runner_can_include_harness_web_tools(monkeypatch):
    captured: dict = {}

    async def fake_create(thread_id, provider, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace()

    monkeypatch.setattr("core.harness.Runner.create", fake_create)

    asyncio.run(
        open_runner(
            "tree-demo",
            object(),
            root=create_node("r", "Root"),
            include_web_tools=True,
            overrides={"image": "unused-for-local"},
        )
    )

    assert captured["kwargs"]["overrides"]["backend"] == "local"
    assert captured["kwargs"]["overrides"]["image"] == "unused-for-local"
    names = [tool.name for tool in captured["kwargs"]["tools"]]
    for tool in HARNESS_WEB_TOOLS:
        assert tool.name in names
