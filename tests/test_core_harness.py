from __future__ import annotations

import asyncio
from pathlib import Path
from types import SimpleNamespace

from core import create_node, open_runner
from pagentv4 import FunctionTool
from pagentv4.tools import HARNESS_WEB_TOOLS


class FakeFiles:
    def __init__(self):
        self.writes: dict[str, str] = {}

    async def write(self, path: str, content: str):
        self.writes[path] = content


def test_open_runner_uses_local_backend_store_and_stages_markdown(
    monkeypatch, tmp_path: Path
):
    captured: dict = {}
    files = FakeFiles()

    async def fake_create(thread_id, provider, **kwargs):
        captured["thread_id"] = thread_id
        captured["provider"] = provider
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            thread_id=thread_id,
            sandbox=SimpleNamespace(files=files),
        )

    monkeypatch.setattr("core.harness.Runner.create", fake_create)

    md = tmp_path / "doc.md"
    md.write_text("# Hello\n\nWorld\n", encoding="utf-8")
    store_dir = tmp_path / "lib"
    provider = object()

    session, runner = asyncio.run(
        open_runner(
            "tree-demo",
            provider,
            store_dir=store_dir,
            md_path=md,
            max_turns=4,
        )
    )

    assert runner.thread_id == "tree-demo"
    assert session.root.id == "root"
    assert session.store is not None
    assert session.source_md_path == md.resolve()
    assert captured["kwargs"]["overrides"] == {"backend": "local"}
    assert captured["kwargs"]["max_turns"] == 4
    names = {tool.name for tool in captured["kwargs"]["tools"]}
    assert "save_tree" in names and "seed_from_markdown" in names
    assert all(isinstance(tool, FunctionTool) for tool in captured["kwargs"]["tools"])
    assert files.writes["source.md"] == "# Hello\n\nWorld\n"
    assert store_dir.joinpath("trees").is_dir()


def test_open_runner_can_include_harness_web_tools(monkeypatch, tmp_path: Path):
    captured: dict = {}

    async def fake_create(thread_id, provider, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(sandbox=SimpleNamespace(files=FakeFiles()))

    monkeypatch.setattr("core.harness.Runner.create", fake_create)

    asyncio.run(
        open_runner(
            "tree-demo",
            object(),
            store_dir=tmp_path / "lib",
            root=create_node("r", "Root"),
            include_web_tools=True,
            overrides={"image": "unused-for-local"},
        )
    )

    assert captured["kwargs"]["overrides"]["backend"] == "local"
    names = [tool.name for tool in captured["kwargs"]["tools"]]
    for tool in HARNESS_WEB_TOOLS:
        assert tool.name in names


def test_open_runner_can_load_existing_tree(monkeypatch, tmp_path: Path):
    from core import TreeStore, markdown_to_tree

    store = TreeStore(tmp_path / "lib")
    root = markdown_to_tree("# A\n\nbody\n", root_title="loaded")
    store.save(root, tree_id="t1", title="loaded")

    async def fake_create(thread_id, provider, **kwargs):
        return SimpleNamespace(sandbox=SimpleNamespace(files=FakeFiles()))

    monkeypatch.setattr("core.harness.Runner.create", fake_create)

    session, _runner = asyncio.run(
        open_runner(
            "tree-demo",
            object(),
            store_dir=tmp_path / "lib",
            tree_id="t1",
        )
    )
    assert session.tree_id == "t1"
    assert session.root.children[0].title == "A"
