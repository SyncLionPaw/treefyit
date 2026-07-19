"""Tree harness —— 与 desktop ``Runner.create(backend=local)`` 对齐。

典型链路：丢一个 markdown → agent 用 tree tools 建树 → ``save_tree`` 持久化
→ 之后 ``load_tree`` / ``search_library`` 再加载、再查询。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pagentv4 import FunctionTool, ProviderProtocol, Runner
from pagentv4.tools import HARNESS_WEB_TOOLS

from .ops import create_node
from .store import TreeStore
from .tools import TreeSession, build_tree_tools
from .tree import TreeNode

DEFAULT_EXTRA_SYSTEM = (
    "You build and maintain a document tree from markdown.\n"
    "Workflow:\n"
    "1) Read `/home/agent/source.md` with sandbox tools when present.\n"
    "2) Call `seed_from_markdown` to create a heading skeleton, then refine "
    "with create_child / update_fields / delete_node / relocate_node.\n"
    "3) Call `save_tree` so the tree is persisted to the library.\n"
    "4) Use `list_saved_trees`, `load_tree`, `search_working_tree`, and "
    "`search_library` for later inspection and reuse.\n"
    "Keep answers concise."
)


async def open_runner(
    thread_id: str,
    provider: ProviderProtocol,
    *,
    store_dir: str | Path,
    root: TreeNode | None = None,
    tree_id: str | None = None,
    md_path: str | Path | None = None,
    backend: str = "local",
    overrides: dict | None = None,
    extra_system: str = DEFAULT_EXTRA_SYSTEM,
    max_turns: int = 16,
    skill_roots: Sequence[str | Path] = (),
    tools: Sequence[FunctionTool] = (),
    include_web_tools: bool = False,
    tool_hooks=None,
) -> tuple[TreeSession, Runner]:
    """打开 local-backend Runner，并挂上可持久化的树工具。

    - ``store_dir``: 树库目录（``trees/*.json``）
    - ``md_path``: 可选，源 markdown；会写入 sandbox ``source.md`` 供 agent 阅读
    - ``tree_id``: 可选，启动时从库中加载已有树
    """
    store = TreeStore(store_dir)
    store.ensure_dirs()

    source = Path(md_path).expanduser().resolve() if md_path is not None else None
    loaded_id = tree_id
    working_root = root

    if loaded_id is not None:
        record = store.load(loaded_id)
        working_root = record.root
        if source is None and record.source_path:
            source = Path(record.source_path)

    if working_root is None:
        title = source.stem if source is not None else "document"
        working_root = create_node("root", title)

    session, tree_tools = build_tree_tools(
        working_root,
        store=store,
        tree_id=loaded_id,
        source_md_path=source,
    )

    merged_overrides = {"backend": backend, **(overrides or {})}
    extra: list[FunctionTool] = list(tools)
    if include_web_tools:
        extra = [*HARNESS_WEB_TOOLS, *extra]

    runner = await Runner.create(
        thread_id,
        provider,
        overrides=merged_overrides,
        extra_system=extra_system,
        max_turns=max_turns,
        skill_roots=skill_roots,
        tools=[*tree_tools, *extra],
        tool_hooks=tool_hooks,
    )

    if source is not None and source.is_file():
        text = source.read_text(encoding="utf-8")
        await runner.sandbox.files.write("source.md", text)

    return session, runner
