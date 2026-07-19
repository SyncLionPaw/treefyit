"""Tree harness —— 与 desktop / ``app.repl`` 对齐的 Runner 入口。

树 CRUD tools 跑在 host 进程里（同 ``pagentv4.tools.HARNESS_WEB_TOOLS``），
通过 ``Runner.create(..., tools=...)`` 注入；sandbox 仍由 local backend 提供
``run_command`` / ``read_file`` 等伴身电脑工具。
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

from pagentv4 import FunctionTool, ProviderProtocol, Runner
from pagentv4.tools import HARNESS_WEB_TOOLS

from .tools import TreeSession, build_tree_tools
from .tree import TreeNode

DEFAULT_EXTRA_SYSTEM = (
    "You can inspect and edit the in-memory document tree with tree tools "
    "(view_outline, view_detail, create_child, update_fields, delete_node, "
    "relocate_node). Use sandbox file/command tools when the task needs the "
    "local workspace under /home/agent."
)


async def open_runner(
    thread_id: str,
    provider: ProviderProtocol,
    *,
    root: TreeNode,
    backend: str = "local",
    overrides: dict | None = None,
    extra_system: str = DEFAULT_EXTRA_SYSTEM,
    max_turns: int = 8,
    skill_roots: Sequence[str | Path] = (),
    tools: Sequence[FunctionTool] = (),
    include_web_tools: bool = False,
    tool_hooks=None,
) -> tuple[TreeSession, Runner]:
    """打开与 desktop 同构的 local-backend Runner，并挂上树 CRUD tools。

    等价于::

        session, tree_tools = build_tree_tools(root)
        runner = await Runner.create(
            thread_id,
            provider,
            overrides={"backend": "local", **overrides},
            tools=[*tree_tools, *tools],
            ...
        )
    """
    session, tree_tools = build_tree_tools(root)
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
    return session, runner
