"""文档树三层：

- ``core.model``：数据结构定义（TreeNode / 节点操作 / 持久化）
- ``core.build``：构建方式（markdown 建树 / 检索）
- ``core.agent``：面向 LLM 的工具（树工具 / 检索工具）

顶层重导出保持向后兼容，``from core import ...`` 仍可用。
"""

from .agent import (
    DEFAULT_EXTRA_SYSTEM,
    SEARCH_EXTRA_SYSTEM,
    TreeSession,
    build_search_tools,
    build_tree_tools,
    index_markdown,
)
from .build import NodeHit, format_hits, markdown_to_tree, search_store, search_tree
from .model import (
    NodeKind,
    TreeNode,
    TreeRecord,
    TreeStore,
    collect_ids,
    create_node,
    get_by_path,
    get_node,
    get_parent,
    mount_node,
    move_node,
    remove_node,
    update_node,
    view_by_path,
    view_detail_by_path,
    view_node,
    view_node_detail,
)

__all__ = [
    "NodeKind",
    "TreeNode",
    "TreeSession",
    "TreeStore",
    "TreeRecord",
    "NodeHit",
    "DEFAULT_EXTRA_SYSTEM",
    "SEARCH_EXTRA_SYSTEM",
    "build_tree_tools",
    "build_search_tools",
    "index_markdown",
    "markdown_to_tree",
    "search_tree",
    "search_store",
    "format_hits",
    "collect_ids",
    "create_node",
    "get_node",
    "get_parent",
    "get_by_path",
    "view_node",
    "view_node_detail",
    "view_by_path",
    "view_detail_by_path",
    "mount_node",
    "update_node",
    "remove_node",
    "move_node",
]
