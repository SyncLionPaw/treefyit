"""Read-only tools for Q&A agents over a single document tree."""

from __future__ import annotations

from pathlib import Path

from pagentv4 import FunctionTool, ToolOutput, tool

from ..build.query import format_hits, search_tree
from ..model.ops import get_node, view_node, view_node_detail
from ..model.store import TreeStore
from ..model.tree import TreeNode

SEARCH_EXTRA_SYSTEM = (
    "You answer questions about one document tree.\n"
    "Use search_document to find relevant sections, then view_detail for "
    "evidence. Ground answers in retrieved content. If nothing matches, say so."
)


def build_search_tools(
    root: TreeNode,
    *,
    tree_id: str | None = None,
) -> list[FunctionTool]:
    """只读问答工具：在给定的一棵树里检索并查看节点。"""
    doc_id = tree_id or "document"

    @tool(
        description=(
            "在文档树里检索与问题相关的章节。"
            "回答问题时先用它找线索，再用 view_detail 取证。"
        )
    )
    def search_document(query: str, limit: int = 8) -> ToolOutput:
        """检索文档树。

        Args:
            query: 从用户问题里提炼的关键词。
            limit: 最多返回几条。
        """
        hits = search_tree(root, query, tree_id=doc_id, limit=limit)
        return ToolOutput.succeed(format_hits(hits))

    @tool(
        description=(
            "查看文档树中某个节点的大纲（结构）。"
            "适合顺着 search 命中的 node_id 往下看子结构时使用。"
            "要读正文请用 view_detail。"
        )
    )
    def view_outline(node_id: str, depth: int = 1) -> ToolOutput:
        """查看节点大纲。

        Args:
            node_id: search 命中的目标节点 id。
            depth: 展开几层子节点。
        """
        try:
            return ToolOutput.succeed(view_node(root, node_id, depth=depth))
        except (KeyError, ValueError) as exc:
            return ToolOutput.fail(str(exc))

    @tool(
        description=(
            "查看文档树中某个节点的详情与正文（超长会截断）。"
            "回答问题时用它取原文作为依据。"
            "只想看结构不看正文时用 view_outline。"
        )
    )
    def view_detail(node_id: str, max_content_chars: int = 2000) -> ToolOutput:
        """查看节点详情与正文。

        Args:
            node_id: search 命中的目标节点 id。
            max_content_chars: 正文最多返回多少字符。
        """
        try:
            get_node(root, node_id)
            return ToolOutput.succeed(
                view_node_detail(root, node_id, max_content_chars=max_content_chars)
            )
        except (KeyError, ValueError) as exc:
            return ToolOutput.fail(str(exc))

    return [search_document, view_outline, view_detail]


def index_markdown(
    md_path: str | Path,
    store_dir: str | Path,
    *,
    tree_id: str | None = None,
    title: str | None = None,
):
    """Deterministic helper: markdown → tree → persist (no LLM required)."""
    from ..build.md import markdown_to_tree

    path = Path(md_path).expanduser().resolve()
    store = TreeStore(store_dir)
    root = markdown_to_tree(
        path.read_text(encoding="utf-8"),
        root_title=title or path.stem,
    )
    return store.save(
        root,
        tree_id=tree_id or path.stem,
        title=title or path.stem,
        source_path=path,
    )
