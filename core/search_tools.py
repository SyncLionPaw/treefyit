"""Read-only library tools for Q&A agents over persisted trees."""

from __future__ import annotations

from pathlib import Path

from pagentv4 import FunctionTool, ToolOutput, tool

from .ops import get_node, view_node, view_node_detail
from .query import format_hits, search_store, search_tree
from .store import TreeStore
from .tree import TreeNode


def build_library_search_tools(
    store: TreeStore,
    *,
    tree_id: str | None = None,
) -> list[FunctionTool]:
    """Tools for agents that answer questions about saved trees.

    If ``tree_id`` is set, ``search_document`` / ``view_*`` target that tree.
    ``search_library`` always searches the whole catalog.
    """
    state = {"tree_id": tree_id, "root": None}

    def active_root() -> tuple[str, TreeNode]:
        current_id = state["tree_id"]
        if current_id is None:
            items = store.list()
            if not items:
                raise ValueError("library empty")
            if len(items) == 1:
                current_id = str(items[0]["tree_id"])
                state["tree_id"] = current_id
            else:
                raise ValueError(
                    "tree_id not selected; call list_saved_trees / use_tree first"
                )
        if state["root"] is None or state["tree_id"] != current_id:
            state["root"] = store.load(current_id).root
            state["tree_id"] = current_id
        return current_id, state["root"]

    @tool()
    def list_saved_trees() -> ToolOutput:
        """List persisted document trees available for search."""
        items = store.list()
        if not items:
            return ToolOutput.succeed("library empty")
        lines = [f"library={len(items)}"]
        for item in items:
            marker = " (active)" if item["tree_id"] == state["tree_id"] else ""
            lines.append(
                f"- {item['tree_id']}: {item['title']} "
                f"(nodes={item['node_count']}){marker}"
            )
        return ToolOutput.succeed("\n".join(lines))

    @tool()
    def use_tree(tree_id: str) -> ToolOutput:
        """Select which persisted tree later search/view tools should use.

        Args:
            tree_id: Library tree id.
        """
        try:
            record = store.load(tree_id)
        except KeyError as exc:
            return ToolOutput.fail(str(exc))
        state["tree_id"] = record.tree_id
        state["root"] = record.root
        return ToolOutput.succeed(
            f"active tree_id={record.tree_id} title={record.title!r}\n"
            + view_node(record.root, record.root.id, depth=1)
        )

    @tool()
    def search_document(query: str, limit: int = 8) -> ToolOutput:
        """Search the active document tree for sections relevant to a question.

        Args:
            query: Keywords from the user question.
            limit: Max hits.
        """
        try:
            current_id, root = active_root()
            hits = search_tree(root, query, tree_id=current_id, limit=limit)
        except (KeyError, ValueError) as exc:
            return ToolOutput.fail(str(exc))
        return ToolOutput.succeed(format_hits(hits))

    @tool()
    def search_library(query: str, limit: int = 8) -> ToolOutput:
        """Search all persisted trees in the library.

        Args:
            query: Keywords from the user question.
            limit: Max hits.
        """
        try:
            hits = search_store(store, query, limit=limit)
        except (KeyError, ValueError, OSError) as exc:
            return ToolOutput.fail(str(exc))
        return ToolOutput.succeed(format_hits(hits))

    @tool()
    def view_outline(node_id: str, depth: int = 1) -> ToolOutput:
        """View outline of a node in the active tree.

        Args:
            node_id: Target node id from search hits.
            depth: How many child levels to include.
        """
        try:
            _tree_id, root = active_root()
            return ToolOutput.succeed(view_node(root, node_id, depth=depth))
        except (KeyError, ValueError) as exc:
            return ToolOutput.fail(str(exc))

    @tool()
    def view_detail(node_id: str, max_content_chars: int = 2000) -> ToolOutput:
        """View detail/content of a node in the active tree.

        Args:
            node_id: Target node id from search hits.
            max_content_chars: Max body characters to return.
        """
        try:
            _tree_id, root = active_root()
            get_node(root, node_id)
            return ToolOutput.succeed(
                view_node_detail(root, node_id, max_content_chars=max_content_chars)
            )
        except (KeyError, ValueError) as exc:
            return ToolOutput.fail(str(exc))

    return [
        list_saved_trees,
        use_tree,
        search_document,
        search_library,
        view_outline,
        view_detail,
    ]


def index_markdown(
    md_path: str | Path,
    store_dir: str | Path,
    *,
    tree_id: str | None = None,
    title: str | None = None,
):
    """Deterministic helper: markdown → tree → persist (no LLM required)."""
    from .md import markdown_to_tree

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
