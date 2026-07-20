"""Tree harness tools —— host 进程内 CRUD / 持久化 / 查询。

供 ``Runner.create(tools=...)`` 注入；与 sandbox 文件工具配合：
读 ``source.md`` → 建树 → ``save_tree`` 落盘 → 他人 ``load_tree`` / ``search_library``。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from pagentv4 import FunctionTool, ToolOutput, tool

from .md import markdown_to_tree
from .ops import (
    create_node,
    get_node,
    get_parent,
    mount_node,
    move_node,
    remove_node,
    update_node,
    view_node,
    view_node_detail,
)
from .query import format_hits, search_store, search_tree
from .store import TreeStore
from .tree import NodeKind, TreeNode


def _parse_kind(kind: str | None) -> NodeKind | None:
    if kind is None or kind == "":
        return None
    try:
        return NodeKind(kind)
    except ValueError as exc:
        raise ValueError(
            f"invalid kind: {kind!r}; expected one of {[k.value for k in NodeKind]}"
        ) from exc


@dataclass
class TreeSession:
    """工作树 + 可选持久化库。"""

    root: TreeNode
    store: TreeStore | None = None
    tree_id: str | None = None
    source_md_path: Path | None = None
    history: list[TreeNode] = field(default_factory=list)

    def snapshot(self) -> None:
        self.history.append(self.root)

    def undo(self) -> bool:
        if not self.history:
            return False
        self.root = self.history.pop()
        return True

    def require_store(self) -> TreeStore:
        if self.store is None:
            raise ValueError("no tree store configured")
        return self.store

    def build_tools(self) -> list[FunctionTool]:
        session = self

        @tool()
        def view_outline(node_id: str, depth: int = 1) -> ToolOutput:
            """View a node outline by id.

            Args:
                node_id: Target node id.
                depth: 0 = node itself; 1 = include direct children.
            """
            try:
                return ToolOutput.succeed(
                    view_node(session.root, node_id, depth=depth)
                )
            except (KeyError, ValueError) as exc:
                return ToolOutput.fail(str(exc))

        @tool()
        def view_detail(node_id: str, max_content_chars: int = 2000) -> ToolOutput:
            """View node detail by id, including truncated content.

            Args:
                node_id: Target node id.
                max_content_chars: Max characters of content to include.
            """
            try:
                return ToolOutput.succeed(
                    view_node_detail(
                        session.root,
                        node_id,
                        max_content_chars=max_content_chars,
                    )
                )
            except (KeyError, ValueError) as exc:
                return ToolOutput.fail(str(exc))

        @tool()
        def create_child(
            parent_id: str,
            id: str,
            title: str,
            kind: str | None = None,
            content: str | None = None,
            summary: str | None = None,
        ) -> ToolOutput:
            """Create a child node under parent_id and attach it.

            Args:
                parent_id: Parent node id.
                id: New node id (must be unique in the tree).
                title: New node title.
                kind: Optional kind: text, image, table, or link.
                content: Optional body text.
                summary: Optional summary.
            """
            try:
                child = create_node(
                    id,
                    title,
                    kind=_parse_kind(kind),
                    content=content,
                    summary=summary,
                )
                new_root = mount_node(session.root, parent_id, child)
            except (KeyError, ValueError) as exc:
                return ToolOutput.fail(str(exc))
            session.snapshot()
            session.root = new_root
            return ToolOutput.succeed(view_node(session.root, parent_id, depth=1))

        @tool()
        def update_fields(
            node_id: str,
            title: str | None = None,
            kind: str | None = None,
            content: str | None = None,
            summary: str | None = None,
            clear_kind: bool = False,
            clear_content: bool = False,
            clear_summary: bool = False,
        ) -> ToolOutput:
            """Update fields on a node by id.

            Args:
                node_id: Target node id.
                title: New title; omit to keep unchanged.
                kind: New kind; omit to keep unchanged.
                content: New content; omit to keep unchanged.
                summary: New summary; omit to keep unchanged.
                clear_kind: Set kind to null.
                clear_content: Set content to null.
                clear_summary: Set summary to null.
            """
            try:
                kwargs: dict = {}
                if title is not None:
                    kwargs["title"] = title
                if clear_kind:
                    kwargs["kind"] = None
                elif kind is not None:
                    kwargs["kind"] = _parse_kind(kind)
                if clear_content:
                    kwargs["content"] = None
                elif content is not None:
                    kwargs["content"] = content
                if clear_summary:
                    kwargs["summary"] = None
                elif summary is not None:
                    kwargs["summary"] = summary

                get_node(session.root, node_id)
                new_root = update_node(session.root, node_id, **kwargs)
            except (KeyError, ValueError) as exc:
                return ToolOutput.fail(str(exc))
            session.snapshot()
            session.root = new_root
            return ToolOutput.succeed(
                view_node_detail(session.root, node_id, max_content_chars=500)
            )

        @tool()
        def delete_node(node_id: str) -> ToolOutput:
            """Delete a node and its subtree by id. Cannot delete the root.

            Args:
                node_id: Node id to delete.
            """
            try:
                parent = get_parent(session.root, node_id)
                parent_hint = parent.id if parent is not None else None
                new_root = remove_node(session.root, node_id)
            except (KeyError, ValueError) as exc:
                return ToolOutput.fail(str(exc))
            session.snapshot()
            session.root = new_root
            if parent_hint is None:
                return ToolOutput.succeed(f"deleted {node_id}")
            return ToolOutput.succeed(
                f"deleted {node_id}\n"
                + view_node(session.root, parent_hint, depth=1)
            )

        @tool()
        def relocate_node(
            node_id: str,
            new_parent_id: str,
            index: int | None = None,
        ) -> ToolOutput:
            """Move a node under a new parent.

            Args:
                node_id: Node id to move.
                new_parent_id: Destination parent id.
                index: Optional child index after move; default append.
            """
            try:
                new_root = move_node(
                    session.root,
                    node_id,
                    new_parent_id,
                    index=index,
                )
            except (KeyError, ValueError) as exc:
                return ToolOutput.fail(str(exc))
            session.snapshot()
            session.root = new_root
            return ToolOutput.succeed(
                view_node(session.root, new_parent_id, depth=1)
            )

        @tool()
        def seed_from_markdown(path: str | None = None) -> ToolOutput:
            """Build a heading tree from a markdown file into the working tree.

            Args:
                path: Host markdown path. Defaults to the session source markdown.
            """
            try:
                md_path = Path(path) if path else session.source_md_path
                if md_path is None:
                    raise ValueError("markdown path not provided")
                md_path = md_path.expanduser().resolve()
                if not md_path.is_file():
                    raise FileNotFoundError(f"markdown not found: {md_path}")
                text = md_path.read_text(encoding="utf-8")
                seeded = markdown_to_tree(
                    text,
                    root_id=session.root.id,
                    root_title=session.root.title or md_path.stem,
                )
            except (OSError, ValueError) as exc:
                return ToolOutput.fail(str(exc))
            session.snapshot()
            session.root = seeded
            session.source_md_path = md_path
            return ToolOutput.succeed(
                "seeded from markdown\n" + view_node(session.root, session.root.id, depth=2)
            )

        @tool()
        def save_tree(tree_id: str | None = None, title: str | None = None) -> ToolOutput:
            """Persist the working tree to the library for later load/search.

            Args:
                tree_id: Optional id; defaults to current session tree_id or a new id.
                title: Optional library title; defaults to root title.
            """
            try:
                store = session.require_store()
                record = store.save(
                    session.root,
                    tree_id=tree_id or session.tree_id,
                    title=title,
                    source_path=session.source_md_path,
                )
            except (KeyError, ValueError, OSError) as exc:
                return ToolOutput.fail(str(exc))
            session.tree_id = record.tree_id
            return ToolOutput.succeed(
                "saved "
                f"tree_id={record.tree_id} title={record.title!r} "
                f"nodes={record.node_count} path={store.tree_path(record.tree_id)}"
            )

        @tool()
        def load_tree(tree_id: str) -> ToolOutput:
            """Load a persisted tree into the working session.

            Args:
                tree_id: Library tree id.
            """
            try:
                store = session.require_store()
                record = store.load(tree_id)
            except (KeyError, ValueError, OSError) as exc:
                return ToolOutput.fail(str(exc))
            session.snapshot()
            session.root = record.root
            session.tree_id = record.tree_id
            if record.source_path:
                session.source_md_path = Path(record.source_path)
            return ToolOutput.succeed(
                f"loaded tree_id={record.tree_id}\n"
                + view_node(session.root, session.root.id, depth=2)
            )

        @tool()
        def list_saved_trees() -> ToolOutput:
            """List trees saved in the persistent library."""
            try:
                store = session.require_store()
                items = store.list()
            except (ValueError, OSError) as exc:
                return ToolOutput.fail(str(exc))
            if not items:
                return ToolOutput.succeed("library empty")
            lines = [f"library={len(items)}"]
            for item in items:
                lines.append(
                    f"- {item['tree_id']}: {item['title']} "
                    f"(nodes={item['node_count']}, updated={item['updated_at']})"
                )
            return ToolOutput.succeed("\n".join(lines))

        @tool()
        def search_working_tree(query: str, limit: int = 8) -> ToolOutput:
            """Search nodes in the in-memory working tree.

            Args:
                query: Keywords to match against title/summary/content.
                limit: Max hits.
            """
            hits = search_tree(
                session.root,
                query,
                tree_id=session.tree_id or "memory",
                limit=limit,
            )
            return ToolOutput.succeed(format_hits(hits))

        @tool()
        def search_library(query: str, limit: int = 8) -> ToolOutput:
            """Search nodes across all persisted trees in the library.

            Args:
                query: Keywords to match against title/summary/content.
                limit: Max hits.
            """
            try:
                store = session.require_store()
                hits = search_store(store, query, limit=limit)
            except (ValueError, OSError, KeyError) as exc:
                return ToolOutput.fail(str(exc))
            return ToolOutput.succeed(format_hits(hits))

        return [
            view_outline,
            view_detail,
            create_child,
            update_fields,
            delete_node,
            relocate_node,
            seed_from_markdown,
            save_tree,
            load_tree,
            list_saved_trees,
            search_working_tree,
            search_library,
        ]


def build_tree_tools(
    root: TreeNode,
    *,
    store: TreeStore | None = None,
    tree_id: str | None = None,
    source_md_path: str | Path | None = None,
) -> tuple[TreeSession, list[FunctionTool]]:
    """便捷工厂：给定根节点，返回 session 与 pagentv4 tool 列表。"""
    session = TreeSession(
        root=root,
        store=store,
        tree_id=tree_id,
        source_md_path=Path(source_md_path).expanduser().resolve()
        if source_md_path is not None
        else None,
    )
    return session, session.build_tools()
