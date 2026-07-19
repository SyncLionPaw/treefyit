"""把 core ops 包装成 pagent tool，供 agent 直接调用。"""

from __future__ import annotations

from dataclasses import dataclass, field

from pagent import tool

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
    """持有一棵可变更的工作树；tool 调用会更新 ``root``。"""

    root: TreeNode
    history: list[TreeNode] = field(default_factory=list)

    def snapshot(self) -> None:
        self.history.append(self.root)

    def undo(self) -> bool:
        if not self.history:
            return False
        self.root = self.history.pop()
        return True

    def build_tools(self) -> list:
        """返回可交给 pagent Agent 的 FunctionTool 列表。"""
        session = self

        @tool()
        def view_outline(node_id: str, depth: int = 1) -> str:
            """View a node outline by id. depth=0 is the node itself; depth=1 includes direct children."""
            try:
                return view_node(session.root, node_id, depth=depth)
            except (KeyError, ValueError) as exc:
                return f"error: {exc}"

        @tool()
        def view_detail(node_id: str, max_content_chars: int = 2000) -> str:
            """View node detail by id, including truncated content and immediate children."""
            try:
                return view_node_detail(
                    session.root,
                    node_id,
                    max_content_chars=max_content_chars,
                )
            except (KeyError, ValueError) as exc:
                return f"error: {exc}"

        @tool()
        def create_child(
            parent_id: str,
            id: str,
            title: str,
            kind: str | None = None,
            content: str | None = None,
            summary: str | None = None,
        ) -> str:
            """Create a child node under parent_id and attach it. Returns the new outline of the parent."""
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
                return f"error: {exc}"
            session.snapshot()
            session.root = new_root
            return view_node(session.root, parent_id, depth=1)

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
        ) -> str:
            """Update fields on a node by id. Omit a field to leave it unchanged; use clear_* to null it."""
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
                return f"error: {exc}"
            session.snapshot()
            session.root = new_root
            return view_node_detail(session.root, node_id, max_content_chars=500)

        @tool()
        def delete_node(node_id: str) -> str:
            """Delete a node and its subtree by id. Cannot delete the root."""
            try:
                parent = get_parent(session.root, node_id)
                parent_hint = parent.id if parent is not None else None
                new_root = remove_node(session.root, node_id)
            except (KeyError, ValueError) as exc:
                return f"error: {exc}"
            session.snapshot()
            session.root = new_root
            if parent_hint is None:
                return f"deleted {node_id}"
            return f"deleted {node_id}\n" + view_node(
                session.root, parent_hint, depth=1
            )

        @tool()
        def relocate_node(
            node_id: str,
            new_parent_id: str,
            index: int | None = None,
        ) -> str:
            """Move a node under a new parent. Optional index inserts at that child position."""
            try:
                new_root = move_node(
                    session.root,
                    node_id,
                    new_parent_id,
                    index=index,
                )
            except (KeyError, ValueError) as exc:
                return f"error: {exc}"
            session.snapshot()
            session.root = new_root
            return view_node(session.root, new_parent_id, depth=1)

        return [
            view_outline,
            view_detail,
            create_child,
            update_fields,
            delete_node,
            relocate_node,
        ]


def build_tree_tools(root: TreeNode) -> tuple[TreeSession, list]:
    """便捷工厂：给定根节点，返回 session 与 tool 列表。"""
    session = TreeSession(root=root)
    return session, session.build_tools()
