"""TreeNode 纯函数操作：按 id 创建、查看、挂载。"""

from __future__ import annotations

from .tree import NodeKind, TreeNode


def create_node(
    id: str,
    title: str,
    *,
    kind: NodeKind | None = None,
    content: str | None = None,
    summary: str | None = None,
    children: list[TreeNode] | None = None,
) -> TreeNode:
    """创造节点。"""
    return TreeNode(
        id=id,
        title=title,
        kind=kind,
        content=content,
        summary=summary,
        children=list(children or []),
    )


def get_node(root: TreeNode, node_id: str) -> TreeNode:
    """按 id 取节点；找不到则抛 ``KeyError``。"""
    if root.id == node_id:
        return root
    for child in root.children:
        try:
            return get_node(child, node_id)
        except KeyError:
            continue
    raise KeyError(f"node not found: {node_id}")


def view_node(root: TreeNode, node_id: str, depth: int = 1) -> str:
    """按 id 查看节点大纲（带深度）。"""
    return get_node(root, node_id).outline(depth=depth)


def view_node_detail(
    root: TreeNode,
    node_id: str,
    *,
    max_content_chars: int | None = None,
) -> str:
    """按 id 查看节点详情（含正文）。"""
    return get_node(root, node_id).detail(max_content_chars=max_content_chars)


def mount_node(root: TreeNode, parent_id: str, child: TreeNode) -> TreeNode:
    """把 ``child`` 挂到 ``parent_id`` 下，返回新的根节点（不修改原树）。"""

    def mount(node: TreeNode) -> tuple[TreeNode, bool]:
        if node.id == parent_id:
            return node.model_copy(update={"children": [*node.children, child]}), True

        new_children: list[TreeNode] = []
        changed = False
        for existing in node.children:
            updated, ok = mount(existing)
            new_children.append(updated)
            changed = changed or ok
        if changed:
            return node.model_copy(update={"children": new_children}), True
        return node, False

    new_root, found = mount(root)
    if not found:
        raise KeyError(f"parent not found: {parent_id}")
    return new_root
