"""TreeNode 纯函数操作：增删改查（按 id / path）。

写操作一律返回新根，不修改原树。
"""

from __future__ import annotations

from typing import Any

from .tree import NodeKind, TreeNode

_UNSET: Any = object()


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


def collect_ids(root: TreeNode) -> set[str]:
    """收集子树全部节点 id。"""
    ids = {root.id}
    for child in root.children:
        ids.update(collect_ids(child))
    return ids


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


def get_by_path(root: TreeNode, path: str) -> TreeNode:
    """按绝对 path 取节点。空字符串或 ``root`` 表示根；其余为 ``0.1.2``。"""
    if path in ("", "root"):
        return root

    node = root
    for part in path.split("."):
        if not part.isdigit():
            raise KeyError(f"invalid path: {path}")
        index = int(part)
        if index < 0 or index >= len(node.children):
            raise KeyError(f"invalid path: {path}")
        node = node.children[index]
    return node


def get_parent(root: TreeNode, node_id: str) -> TreeNode | None:
    """返回节点的父节点；根节点或找不到时返回 ``None``。"""
    if root.id == node_id:
        return None
    for child in root.children:
        if child.id == node_id:
            return root
        parent = get_parent(child, node_id)
        if parent is not None:
            return parent
    return None


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


def view_by_path(root: TreeNode, path: str, depth: int = 1) -> str:
    """按 path 查看节点大纲。"""
    return get_by_path(root, path).outline(depth=depth)


def view_detail_by_path(
    root: TreeNode,
    path: str,
    *,
    max_content_chars: int | None = None,
) -> str:
    """按 path 查看节点详情。"""
    return get_by_path(root, path).detail(max_content_chars=max_content_chars)


def mount_node(root: TreeNode, parent_id: str, child: TreeNode) -> TreeNode:
    """把 ``child`` 挂到 ``parent_id`` 下，返回新根。"""
    overlap = collect_ids(root) & collect_ids(child)
    if overlap:
        raise ValueError(f"duplicate node id: {sorted(overlap)[0]}")

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


def update_node(
    root: TreeNode,
    node_id: str,
    *,
    title: str | None = None,
    kind: NodeKind | None | Any = _UNSET,
    content: str | None | Any = _UNSET,
    summary: str | None | Any = _UNSET,
) -> TreeNode:
    """按 id 更新字段；``None`` 标题表示不改，其余字段传 ``None`` 可清空。"""

    def patch(node: TreeNode) -> tuple[TreeNode, bool]:
        if node.id == node_id:
            updates: dict[str, Any] = {}
            if title is not None:
                updates["title"] = title
            if kind is not _UNSET:
                updates["kind"] = kind
            if content is not _UNSET:
                updates["content"] = content
            if summary is not _UNSET:
                updates["summary"] = summary
            if not updates:
                return node, True
            return node.model_copy(update=updates), True

        new_children: list[TreeNode] = []
        changed = False
        for existing in node.children:
            updated, ok = patch(existing)
            new_children.append(updated)
            changed = changed or ok
        if changed:
            return node.model_copy(update={"children": new_children}), True
        return node, False

    new_root, found = patch(root)
    if not found:
        raise KeyError(f"node not found: {node_id}")
    return new_root


def remove_node(root: TreeNode, node_id: str) -> TreeNode:
    """按 id 删除节点及其子树；不能删根。"""
    if root.id == node_id:
        raise ValueError("cannot remove root node")

    def prune(node: TreeNode) -> tuple[TreeNode, bool]:
        kept: list[TreeNode] = []
        removed = False
        for child in node.children:
            if child.id == node_id:
                removed = True
                continue
            updated, ok = prune(child)
            kept.append(updated)
            removed = removed or ok
        if removed:
            return node.model_copy(update={"children": kept}), True
        return node, False

    new_root, found = prune(root)
    if not found:
        raise KeyError(f"node not found: {node_id}")
    return new_root


def move_node(
    root: TreeNode,
    node_id: str,
    new_parent_id: str,
    *,
    index: int | None = None,
) -> TreeNode:
    """把节点移到新父节点下；``index`` 为插入位置，默认追加到末尾。"""
    if node_id == root.id:
        raise ValueError("cannot move root node")
    if node_id == new_parent_id:
        raise ValueError("cannot move node under itself")

    node = get_node(root, node_id)
    get_node(root, new_parent_id)  # ensure parent exists before detach
    if new_parent_id in collect_ids(node):
        raise ValueError("cannot move node under its descendant")

    detached = remove_node(root, node_id)

    def attach(current: TreeNode) -> tuple[TreeNode, bool]:
        if current.id == new_parent_id:
            children = list(current.children)
            insert_at = len(children) if index is None else index
            if insert_at < 0 or insert_at > len(children):
                raise ValueError(f"invalid index: {insert_at}")
            children.insert(insert_at, node)
            return current.model_copy(update={"children": children}), True

        new_children: list[TreeNode] = []
        changed = False
        for child in current.children:
            updated, ok = attach(child)
            new_children.append(updated)
            changed = changed or ok
        if changed:
            return current.model_copy(update={"children": new_children}), True
        return current, False

    new_root, found = attach(detached)
    if not found:
        raise KeyError(f"parent not found: {new_parent_id}")
    return new_root
