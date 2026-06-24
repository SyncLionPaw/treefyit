"""Tree navigation tools — designed for agent use.

Agents reference trees by ID — they never pass raw tree data around.
Trees are registered in a module-level store after ``build_tree()`` or
via ``register()``.

Usage::

    from src.tools import overview, inspect, get_children

    overview("paper")             # shape of the whole tree
    inspect("paper", "0.1")      # full details of node 0.1
    get_children("paper", "0.1") # list children of node 0.1
"""

from __future__ import annotations

from src.tree.model import Tree, TreeNode, to_wire_tree

__all__ = ["register", "unregister", "list_trees", "overview", "inspect", "get_children"]

# ---------------------------------------------------------------------------
# Tree registry
# ---------------------------------------------------------------------------

_registry: dict[str, Tree] = {}


def register(
    tree_id: str,
    tree: Tree,
    *,
    filename: str = "",
    doc_kind: str = "",
) -> None:
    """Store a tree so agents can reference it by *tree_id*."""
    public_tree = to_wire_tree(tree)
    _registry[tree_id] = public_tree
    from src.tools.forest import index_tree

    index_tree(tree_id, public_tree, filename=filename, doc_kind=doc_kind)


def unregister(tree_id: str) -> None:
    """Remove a tree from the registry."""
    _registry.pop(tree_id, None)
    from src.tools.forest import remove_tree

    remove_tree(tree_id)


def list_trees() -> list[dict]:
    """Return all registered trees (id + overview)."""
    return [
        {"tree_id": tid, "node_count": _count(tree), "max_depth": _max_depth(tree)}
        for tid, tree in _registry.items()
    ]


# ---------------------------------------------------------------------------
# Public tools
# ---------------------------------------------------------------------------


def overview(tree_id: str) -> dict:
    """Return a structural overview of the tree.

    Returns:
        ``{"tree_id", "node_count", "max_depth", "roots": [{"path", "title", "summary", "children_count"}, ...]}``
        or ``{"error": ...}`` if the tree_id is unknown.
    """
    tree = _registry.get(tree_id)
    if tree is None:
        return {"error": f"unknown tree_id: {tree_id}"}

    roots: list[dict] = []
    for i, node in enumerate(tree):
        roots.append(_summarize_node(node, str(i)))

    return {
        "tree_id": tree_id,
        "node_count": _count(tree),
        "max_depth": _max_depth(tree),
        "roots": roots,
    }


def inspect(tree_id: str, path: str = "0") -> dict:
    """Return detailed information about a single node.

    Args:
        tree_id: Registered tree identifier.
        path: Dot-separated index path (e.g. ``"0"``, ``"0.1"``, ``"0.1.2"``).

    Returns:
        ``{"tree_id", "path", "title", "text", "summary", "children_count", "children"}``
        or ``{"error": ...}``.
    """
    tree = _registry.get(tree_id)
    if tree is None:
        return {"error": f"unknown tree_id: {tree_id}"}

    node = _resolve(tree, path)
    if node is None:
        return {"error": f"invalid path: {path}"}

    children = node.get("children", [])
    return {
        "tree_id": tree_id,
        "path": path,
        "title": node.get("title", ""),
        "text": node.get("text", ""),
        "summary": node.get("summary", ""),
        "children_count": len(children),
        "children": [_child_path(path, i) for i in range(len(children))],
    }


def get_children(tree_id: str, path: str = "0") -> dict:
    """List the immediate children of a node.

    Args:
        tree_id: Registered tree identifier.
        path: Dot-separated index path.

    Returns:
        ``{"tree_id", "path", "title", "children_count", "children": [{"path", "title", "summary", "children_count"}, ...]}``
    """
    tree = _registry.get(tree_id)
    if tree is None:
        return {"error": f"unknown tree_id: {tree_id}"}

    node = _resolve(tree, path)
    if node is None:
        return {"error": f"invalid path: {path}"}

    children = node.get("children", [])
    result: list[dict] = []
    for i, child in enumerate(children):
        gc = child.get("children", [])
        result.append({
            "path": _child_path(path, i),
            "title": child.get("title", ""),
            "summary": child.get("summary", ""),
            "children_count": len(gc),
        })

    return {
        "tree_id": tree_id,
        "path": path,
        "title": node.get("title", ""),
        "children_count": len(children),
        "children": result,
    }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _resolve(tree: Tree, path: str) -> TreeNode | None:
    """Walk *tree* by dot-separated index path.  Returns the node or None."""
    indices = [int(x) for x in path.split(".")]
    current: Tree = tree
    node: TreeNode | None = None
    for idx in indices:
        if idx < 0 or idx >= len(current):
            return None
        node = current[idx]
        current = node.get("children", [])
    return node


def _child_path(parent_path: str, index: int) -> str:
    return f"{parent_path}.{index}"


def _summarize_node(node: TreeNode, path: str) -> dict:
    children = node.get("children", [])
    return {
        "path": path,
        "title": node.get("title", ""),
        "summary": node.get("summary", ""),
        "children_count": len(children),
    }


def _count(nodes: Tree) -> int:
    n = 0
    for node in nodes:
        n += 1
        if "children" in node:
            n += _count(node["children"])
    return n


def _max_depth(nodes: Tree, depth: int = 1) -> int:
    max_d = depth
    for node in nodes:
        if "children" in node:
            max_d = max(max_d, _max_depth(node["children"], depth + 1))
    return max_d
