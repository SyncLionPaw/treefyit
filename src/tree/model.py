"""Canonical tree data model.

The project keeps document trees in a JSON-friendly shape so the same schema can
be implemented in other languages without carrying Python runtime details.

Public node fields:
    - ``title``: section title
    - ``text``: original section text
    - ``summary``: generated summary
    - ``line_num``: source line number when available
    - ``node_id``: stable per-build node identifier
    - ``children``: nested child nodes

Runtime-only fields must stay out of the wire format.  By convention they start
with ``_`` and are stripped by :func:`to_wire_tree`.
"""

from __future__ import annotations

from typing import Any, NotRequired, Required, TypedDict, cast

TREE_SCHEMA = "treefyit.tree.v1"
_PUBLIC_NODE_FIELDS = frozenset(
    {"title", "text", "summary", "line_num", "node_id", "children"}
)


class TreeNode(TypedDict, total=False):
    """Stable, JSON-serializable node schema."""

    title: Required[str]
    text: NotRequired[str]
    summary: NotRequired[str]
    line_num: NotRequired[int]
    node_id: NotRequired[str]
    children: NotRequired[list["TreeNode"]]


class FlatNode(TypedDict, total=False):
    """Flat parser output before hierarchical assembly."""

    title: Required[str]
    level: Required[int]
    line_num: NotRequired[int]
    text: NotRequired[str]
    summary: NotRequired[str]


type Tree = list[TreeNode]


def to_wire_tree(tree: list[dict[str, Any]] | Tree) -> Tree:
    """Deep-copy *tree* into the canonical public schema."""

    return [to_wire_node(node) for node in tree]


def to_wire_node(node: dict[str, Any] | TreeNode) -> TreeNode:
    """Deep-copy one node into the canonical public schema."""

    out: TreeNode = {"title": str(node.get("title", ""))}
    for field in _PUBLIC_NODE_FIELDS - {"title", "children"}:
        value = node.get(field)
        if value is None:
            continue
        if field == "line_num":
            out[field] = int(value)
        else:
            out[field] = str(value)

    children = node.get("children")
    if isinstance(children, list) and children:
        out["children"] = [to_wire_node(child) for child in children]
    return out


def from_wire_tree(data: Any) -> Tree:
    """Normalize arbitrary JSON-like input into the canonical tree schema."""

    if not isinstance(data, list):
        raise TypeError("tree must be a list")
    return [from_wire_node(node) for node in data]


def from_wire_node(data: Any) -> TreeNode:
    """Normalize one JSON-like node into the canonical schema."""

    if not isinstance(data, dict):
        raise TypeError("tree node must be an object")

    title = data.get("title", "")
    if not isinstance(title, str):
        title = str(title)

    node: TreeNode = {"title": title}
    for field in ("text", "summary", "node_id"):
        value = data.get(field)
        if value is not None:
            node[field] = str(value)

    line_num = data.get("line_num")
    if line_num is not None:
        node["line_num"] = int(line_num)

    children = data.get("children")
    if isinstance(children, list) and children:
        node["children"] = [from_wire_node(child) for child in children]
    return node


def is_runtime_field(name: str) -> bool:
    """Return whether *name* is an internal-only node field."""

    return name.startswith("_") or name not in _PUBLIC_NODE_FIELDS


def clone_tree(tree: Tree) -> Tree:
    """Return a deep copy of a canonical tree."""

    return cast(Tree, from_wire_tree(tree))
