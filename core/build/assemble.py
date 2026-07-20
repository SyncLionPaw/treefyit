"""Assemble nested TreeNode from flat sections."""

from __future__ import annotations

from core.ops import create_node
from core.tree import NodeKind, TreeNode

from .types import Section


def _kind_for(section: Section) -> NodeKind | None:
    leaf = str(section.get("leaf_type") or "").strip().lower()
    if leaf == "table":
        return NodeKind.TABLE
    if leaf == "image":
        return NodeKind.IMAGE
    if leaf == "link":
        return NodeKind.LINK
    if leaf == "text":
        return NodeKind.TEXT
    return None


def _content_for(section: Section) -> str | None:
    text = str(section.get("text") or "").strip()
    if text:
        return text
    url = section.get("url") or section.get("uri")
    if url:
        return str(url)
    return None


def assemble_tree(
    sections: list[Section],
    *,
    root_id: str = "root",
    root_title: str = "Document",
) -> TreeNode:
    """Nest flat sections into a TreeNode hierarchy by heading level."""
    root = create_node(root_id, root_title, kind=NodeKind.TEXT)
    if not sections:
        return root

    stack: list[tuple[TreeNode, int]] = [(root, 0)]
    counters: list[int] = []

    for section in sections:
        level = max(1, int(section.get("level") or 1))

        while stack and stack[-1][1] >= level:
            stack.pop()

        while len(counters) < level:
            counters.append(0)
        del counters[level:]
        counters[level - 1] += 1
        node_id = "n" + ".".join(str(n) for n in counters[:level])

        node = create_node(
            node_id,
            str(section.get("title") or "Untitled").strip() or "Untitled",
            kind=_kind_for(section),
            content=_content_for(section),
            summary=section.get("summary"),
        )
        stack[-1][0].children.append(node)
        stack.append((node, level))

    return root
