"""轻量 Markdown → TreeNode：按 ATX 标题层级搭骨架，正文挂到对应节点。"""

from __future__ import annotations

import re

from .ops import create_node
from .tree import NodeKind, TreeNode

heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$")


def markdown_to_tree(
    text: str,
    *,
    root_id: str = "root",
    root_title: str = "document",
) -> TreeNode:
    """把 markdown 解析成一棵树；无标题时整篇作为根节点 content。"""
    lines = text.splitlines()
    root = create_node(root_id, root_title, kind=NodeKind.TEXT)
    stack: list[tuple[TreeNode, int]] = [(root, 0)]
    buffer: list[str] = []
    counters = [0]

    def flush() -> None:
        node = stack[-1][0]
        body = "\n".join(buffer).strip()
        buffer.clear()
        if body:
            node.content = body

    def next_id(level: int) -> str:
        while len(counters) < level:
            counters.append(0)
        del counters[level:]
        counters[level - 1] += 1
        return "n" + ".".join(str(n) for n in counters[:level])

    for line in lines:
        match = heading_pattern.match(line)
        if match is None:
            buffer.append(line)
            continue

        flush()
        level = len(match.group(1))
        title = match.group(2).strip()
        node = create_node(next_id(level), title, kind=NodeKind.TEXT)

        while stack and stack[-1][1] >= level:
            stack.pop()
        parent = stack[-1][0]
        parent.children.append(node)
        stack.append((node, level))

    flush()

    if not root.children and root.content is None:
        body = text.strip()
        if body:
            root.content = body
    return root
