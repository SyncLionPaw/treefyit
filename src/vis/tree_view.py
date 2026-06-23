"""Terminal tree visualization with box-drawing characters."""

from __future__ import annotations

import re

_ATX_HEADING = re.compile(r"^#{1,6}\s")
_SETEXT_UNDERLINE = re.compile(r"^(=+|-+)\s*$")


def render(tree: list[dict], max_text: int = 0) -> str:
    """Render tree as a terminal string with box-drawing characters.

    Example output::

        白茶简介
        ├── 历史与产地
        │   └── 核心产区
        └── 主要品类

    Args:
        tree: nested tree from build_tree().
        max_text: max chars for an optional trailing snippet per node.
            ``0`` (default) — titles only, clean structure view.

    Returns:
        Multi-line string suitable for printing.
    """
    lines: list[str] = []
    _render(tree, lines, prefix="", max_text=max_text, is_root=True)
    return "\n".join(lines)


def show(tree: list[dict], max_text: int = 0) -> None:
    """Print the tree to stdout.  Convenience wrapper around :func:`render`."""
    print(render(tree, max_text=max_text))


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

_BRANCH = "├── "
_LAST = "└── "
_PIPE = "│   "
_BLANK = "    "


def _render(
    nodes: list[dict],
    lines: list[str],
    prefix: str,
    max_text: int,
    is_root: bool = False,
) -> None:
    for i, node in enumerate(nodes):
        is_last = i == len(nodes) - 1

        title = node.get("title", "")
        if is_root:
            line = f"{prefix}{title}"
        else:
            connector = _LAST if is_last else _BRANCH
            line = f"{prefix}{connector}{title}"

        snippet = _snippet(node, max_text)
        if snippet:
            line += f"  — {snippet}"

        lines.append(line)

        if node.get("children"):
            child_prefix = "    " if is_root else prefix + (_BLANK if is_last else _PIPE)
            _render(node["children"], lines, child_prefix, max_text)


def _snippet(node: dict, max_text: int) -> str:
    if max_text <= 0:
        return ""

    raw = node.get("summary") or node.get("text", "")
    if not raw:
        return ""

    body = _strip_heading_from_body(raw, node.get("title", ""))
    body = re.sub(r"[*_`#|\[\]()]", "", body)
    body = re.sub(r"\s+", " ", body).strip()
    if not body:
        return ""

    if len(body) <= max_text:
        return body
    return body[: max_text - 1].rstrip() + "…"


def _strip_heading_from_body(text: str, title: str) -> str:
    """Drop the heading line(s) already shown in the tree title."""
    lines = text.splitlines()
    while lines:
        stripped = lines[0].strip()
        if not stripped:
            lines.pop(0)
            continue
        if _ATX_HEADING.match(stripped):
            lines.pop(0)
            continue
        if title and stripped == title.strip():
            lines.pop(0)
            if lines and _SETEXT_UNDERLINE.match(lines[0].strip()):
                lines.pop(0)
            continue
        if _SETEXT_UNDERLINE.match(stripped):
            lines.pop(0)
            continue
        break
    return "\n".join(lines).strip()
