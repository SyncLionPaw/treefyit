"""Terminal tree visualization with box-drawing characters."""

from __future__ import annotations


def render(tree: list[dict], max_text: int = 80) -> str:
    """Render tree as a terminal string with box-drawing characters.

    Example output::

        Introduction
        ├── Background
        │   ├── Prior Work
        │   └── Motivation
        └── Methods

    Args:
        tree: nested tree from build_tree().
        max_text: max chars for the text preview column.

    Returns:
        Multi-line string suitable for printing.
    """
    lines: list[str] = []
    _render(tree, lines, prefix="", max_text=max_text, is_root=True)
    return "\n".join(lines)


def show(tree: list[dict], max_text: int = 80) -> None:
    """Print the tree to stdout.  Convenience wrapper around :func:`render`."""
    print(render(tree, max_text=max_text))


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------

_BRANCH = "├── "
_LAST = "└── "
_PIPE = "│   "
_BLANK = "    "


def _render(nodes: list[dict], lines: list[str], prefix: str, max_text: int, is_root: bool = False) -> None:
    for i, node in enumerate(nodes):
        is_last = i == len(nodes) - 1
        has_children = bool(node.get("children"))

        # Title line — no connector at root level
        title = node.get("title", "")
        if is_root:
            line = f"{prefix}{title}"
        else:
            connector = _LAST if is_last else _BRANCH
            line = f"{prefix}{connector}{title}"

        # Append summary (preferred) or text preview
        preview = node.get("summary") or node.get("text", "")
        if preview:
            preview = preview[:max_text].replace("\n", " ").strip()
            line += f"  — {preview}"

        lines.append(line)

        # Recurse
        if has_children:
            if is_root:
                child_prefix = "    "
            else:
                child_prefix = prefix + (_BLANK if is_last else _PIPE)
            _render(node["children"], lines, child_prefix, max_text)
