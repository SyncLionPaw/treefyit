"""Generate Mermaid graph diagrams from tree structures."""

from __future__ import annotations


def to_mermaid(tree: list[dict]) -> str:
    """Generate a Mermaid ``flowchart TD`` diagram with circular nodes.

    Renders in any Mermaid-compatible viewer (GitHub, VS Code, browsers).
    """
    lines = [
        "%%{init: {'flowchart': {'nodeSpacing': 30, 'rankSpacing': 70, 'curve': 'basis'}, 'theme': 'default'}}%%",
        "flowchart TD",
        "    classDef root fill:#3b82f6,stroke:#2563eb,color:#fff,stroke-width:2px",
        "    classDef branch fill:#f0f9ff,stroke:#7dd3fc,color:#0c4a6e,stroke-width:1.5px",
        "    classDef leaf fill:#f8fafc,stroke:#cbd5e1,color:#475569,stroke-width:1px",
    ]
    counter = 0
    _render_nodes(tree, lines, parent=None, counter=counter, level=0)
    return "\n".join(lines)


def _render_nodes(
    nodes: list[dict],
    lines: list[str],
    parent: str | None,
    counter: int,
    level: int,
) -> int:
    for node in nodes:
        cid = f"N{counter}"
        counter += 1

        title = _escape(node.get("title", ""))
        display = title if len(title) <= 18 else title[:16] + ".."
        has_children = "children" in node

        lines.append(f'    {cid}(("{display}"))')

        if parent:
            lines.append(f"    {parent} --> {cid}")

        if parent is None:
            lines.append(f"    class {cid} root")
        elif has_children:
            lines.append(f"    class {cid} branch")
        else:
            lines.append(f"    class {cid} leaf")

        if has_children:
            counter = _render_nodes(
                node["children"], lines, parent=cid, counter=counter, level=level + 1
            )
    return counter


def _escape(text: str) -> str:
    """Escape special characters for Mermaid labels."""
    return text.replace('"', "'").replace("\n", " ").strip()
