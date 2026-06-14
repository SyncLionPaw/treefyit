"""Export tree to an HTML page with Mermaid graph + collapsible details."""

from __future__ import annotations

from .mermaid import to_mermaid


def to_html(tree: list[dict], title: str = "Document Tree") -> str:
    """Render tree as an HTML page with Mermaid diagram and detail sections.

    Args:
        tree: nested tree from build_tree().
        title: page title.

    Returns:
        Self-contained HTML string.  Open in any browser.
    """
    mermaid_code = to_mermaid(tree)
    detail_body = _render_detail(tree)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>{title}</title>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<script>mermaid.initialize({{ startOnLoad: true, theme: 'default' }});</script>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; margin: 20px; max-width: 1200px; }}
.mermaid {{ margin: 20px 0; }}
.detail {{ margin-top: 40px; }}
.detail details {{ margin: 4px 0; }}
.detail summary {{ cursor: pointer; padding: 4px 8px; border-radius: 4px; }}
.detail summary:hover {{ background: #f0f0f0; }}
.detail summary .title {{ font-weight: 600; }}
.detail summary .summary {{ color: #555; font-size: 0.9em; margin-left: 8px; }}
.detail .children {{ padding-left: 24px; border-left: 2px solid #e0e0e0; margin-left: 12px; }}
.detail .leaf {{ padding: 4px 8px; color: #555; }}
.detail .leaf .title {{ font-weight: 600; color: #333; }}
.detail .text {{ color: #999; font-size: 0.85em; margin-top: 2px; padding-left: 8px; white-space: pre-wrap; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="mermaid">
{mermaid_code}
</div>
<hr>
<h2>Document Details</h2>
<div class="detail">{detail_body}
</div>
</body>
</html>"""


def save_html(tree: list[dict], path: str, title: str = "Document Tree") -> None:
    """Write tree to an HTML file."""
    with open(path, "w", encoding="utf-8") as f:
        f.write(to_html(tree, title=title))


# ---------------------------------------------------------------------------
# Detail (collapsible)
# ---------------------------------------------------------------------------

def _render_detail(nodes: list[dict]) -> str:
    return "\n".join(_render_node(n) for n in nodes)


def _render_node(node: dict) -> str:
    title = node.get("title", "")
    summary = node.get("summary", "")
    text = node.get("text", "")
    children = node.get("children")

    summary_html = f'<span class="summary">— {summary[:120]}</span>' if summary else ""

    if children:
        kids = "\n".join(_render_node(c) for c in children)
        return f"""<details open>
<summary><span class="title">{title}</span>{summary_html}</summary>
<div class="children">{kids}
</div></details>"""
    else:
        text_html = f'<div class="text">{text[:500]}</div>' if text else ""
        return f"""<div class="leaf"><span class="title">{title}</span>{summary_html}{text_html}</div>"""
