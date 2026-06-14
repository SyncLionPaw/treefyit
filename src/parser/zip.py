"""ZIP parser — extract multiple documents from a ZIP archive.

Each file in the ZIP is parsed individually. An optional top-level root node
wraps all documents, named after the ZIP file itself.
"""

from __future__ import annotations

import zipfile
from pathlib import Path


def parse_zip(
    path: str,
    parser: str = "auto",
    **kwargs,
) -> list[dict]:
    """Parse a ZIP archive containing multiple documents.

    Args:
        path: Path to a ``.zip`` file.
        parser: Which parser to use for each file (``"auto"``, ``"md"``, ``"html"``).
        **kwargs: Passed through to the individual parsers.

    Returns:
        A list of nodes. If the ZIP contains multiple files, they are wrapped
        in a root node whose title is the ZIP filename.
    """
    zip_name = Path(path).stem
    children: list[dict] = []

    with zipfile.ZipFile(path, "r") as zf:
        for info in sorted(zf.infolist(), key=lambda i: i.filename):
            if info.is_dir():
                continue

            name = info.filename
            ext = Path(name).suffix.lower()

            # Only process supported file types
            if ext not in (".md", ".markdown", ".html", ".htm", ".txt"):
                continue

            text = zf.read(name).decode("utf-8", errors="replace")

            # Parse based on extension
            nodes = _parse_text(text, ext, name, parser, kwargs)
            if nodes:
                # Wrap each file's nodes in a single root for that file
                if len(nodes) == 1:
                    file_node = nodes[0]
                else:
                    file_node = {
                        "title": Path(name).stem,
                        "text": "",
                        "children": nodes,
                    }
                file_node["source"] = name
                children.append(file_node)

    if not children:
        return []

    # Single file → no wrapper needed
    if len(children) == 1:
        return children

    # Multiple files → wrap in a root node named after the ZIP
    root = {
        "title": zip_name,
        "text": "",
        "children": children,
    }
    return [root]


def _parse_text(
    text: str,
    ext: str,
    name: str,
    parser: str,
    kwargs: dict,
) -> list[dict]:
    """Parse text content based on file extension or parser mode."""
    import tempfile
    import os

    # Write to temp file so existing parsers can read it
    with tempfile.NamedTemporaryFile(
        suffix=ext, mode="w", delete=False, encoding="utf-8"
    ) as f:
        f.write(text)
        tmp = f.name

    try:
        if ext in (".html", ".htm"):
            from src.parser.html import parse_html
            return parse_html(tmp)
        elif ext in (".md", ".markdown", ".txt"):
            from src.parser.md import parse_md
            return parse_md(tmp)
        else:
            return []
    finally:
        os.unlink(tmp)
