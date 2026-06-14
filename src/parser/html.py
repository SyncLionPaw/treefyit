"""HTML parser — let ``markitdown`` do the heavy lifting, parse the resulting
Markdown with :func:`src.parser.md.parse_md`, and keep a tiny regex fallback
for environments where markitdown / its dependencies aren't available.
"""

from __future__ import annotations

import logging
import os
import re
import tempfile

from src.parser.md import parse_md

logger = logging.getLogger(__name__)


def parse_html(path: str) -> list[dict]:
    """Parse an HTML file into a list of section nodes.

    The returned schema matches :func:`src.parser.md.parse_md`:

        {"title": str, "level": int, "line_num": int, "text": str}
    """
    with open(path, encoding="utf-8", errors="replace") as f:
        raw = f.read()

    nodes = _try_markitdown(raw)
    if nodes:
        return nodes

    logger.warning(
        "markitdown not available for %s; falling back to minimal regex "
        "extractor (headers + text between them). Install markitdown for a "
        "much better result on real HTML.",
        path,
    )
    nodes = _try_regex(raw)
    return nodes or [
        {"title": "Document", "level": 1, "line_num": 0, "text": _strip_tags(raw)}
    ]


# ---------------------------------------------------------------------------
# Stage 1: markitdown — Microsoft's HTML / PDF → Markdown library.
# It handles tables, code blocks, lists, figures, math, boilerplate cleanup,
# entity decoding, etc. out of the box; we just feed its Markdown output into
# :func:`src.parser.md.parse_md` so all downstream logic (hierarchy inference,
# numbering, summaries) stays the same across md/html/pdf inputs.
# ---------------------------------------------------------------------------


def _try_markitdown(raw: str) -> list[dict]:
    try:
        from markitdown import MarkItDown  # type: ignore
    except Exception as exc:  # noqa: BLE001
        logger.debug("markitdown not installed: %s", exc)
        return []

    try:
        md = MarkItDown()
        # markitdown.convert() accepts a path on disk; write once so its own
        # preprocessing (encoding detection, JS cleanup, etc.) runs natively.
        with tempfile.NamedTemporaryFile(
            "w", suffix=".html", delete=False, encoding="utf-8"
        ) as tmp:
            tmp.write(raw)
            tmp_path = tmp.name

        try:
            result = md.convert(tmp_path)
        finally:
            _silent_unlink(tmp_path)

        markdown_text = (
            result.text_content if hasattr(result, "text_content") else str(result)
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("markitdown failed: %s", exc)
        return []

    if not markdown_text or not markdown_text.strip():
        return []

    # Persist the markdown to disk so parse_md can read it (keeps all
    # heading-inference / text-filling logic in one place).
    with tempfile.NamedTemporaryFile(
        "w", suffix=".md", delete=False, encoding="utf-8"
    ) as tmp:
        tmp.write(markdown_text)
        md_path = tmp.name

    try:
        return parse_md(md_path)
    finally:
        _silent_unlink(md_path)


# ---------------------------------------------------------------------------
# Stage 2 (fallback): very small regex extractor, only used when markitdown
# is unavailable. It's intentionally tiny — we prefer deferring to a real
# library and keep this only to avoid total breakage.
# ---------------------------------------------------------------------------


def _try_regex(raw: str) -> list[dict]:
    # Drop script/style blocks so we don't pick up JS strings as "content".
    cleaned = re.sub(
        r"<(script|style)[^>]*>.*?</\1>",
        "",
        raw,
        flags=re.DOTALL | re.IGNORECASE,
    )

    title = ""
    m = re.search(r"<title[^>]*>(.*?)</title>", cleaned, re.IGNORECASE | re.DOTALL)
    if m:
        title = _strip_tags(m.group(1)).strip()

    # Headings in document order.
    nodes: list[dict] = []
    positions: list[int] = []
    for m in re.finditer(
        r"<h([1-6])[^>]*>(.*?)</h\1>", cleaned, re.IGNORECASE | re.DOTALL
    ):
        level = int(m.group(1))
        t = _strip_tags(m.group(2)).strip()
        if not t:
            continue
        nodes.append({"title": t, "level": level, "line_num": m.start(), "text": ""})
        positions.append(m.start())

    if not nodes:
        return []

    if title and not any(n["level"] == 1 for n in nodes):
        nodes.insert(0, {"title": title, "level": 1, "line_num": 0, "text": ""})
        positions.insert(0, 0)

    # Assign text between headings.
    for i, node in enumerate(nodes):
        start = positions[i]
        end = (
            positions[i + 1]
            if i + 1 < len(nodes) and positions[i + 1] > start
            else len(cleaned)
        )
        node["text"] = _strip_tags(cleaned[start:end])

    # Reuse the md parser's hierarchy inference so numbering behaves.
    from src.parser.md import _infer_levels

    _infer_levels(nodes)
    return nodes


def _strip_tags(html: str) -> str:
    text = re.sub(r"<[^>]+>", " ", html)
    # Basic HTML entity handling — markitdown handles this fully for us.
    # The fallback only needs a small subset to be usable.
    text = (
        text.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&apos;", "'")
        .replace("&nbsp;", " ")
    )
    return re.sub(r"\s+", " ", text).strip()


def _silent_unlink(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass
