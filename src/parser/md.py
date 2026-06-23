"""Markdown parser — extract headers and their text content.

Heading detection uses ``markdown-it-py`` (CommonMark).  Domain-specific
post-processing keeps PDF-style numbering inference and plain-text 章回 fallback.
"""

from __future__ import annotations

import logging
import re

from markdown_it import MarkdownIt

logger = logging.getLogger(__name__)

_MD = MarkdownIt("commonmark")


def parse_md(path: str) -> list[dict]:
    """Parse a Markdown file into a list of section nodes.

    Each node: {"title": str, "level": int, "line_num": int, "text": str}

    Headers inside fenced code blocks are ignored by the Markdown parser.
    Hierarchy is inferred from numbering when header levels are flat.
    """
    logger.info("[md] parse_md %s", path)
    with open(path, encoding="utf-8") as f:
        content = f.read()
    return parse_md_text(content)


def parse_md_text(content: str) -> list[dict]:
    """Parse Markdown *content* into flat section nodes."""
    lines = content.split("\n")
    nodes = _extract_markdown_headings(content)
    logger.info("[md] extracted %d markdown headings", len(nodes))

    if not nodes:
        nodes = _extract_plaintext_chapters(lines)
        logger.info("[md] plaintext chapter fallback → %d nodes", len(nodes))

    _infer_levels(nodes)
    level_counts: dict[int, int] = {}
    for n in nodes:
        level_counts[n["level"]] = level_counts.get(n["level"], 0) + 1
    logger.info("[md] inferred levels: %s", level_counts)

    _fill_text(nodes, lines)
    total_text = sum(len(n.get("text", "")) for n in nodes)
    logger.info("[md] filled text for %d nodes, total %d chars", len(nodes), total_text)
    return nodes


# ---------------------------------------------------------------------------
# Header extraction (markdown-it-py)
# ---------------------------------------------------------------------------


def _extract_markdown_headings(text: str) -> list[dict]:
    """Extract ATX and setext headings via CommonMark token stream."""
    if not text.strip():
        return []

    tokens = _MD.parse(text)
    nodes: list[dict] = []

    for i, tok in enumerate(tokens):
        if tok.type != "heading_open":
            continue

        level = int(tok.tag[1])  # h1 … h6
        line_num = tok.map[0] + 1 if tok.map else 0

        title = ""
        if i + 1 < len(tokens) and tokens[i + 1].type == "inline":
            title = tokens[i + 1].content.strip()

        if title:
            nodes.append({"title": title, "level": level, "line_num": line_num})

    return nodes


_CHAPTER_HUI = re.compile(r"^第.+回\s*.+$")
_CHAPTER_EN = re.compile(r"^Chapter\s+\d+\b", re.IGNORECASE)
CHAPTER_NOVEL_MIN = 5


def is_chapter_heading(title: str) -> bool:
    stripped = title.strip()
    return bool(_CHAPTER_HUI.match(stripped) or _CHAPTER_EN.match(stripped))


def looks_like_chapter_novel(
    nodes: list[dict], *, min_chapters: int = CHAPTER_NOVEL_MIN
) -> bool:
    """True when most top-level sections look like 章回 / Chapter headings."""
    if len(nodes) < min_chapters:
        return False
    chapter_count = sum(1 for n in nodes if is_chapter_heading(n.get("title", "")))
    return chapter_count >= min_chapters and chapter_count >= len(nodes) * 0.5


def _extract_plaintext_chapters(lines: list[str]) -> list[dict]:
    """Extract chapter headings from plain text without Markdown markup."""
    nodes: list[dict] = []
    for i, raw in enumerate(lines, 1):
        stripped = raw.strip()
        if not stripped:
            continue
        if _CHAPTER_HUI.match(stripped) or _CHAPTER_EN.match(stripped):
            nodes.append(
                {
                    "title": stripped,
                    "level": 1,
                    "line_num": i,
                }
            )

    if not nodes:
        logger.info("[md] plaintext file detected; deferring to semantic extraction")
    return nodes


# ---------------------------------------------------------------------------
# Hierarchy inference from numbering
# ---------------------------------------------------------------------------


def _infer_levels(nodes: list[dict]) -> None:
    """Fix levels when all sections use the same header level (flat structure).

    Detects numbering like "2.1", "3.1.1" and promotes subsections to deeper
    levels, overriding the markdown header level.
    """
    numbered = [_parse_number(n["title"]) for n in nodes]
    numbered = [n for n in numbered if n is not None]
    if not numbered:
        return

    max_dots = max(len(n) - 1 for n in numbered)
    if max_dots == 0:
        return

    base_level: int | None = None
    for node in nodes:
        num = _parse_number(node["title"])
        if num and len(num) == 1:
            base_level = node["level"]
            break

    if base_level is None:
        from collections import Counter

        levels = Counter(n["level"] for n in nodes if _parse_number(n["title"]))
        if levels:
            base_level = levels.most_common(1)[0][0]
        else:
            return

    for node in nodes:
        num = _parse_number(node["title"])
        if num:
            node["level"] = base_level + len(num) - 1

    n = len(nodes)
    i = 0
    while i < n:
        if _parse_number(nodes[i]["title"]):
            i += 1
            continue

        j = i + 1
        run_numbers: list[int] = []
        while j < n:
            num = _parse_number(nodes[j]["title"])
            if num and len(num) == 1 and nodes[j]["level"] == nodes[i]["level"]:
                run_numbers.append(num[0])
                j += 1
                continue
            break

        if len(run_numbers) >= 2 and _is_numbered_run_container(nodes[i]["title"]):
            nodes[i]["level"] = nodes[i]["level"] - 1
            for k in range(i + 1, j):
                nodes[k]["level"] = nodes[k]["level"] + 1
            i = j
        else:
            i += 1


def _parse_number(title: str) -> tuple[int, ...] | None:
    """Extract a hierarchical number from a section title."""
    m = re.match(r"^(\d+(?:\.\d+)*)\b", title.strip())
    if not m:
        return None
    return tuple(int(x) for x in m.group(1).split("."))


def _is_numbered_run_container(title: str) -> bool:
    """Return true for unnumbered headings that can own numbered children."""
    normalized = re.sub(r"[^a-z]+", " ", title.lower()).strip()
    if not normalized:
        return False
    blocked = {
        "abstract",
        "summary",
        "keywords",
        "acknowledgements",
        "acknowledgments",
        "references",
        "bibliography",
    }
    if normalized in blocked:
        return False
    return normalized in {
        "appendix",
        "appendices",
        "supplementary material",
        "supplemental material",
    }


# ---------------------------------------------------------------------------
# Text extraction
# ---------------------------------------------------------------------------


def _fill_text(nodes: list[dict], lines: list[str]) -> None:
    """Assign text content to each node (from its header line to the next header)."""
    for i, node in enumerate(nodes):
        start = node["line_num"] - 1
        if i + 1 < len(nodes):
            end = nodes[i + 1]["line_num"] - 1
        else:
            end = len(lines)
        node["text"] = "\n".join(lines[start:end]).strip()
