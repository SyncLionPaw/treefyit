"""Markdown parser — extract headers and their text content.

Handles two common cases:
1. Well-formed MD with ``#`` / ``##`` / ``###`` — uses header level directly.
2. Flat MD (all ``##``) with numbering like "2.1", "3.1.1" — infers hierarchy
   from the numbering pattern, which is common in academic papers converted
   from PDF.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


def parse_md(path: str) -> list[dict]:
    """Parse a Markdown file into a list of section nodes.

    Each node: {"title": str, "level": int, "line_num": int, "text": str}

    Headers inside ```code blocks``` are ignored.
    Hierarchy is inferred from numbering when header levels are flat.
    """
    logger.info("[md] parse_md %s", path)
    with open(path, encoding="utf-8") as f:
        content = f.read()

    lines = content.split("\n")
    nodes = _extract_headers(lines)
    logger.info("[md] extracted %d headers", len(nodes))

    _infer_levels(nodes)
    level_counts = {}
    for n in nodes:
        level_counts[n["level"]] = level_counts.get(n["level"], 0) + 1
    logger.info("[md] inferred levels: %s", level_counts)

    _fill_text(nodes, lines)
    total_text = sum(len(n.get("text", "")) for n in nodes)
    logger.info("[md] filled text for %d nodes, total %d chars", len(nodes), total_text)
    return nodes


# ---------------------------------------------------------------------------
# Header extraction
# ---------------------------------------------------------------------------


def _extract_headers(lines: list[str]) -> list[dict]:
    """Extract all headers (ignoring code blocks).

    Falls back to common plain-text chapter patterns when no Markdown headers
    are found — e.g. Chinese novels ("第一回 ..."), English chapters, etc.
    """
    nodes: list[dict] = []
    in_code = False

    for i, raw in enumerate(lines, 1):
        stripped = raw.strip()

        if stripped.startswith("```"):
            in_code = not in_code
            continue

        if in_code or not stripped:
            continue

        m = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if m:
            nodes.append(
                {
                    "title": m.group(2).strip(),
                    "level": len(m.group(1)),
                    "line_num": i,
                }
            )

    # Fallback: if no Markdown headers, try plain-text chapter patterns.
    if not nodes:
        nodes = _extract_plaintext_chapters(lines)

    return nodes


def _extract_plaintext_chapters(lines: list[str]) -> list[dict]:
    """Extract chapter headings from plain text without Markdown markup.

    We intentionally do NOT use hand-written regexes here.  Plain-text
    documents (e.g. ``.txt`` novels) are handled by the **semantic** pipeline
    in ``src.tree.semantic`` which calls an LLM to infer structure.  This
    keeps ``md.py`` focused on Markdown parsing and avoids brittle rules.
    """
    logger.info("[md] plaintext file detected; deferring to semantic extraction")
    return []


# ---------------------------------------------------------------------------
# Hierarchy inference from numbering
# ---------------------------------------------------------------------------


def _infer_levels(nodes: list[dict]) -> None:
    """Fix levels when all sections use the same header level (flat structure).

    Detects numbering like "2.1", "3.1.1" and promotes subsections to deeper
    levels, overriding the markdown header level.
    """
    # Only act if most numbered headers share the same markdown level
    numbered = [_parse_number(n["title"]) for n in nodes]
    numbered = [n for n in numbered if n is not None]
    if not numbered:
        return

    # Count how many dots in each number
    max_dots = max(len(n) - 1 for n in numbered)

    if max_dots == 0:
        return  # no subsections detected

    # Determine the base markdown level (the level of "X" without dots)
    base_level: int | None = None
    for node in nodes:
        num = _parse_number(node["title"])
        if num and len(num) == 1:
            base_level = node["level"]
            break

    if base_level is None:
        # Fallback: use the most common level
        from collections import Counter

        levels = Counter(n["level"] for n in nodes if _parse_number(n["title"]))
        if levels:
            base_level = levels.most_common(1)[0][0]
        else:
            return

    # Apply: "X" → base_level, "X.Y" → base_level+1, "X.Y.Z" → base_level+2
    for node in nodes:
        num = _parse_number(node["title"])
        if num:
            node["level"] = base_level + len(num) - 1

    # ------------------------------------------------------------------
    # Pattern: unnumbered header followed by a run of numbered headers
    # with the same numbering depth (e.g. "Appendix" → "6 …", "7 …", …).
    # Promote the unnumbered header so it becomes the parent of the run,
    # and deepen the run by one level so _build nests them correctly.
    # ------------------------------------------------------------------
    n = len(nodes)
    i = 0
    while i < n:
        if _parse_number(nodes[i]["title"]):
            i += 1
            continue

        # Find the run of consecutive single-depth numbered nodes that
        # follow this unnumbered node and share its current (raw) level.
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
            # Promote this unnumbered header by one level — it becomes
            # a section container for the run, regardless of whether the
            # run numbers are strictly consecutive (e.g. "A, B, C" or
            # "6, 7, 8" both qualify).
            nodes[i]["level"] = nodes[i]["level"] - 1
            for k in range(i + 1, j):
                nodes[k]["level"] = nodes[k]["level"] + 1
            i = j
        else:
            i += 1


def _parse_number(title: str) -> tuple[int, ...] | None:
    """Extract a hierarchical number from a section title.

    Examples:
        "2.1 Object Counting" → (2, 1)
        "3.1.2 Something"     → (3, 1, 2)
        "Introduction"        → None
    """
    m = re.match(r"^(\d+(?:\.\d+)*)\b", title.strip())
    if not m:
        return None
    return tuple(int(x) for x in m.group(1).split("."))


def _is_numbered_run_container(title: str) -> bool:
    """Return true for unnumbered headings that can own numbered children.

    PDF-to-Markdown often emits all headings at the same ``##`` level.  We only
    apply the "unnumbered heading owns the following numbered run" heuristic to
    container-like headings.  Front/back matter such as Abstract must stay as a
    sibling of ``1. Introduction`` rather than becoming its parent.
    """
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
        start = node["line_num"] - 1  # 0-indexed
        if i + 1 < len(nodes):
            end = nodes[i + 1]["line_num"] - 1
        else:
            end = len(lines)
        node["text"] = "\n".join(lines[start:end]).strip()
