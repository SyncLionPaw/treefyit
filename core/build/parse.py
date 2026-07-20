"""parse: text/markdown -> flat sections (no LLM)."""

from __future__ import annotations

import re

from .types import Section

atx_heading_pattern = re.compile(r"^(#{1,6})\s+(.+?)\s*$")
setext_h1_pattern = re.compile(r"^=+\s*$")
setext_h2_pattern = re.compile(r"^-{2,}\s*$")
chapter_prefixes = ("chapter ", "第")


def parse_text(text: str, *, format: str = "md") -> list[Section]:
    """Normalize markdown/plaintext into flat sections with raw levels."""
    lines = text.splitlines()
    fmt = format.lower().strip()
    sections: list[Section] = []
    if fmt in {"md", "markdown", "text"}:
        sections = extract_markdown_sections(lines)
    if not sections and fmt in {"txt", "text", "plaintext", "md", "markdown"}:
        sections = extract_plaintext_sections(lines)
    if not sections:
        return build_single_section(text)

    fill_section_text(sections, lines)
    return sections


def parse_text_sections(text: str) -> list[Section]:
    """Backward-compatible alias for ``parse_text``."""
    return parse_text(text, format="md")


def extract_markdown_sections(lines: list[str]) -> list[Section]:
    sections: list[Section] = []
    index = 0
    while index < len(lines):
        line = lines[index]
        atx = atx_heading_pattern.match(line)
        if atx:
            sections.append(
                {
                    "title": atx.group(2).strip(),
                    "level": len(atx.group(1)),
                    "line_num": index + 1,
                }
            )
            index += 1
            continue

        if index + 1 < len(lines):
            nxt = lines[index + 1]
            title = line.strip()
            if title and setext_h1_pattern.match(nxt):
                sections.append({"title": title, "level": 1, "line_num": index + 1})
                index += 2
                continue
            if title and setext_h2_pattern.match(nxt):
                sections.append({"title": title, "level": 2, "line_num": index + 1})
                index += 2
                continue
        index += 1
    return sections


def extract_plaintext_sections(lines: list[str]) -> list[Section]:
    sections: list[Section] = []
    for line_num, raw in enumerate(lines, start=1):
        title = raw.strip()
        if not title:
            continue
        if looks_like_chapter_title(title):
            sections.append({"title": title, "level": 1, "line_num": line_num})
    return sections


def looks_like_chapter_title(title: str) -> bool:
    lowered = title.lower()
    return lowered.startswith(chapter_prefixes) and len(title.split()) <= 8


def fill_section_text(sections: list[Section], lines: list[str]) -> None:
    total = len(lines)
    for index, section in enumerate(sections):
        start = max(int(section.get("line_num", 1)), 1)
        end = total + 1
        if index + 1 < len(sections):
            end = max(int(sections[index + 1].get("line_num", total + 1)), start)
        # body starts after the heading line
        body = lines[start : end - 1]
        section["text"] = "\n".join(body).strip()


def build_single_section(text: str) -> list[Section]:
    stripped = text.strip()
    if not stripped:
        return []
    first = stripped.splitlines()[0].strip() or "Document"
    return [{"title": first, "level": 1, "line_num": 1, "text": stripped}]
