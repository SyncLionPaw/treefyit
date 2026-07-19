"""Parsing and intermediate tree assembly.

This module normalizes different source formats into flat sections or Markdown-derived
sections, and provides the stack-based nested dict tree builder. It does not own the
final typed model conversion or hierarchy inference policy.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from pathlib import Path

from bs4 import BeautifulSoup
from markdown_it import MarkdownIt
from markitdown import MarkItDown


markdown_parser = MarkdownIt("commonmark")
chapter_prefixes = ("chapter ", "第")


def build_legacy_tree_from_sections(sections: list[dict]) -> list[dict]:
    stack: list[tuple[dict, int]] = []
    roots: list[dict] = []

    for section in sections:
        node = build_legacy_node(section)
        level = max(int(section.get("level", 1)), 1)

        while stack and stack[-1][1] >= level:
            stack.pop()

        if not stack:
            roots.append(node)
        else:
            stack[-1][0]["children"].append(node)

        stack.append((node, level))

    clean_children(roots)
    return roots


def build_legacy_node(section: dict) -> dict:
    node = {
        "title": section["title"],
        "text": section.get("text", ""),
        "children": [],
    }
    for key in ("leaf_type", "content_kind", "url", "uri", "media_type", "summary"):
        value = section.get(key)
        if value is not None:
            node[key] = value
    return node


def parse_text_sections(text: str) -> list[dict]:
    lines = text.splitlines()
    sections = extract_markdown_sections(text)
    if not sections:
        sections = extract_plaintext_sections(lines)
    if not sections:
        return build_single_section(text)

    fill_section_text(sections, lines)
    return sections


def parse_html_sections(path: Path) -> list[dict]:
    markdown_text = convert_document_to_markdown(path)
    if markdown_text:
        return parse_text_sections(markdown_text)
    fallback_sections = parse_html_fallback(path)
    script_sections = parse_script_rendered_html_sections(path)
    if should_use_script_sections(fallback_sections, script_sections):
        return script_sections
    return fallback_sections


def parse_pdf_sections(path: Path) -> list[dict]:
    markdown_text = convert_document_to_markdown(path)
    return parse_text_sections(markdown_text)


def parse_file_text(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".html", ".htm"}:
        return sections_to_text(parse_html_sections(path))
    if suffix == ".pdf":
        return sections_to_text(parse_pdf_sections(path))
    return path.read_text(encoding="utf-8", errors="replace").strip()


def sections_to_text(sections: list[dict]) -> str:
    chunks: list[str] = []
    for section in sections:
        title = str(section.get("title") or "").strip()
        text = str(section.get("text") or "").strip()
        if title:
            chunks.append(title)
        if text:
            chunks.append(text)
    return "\n\n".join(chunks).strip()


def clean_children(nodes: list[dict]) -> None:
    for node in nodes:
        children = node.get("children", [])
        if children:
            clean_children(children)
            continue
        node.pop("children", None)


def convert_document_to_markdown(path: Path) -> str:
    result = MarkItDown().convert(str(path))
    return getattr(result, "text_content", str(result)).strip()


def extract_markdown_sections(text: str) -> list[dict]:
    if not text.strip():
        return []

    tokens = markdown_parser.parse(text)
    sections: list[dict] = []

    for index, token in enumerate(tokens):
        if token.type != "heading_open":
            continue

        title = extract_heading_title(tokens, index)
        if not title:
            continue

        sections.append(
            {
                "title": title,
                "level": int(token.tag[1]),
                "line_num": (token.map[0] + 1) if token.map else 1,
            }
        )

    return sections


def extract_heading_title(tokens: list, index: int) -> str:
    if index + 1 >= len(tokens):
        return ""
    next_token = tokens[index + 1]
    if next_token.type != "inline":
        return ""
    return next_token.content.strip()


def extract_plaintext_sections(lines: list[str]) -> list[dict]:
    sections: list[dict] = []

    for line_num, raw_line in enumerate(lines, start=1):
        title = raw_line.strip()
        if not title:
            continue
        if looks_like_chapter_title(title):
            sections.append({"title": title, "level": 1, "line_num": line_num})

    return sections


def looks_like_chapter_title(title: str) -> bool:
    lowered = title.lower()
    return lowered.startswith(chapter_prefixes) and len(title.split()) <= 8


def fill_section_text(sections: list[dict], lines: list[str]) -> None:
    total_lines = len(lines)

    for index, section in enumerate(sections):
        start = max(int(section.get("line_num", 1)), 1)
        end = total_lines + 1
        if index + 1 < len(sections):
            end = max(int(sections[index + 1].get("line_num", total_lines + 1)), start)

        body = lines[start : end - 1]
        section["text"] = "\n".join(body).strip()


def build_single_section(text: str) -> list[dict]:
    stripped = text.strip()
    if not stripped:
        return []

    first_line = stripped.splitlines()[0].strip() or "Document"
    return [{"title": first_line, "level": 1, "line_num": 1, "text": stripped}]


def parse_html_fallback(path: Path) -> list[dict]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "lxml")
    headings = soup.find_all(["h1", "h2", "h3", "h4", "h5", "h6"])
    if not headings:
        title = (
            soup.title.string.strip() if soup.title and soup.title.string else path.name
        )
        text = soup.get_text("\n", strip=True)
        return [{"title": title, "level": 1, "line_num": 1, "text": text}]

    sections: list[dict] = []
    for heading in headings:
        title = heading.get_text(" ", strip=True)
        if not title:
            continue
        level = int(heading.name[1])
        text = collect_heading_text(heading)
        sections.append({"title": title, "level": level, "line_num": 1, "text": text})

    return sections


def parse_script_rendered_html_sections(path: Path) -> list[dict]:
    soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="replace"), "lxml")
    script_text = "\n".join(script.get_text() for script in soup.find_all("script"))
    if not script_text:
        return []

    anchors = extract_react_rendered_anchors(script_text)
    if not anchors:
        return []

    text_items = extract_react_rendered_text_items(script_text)
    sections: list[dict] = []
    for index, anchor in enumerate(anchors):
        next_line = anchors[index + 1]["line_num"] if index + 1 < len(anchors) else None
        body = [
            item["text"]
            for item in text_items
            if item["line_num"] > anchor["line_num"]
            and (next_line is None or item["line_num"] < next_line)
            and item["text"] != anchor["title"]
        ]
        if anchor.get("text"):
            body.insert(0, str(anchor["text"]))
        sections.append(
            {
                "title": anchor["title"],
                "level": anchor["level"],
                "line_num": anchor["line_num"],
                "text": "\n".join(dedupe_keep_order(body)).strip(),
            }
        )
    return sections


def should_use_script_sections(
    fallback_sections: list[dict], script_sections: list[dict]
) -> bool:
    if len(script_sections) <= len(fallback_sections):
        return False
    fallback_text = "\n".join(
        f"{section.get('title', '')}\n{section.get('text', '')}"
        for section in fallback_sections
    ).strip()
    return len(fallback_text) < 300


def extract_react_rendered_anchors(script_text: str) -> list[dict]:
    anchors: list[dict] = []

    heading_pattern = re.compile(
        r"\(`h([1-6])`,\{\"data-source\":`src/index\.tsx:(\d+):\d+`"
        r'(?:(?!"data-source":`).){0,1200}?children:`((?:\\`|[^`])*)`',
        re.DOTALL,
    )
    for match in heading_pattern.finditer(script_text):
        anchors.append(
            {
                "line_num": int(match.group(2)),
                "level": int(match.group(1)),
                "title": clean_javascript_text(match.group(3)),
                "text": "",
            }
        )

    section_pattern = re.compile(
        r'We,\{"data-source":`src/index\.tsx:(\d+):\d+`'
        r"(?:(?!\}\)\}\)).){0,500}?id:`[^`]+`"
        r"(?:(?!\}\)\}\)).){0,300}?eyebrow:`((?:\\`|[^`])*)`"
        r"(?:(?!\}\)\}\)).){0,300}?title:`((?:\\`|[^`])*)`"
        r"(?:(?!\}\)\}\)).){0,500}?desc:`((?:\\`|[^`])*)`",
        re.DOTALL,
    )
    for match in section_pattern.finditer(script_text):
        eyebrow = clean_javascript_text(match.group(2))
        desc = clean_javascript_text(match.group(4))
        anchors.append(
            {
                "line_num": int(match.group(1)),
                "level": 2,
                "title": clean_javascript_text(match.group(3)),
                "text": "\n".join(part for part in (eyebrow, desc) if part),
            }
        )

    card_pattern = re.compile(
        r'\bD,\{"data-source":`src/index\.tsx:(\d+):\d+`'
        r'(?:(?!"data-source":`).){0,500}?title:`((?:\\`|[^`])*)`',
        re.DOTALL,
    )
    for match in card_pattern.finditer(script_text):
        anchors.append(
            {
                "line_num": int(match.group(1)),
                "level": 3,
                "title": clean_javascript_text(match.group(2)),
                "text": "",
            }
        )

    cleaned = [
        anchor
        for anchor in anchors
        if anchor["title"] and not looks_like_internal_render_text(anchor["title"])
    ]
    cleaned.sort(key=lambda anchor: (anchor["line_num"], anchor["level"]))
    return dedupe_anchors(cleaned)


def extract_react_rendered_text_items(script_text: str) -> list[dict]:
    text_pattern = re.compile(
        r'"data-source":`src/index\.tsx:(\d+):\d+`'
        r'(?:(?!"data-source":`).){0,1200}?children:`((?:\\`|[^`])*)`',
        re.DOTALL,
    )
    items: list[dict] = []
    for match in text_pattern.finditer(script_text):
        text = clean_javascript_text(match.group(2))
        if not text or looks_like_internal_render_text(text):
            continue
        items.append({"line_num": int(match.group(1)), "text": text})
    items.sort(key=lambda item: item["line_num"])
    return items


def clean_javascript_text(text: str) -> str:
    return text.replace("\\`", "`").replace("\\n", "\n").replace("\\\\", "\\").strip()


def looks_like_internal_render_text(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if stripped in {"Render Error", "Stack trace", "Retry", "Copy"}:
        return True
    if len(stripped) > 1200:
        return True
    return False


def dedupe_anchors(anchors: list[dict]) -> list[dict]:
    seen: set[tuple[int, str]] = set()
    deduped: list[dict] = []
    for anchor in anchors:
        key = (anchor["line_num"], anchor["title"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(anchor)
    return deduped


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def collect_heading_text(heading) -> str:
    chunks: list[str] = []
    for sibling in iter_following_siblings(heading):
        name = getattr(sibling, "name", None)
        if name in {"h1", "h2", "h3", "h4", "h5", "h6"}:
            break
        text = sibling.get_text(" ", strip=True) if hasattr(sibling, "get_text") else ""
        if text:
            chunks.append(text)
    return "\n".join(chunks).strip()


def iter_following_siblings(tag) -> Iterable:
    current = tag.next_sibling
    while current is not None:
        yield current
        current = current.next_sibling


__all__ = [
    "build_legacy_tree_from_sections",
    "parse_file_text",
    "parse_html_sections",
    "parse_pdf_sections",
    "parse_text_sections",
]
