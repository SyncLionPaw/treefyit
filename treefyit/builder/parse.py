"""Parsing and intermediate tree assembly.

This module normalizes different source formats into flat sections or Markdown-derived
sections, and provides the stack-based nested dict tree builder. It does not own the
final typed model conversion or hierarchy inference policy.
"""

from __future__ import annotations

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
    return parse_html_fallback(path)


def parse_pdf_sections(path: Path) -> list[dict]:
    markdown_text = convert_document_to_markdown(path)
    return parse_text_sections(markdown_text)


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

        body = lines[start:end - 1]
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
        title = soup.title.string.strip() if soup.title and soup.title.string else path.name
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
    "parse_html_sections",
    "parse_pdf_sections",
    "parse_text_sections",
]
