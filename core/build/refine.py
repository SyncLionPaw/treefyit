"""refine: split oversized sections and detect table/image/link kinds (no LLM)."""

from __future__ import annotations

import re

from .types import Section

url_pattern = re.compile(r"https?://[^\s)>\]]+")
image_pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")

DEFAULT_SPLIT_THRESHOLD = 400
DEFAULT_MAX_PARTS = 4


def refine_sections(
    sections: list[Section],
    *,
    split_threshold: int = DEFAULT_SPLIT_THRESHOLD,
    max_parts: int = DEFAULT_MAX_PARTS,
) -> list[Section]:
    refined: list[Section] = []
    for section in sections:
        normalized = normalize_section(section)
        refined.extend(
            refine_one(
                normalized,
                split_threshold=split_threshold,
                max_parts=max_parts,
            )
        )
    return refined


def normalize_section(section: Section) -> Section:
    return {
        "title": str(section.get("title", "")).strip() or "Untitled",
        "level": max(int(section.get("level", 1)), 1),
        "line_num": max(int(section.get("line_num", 1)), 1),
        "text": str(section.get("text", "")).strip(),
        **{
            key: section[key]
            for key in ("summary", "leaf_type", "content_kind", "url", "uri", "media_type")
            if key in section
        },
    }


def refine_one(
    section: Section,
    *,
    split_threshold: int,
    max_parts: int,
) -> list[Section]:
    refined = dict(section)
    refined["text"] = str(refined.get("text", "")).strip()
    apply_content_detection(refined)
    return split_long_section(
        refined,  # type: ignore[arg-type]
        split_threshold=split_threshold,
        max_parts=max_parts,
    )


def apply_content_detection(section: dict) -> None:
    text = str(section.get("text", "")).strip()
    if not text:
        return

    image_match = image_pattern.search(text)
    if image_match:
        section["leaf_type"] = "image"
        section["content_kind"] = "resource"
        section["uri"] = image_match.group(1)
        return

    if looks_like_table(text):
        section["leaf_type"] = "table"
        section["content_kind"] = "text"
        return

    urls = url_pattern.findall(text)
    if len(urls) == 1 and is_link_like_text(text, urls[0]):
        section["leaf_type"] = "link"
        section["content_kind"] = "url"
        section["url"] = urls[0]
        return

    section["leaf_type"] = "text"
    section["content_kind"] = "text"


def split_long_section(
    section: Section,
    *,
    split_threshold: int,
    max_parts: int,
) -> list[Section]:
    text = str(section.get("text", "")).strip()
    if len(text) < split_threshold:
        return [section]

    paragraphs = [chunk.strip() for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
    if len(paragraphs) < 2:
        return [section]

    paragraphs = paragraphs[:max_parts]
    parent = dict(section)
    parent["summary"] = build_parent_summary(parent, paragraphs)
    parent["text"] = ""
    parent["leaf_type"] = None
    parent["content_kind"] = None
    parent.pop("url", None)
    parent.pop("uri", None)
    parent.pop("media_type", None)

    parts: list[Section] = [parent]  # type: ignore[list-item]
    for index, paragraph in enumerate(paragraphs, start=1):
        child: Section = {
            "title": f"{parent['title']} / Part {index}",
            "level": int(parent["level"]) + 1,
            "line_num": int(parent.get("line_num", 1)),
            "text": paragraph,
        }
        apply_content_detection(child)
        parts.append(child)
    return parts


def build_parent_summary(
    section: dict,
    paragraphs: list[str],
    *,
    max_length: int = 160,
) -> str | None:
    summary = str(section.get("summary") or "").strip()
    if summary:
        return summary
    text = " ".join(" ".join(paragraphs).split())
    if not text:
        return None
    if len(text) <= max_length:
        return text
    return text[: max_length - 1].rstrip() + "…"


def looks_like_table(text: str) -> bool:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    pipe_lines = [line for line in lines if line.count("|") >= 2]
    if len(pipe_lines) < 2:
        return False
    return any(re.fullmatch(r"[\|\s:\-]+", line) for line in pipe_lines[1:3])


def is_link_like_text(text: str, url: str) -> bool:
    trimmed = text.strip()
    if trimmed == url:
        return True
    remainder = trimmed.replace(url, "").strip(" -:\n\t")
    return len(remainder) <= 24
