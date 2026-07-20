"""Section refinement strategies.

This module refines flat sections after hierarchy inference and before tree
assembly. It can split coarse sections, clean weak content blocks, and detect
special content types. It does not build nested trees or convert sections into
the final typed model.
"""

from __future__ import annotations

import re
from typing import Any, Protocol

from src.config import get_settings
from src.llm.refine import refine_section
from src.model.tree import LeafType


type Section = dict[str, Any]


url_pattern = re.compile(r"https?://[^\s)>\]]+")
image_pattern = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")


class SectionRefiner(Protocol):
    def refine(
        self,
        sections: list[Section],
        *,
        text: str | None = None,
        source_kind: str | None = None,
    ) -> list[Section]: ...


class NoopSectionRefiner:
    def refine(
        self,
        sections: list[Section],
        *,
        text: str | None = None,
        source_kind: str | None = None,
    ) -> list[Section]:
        return [dict(section) for section in sections]


class LLMSectionRefiner:
    def __init__(
        self,
        *,
        model: str | None = None,
        split_threshold: int | None = None,
        max_parts: int | None = None,
        max_tokens: int | None = None,
    ) -> None:
        settings = get_settings().builder
        self.model = model
        self.split_threshold = (
            split_threshold
            if split_threshold is not None
            else settings.refine_split_threshold
        )
        self.max_parts = (
            max_parts if max_parts is not None else settings.refine_max_parts
        )
        self.max_tokens = max_tokens
        self.rule_based_refiner = RuleBasedSectionRefiner(
            split_threshold=self.split_threshold,
            max_parts=self.max_parts,
        )

    def refine(
        self,
        sections: list[Section],
        *,
        text: str | None = None,
        source_kind: str | None = None,
    ) -> list[Section]:
        refined_sections: list[Section] = []

        for section in sections:
            normalized = normalize_section(section)
            refined_sections.extend(
                self.refine_one(normalized, source_kind=source_kind)
            )

        return refined_sections

    def refine_one(
        self,
        section: Section,
        *,
        source_kind: str | None = None,
    ) -> list[Section]:
        text = str(section.get("text", "")).strip()
        if not text:
            refined = dict(section)
            apply_content_detection(refined)
            return [refined]

        if len(text) < self.split_threshold:
            return self.rule_based_refiner.refine_one(section)

        llm_sections = refine_section(
            title=str(section.get("title", "")).strip() or "Untitled",
            content=text,
            source_kind=source_kind,
            model=self.model,
            max_parts=self.max_parts,
            max_tokens=self.max_tokens,
        )
        if not llm_sections:
            return self.rule_based_refiner.refine_one(section)

        mapped_sections = map_llm_sections(section, llm_sections)
        if not mapped_sections:
            return self.rule_based_refiner.refine_one(section)

        return mapped_sections


class RuleBasedSectionRefiner:
    def __init__(
        self,
        *,
        split_threshold: int | None = None,
        max_parts: int | None = None,
    ) -> None:
        settings = get_settings().builder
        self.split_threshold = (
            split_threshold
            if split_threshold is not None
            else settings.refine_split_threshold
        )
        self.max_parts = (
            max_parts if max_parts is not None else settings.refine_max_parts
        )

    def refine(
        self,
        sections: list[Section],
        *,
        text: str | None = None,
        source_kind: str | None = None,
    ) -> list[Section]:
        refined_sections: list[Section] = []

        for section in sections:
            normalized = normalize_section(section)
            refined_sections.extend(self.refine_one(normalized))

        return refined_sections

    def refine_one(self, section: Section) -> list[Section]:
        refined = dict(section)
        text = str(refined.get("text", "")).strip()
        refined["text"] = text

        apply_content_detection(refined)
        split_sections = split_long_section(
            refined,
            split_threshold=self.split_threshold,
            max_parts=self.max_parts,
        )
        return split_sections


def normalize_section(section: Section) -> Section:
    normalized = dict(section)
    normalized["title"] = str(section.get("title", "")).strip() or "Untitled"
    normalized["level"] = max(int(section.get("level", 1)), 1)
    normalized["line_num"] = max(int(section.get("line_num", 1)), 1)
    normalized["text"] = str(section.get("text", "")).strip()
    return normalized


def apply_content_detection(section: Section) -> None:
    text = str(section.get("text", "")).strip()
    if not text:
        return

    image_match = image_pattern.search(text)
    if image_match:
        section["leaf_type"] = LeafType.IMAGE.value
        section["content_kind"] = "resource"
        section["uri"] = image_match.group(1)
        return

    if looks_like_table(text):
        section["leaf_type"] = LeafType.TABLE.value
        section["content_kind"] = "text"
        return

    urls = url_pattern.findall(text)
    if len(urls) == 1 and is_link_like_text(text, urls[0]):
        section["leaf_type"] = LeafType.LINK.value
        section["content_kind"] = "url"
        section["url"] = urls[0]
        return

    section["leaf_type"] = LeafType.TEXT.value
    section["content_kind"] = "text"


def map_llm_sections(base_section: Section, llm_sections: list[dict]) -> list[Section]:
    base_level = max(int(base_section.get("level", 1)), 1)
    base_line_num = max(int(base_section.get("line_num", 1)), 1)
    mapped_sections: list[Section] = []

    for llm_section in llm_sections:
        title = str(llm_section.get("title", "")).strip()
        if not title:
            continue

        level_delta = llm_section.get("level_delta", 0)
        if level_delta not in {0, 1}:
            level_delta = 0

        mapped: Section = {
            "title": title,
            "level": base_level + int(level_delta),
            "line_num": base_line_num,
            "text": str(llm_section.get("text", "")).strip(),
        }

        for key in (
            "summary",
            "leaf_type",
            "content_kind",
            "url",
            "uri",
            "media_type",
        ):
            value = llm_section.get(key)
            if value is not None:
                mapped[key] = value

        if "leaf_type" not in mapped or "content_kind" not in mapped:
            apply_content_detection(mapped)

        mapped_sections.append(mapped)

    return mapped_sections


def split_long_section(
    section: Section,
    *,
    split_threshold: int,
    max_parts: int,
) -> list[Section]:
    text = str(section.get("text", "")).strip()
    if len(text) < split_threshold:
        return [section]

    paragraphs = [
        chunk.strip() for chunk in re.split(r"\n\s*\n", text) if chunk.strip()
    ]
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

    refined_sections: list[Section] = [parent]
    for index, paragraph in enumerate(paragraphs, start=1):
        child = {
            "title": f"{parent['title']} / Part {index}",
            "level": parent["level"] + 1,
            "line_num": parent.get("line_num", 1),
            "text": paragraph,
        }
        apply_content_detection(child)
        refined_sections.append(child)

    return refined_sections


def build_parent_summary(
    section: Section,
    paragraphs: list[str],
    *,
    max_length: int = 160,
) -> str | None:
    summary = str(section.get("summary", "")).strip()
    if summary:
        return summary

    text = " ".join(paragraphs)
    normalized = " ".join(text.split())
    if not normalized:
        return None
    if len(normalized) <= max_length:
        return normalized
    return normalized[: max_length - 1].rstrip() + "…"


def looks_like_table(text: str) -> bool:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    if len(lines) < 2:
        return False
    pipe_lines = [line for line in lines if line.count("|") >= 2]
    if len(pipe_lines) < 2:
        return False
    separator_like = any(re.fullmatch(r"[\|\s:\-]+", line) for line in pipe_lines[1:3])
    return separator_like


def is_link_like_text(text: str, url: str) -> bool:
    trimmed = text.strip()
    if trimmed == url:
        return True
    trimmed_without_url = trimmed.replace(url, "").strip(" -:\n\t")
    return len(trimmed_without_url) <= 24


__all__ = [
    "LLMSectionRefiner",
    "NoopSectionRefiner",
    "RuleBasedSectionRefiner",
    "SectionRefiner",
]
