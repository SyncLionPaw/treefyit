"""LLM-backed section refinement helpers."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from treefyit.llm.client import complete
from treefyit.llm.prompts import (
    SECTION_REFINER_SYSTEM_PROMPT,
    build_section_refine_prompt,
)


class RefinedSectionItem(BaseModel):
    title: str = Field(min_length=1)
    level_delta: Literal[0, 1] = 0
    text: str = ""
    summary: str | None = None
    leaf_type: Literal["text", "image", "table", "link"] | None = None
    content_kind: Literal["text", "url", "resource"] | None = None
    url: str | None = None
    uri: str | None = None
    media_type: str | None = None


class RefinedSectionResponse(BaseModel):
    sections: list[RefinedSectionItem] = Field(default_factory=list)


def refine_section(
    *,
    title: str,
    content: str,
    source_kind: str | None = None,
    model: str | None = None,
    max_parts: int = 4,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> list[dict[str, Any]]:
    prompt = build_section_refine_prompt(
        title=title,
        content=content,
        source_kind=source_kind,
        max_parts=max_parts,
    )
    response_text = complete(
        prompt,
        system=SECTION_REFINER_SYSTEM_PROMPT,
        model=model,
        max_tokens=max_tokens,
        **kwargs,
    )
    payload = parse_refined_section_response(response_text)
    return [item.model_dump(exclude_none=True) for item in payload.sections]


def parse_refined_section_response(response_text: str) -> RefinedSectionResponse:
    blocks = extract_section_blocks(response_text)
    if blocks:
        items = [parse_section_block(block) for block in blocks]
        return RefinedSectionResponse(sections=items)

    json_text = extract_json_text(response_text)
    return RefinedSectionResponse.model_validate_json(json_text)


def extract_section_blocks(response_text: str) -> list[str]:
    stripped = strip_code_fence(response_text)
    return re.findall(r"\[SECTION\]\s*(.*?)\s*\[/SECTION\]", stripped, flags=re.S)


def parse_section_block(block_text: str) -> RefinedSectionItem:
    text = extract_block_value(block_text, "TEXT")
    summary = extract_block_value(block_text, "SUMMARY")
    header_text = remove_named_blocks(block_text, ["TEXT", "SUMMARY"])
    fields = parse_header_fields(header_text)

    title = fields.get("title", "").strip()
    if not title:
        raise ValueError("Refined section block missing title")

    level_delta_text = fields.get("level_delta", "0").strip()
    level_delta = 0 if level_delta_text != "1" else 1

    return RefinedSectionItem(
        title=title,
        level_delta=level_delta,
        text=text,
        summary=summary or None,
        leaf_type=normalize_optional_literal(
            fields.get("leaf_type"),
            {"text", "image", "table", "link"},
        ),
        content_kind=normalize_optional_literal(
            fields.get("content_kind"),
            {"text", "url", "resource"},
        ),
        url=normalize_optional_text(fields.get("url")),
        uri=normalize_optional_text(fields.get("uri")),
        media_type=normalize_optional_text(fields.get("media_type")),
    )


def extract_json_text(response_text: str) -> str:
    stripped = strip_code_fence(response_text)
    return stripped


def strip_code_fence(response_text: str) -> str:
    stripped = response_text.strip()
    if not stripped.startswith("```"):
        return stripped

    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines).strip()


def extract_block_value(block_text: str, name: str) -> str:
    pattern = rf"\[{name}\]\s*(.*?)\s*\[/{name}\]"
    match = re.search(pattern, block_text, flags=re.S)
    if not match:
        if name == "TEXT":
            raise ValueError("Refined section block missing TEXT")
        return ""
    return match.group(1).strip()


def remove_named_blocks(block_text: str, names: list[str]) -> str:
    header_text = block_text
    for name in names:
        pattern = rf"\[{name}\]\s*.*?\s*\[/{name}\]"
        header_text = re.sub(pattern, "", header_text, flags=re.S)
    return header_text


def parse_header_fields(header_text: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    for raw_line in header_text.splitlines():
        line = raw_line.strip()
        if not line or ":" not in line:
            continue
        key, value = line.split(":", 1)
        fields[key.strip()] = value.strip()
    return fields


def normalize_optional_text(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped


def normalize_optional_literal(
    value: str | None,
    allowed: set[str],
) -> str | None:
    normalized = normalize_optional_text(value)
    if normalized is None:
        return None
    if normalized not in allowed:
        return None
    return normalized


__all__ = [
    "RefinedSectionItem",
    "RefinedSectionResponse",
    "extract_section_blocks",
    "extract_json_text",
    "parse_section_block",
    "parse_refined_section_response",
    "refine_section",
]
