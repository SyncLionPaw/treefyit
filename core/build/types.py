"""Flat section type used across the rule-based build pipeline."""

from __future__ import annotations

from typing import Any, TypedDict


class Section(TypedDict, total=False):
    title: str
    level: int
    line_num: int
    text: str
    summary: str | None
    leaf_type: str | None
    content_kind: str | None
    url: str | None
    uri: str | None
    media_type: str | None


def as_section(data: dict[str, Any]) -> Section:
    return Section(**data)  # type: ignore[misc]
