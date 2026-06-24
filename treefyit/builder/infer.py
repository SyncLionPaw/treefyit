"""Hierarchy inference strategies.

This module adjusts flat section levels before tree assembly. It defines the pluggable
inferer interface and default implementations, but it does not parse files or build trees.
"""

from __future__ import annotations

import re
from typing import Any, Protocol


type Section = dict[str, Any]


number_prefix_pattern = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:[.)]|\s)\s*")
chinese_main_prefix_pattern = re.compile(r"^\s*[一二三四五六七八九十]+、")
chinese_sub_prefix_pattern = re.compile(r"^\s*[（(][一二三四五六七八九十]+[)）]")


class LevelInferer(Protocol):
    def infer(
        self,
        sections: list[Section],
        *,
        text: str | None = None,
        source_kind: str | None = None,
    ) -> list[Section]: ...


class NoopLevelInferer:
    def infer(
        self,
        sections: list[Section],
        *,
        text: str | None = None,
        source_kind: str | None = None,
    ) -> list[Section]:
        return [dict(section) for section in sections]


class LLMLevelInferer:
    def infer(
        self,
        sections: list[Section],
        *,
        text: str | None = None,
        source_kind: str | None = None,
    ) -> list[Section]:
        raise NotImplementedError(
            "LLMLevelInferer is not implemented yet. Provide an LLM-backed inferer."
        )


class RuleBasedLevelInferer:
    def infer(
        self,
        sections: list[Section],
        *,
        text: str | None = None,
        source_kind: str | None = None,
    ) -> list[Section]:
        if not sections:
            return []

        min_base_level = min(
            max(int(section.get("level", 1)), 1) for section in sections
        )
        inferred_sections: list[Section] = []

        for section in sections:
            normalized_base_level = max(
                int(section.get("level", 1)) - min_base_level + 1, 1
            )
            title = str(section.get("title", "")).strip()
            inferred_title_level = infer_title_level(title)
            final_level = normalized_base_level
            if inferred_title_level is not None:
                final_level = max(normalized_base_level, inferred_title_level)

            inferred_section = dict(section)
            inferred_section["level"] = final_level
            inferred_sections.append(inferred_section)

        return inferred_sections


def infer_title_level(title: str) -> int | None:
    if not title:
        return None

    number_match = number_prefix_pattern.match(title)
    if number_match:
        return len(number_match.group(1).split("."))
    if chinese_sub_prefix_pattern.match(title):
        return 2
    if chinese_main_prefix_pattern.match(title):
        return 1
    return None


__all__ = [
    "LevelInferer",
    "LLMLevelInferer",
    "NoopLevelInferer",
    "RuleBasedLevelInferer",
]
