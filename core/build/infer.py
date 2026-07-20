"""infer: repair section hierarchy with numbering / Chinese title heuristics."""

from __future__ import annotations

import re

from .types import Section

number_prefix_pattern = re.compile(r"^\s*(\d+(?:\.\d+)*)(?:[.)]|\s)\s*")
chinese_main_prefix_pattern = re.compile(r"^\s*[一二三四五六七八九十]+、")
chinese_sub_prefix_pattern = re.compile(r"^\s*[（(][一二三四五六七八九十]+[)）]")


def infer_levels(sections: list[Section]) -> list[Section]:
    if not sections:
        return []

    min_base = min(max(int(section.get("level", 1)), 1) for section in sections)
    inferred: list[Section] = []
    for section in sections:
        base = max(int(section.get("level", 1)) - min_base + 1, 1)
        title = str(section.get("title", "")).strip()
        from_title = infer_title_level(title)
        level = base if from_title is None else max(base, from_title)
        item = dict(section)
        item["level"] = level
        inferred.append(item)  # type: ignore[arg-type]
    return inferred


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
