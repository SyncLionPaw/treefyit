"""Tests for core rule-based build pipeline (parse → infer → refine → assemble)."""

from __future__ import annotations

from pathlib import Path

from core import build_tree_from_file, build_tree_from_text
from core.build import infer_levels, parse_text, refine_sections
from core.tree import NodeKind


SAMPLE_MD = """# 白茶简介

白茶是中国六大茶类之一。

## 历史与产地

白茶历史可追溯至唐代。

### 核心产区

福鼎与政和。

## 冲泡建议

水温 85 度。
"""


def test_build_tree_from_markdown_headings():
    root = build_tree_from_text(SAMPLE_MD, root_title="tea", refine=False)
    assert root.title == "tea"
    assert [c.title for c in root.children] == ["白茶简介"]
    intro = root.children[0]
    assert intro.content and "六大茶类" in intro.content
    assert [c.title for c in intro.children] == ["历史与产地", "冲泡建议"]
    assert intro.children[0].children[0].title == "核心产区"


def test_infer_levels_from_numbering():
    sections = parse_text(
        """# 1. Intro

alpha

# 1.1 Nested

beta

# 2. Next

gamma
"""
    )
    inferred = infer_levels(sections)
    levels = [s["level"] for s in inferred]
    assert levels[0] == 1
    assert levels[1] == 2
    assert levels[2] == 1


def test_infer_chinese_title_prefixes():
    sections = [
        {"title": "一、概述", "level": 1, "line_num": 1, "text": "a"},
        {"title": "（一）细节", "level": 1, "line_num": 2, "text": "b"},
    ]
    inferred = infer_levels(sections)
    assert inferred[0]["level"] == 1
    assert inferred[1]["level"] == 2


def test_refine_splits_long_section():
    long_body = "\n\n".join(f"Paragraph {i} " + ("x" * 80) for i in range(1, 6))
    sections = [
        {
            "title": "Long",
            "level": 1,
            "line_num": 1,
            "text": long_body,
        }
    ]
    refined = refine_sections(sections, split_threshold=100, max_parts=3)
    assert refined[0]["title"] == "Long"
    assert refined[0]["text"] == ""
    assert refined[0].get("summary")
    children = refined[1:]
    assert len(children) == 3
    assert all(c["level"] == 2 for c in children)
    assert children[0]["title"].endswith("Part 1")


def test_refine_detects_table_kind():
    sections = [
        {
            "title": "Grades",
            "level": 1,
            "line_num": 1,
            "text": (
                "| Grade | Note |\n"
                "|------|------|\n"
                "| A | top |\n"
            ),
        }
    ]
    refined = refine_sections(sections)
    assert refined[0]["leaf_type"] == "table"
    root = build_tree_from_text(
        "# Grades\n\n| Grade | Note |\n|------|------|\n| A | top |\n",
        refine=True,
    )
    grades = root.children[0]
    assert grades.kind == NodeKind.TABLE


def test_build_white_tea_file():
    path = Path("examples/agent_tree/white_tea.md")
    root = build_tree_from_file(path, refine=True)
    titles = [c.title for c in root.children]
    assert "White Tea Guide" in titles or root.children[0].title == "White Tea Guide"
    guide = root.children[0]
    child_titles = {c.title for c in guide.children}
    assert "History and Origins" in child_titles
    assert "Main Grades" in child_titles
    grades = next(c for c in guide.children if c.title == "Main Grades")
    assert grades.kind == NodeKind.TABLE
    assert grades.content and "Silver Needle" in grades.content
