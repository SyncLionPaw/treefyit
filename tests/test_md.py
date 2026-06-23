"""Markdown structure extraction tests — 30 cases, fixtures + edge cases."""

from __future__ import annotations

from pathlib import Path

from src.parser.md import (
    looks_like_chapter_novel,
    parse_md,
    parse_md_text,
)
from src.tree.builder import build_nodes

TESTFILE = Path(__file__).resolve().parent / "testfile"


def _parse(path: str) -> list[dict]:
    return parse_md(str(TESTFILE / path))


def _titles(path: str) -> list[str]:
    return [n["title"] for n in _parse(path)]


def _levels(path: str) -> list[int]:
    return [n["level"] for n in _parse(path)]


# ---- 1–7: core markdown-it behavior ----


def test_01_atx_headings_and_nesting():
    nodes = parse_md_text("# Doc\n\n## A\n\n### A.1\n")
    assert [n["title"] for n in nodes] == ["Doc", "A", "A.1"]
    assert [n["level"] for n in nodes] == [1, 2, 3]
    tree = build_nodes(nodes)
    assert tree[0]["children"][0]["children"][0]["title"] == "A.1"


def test_02_ignores_headings_in_fenced_code():
    nodes = parse_md_text("# Real\n\n```\n# fake\n## fake\n```\n\n## OK\n")
    assert [n["title"] for n in nodes] == ["Real", "OK"]


def test_03_setext_headings_inline():
    nodes = parse_md_text("Top\n=====\n\nBody\n\nSub\n-----\n")
    assert [(n["title"], n["level"]) for n in nodes] == [
        ("Top", 1),
        ("Sub", 2),
    ]


def test_04_atx_closing_hashes():
    nodes = parse_md_text("## Title ##\n\nbody\n")
    assert nodes[0]["title"] == "Title"


def test_05_h6_deepest_atx_level():
    nodes = parse_md_text("###### Six\n\ndeep\n")
    assert nodes[0]["level"] == 6


def test_06_skipped_heading_levels():
    nodes = parse_md_text("# Root\n\n#### Jump\n")
    assert [n["level"] for n in nodes] == [1, 4]


def test_07_inline_hash_not_a_heading():
    nodes = parse_md_text("hello # world\n\n# Real\n")
    assert [n["title"] for n in nodes] == ["Real"]


# ---- 8–10: numbering & plain text ----


def test_08_numbering_inference_flat_headers():
    nodes = parse_md_text("## 1 Intro\n\n## 2.1 Sub\n\n## 2.2 Sub2\n")
    assert [n["level"] for n in nodes] == [2, 3, 3]
    tree = build_nodes(nodes)
    assert len(tree) == 1 and tree[0]["title"] == "1 Intro"


def test_09_plaintext_chapter_hui_inline():
    md = "\n".join(f"第{i:02d}回 标题{i}" for i in range(1, 8))
    nodes = parse_md_text(md)
    assert len(nodes) == 7 and all(n["level"] == 1 for n in nodes)


def test_10_heading_with_inline_formatting():
    nodes = parse_md_text("# Hello **bold** and `code`\n")
    assert nodes[0]["title"] == "Hello **bold** and `code`"


# ---- 11–17: standard fixtures ----


def test_11_short_md_white_tea():
    assert len(_titles("short.md")) == 7
    assert _titles("short.md")[0] == "白茶简介"
    tree = build_nodes(_parse("short.md"))
    assert tree[0]["children"][0]["children"][0]["title"] == "核心产区"


def test_12_api_doc_nested_endpoints():
    tree = build_nodes(_parse("api-doc.md"))
    build = next(c for c in tree[0]["children"] if c["title"] == "Build")
    assert {c["title"] for c in build["children"]} == {
        "POST /api/build",
        "POST /api/build/stream",
    }


def test_13_code_heavy_fixture():
    assert _titles("code-heavy.md") == [
        "Parser Test Document",
        "Valid Section",
        "Another Valid Section",
        "Final Section",
    ]


def test_14_numbered_paper_pdf_style():
    tree = build_nodes(_parse("numbered-paper.md"))
    related = next(n for n in tree if n["title"] == "2 Related Work")
    assert {c["title"] for c in related["children"]} == {
        "2.1 Rule-based Methods",
        "2.2 Neural Methods",
    }


def test_15_chapters_txt_sanguo_style():
    nodes = _parse("chapters.txt")
    assert len(nodes) == 6 and looks_like_chapter_novel(nodes)
    assert len(build_nodes(nodes)) == 6


def test_16_setext_fixture_file():
    assert _titles("setext.md") == ["White Tea Guide", "Processing Steps", "Storage Notes"]


def test_17_deep_nested_fixture():
    titles = _titles("deep-nested.md")
    assert titles[0] == "Book" and "1.1 Tokens" in titles


# ---- 18–30: weird edge fixtures ----


def test_18_empty_and_whitespace():
    assert parse_md_text("") == []
    assert parse_md_text("   \n\n  \t\n") == []


def test_19_no_headings_unstructured():
    assert _parse("edge-no-headings.md") == []


def test_20_multiple_h1_siblings():
    nodes = parse_md_text("# One\n\n# Two\n\n# Three\n")
    assert len(build_nodes(nodes)) == 3


def test_21_chapter_en_plaintext_fixture():
    nodes = _parse("edge-chapters-en.txt")
    assert len(nodes) == 6
    assert nodes[0]["title"].startswith("Chapter 1")
    assert looks_like_chapter_novel(nodes)


def test_22_preface_not_counted_as_chapter():
    nodes = _parse("edge-preface-chapters.txt")
    assert len(nodes) == 5
    assert nodes[0]["title"] == "第一回 起"
    assert looks_like_chapter_novel(nodes)
    assert "前言" not in [n["title"] for n in nodes]


def test_23_duplicate_section_titles():
    titles = _titles("edge-duplicate-titles.md")
    assert titles.count("Summary") == 3
    tree = build_nodes(_parse("edge-duplicate-titles.md"))
    assert tree[0]["children"][0]["title"] == "Summary"


def test_24_unicode_and_emoji_titles():
    titles = _titles("edge-unicode.md")
    assert "茶·文化 🍵" in titles
    assert any("Ελληνικά" in t for t in titles)
    assert any("Русский" in t for t in titles)


def test_25_mixed_setext_and_atx():
    titles = _titles("edge-mixed-headings.md")
    assert titles[0] == "Main Title"
    assert "ATX inside setext doc" in titles
    assert "Sub section" in titles


def test_26_table_only_sections_still_have_text():
    nodes = _parse("edge-table-only.md")
    table_node = next(n for n in nodes if n["title"] == "Table Section")
    assert "| Key | Value |" in table_node["text"]
    empty = next(n for n in nodes if n["title"] == "Empty Section")
    assert empty["text"].strip() == "## Empty Section"


def test_27_blockquote_heading_leaks_through():
    """markdown-it emits headings inside blockquotes — known quirk."""
    titles = _titles("edge-blockquote.md")
    assert "Looks like a heading" in titles
    assert "Real Section" in titles


def test_28_heading_level_ladder():
    nodes = _parse("edge-heading-ladder.md")
    assert [n["level"] for n in nodes[:4]] == [1, 6, 5, 4]
    assert _titles("edge-heading-ladder.md")[-1] == "Another root-level h1"


def test_29_numeric_punctuation_titles():
    titles = _titles("edge-numeric-titles.md")
    assert titles[0] == "API v2.0 (2024)"
    assert "10.0 Breaking Changes" in titles
    assert "3.14 Pi Section" in titles


def test_30_chapter_novel_false_on_markdown_doc():
    assert not looks_like_chapter_novel(_parse("short.md"))
    assert not looks_like_chapter_novel(_parse("api-doc.md"))


ALL_TESTS = [
    test_01_atx_headings_and_nesting,
    test_02_ignores_headings_in_fenced_code,
    test_03_setext_headings_inline,
    test_04_atx_closing_hashes,
    test_05_h6_deepest_atx_level,
    test_06_skipped_heading_levels,
    test_07_inline_hash_not_a_heading,
    test_08_numbering_inference_flat_headers,
    test_09_plaintext_chapter_hui_inline,
    test_10_heading_with_inline_formatting,
    test_11_short_md_white_tea,
    test_12_api_doc_nested_endpoints,
    test_13_code_heavy_fixture,
    test_14_numbered_paper_pdf_style,
    test_15_chapters_txt_sanguo_style,
    test_16_setext_fixture_file,
    test_17_deep_nested_fixture,
    test_18_empty_and_whitespace,
    test_19_no_headings_unstructured,
    test_20_multiple_h1_siblings,
    test_21_chapter_en_plaintext_fixture,
    test_22_preface_not_counted_as_chapter,
    test_23_duplicate_section_titles,
    test_24_unicode_and_emoji_titles,
    test_25_mixed_setext_and_atx,
    test_26_table_only_sections_still_have_text,
    test_27_blockquote_heading_leaks_through,
    test_28_heading_level_ladder,
    test_29_numeric_punctuation_titles,
    test_30_chapter_novel_false_on_markdown_doc,
]


if __name__ == "__main__":
    assert len(ALL_TESTS) == 30
    for fn in ALL_TESTS:
        fn()
        print(f"  ok  {fn.__name__}")
    print("all 30 passed")
