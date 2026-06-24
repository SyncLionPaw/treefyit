from __future__ import annotations

from pathlib import Path

import pytest

from treefyit.builder import (
    BuildOptions,
    LLMLevelInferer,
    LLMSectionRefiner,
    RuleBasedLevelInferer,
    RuleBasedSectionRefiner,
    build_tree_from_file,
    build_tree_from_sections,
    build_tree_from_text,
)
from treefyit.builder import source
from treefyit.model.tree import LeafType, TextContent, Tree, UrlContent


FIXTURE_DIR = Path(__file__).parent / "testfile"


def test_build_options_defaults_to_lightweight_mode():
    assert BuildOptions().summarize is False


def test_build_tree_from_sections_returns_tree_model():
    sections = [
        {"title": "Intro", "level": 1, "text": "hello world"},
        {"title": "Detail", "level": 2, "text": "more text"},
    ]

    tree = build_tree_from_sections(
        sections,
        BuildOptions(),
    )

    assert isinstance(tree, Tree)
    assert tree.node_id == "document"
    assert tree.depth == 0
    assert tree.subtree_size == 3
    assert tree.leaf_count == 1
    assert tree.children[0].title == "Intro"
    assert isinstance(tree.children[0].content, TextContent)
    assert tree.children[0].depth == 1
    assert tree.children[0].children[0].depth == 2


def test_build_tree_from_text_returns_tree_model():
    text = "# Intro\n\nHello world.\n\n## Detail\n\nMore text."

    tree = build_tree_from_text(
        text,
        filename="sample.md",
        options=BuildOptions(),
    )

    assert isinstance(tree, Tree)
    assert tree.node_id == "sample"
    assert tree.title == "sample.md"
    assert tree.depth == 0
    assert tree.children[0].title == "Intro"
    assert isinstance(tree.children[0].content, TextContent)
    assert "Hello world." in tree.children[0].content.text


def test_build_tree_from_file_supports_markdown_and_html():
    markdown_tree = build_tree_from_file(
        FIXTURE_DIR / "short.md",
        BuildOptions(),
    )
    html_tree = build_tree_from_file(
        FIXTURE_DIR / "tea.html",
        BuildOptions(),
    )

    assert markdown_tree.node_id == "short"
    assert markdown_tree.children
    assert html_tree.node_id == "tea"
    assert html_tree.children


def test_build_tree_from_file_rejects_zip():
    with pytest.raises(NotImplementedError):
        build_tree_from_file(
            "demo.zip",
            BuildOptions(),
        )


def test_rule_based_level_inferer_nests_numbered_sections():
    tree = build_tree_from_file(
        FIXTURE_DIR / "numbered-paper.md",
        BuildOptions(),
        level_inferer=RuleBasedLevelInferer(),
    )

    related_work = tree.children[2]
    experiments = tree.children[3]

    assert related_work.title == "2 Related Work"
    assert [child.title for child in related_work.children] == [
        "2.1 Rule-based Methods",
        "2.2 Neural Methods",
    ]
    assert experiments.title == "3 Experiments"
    assert [child.title for child in experiments.children] == [
        "3.1 Dataset",
        "3.2 Metrics",
    ]


def test_custom_level_inferer_is_pluggable():
    text = "# Root\n\nBody\n\n## Child\n\nDetail"

    class FlatLevelInferer:
        def infer(self, sections, *, text=None, source_kind=None):
            flattened = []
            for section in sections:
                item = dict(section)
                item["level"] = 1
                flattened.append(item)
            return flattened

    tree = build_tree_from_text(
        text,
        filename="sample.md",
        options=BuildOptions(),
        level_inferer=FlatLevelInferer(),
    )

    assert [child.title for child in tree.children] == ["Root", "Child"]


def test_llm_level_inferer_is_placeholder():
    with pytest.raises(NotImplementedError):
        LLMLevelInferer().infer([{"title": "Intro", "level": 1, "text": "body"}])


def test_rule_based_refiner_splits_long_section_into_children():
    text = "# Overview\n\n" + "\n\n".join(
        [
            "First paragraph about the system design." * 8,
            "Second paragraph about deployment details." * 8,
        ]
    )

    tree = build_tree_from_text(
        text,
        filename="long.md",
        options=BuildOptions(),
        section_refiner=RuleBasedSectionRefiner(split_threshold=120),
    )

    overview = tree.children[0]
    assert overview.title == "Overview"
    assert overview.content is None
    assert overview.summary is not None
    assert "First paragraph about the system design." in overview.summary
    assert [child.title for child in overview.children] == [
        "Overview / Part 1",
        "Overview / Part 2",
    ]
    assert isinstance(overview.children[0].content, TextContent)


def test_rule_based_refiner_preserves_existing_parent_summary_when_splitting():
    sections = [
        {
            "title": "Overview",
            "level": 1,
            "text": "\n\n".join(
                [
                    "First paragraph about the system design." * 8,
                    "Second paragraph about deployment details." * 8,
                ]
            ),
            "summary": "Existing overview summary",
        }
    ]

    tree = build_tree_from_sections(
        sections,
        BuildOptions(),
        section_refiner=RuleBasedSectionRefiner(split_threshold=120),
    )

    overview = tree.children[0]
    assert overview.summary == "Existing overview summary"
    assert overview.content is None
    assert [child.title for child in overview.children] == [
        "Overview / Part 1",
        "Overview / Part 2",
    ]


def test_rule_based_refiner_detects_table_sections():
    tree = build_tree_from_file(
        FIXTURE_DIR / "edge-table-only.md",
        BuildOptions(),
        section_refiner=RuleBasedSectionRefiner(),
    )

    table_section = tree.children[0]
    assert table_section.title == "Metrics"
    table_child = table_section.children[0]
    assert table_child.leaf_type == LeafType.TABLE
    assert isinstance(table_child.content, TextContent)


def test_rule_based_refiner_detects_link_sections():
    tree = build_tree_from_text(
        "# Links\n\n## Homepage\n\nhttps://example.com/docs",
        filename="links.md",
        options=BuildOptions(),
        section_refiner=RuleBasedSectionRefiner(),
    )

    homepage = tree.children[0].children[0]
    assert homepage.leaf_type == LeafType.LINK
    assert isinstance(homepage.content, UrlContent)
    assert str(homepage.content.url) == "https://example.com/docs"


def test_build_summary_runs_bottom_up(monkeypatch: pytest.MonkeyPatch):
    calls = []

    def fake_summarize_text(*, title, content="", child_summaries=None, **kwargs):
        calls.append(
            {
                "title": title,
                "content": content,
                "child_summaries": list(child_summaries or []),
            }
        )
        if child_summaries:
            return f"{title} <- {' | '.join(child_summaries)}"
        return f"{title} <- self"

    monkeypatch.setattr(
        "treefyit.builder.summarize.summarize_text", fake_summarize_text
    )

    tree = build_tree_from_text(
        "# Root\n\nRoot body\n\n## Child\n\nChild body",
        filename="summary.md",
        options=BuildOptions(summarize=True),
    )

    assert [call["title"] for call in calls] == ["Child", "Root"]
    assert calls[1]["child_summaries"] == ["Child <- self"]
    assert tree.children[0].summary == "Root <- Child <- self"
    assert tree.children[0].children[0].summary == "Child <- self"


def test_build_summary_uses_child_summaries_for_parent(monkeypatch: pytest.MonkeyPatch):
    def fake_summarize_text(*, title, content="", child_summaries=None, **kwargs):
        if title == "Parent":
            assert child_summaries == ["Leaf A summary", "Leaf B summary"]
            return "Parent summary"
        if title == "Leaf A":
            return "Leaf A summary"
        if title == "Leaf B":
            return "Leaf B summary"
        return f"{title} summary"

    monkeypatch.setattr(
        "treefyit.builder.summarize.summarize_text", fake_summarize_text
    )

    tree = build_tree_from_text(
        "# Parent\n\nParent body\n\n## Leaf A\n\nA body\n\n## Leaf B\n\nB body",
        filename="summary.md",
        options=BuildOptions(summarize=True),
    )

    assert tree.children[0].summary == "Parent summary"


def test_custom_section_refiner_is_pluggable():
    class FlatSectionRefiner:
        def refine(self, sections, *, text=None, source_kind=None):
            refined = []
            for section in sections:
                item = dict(section)
                item["title"] = f"Refined: {item['title']}"
                refined.append(item)
            return refined

    tree = build_tree_from_text(
        "# Root\n\nBody",
        filename="custom.md",
        options=BuildOptions(),
        section_refiner=FlatSectionRefiner(),
    )

    assert tree.children[0].title == "Refined: Root"


def test_llm_section_refiner_maps_llm_sections(monkeypatch: pytest.MonkeyPatch):
    def fake_refine_section(**kwargs):
        return [
            {"title": "Overview", "level_delta": 0, "text": ""},
            {"title": "Overview / A", "level_delta": 1, "text": "Part A"},
            {"title": "Overview / B", "level_delta": 1, "text": "Part B"},
        ]

    monkeypatch.setattr("treefyit.builder.refine.refine_section", fake_refine_section)

    tree = build_tree_from_text(
        "# Overview\n\n"
        + "\n\n".join(["First paragraph." * 20, "Second paragraph." * 20]),
        filename="llm-refine.md",
        options=BuildOptions(),
        section_refiner=LLMSectionRefiner(split_threshold=20),
    )

    overview = tree.children[0]
    assert overview.title == "Overview"
    assert [child.title for child in overview.children] == [
        "Overview / A",
        "Overview / B",
    ]


def test_llm_section_refiner_falls_back_to_rule_based(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("treefyit.builder.refine.refine_section", lambda **kwargs: [])

    tree = build_tree_from_text(
        "# Overview\n\n"
        + "\n\n".join(["First paragraph." * 20, "Second paragraph." * 20]),
        filename="llm-refine.md",
        options=BuildOptions(),
        section_refiner=LLMSectionRefiner(split_threshold=20, max_parts=2),
    )

    overview = tree.children[0]
    assert [child.title for child in overview.children] == [
        "Overview / Part 1",
        "Overview / Part 2",
    ]


def test_detect_source_kind_prefers_libmagic_over_suffix(
    monkeypatch: pytest.MonkeyPatch,
):
    monkeypatch.setattr(source, "detect_mime_type", lambda path: "text/html")
    assert source.detect_source_kind(Path("demo.txt")) == "html"


def test_detect_source_kind_falls_back_to_suffix(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(source, "detect_mime_type", lambda path: None)
    assert source.detect_source_kind(Path("demo.zip")) == "zip"
