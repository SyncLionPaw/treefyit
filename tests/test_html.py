"""HTML parser tests — MarkItDown → parse_md pipeline + regex fallback."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from src.parser.html import _try_regex, parse_html
from src.tree.builder import build_nodes

TESTFILE = Path(__file__).resolve().parent / "testfile"


def _parse(name: str) -> list[dict]:
    return parse_html(str(TESTFILE / name))


def _titles(name: str) -> list[str]:
    return [n["title"] for n in _parse(name)]


def test_tea_html_basic_structure():
    titles = _titles("tea.html")
    assert titles == ["白茶", "历史", "冲泡"]

    tree = build_nodes(_parse("tea.html"))
    assert tree[0]["title"] == "白茶"
    assert len(tree[0]["children"]) == 2

    brew = next(c for c in tree[0]["children"] if c["title"] == "冲泡")
    assert "| 步骤 | 说明 |" in brew.get("text", "") or "温杯" in brew.get("text", "")


def test_api_doc_html_nested_endpoints():
    titles = _titles("api-doc.html")
    assert titles[0] == "Treefyit API"
    assert "POST /api/build" in titles
    assert "POST /api/chat" in titles

    tree = build_nodes(_parse("api-doc.html"))
    build = next(c for c in tree[0]["children"] if c["title"] == "Build")
    assert {c["title"] for c in build["children"]} == {
        "POST /api/build",
        "POST /api/build/stream",
    }


def test_script_and_pre_do_not_add_fake_headings():
    titles = _titles("edge-html-script.html")
    assert titles == ["Safe Document", "Real Section", "Final"]
    assert not any("Script" in t for t in titles)
    assert not any("Escaped" in t for t in titles)


def test_layout_sidebar_headings_included():
    titles = _titles("edge-html-layout.html")
    assert titles == ["Article Title", "Sidebar Box", "Main Content"]
    assert "Site Nav" not in " ".join(titles)


def test_no_headings_fallback_document_node():
    nodes = _parse("edge-html-no-headings.html")
    assert len(nodes) == 1
    assert nodes[0]["title"] == "Document"
    assert "No heading tags" in nodes[0]["text"]


def test_numbered_flat_h2_inference():
    nodes = _parse("edge-html-numbered.html")
    assert [n["level"] for n in nodes] == [2, 3, 3]

    tree = build_nodes(nodes)
    assert len(tree) == 1
    assert tree[0]["title"] == "1 Introduction"
    assert {c["title"] for c in tree[0]["children"]} == {
        "2.1 Methods",
        "2.2 Results",
    }


def test_regex_fallback_matches_markitdown_for_tea():
    raw = (TESTFILE / "tea.html").read_text(encoding="utf-8")
    md_nodes = _parse("tea.html")
    regex_nodes = _try_regex(raw)
    assert [n["title"] for n in md_nodes] == [n["title"] for n in regex_nodes]
    assert [n["level"] for n in md_nodes] == [n["level"] for n in regex_nodes]


def test_regex_fallback_when_markitdown_unavailable():
    raw = (TESTFILE / "api-doc.html").read_text(encoding="utf-8")

    with patch("src.parser.html._try_markitdown", return_value=[]):
        nodes = parse_html(str(TESTFILE / "api-doc.html"))

    expected = _try_regex(raw)
    assert [n["title"] for n in nodes] == [n["title"] for n in expected]


def test_api_doc_html_core_endpoints_present():
    html_titles = _titles("api-doc.html")
    for title in [
        "Treefyit API",
        "Build",
        "POST /api/build",
        "POST /api/build/stream",
        "Chat",
        "POST /api/chat",
    ]:
        assert title in html_titles


def test_unicode_title_in_html():
    nodes = _parse("tea.html")
    assert nodes[0]["title"] == "白茶"
    assert "六大茶类" in nodes[0]["text"] or "白茶" in nodes[0]["text"]


def test_css_heavy_page_real_headings_only():
    """Full CSS layout: only semantic h1-h6 enter the tree."""
    titles = _titles("edge-html-css-hell.html")
    assert titles == [
        "白茶完全指南",
        "历史与产地",
        "福鼎与政和",
        "主要品类",
        "Hidden Section Title",
        "冲泡参数",
        "老白茶煮饮",
    ]

    # div/CSS fake headings must not leak in
    joined = " ".join(titles)
    for fake in ("限时特惠", "热门标签", "相关阅读", "JS 动态", "WhiteTea"):
        assert fake not in joined


def test_css_heavy_nested_tree_shape():
    tree = build_nodes(_parse("edge-html-css-hell.html"))
    assert tree[0]["title"] == "白茶完全指南"
    assert len(tree[0]["children"]) == 4
    assert {c["title"] for c in tree[0]["children"]} == {
        "历史与产地",
        "主要品类",
        "Hidden Section Title",
        "冲泡参数",
    }
    history = next(c for c in tree[0]["children"] if c["title"] == "历史与产地")
    assert history["children"][0]["title"] == "福鼎与政和"

    brew = next(c for c in tree[0]["children"] if c["title"] == "冲泡参数")
    assert brew["children"][0]["title"] == "老白茶煮饮"


def test_css_heavy_table_preserved_in_section():
    nodes = _parse("edge-html-css-hell.html")
    types = next(n for n in nodes if n["title"] == "主要品类")
    assert "白毫银针" in types["text"]
    assert "白牡丹" in types["text"]


def test_css_heavy_sr_only_h2_still_extracted():
    """Visually hidden but real <h2> — parser keeps it."""
    assert "Hidden Section Title" in _titles("edge-html-css-hell.html")


ALL_TESTS = [
    test_tea_html_basic_structure,
    test_api_doc_html_nested_endpoints,
    test_script_and_pre_do_not_add_fake_headings,
    test_layout_sidebar_headings_included,
    test_no_headings_fallback_document_node,
    test_numbered_flat_h2_inference,
    test_regex_fallback_matches_markitdown_for_tea,
    test_regex_fallback_when_markitdown_unavailable,
    test_api_doc_html_core_endpoints_present,
    test_unicode_title_in_html,
    test_css_heavy_page_real_headings_only,
    test_css_heavy_nested_tree_shape,
    test_css_heavy_table_preserved_in_section,
    test_css_heavy_sr_only_h2_still_extracted,
]


if __name__ == "__main__":
    for fn in ALL_TESTS:
        fn()
        print(f"  ok  {fn.__name__}")
    print(f"all {len(ALL_TESTS)} passed")
