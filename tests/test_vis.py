"""Tree visualization tests."""

from __future__ import annotations

from pathlib import Path

from src.parser.md import parse_md
from src.tree.builder import build_nodes
from src.vis.tree_view import render

TESTFILE = Path(__file__).resolve().parent / "testfile"


def test_render_titles_only():
    tree = build_nodes(parse_md(str(TESTFILE / "short.md")))
    out = render(tree, max_text=0)
    assert "白茶简介" in out
    assert "核心产区" in out
    assert "  — " not in out
    assert "├──" in out


def test_render_with_snippet_strips_heading():
    tree = build_nodes(parse_md(str(TESTFILE / "short.md")))
    out = render(tree, max_text=48)
    assert "  — " in out
    assert "# 白茶" not in out
    assert "六大茶类" in out or "轻微发酵" in out


def test_render_chapter_novel_flat_list():
    tree = build_nodes(parse_md(str(TESTFILE / "chapters.txt")))
    out = render(tree, max_text=0)
    assert out.count("第一回") == 1
    assert out.count("第六回") == 1
    assert "├──" not in out  # six roots, no nesting connectors at top


if __name__ == "__main__":
    test_render_titles_only()
    test_render_with_snippet_strips_heading()
    test_render_chapter_novel_flat_list()
    print("ok")
