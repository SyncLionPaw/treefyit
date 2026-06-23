"""Forest catalog — multi-tree routing."""

from __future__ import annotations

from src.tools import (
    find_sections,
    find_trees,
    forest_catalog,
    register,
    unregister,
)


def _tea_tree() -> list[dict]:
    return [
        {
            "title": "白茶",
            "summary": "中国六大茶类之一，微发酵茶",
            "text": "白茶属微发酵茶，主要产于福建福鼎、政和等地。",
            "children": [
                {
                    "title": "制作工艺",
                    "summary": "萎凋、干燥",
                    "text": "白茶工艺简单，不炒不揉。",
                    "children": [],
                }
            ],
        }
    ]


def _novel_tree() -> list[dict]:
    return [
        {
            "title": "第一回 宴桃园豪杰三结义",
            "summary": "刘关张桃园结义",
            "text": "话说天下大势，分久必合，合久必分。",
            "children": [],
        },
        {
            "title": "第二回 张翼德怒鞭督邮",
            "summary": "张飞鞭打督邮",
            "text": "且说董卓专权，天下诸侯起兵讨董。",
            "children": [],
        },
        {
            "title": "第七十四回 关云长水淹七军",
            "summary": "关羽水淹于禁七军",
            "text": "关云长放水淹了于禁、庞德之军。",
            "children": [],
        },
    ]


def setup_function():
    unregister("tea")
    unregister("novel")


def teardown_function():
    unregister("tea")
    unregister("novel")


def test_forest_catalog_lists_trees():
    register("tea", _tea_tree(), filename="short.md", doc_kind="markdown")
    register("novel", _novel_tree(), filename="三国演义.txt", doc_kind="chapter_novel")

    cat = forest_catalog()
    assert cat["tree_count"] == 2
    ids = {t["tree_id"] for t in cat["trees"]}
    assert ids == {"tea", "novel"}
    tea = next(t for t in cat["trees"] if t["tree_id"] == "tea")
    assert tea["filename"] == "short.md"
    assert tea["roots"][0]["title"] == "白茶"


def test_find_trees_routes_by_topic():
    register("tea", _tea_tree(), filename="short.md")
    register("novel", _novel_tree(), filename="三国演义.txt")

    hits = find_trees("关羽 水淹七军", limit=3)["hits"]
    assert hits
    assert hits[0]["tree_id"] == "novel"
    assert hits[0]["score"] > 0

    tea_hits = find_trees("白茶 福鼎", limit=3)["hits"]
    assert tea_hits[0]["tree_id"] == "tea"


def test_find_sections_cross_tree():
    register("tea", _tea_tree(), filename="short.md")
    register("novel", _novel_tree(), filename="三国演义.txt")

    hits = find_sections("水淹七军", limit=5)["hits"]
    assert any(h["tree_id"] == "novel" and "水淹" in h["title"] for h in hits)

    tea_hits = find_sections("萎凋", limit=5)["hits"]
    assert any(h["tree_id"] == "tea" for h in tea_hits)


def test_unregister_removes_from_catalog():
    register("tea", _tea_tree(), filename="short.md")
    assert forest_catalog()["tree_count"] == 1
    unregister("tea")
    assert forest_catalog()["tree_count"] == 0
