from __future__ import annotations

from pathlib import Path

from core import TreeStore, markdown_to_tree, search_store, search_tree


SAMPLE = """# 白茶简介

白茶是中国六大茶类之一。

## 历史与产地

白茶历史可追溯至唐代。

### 核心产区

福鼎与政和。

## 冲泡建议

水温 85 度。
"""


def test_markdown_to_tree_builds_heading_hierarchy():
    root = markdown_to_tree(SAMPLE, root_title="tea")
    assert root.title == "tea"
    assert [c.title for c in root.children] == ["白茶简介"]
    intro = root.children[0]
    assert intro.content and "六大茶类" in intro.content
    assert [c.title for c in intro.children] == ["历史与产地", "冲泡建议"]
    assert intro.children[0].children[0].title == "核心产区"


def test_search_tree_and_store(tmp_path: Path):
    root = markdown_to_tree(SAMPLE, root_title="tea")
    hits = search_tree(root, "政和", tree_id="memory")
    assert hits
    assert any("政和" in hit.snippet or hit.title == "核心产区" for hit in hits)

    store = TreeStore(tmp_path / "lib")
    store.save(root, tree_id="tea-1", title="tea")
    lib_hits = search_store(store, "冲泡")
    assert lib_hits
    assert lib_hits[0].tree_id == "tea-1"
