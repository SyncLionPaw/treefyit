from __future__ import annotations

from types import SimpleNamespace

from treefyit.model.forest import Forest
from treefyit.model.tree import TextContent, Tree
from treefyit.query import query as query_module
from treefyit.query.query import (
    InMemoryForestQuery,
    build_tree_index,
    score_nodes_bm25,
    tokenize,
)


def build_sample_forest() -> Forest:
    retrieval_tree = Tree(
        node_id="doc-retrieval",
        title="Retrieval Paper",
        content=TextContent(text="A paper about document understanding."),
        children=[
            Tree(
                node_id="doc-retrieval-0",
                title="Introduction",
                content=TextContent(text="General introduction."),
            ),
            Tree(
                node_id="doc-retrieval-1",
                title="Dense Retrieval",
                content=TextContent(
                    text="This section covers retrieval, reranking, and recall."
                ),
                summary="retrieval section",
            ),
        ],
    )
    vision_tree = Tree(
        node_id="doc-vision",
        title="Vision Notes",
        content=TextContent(text="Image parsing and visual understanding."),
        children=[
            Tree(
                node_id="doc-vision-0",
                title="Tables",
                content=TextContent(text="How to parse tables from html and pdf."),
            )
        ],
    )
    chinese_tree = Tree(
        node_id="doc-chinese",
        title="中文笔记",
        content=TextContent(text="这里讨论摘要生成和结构化处理。"),
    )
    return Forest(
        forest_id="forest-demo",
        trees=[retrieval_tree, vision_tree, chinese_tree],
    )


def test_find_trees_recalls_tree_from_descendant_text():
    forest = build_sample_forest()
    query = InMemoryForestQuery(forest)

    result = query.find_trees("reranking recall")

    assert result
    assert result[0].tree_id == "doc-retrieval"
    assert result[0].tree_title == "Retrieval Paper"


def test_find_nodes_returns_matching_node():
    forest = build_sample_forest()
    query = InMemoryForestQuery(forest)

    result = query.find_nodes("retrieval")

    assert result
    assert result[0].tree_id == "doc-retrieval"
    assert result[0].node_id == "doc-retrieval-1"
    assert result[0].path == "1"
    assert result[0].tree_title == "Retrieval Paper"
    assert "retrieval" in result[0].snippet.lower()


def test_find_trees_supports_chinese_tokens():
    forest = build_sample_forest()
    query = InMemoryForestQuery(forest)

    result = query.find_trees("摘要")

    assert result
    assert result[0].tree_id == "doc-chinese"


def test_empty_query_returns_no_hits():
    forest = build_sample_forest()
    query = InMemoryForestQuery(forest)

    tree_result = query.find_trees("   ")
    node_result = query.find_nodes("   ")

    assert tree_result == []
    assert node_result == []


def test_build_tree_index_collects_node_term_statistics():
    tree = build_sample_forest().trees[0]

    index = build_tree_index(tree)

    assert index.tree_id == "doc-retrieval"
    assert index.corpus.document_count == 3
    assert index.corpus.document_frequency["retrieval"] >= 1
    assert index.tree_term_frequency["retrieval"] >= 2

    dense_retrieval = next(
        document
        for document in index.node_documents
        if document.node_id == "doc-retrieval-1"
    )
    assert dense_retrieval.summary_term_frequency["retrieval"] == 1
    assert dense_retrieval.content_term_frequency["retrieval"] == 1
    assert dense_retrieval.term_frequency["retrieval"] == 3
    assert index.postings["retrieval"]


def test_score_nodes_bm25_returns_best_matching_node():
    tree = build_sample_forest().trees[0]
    index = build_tree_index(tree)

    result = score_nodes_bm25(index, "reranking recall")

    assert result
    assert result[0].node_id == "doc-retrieval-1"
    assert result[0].path == "1"
    assert result[0].score > 0


def test_tokenize_falls_back_without_jieba(monkeypatch):
    monkeypatch.setattr(query_module, "jieba", None)

    tokens = tokenize("这里讨论摘要生成和结构化处理。")

    assert "摘" in tokens
    assert "要" in tokens
    assert "摘要" not in tokens
    assert "结构化" not in tokens


def test_tokenize_uses_word_segmentation_for_chinese_text(monkeypatch):
    fake_jieba = SimpleNamespace(
        lcut=lambda text: ["这里", "讨论", "摘要", "生成", "和", "结构化", "处理"]
    )
    monkeypatch.setattr(query_module, "jieba", fake_jieba)

    tokens = tokenize("这里讨论摘要生成和结构化处理。")

    assert "摘要" in tokens
    assert "生成" in tokens
    assert "结构化" in tokens
    assert "处理" in tokens
