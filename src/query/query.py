"""In-memory query over a forest of document trees.

This module keeps query behavior outside the data model. It builds lightweight
in-memory indexes from a ``Forest`` and supports tree-level recall plus node-
level search.
"""

from __future__ import annotations

from collections import Counter, defaultdict
import re
from math import log
from typing import Any

import jieba
from pydantic import BaseModel, Field
from rank_bm25 import BM25Okapi

from src.model.forest import Forest
from src.model.tree import NodeContent, Tree


class TreeQueryHit(BaseModel):
    tree_id: str
    tree_title: str
    score: float
    node_count: int = Field(ge=1)
    root_titles: list[str] = Field(default_factory=list)


class NodeQueryHit(BaseModel):
    tree_id: str
    tree_title: str
    node_id: str
    path: str
    title: str
    score: float
    summary: str | None = None
    snippet: str = ""


class Posting(BaseModel):
    term_frequency: int = Field(ge=1)
    node_id: str
    path: str


class NodeDocumentStats(BaseModel):
    node_id: str
    path: str
    title: str
    summary: str | None = None
    text: str = ""
    title_term_frequency: dict[str, int] = Field(default_factory=dict)
    summary_term_frequency: dict[str, int] = Field(default_factory=dict)
    content_term_frequency: dict[str, int] = Field(default_factory=dict)
    term_frequency: dict[str, int] = Field(default_factory=dict)
    document_length: int = Field(default=0, ge=0)


class CorpusStats(BaseModel):
    document_count: int = Field(default=0, ge=0)
    average_document_length: float = Field(default=0.0, ge=0.0)
    document_frequency: dict[str, int] = Field(default_factory=dict)


class TreeIndex(BaseModel):
    tree_id: str
    tree_title: str
    node_documents: list[NodeDocumentStats] = Field(default_factory=list)
    postings: dict[str, list[Posting]] = Field(default_factory=dict)
    corpus: CorpusStats = Field(default_factory=CorpusStats)
    tree_term_frequency: dict[str, int] = Field(default_factory=dict)
    tree_document_length: int = Field(default=0, ge=0)


token_chunk_pattern = re.compile(r"[\u4e00-\u9fff]+|[a-z0-9]+")


class InMemoryForestQuery:
    def __init__(self, forest: Forest) -> None:
        self.forest = forest
        self.tree_rows: list[dict[str, Any]] = []
        self.node_rows: list[dict[str, Any]] = []
        self.rebuild()

    def set_forest(self, forest: Forest) -> None:
        self.forest = forest
        self.rebuild()

    def rebuild(self) -> None:
        self.tree_rows = []
        self.node_rows = []

        for tree in self.forest.trees:
            self.tree_rows.append(build_tree_row(tree))
            self.node_rows.extend(flatten_tree_nodes(tree))

    def find_trees(self, query: str, *, limit: int = 5) -> list[TreeQueryHit]:
        normalized_query = query.strip()
        if not normalized_query:
            return []
        if not self.tree_rows:
            return []

        corpus = [tokenize(row["search_text"]) for row in self.tree_rows]
        scores = combined_scores(corpus, tokenize(normalized_query))
        ranked = sorted(
            zip(self.tree_rows, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        hits: list[TreeQueryHit] = []
        for row, score in ranked[:limit]:
            if score <= 0:
                continue
            hits.append(
                TreeQueryHit(
                    tree_id=row["tree_id"],
                    tree_title=row["tree_title"],
                    score=round(float(score), 3),
                    node_count=row["node_count"],
                    root_titles=row["root_titles"][:8],
                )
            )
        return hits

    def find_nodes(self, query: str, *, limit: int = 8) -> list[NodeQueryHit]:
        normalized_query = query.strip()
        if not normalized_query:
            return []
        if not self.node_rows:
            return []

        corpus = [tokenize(row["search_text"]) for row in self.node_rows]
        scores = combined_scores(corpus, tokenize(normalized_query))
        ranked = sorted(
            zip(self.node_rows, scores),
            key=lambda item: item[1],
            reverse=True,
        )

        hits: list[NodeQueryHit] = []
        for row, score in ranked[: limit * 2]:
            if score <= 0:
                continue
            hits.append(
                NodeQueryHit(
                    tree_id=row["tree_id"],
                    tree_title=row["tree_title"],
                    node_id=row["node_id"],
                    path=row["path"],
                    title=row["title"],
                    score=round(float(score), 3),
                    summary=row["summary"],
                    snippet=snippet_from_row(row),
                )
            )
            if len(hits) >= limit:
                break

        return hits


def build_tree_row(tree: Tree) -> dict[str, Any]:
    root_titles = [child.title for child in tree.children]
    node_rows = flatten_tree_nodes(tree)
    search_text = " ".join(
        part
        for part in [
            tree.node_id,
            tree.title,
            tree.summary or "",
            *root_titles,
            *[row["title"] for row in node_rows],
            *[row["summary"] or "" for row in node_rows],
            *[row["text"] for row in node_rows],
        ]
        if part
    )
    return {
        "tree_id": tree.node_id,
        "tree_title": tree.title,
        "node_count": len(node_rows),
        "root_titles": root_titles,
        "search_text": search_text,
    }


def build_tree_index(tree: Tree) -> TreeIndex:
    rows = flatten_tree_nodes(tree)
    node_documents: list[NodeDocumentStats] = []
    postings: dict[str, list[Posting]] = defaultdict(list)
    document_frequency: Counter[str] = Counter()
    tree_term_frequency: Counter[str] = Counter()
    total_document_length = 0

    for row in rows:
        title_term_frequency = count_terms(str(row.get("title") or ""))
        summary_term_frequency = count_terms(str(row.get("summary") or ""))
        content_term_frequency = count_terms(str(row.get("text") or ""))
        term_frequency = merge_term_frequencies(
            title_term_frequency,
            summary_term_frequency,
            content_term_frequency,
        )
        document_length = sum(term_frequency.values())
        total_document_length += document_length

        node_document = NodeDocumentStats(
            node_id=row["node_id"],
            path=row["path"],
            title=row["title"],
            summary=row["summary"],
            text=row["text"],
            title_term_frequency=dict(title_term_frequency),
            summary_term_frequency=dict(summary_term_frequency),
            content_term_frequency=dict(content_term_frequency),
            term_frequency=dict(term_frequency),
            document_length=document_length,
        )
        node_documents.append(node_document)

        for term, frequency in term_frequency.items():
            postings[term].append(
                Posting(
                    term_frequency=frequency,
                    node_id=row["node_id"],
                    path=row["path"],
                )
            )
            document_frequency[term] += 1

        tree_term_frequency.update(term_frequency)

    document_count = len(node_documents)
    average_document_length = (
        total_document_length / document_count if document_count else 0.0
    )

    return TreeIndex(
        tree_id=tree.node_id,
        tree_title=tree.title,
        node_documents=node_documents,
        postings=dict(postings),
        corpus=CorpusStats(
            document_count=document_count,
            average_document_length=average_document_length,
            document_frequency=dict(document_frequency),
        ),
        tree_term_frequency=dict(tree_term_frequency),
        tree_document_length=total_document_length,
    )


def flatten_tree_nodes(tree: Tree) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    append_tree_node_rows(tree, rows, tree.node_id, tree.title, "root")
    return rows


def append_tree_node_rows(
    node: Tree,
    rows: list[dict[str, Any]],
    tree_id: str,
    tree_title: str,
    path: str,
) -> None:
    text = content_to_search_text(node.content)
    rows.append(
        {
            "tree_id": tree_id,
            "tree_title": tree_title,
            "node_id": node.node_id,
            "path": path,
            "title": node.title,
            "summary": node.summary,
            "text": text[:600],
            "search_text": " ".join(
                part for part in [node.title, node.summary or "", text[:600]] if part
            ),
        }
    )

    for index, child in enumerate(node.children):
        child_path = str(index) if path == "root" else f"{path}.{index}"
        append_tree_node_rows(child, rows, tree_id, tree_title, child_path)


def content_to_search_text(content: NodeContent | None) -> str:
    if content is None:
        return ""
    if content.kind == "text":
        return content.text
    if content.kind == "url":
        return str(content.url)
    return str(content.uri)


def snippet_from_row(row: dict[str, Any], max_len: int = 160) -> str:
    for key in ("summary", "text", "title"):
        value = str(row.get(key) or "").strip()
        if not value:
            continue
        if len(value) <= max_len:
            return value
        return value[: max_len - 1] + "…"
    return ""


def tokenize(text: str) -> list[str]:
    lowered = text.lower()
    chunks = token_chunk_pattern.findall(lowered)
    if chunks:
        tokens: list[str] = []
        for chunk in chunks:
            if chunk.isascii():
                tokens.append(chunk)
                continue
            tokens.extend(segment_chinese_text(chunk))
        if tokens:
            return tokens
    stripped = lowered.strip()
    if stripped:
        return [stripped]
    return []


def segment_chinese_text(text: str) -> list[str]:
    stripped = text.strip()
    if not stripped:
        return []
    if jieba is not None:
        tokens = [token.strip() for token in jieba.lcut(stripped) if token.strip()]
        if tokens:
            return tokens
    return list(stripped)


def count_terms(text: str) -> Counter[str]:
    return Counter(tokenize(text))


def merge_term_frequencies(*parts: Counter[str]) -> Counter[str]:
    merged: Counter[str] = Counter()
    for part in parts:
        merged.update(part)
    return merged


def score_nodes_bm25(
    index: TreeIndex,
    query: str,
    *,
    limit: int = 8,
    k1: float = 1.5,
    b: float = 0.75,
) -> list[NodeQueryHit]:
    normalized_query = query.strip()
    if not normalized_query:
        return []
    if not index.node_documents:
        return []

    query_term_frequency = count_terms(normalized_query)
    if not query_term_frequency:
        return []

    node_documents = {document.node_id: document for document in index.node_documents}
    scores: dict[str, float] = defaultdict(float)
    average_document_length = index.corpus.average_document_length or 1.0
    document_count = index.corpus.document_count

    for term, query_frequency in query_term_frequency.items():
        postings = index.postings.get(term)
        if not postings:
            continue

        document_frequency = index.corpus.document_frequency.get(term, 0)
        if document_frequency <= 0:
            continue

        idf = log(
            1 + (document_count - document_frequency + 0.5) / (document_frequency + 0.5)
        )
        for posting in postings:
            document = node_documents[posting.node_id]
            document_length = max(document.document_length, 1)
            numerator = posting.term_frequency * (k1 + 1)
            denominator = posting.term_frequency + k1 * (
                1 - b + b * document_length / average_document_length
            )
            if denominator == 0:
                continue
            scores[posting.node_id] += query_frequency * idf * numerator / denominator

    ranked_documents = sorted(
        (
            (node_documents[node_id], score)
            for node_id, score in scores.items()
            if score > 0
        ),
        key=lambda item: item[1],
        reverse=True,
    )

    hits: list[NodeQueryHit] = []
    for document, score in ranked_documents[:limit]:
        hits.append(
            NodeQueryHit(
                tree_id=index.tree_id,
                tree_title=index.tree_title,
                node_id=document.node_id,
                path=document.path,
                title=document.title,
                score=round(float(score), 3),
                summary=document.summary,
                snippet=snippet_from_row(
                    {
                        "summary": document.summary,
                        "text": document.text,
                        "title": document.title,
                    }
                ),
            )
        )

    return hits


def combined_scores(corpus: list[list[str]], query_tokens: list[str]) -> list[float]:
    if not corpus or not query_tokens:
        return [0.0] * len(corpus)

    bm25_scores = BM25Okapi(corpus).get_scores(query_tokens)
    query_set = set(query_tokens)
    overlap_scores = [
        sum(1 for token in query_set if token in set(document_tokens)) / len(query_set)
        for document_tokens in corpus
    ]

    max_bm25 = max(bm25_scores)
    if max_bm25 != 0:
        return list(bm25_scores)
    if max(overlap_scores) > 0:
        return overlap_scores
    return overlap_scores


__all__ = [
    "CorpusStats",
    "InMemoryForestQuery",
    "NodeDocumentStats",
    "NodeQueryHit",
    "Posting",
    "TreeQueryHit",
    "TreeIndex",
    "build_tree_index",
    "count_terms",
    "merge_term_frequencies",
    "score_nodes_bm25",
]
