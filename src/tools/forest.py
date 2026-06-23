"""Forest catalog — index many trees and route agents to the right book.

Each *build* is one tree in the forest.  The catalog (林场目录) summarizes every
tree; :func:`find_trees` ranks trees by topic; :func:`find_sections` ranks
individual nodes across the whole forest.
"""

from __future__ import annotations

import re
from typing import Any

from src.tools.query import _count, _max_depth, _registry, _summarize_node

__all__ = [
    "index_tree",
    "remove_tree",
    "forest_catalog",
    "find_trees",
    "find_sections",
]

# tree_id → catalog metadata + search blob
_catalog: dict[str, dict[str, Any]] = {}
# flat node rows for cross-tree section search
_node_rows: list[dict[str, str]] = []
_tree_ids: list[str] = []


def index_tree(
    tree_id: str,
    tree: list[dict],
    *,
    filename: str = "",
    doc_kind: str = "",
) -> None:
    """Register or refresh one tree in the forest catalog."""
    roots_preview = []
    root_parts: list[str] = []
    for i, node in enumerate(tree):
        entry = _summarize_node(node, str(i))
        roots_preview.append(entry)
        root_parts.append(entry["title"])
        summary = (entry.get("summary") or "").strip()
        if summary:
            root_parts.append(summary)

    search_text = " ".join(
        part
        for part in [filename, doc_kind, tree_id, *root_parts]
        if part
    )

    _catalog[tree_id] = {
        "tree_id": tree_id,
        "filename": filename or tree_id,
        "doc_kind": doc_kind,
        "node_count": _count(tree),
        "max_depth": _max_depth(tree),
        "roots": roots_preview,
        "search_text": search_text,
    }

    _node_rows[:] = [row for row in _node_rows if row["tree_id"] != tree_id]
    _node_rows.extend(_flatten_nodes(tree_id, tree, filename))
    _rebuild_indexes()


def remove_tree(tree_id: str) -> None:
    _catalog.pop(tree_id, None)
    _node_rows[:] = [row for row in _node_rows if row["tree_id"] != tree_id]
    _rebuild_indexes()


def forest_catalog() -> dict:
    """Return the forest catalog — one entry per registered tree."""
    trees = sorted(_catalog.values(), key=lambda x: x.get("filename", ""))
    return {
        "tree_count": len(trees),
        "trees": [
            {
                "tree_id": t["tree_id"],
                "filename": t["filename"],
                "doc_kind": t.get("doc_kind") or "",
                "node_count": t["node_count"],
                "max_depth": t["max_depth"],
                "roots": t["roots"],
            }
            for t in trees
        ],
    }


def _combined_scores(corpus: list[list[str]], query_tokens: list[str]) -> list[float]:
    """BM25 ranking with token-overlap fallback for tiny corpora (N≤2)."""
    from rank_bm25 import BM25Okapi

    if not corpus or not query_tokens:
        return [0.0] * len(corpus)

    bm = BM25Okapi(corpus).get_scores(query_tokens)
    qset = set(query_tokens)
    overlap = [
        sum(1 for token in qset if token in set(doc)) / len(qset) for doc in corpus
    ]

    if max(bm) > 0:
        return list(bm)
    if max(bm) < 0:
        return list(bm)
    if max(overlap) > 0:
        return overlap
    return overlap


def find_trees(query: str, *, limit: int = 5) -> dict:
    """BM25 over tree-level catalog text — which books match a topic?"""
    query = query.strip()
    if not query:
        return {"error": "empty query", "hits": []}
    if not _tree_ids:
        return {"query": query, "hits": []}

    corpus = [_tokenize(_catalog[tid]["search_text"]) for tid in _tree_ids]
    scores = _combined_scores(corpus, _tokenize(query))
    ranked = sorted(zip(_tree_ids, scores), key=lambda x: x[1], reverse=True)
    hits = []
    for tree_id, score in ranked[:limit]:
        if score == 0:
            continue
        meta = _catalog[tree_id]
        hits.append(
            {
                "tree_id": tree_id,
                "filename": meta["filename"],
                "doc_kind": meta.get("doc_kind") or "",
                "score": round(float(score), 3),
                "node_count": meta["node_count"],
                "root_titles": [r["title"] for r in meta["roots"][:8]],
            }
        )
    return {"query": query, "hits": hits}


def find_sections(query: str, *, limit: int = 8) -> dict:
    """BM25 over all nodes in the forest — precise (tree_id, path) hits."""
    query = query.strip()
    if not query:
        return {"error": "empty query", "hits": []}
    if not _node_rows:
        return {"query": query, "hits": []}

    corpus = [_tokenize(row["search_text"]) for row in _node_rows]
    scores = _combined_scores(corpus, _tokenize(query))
    ranked = sorted(
        zip(range(len(_node_rows)), scores),
        key=lambda x: x[1],
        reverse=True,
    )
    hits = []
    for idx, score in ranked[: limit * 2]:
        if score == 0:
            continue
        row = _node_rows[idx]
        hits.append(
            {
                "tree_id": row["tree_id"],
                "filename": row["filename"],
                "path": row["path"],
                "title": row["title"],
                "score": round(float(score), 3),
                "snippet": _snippet(row),
            }
        )
        if len(hits) >= limit:
            break
    return {"query": query, "hits": hits}


def _flatten_nodes(
    tree_id: str,
    tree: list[dict],
    filename: str,
    prefix: str = "",
) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for i, node in enumerate(tree):
        path = f"{prefix}{i}" if prefix else str(i)
        title = node.get("title", "")
        summary = node.get("summary", "")
        text = (node.get("text") or "")[:600]
        rows.append(
            {
                "tree_id": tree_id,
                "filename": filename or tree_id,
                "path": path,
                "title": title,
                "summary": summary,
                "text": text,
                "search_text": " ".join(
                    x for x in [filename, title, summary, text] if x
                ),
            }
        )
        children = node.get("children") or []
        if children:
            rows.extend(_flatten_nodes(tree_id, children, filename, f"{path}."))
    return rows


def _snippet(row: dict[str, str], max_len: int = 160) -> str:
    for key in ("summary", "text", "title"):
        value = (row.get(key) or "").strip()
        if value:
            return value if len(value) <= max_len else value[: max_len - 1] + "…"
    return ""


def _tokenize(text: str) -> list[str]:
    text = text.lower()
    parts = re.findall(r"[\u4e00-\u9fff]|[a-z0-9]+", text)
    return parts or [text.strip() or ""]


def _rebuild_indexes() -> None:
    global _tree_ids
    _tree_ids = list(_catalog.keys())


def sync_from_registry(
    meta_by_id: dict[str, dict[str, str]] | None = None,
) -> None:
    """Rebuild the entire catalog from the tree registry (startup)."""
    _catalog.clear()
    _node_rows.clear()
    meta_by_id = meta_by_id or {}
    for tree_id, tree in _registry.items():
        meta = meta_by_id.get(tree_id, {})
        index_tree(
            tree_id,
            tree,
            filename=meta.get("filename", ""),
            doc_kind=meta.get("doc_kind", ""),
        )
