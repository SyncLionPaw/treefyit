"""对内存树 / 持久化库做简单节点检索，供再次查询使用。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .store import TreeStore
from .tree import TreeNode

token_pattern = re.compile(r"[\u4e00-\u9fff]+|[a-z0-9]+", re.I)


@dataclass(frozen=True)
class NodeHit:
    tree_id: str
    node_id: str
    title: str
    score: float
    snippet: str
    path: str


def tokenize(text: str) -> list[str]:
    return [part.casefold() for part in token_pattern.findall(text)]


def score_node(node: TreeNode, terms: list[str]) -> float:
    if not terms:
        return 0.0
    title = tokenize(node.title)
    summary = tokenize(node.summary or "")
    content = tokenize(node.content or "")
    score = 0.0
    for term in terms:
        if term in title:
            score += 3.0
        if term in summary:
            score += 2.0
        if term in content:
            score += 1.0
        # substring boost for CJK fragments
        blob = f"{node.title}\n{node.summary or ''}\n{node.content or ''}".casefold()
        if term in blob:
            score += 0.5
    return score


def snippet_for(node: TreeNode, limit: int = 160) -> str:
    text = (node.summary or node.content or "").strip().replace("\n", " ")
    if len(text) <= limit:
        return text
    return text[: limit - 1] + "…"


def search_tree(
    root: TreeNode,
    query: str,
    *,
    tree_id: str = "memory",
    limit: int = 8,
) -> list[NodeHit]:
    terms = tokenize(query)
    if not terms:
        return []

    hits: list[NodeHit] = []

    def walk(node: TreeNode, path: str) -> None:
        score = score_node(node, terms)
        if score > 0:
            hits.append(
                NodeHit(
                    tree_id=tree_id,
                    node_id=node.id,
                    title=node.title,
                    score=score,
                    snippet=snippet_for(node),
                    path=path,
                )
            )
        for index, child in enumerate(node.children):
            child_path = str(index) if path == "root" else f"{path}.{index}"
            walk(child, child_path)

    walk(root, "root")
    hits.sort(key=lambda hit: (-hit.score, hit.path))
    return hits[: max(limit, 0)]


def search_store(store: TreeStore, query: str, *, limit: int = 8) -> list[NodeHit]:
    hits: list[NodeHit] = []
    for item in store.list():
        tree_id = str(item["tree_id"])
        record = store.load(tree_id)
        hits.extend(search_tree(record.root, query, tree_id=tree_id, limit=limit))
    hits.sort(key=lambda hit: (-hit.score, hit.tree_id, hit.path))
    return hits[: max(limit, 0)]


def format_hits(hits: list[NodeHit]) -> str:
    if not hits:
        return "no matches"
    lines = [f"hits={len(hits)}"]
    for hit in hits:
        lines.append(
            f"- tree={hit.tree_id} path={hit.path} id={hit.node_id} "
            f"title={hit.title!r} score={hit.score:.1f}"
        )
        if hit.snippet:
            lines.append(f"  snippet: {hit.snippet}")
    return "\n".join(lines)
