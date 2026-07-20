"""Q&A-oriented tree ranking / hierarchical unlock."""

from __future__ import annotations

from dataclasses import dataclass

from core.ops import get_node, view_node_detail
from core.query import tokenize
from core.tree import TreeNode


@dataclass(frozen=True)
class RankedNode:
    node_id: str
    title: str
    path: str
    score: float
    depth: int
    evidence: str


def qa_score(node: TreeNode, terms: list[str], *, depth: int) -> float:
    """Content-first score with depth bonus; soft-penalize root-level topic titles."""
    if not terms:
        return 0.0
    title_tokens = tokenize(node.title)
    content_tokens = tokenize(node.content or "")
    summary_tokens = tokenize(node.summary or "")
    blob = f"{node.title}\n{node.summary or ''}\n{node.content or ''}".casefold()

    score = 0.0
    title_hits = 0
    content_hits = 0
    for term in terms:
        if term in content_tokens:
            score += 3.0
            content_hits += 1
        if term in summary_tokens:
            score += 2.0
        if term in title_tokens:
            score += 1.0
            title_hits += 1
        if term in blob:
            score += 0.25

    # Prefer specific sections over broad parents that only match topic words.
    if depth <= 1 and content_hits == 0 and title_hits > 0:
        score *= 0.35
    score += 0.4 * depth
    if node.content:
        score += 0.2
    return score


def iter_nodes(root: TreeNode, path: str = "root", depth: int = 0):
    yield root, path, depth
    for index, child in enumerate(root.children):
        child_path = str(index) if path == "root" else f"{path}.{index}"
        yield from iter_nodes(child, child_path, depth + 1)


def search_tree_qa(root: TreeNode, query: str, *, limit: int = 8) -> list[RankedNode]:
    terms = tokenize(query)
    ranked: list[RankedNode] = []
    for node, path, depth in iter_nodes(root):
        score = qa_score(node, terms, depth=depth)
        if score <= 0:
            continue
        ranked.append(
            RankedNode(
                node_id=node.id,
                title=node.title,
                path=path,
                score=score,
                depth=depth,
                evidence=view_node_detail(root, node.id, max_content_chars=900),
            )
        )
    ranked.sort(key=lambda item: (-item.score, item.path))
    return ranked[: max(limit, 0)]


def unlock_tree_evidence(root: TreeNode, query: str) -> RankedNode | None:
    """Hierarchical unlock: rank nodes, then refine inside the top parent's subtree."""
    ranked = search_tree_qa(root, query, limit=12)
    if not ranked:
        return None

    top = ranked[0]
    node = get_node(root, top.node_id)
    if not node.children:
        return top

    # Unlock: re-rank only the chosen branch (parent + descendants).
    branch = [item for item in ranked if item.path == top.path or item.path.startswith(top.path + ".")]
    if not branch:
        return top
    branch.sort(key=lambda item: (-item.score, -item.depth, item.path))
    # Prefer a deeper child when scores are close (within 15%).
    best = branch[0]
    for item in branch[1:]:
        if item.depth > best.depth and item.score >= best.score * 0.85:
            best = item
    return best
