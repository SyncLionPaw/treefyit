"""Document-type detection and build strategy selection."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

from src.parser.md import looks_like_chapter_novel

_MD_HEADER = re.compile(r"^#{1,6}\s+\S", re.MULTILINE)
_NUMBERED_TITLE = re.compile(r"^(\d+(?:\.\d+)*)\b")


class DocKind(str, Enum):
    """How the source document is organized."""

    CHAPTER_NOVEL = "chapter_novel"  # 第X回 / Chapter N plain-text novels
    MARKDOWN = "markdown"  # # / ## headers with real hierarchy
    NUMBERED = "numbered"  # flat ## with 2.1 / 3.1.1 numbering (papers, specs)
    UNSTRUCTURED = "unstructured"  # no detectable outline — needs LLM extraction
    HTML = "html"
    ZIP = "zip"


@dataclass(frozen=True)
class StructurePlan:
    """Chosen path for turning parsed nodes into a tree."""

    kind: DocKind
    use_semantic_extract: bool
    refine: bool


def has_markdown_headers(text: str) -> bool:
    return bool(_MD_HEADER.search(text))


def has_numbered_sections(nodes: list[dict]) -> bool:
    return any(_NUMBERED_TITLE.match(n.get("title", "").strip()) for n in nodes)


def classify_structure(nodes: list[dict], text: str) -> DocKind:
    """Infer document organization from parse_md output and raw text."""
    if has_markdown_headers(text):
        if has_numbered_sections(nodes):
            return DocKind.NUMBERED
        return DocKind.MARKDOWN

    if looks_like_chapter_novel(nodes):
        return DocKind.CHAPTER_NOVEL

    if nodes:
        return DocKind.MARKDOWN

    return DocKind.UNSTRUCTURED


def plan_structure(
    *,
    doc_kind: DocKind,
    is_short_pdf: bool,
) -> StructurePlan:
    """Decide whether to use rule-based nodes or full LLM structure extraction."""
    if doc_kind in (DocKind.HTML, DocKind.ZIP):
        return StructurePlan(kind=doc_kind, use_semantic_extract=False, refine=False)

    use_semantic = doc_kind == DocKind.UNSTRUCTURED or is_short_pdf

    refine = doc_kind in (
        DocKind.CHAPTER_NOVEL,
        DocKind.MARKDOWN,
        DocKind.NUMBERED,
        DocKind.UNSTRUCTURED,
    )

    return StructurePlan(
        kind=doc_kind,
        use_semantic_extract=use_semantic,
        refine=refine,
    )


def refine_max_leaf_depth(tree: list[dict], doc_kind: DocKind) -> int | None:
    """Depth filter for refine: one extra level, never recursive subdivision."""
    from src.tree.semantic import deepest_leaf_depth

    if doc_kind == DocKind.CHAPTER_NOVEL:
        return 1

    depth = deepest_leaf_depth(tree)
    return depth if depth else None
