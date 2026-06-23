"""Shared structure extraction: pick strategy by document type, build tree."""

from __future__ import annotations

import logging
from pathlib import Path

from src.tree.doc_kind import (
    DocKind,
    classify_structure,
    plan_structure,
    refine_max_leaf_depth,
)
from src.tree.semantic import (
    attach_text_ranges,
    extract_structure,
    refine_tree_granularity,
)

logger = logging.getLogger(__name__)

SEMANTIC_OUTPUT_TOKEN_ESTIMATE = 2000
AUTO_SEMANTIC_MAX_TOKENS = 40_000


async def build_tree_structure(
    *,
    text: str,
    source_path: Path,
    model: str,
    mode: str,
    input_tokens: int,
    is_pdf: bool = False,
    semantic_progress=None,
) -> tuple[list[dict], DocKind, bool, int]:
    """Parse *source_path*, classify document type, build nested tree.

    Returns ``(tree, doc_kind, semantic_used, output_token_estimate)``.
    """
    from src.parser.md import parse_md
    from src.tree.builder import build_nodes

    nodes = parse_md(str(source_path))
    doc_kind = classify_structure(nodes, text)
    is_short_pdf = (
        mode == "auto" and is_pdf and input_tokens <= AUTO_SEMANTIC_MAX_TOKENS
    )
    plan = plan_structure(
        doc_kind=doc_kind,
        is_short_pdf=is_short_pdf,
    )

    output_tokens = 0
    semantic_used = False

    if plan.use_semantic_extract:
        logger.info(
            "[structure] %s → LLM extract (mode=%s)",
            doc_kind.value,
            mode,
        )
        nodes = await extract_structure(
            text, model=model, progress=semantic_progress
        )
        attach_text_ranges(nodes, text)
        semantic_used = True
        output_tokens += SEMANTIC_OUTPUT_TOKEN_ESTIMATE
        if doc_kind == DocKind.UNSTRUCTURED:
            pass  # kind stays unstructured
        else:
            doc_kind = DocKind.UNSTRUCTURED
    else:
        logger.info(
            "[structure] %s → rule-based (%d sections, mode=%s)",
            doc_kind.value,
            len(nodes),
            mode,
        )

    tree = build_nodes(nodes)

    if plan.refine and tree:
        depth = refine_max_leaf_depth(tree, doc_kind)
        expanded = await refine_tree_granularity(
            tree, model=model, max_leaf_depth=depth
        )
        logger.info(
            "[structure] refined %d leaves (kind=%s, max_depth=%s)",
            expanded,
            doc_kind.value,
            depth,
        )

    return tree, doc_kind, semantic_used, output_tokens
