"""Async document → tree pipeline shared by the HTTP server."""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from src.llm import count_tokens
from src.parser.md import parse_md
from src.tree.builder import (
    AUTO_SEMANTIC_MAX_TOKENS,
    THINNING_THRESHOLD,
    assign_node_ids,
    build_nodes,
    flatten_tree,
    summarize_tree,
    thin_tree,
)
from src.tree.semantic import (
    attach_text_ranges,
    extract_structure,
    refine_tree_granularity,
)
from src.tree.verify import verify_tree

logger = logging.getLogger(__name__)

ProgressFn = Callable[[dict], Awaitable[None]] | None

SEMANTIC_OUTPUT_TOKEN_ESTIMATE = 2000


@dataclass(frozen=True)
class UploadKind:
    is_pdf: bool
    is_zip: bool
    is_html: bool


@dataclass
class BuildOutput:
    tree: list[dict]
    input_tokens: int
    output_tokens: int
    verify_result: dict | None
    semantic_used: bool


def classify_upload(filename: str) -> UploadKind:
    lname = filename.lower()
    return UploadKind(
        is_pdf=lname.endswith(".pdf"),
        is_zip=lname.endswith(".zip"),
        is_html=lname.endswith((".html", ".htm")),
    )


def text_from_upload(raw: bytes, filename: str) -> str:
    """Decode upload bytes to plain text (PDF is converted via MinerU)."""
    kind = classify_upload(filename)
    if kind.is_pdf:
        from src.parser.pdf import parse_pdf

        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tf:
            tf.write(raw)
            tmp_pdf = tf.name
        try:
            return parse_pdf(tmp_pdf)
        finally:
            Path(tmp_pdf).unlink(missing_ok=True)
    return raw.decode("utf-8")


def temp_upload_file(bid: str, filename: str, *, suffix: str | None = None) -> Path:
    """Return a safe temp path for a one-off parser input file."""
    ext = suffix if suffix is not None else Path(filename).suffix or ".txt"
    fd, path = tempfile.mkstemp(prefix=f"treefyit_{bid}_", suffix=ext)
    os.close(fd)
    return Path(path)


async def build_tree_from_upload(
    *,
    raw: bytes,
    text: str,
    filename: str,
    bid: str,
    model: str,
    mode: str,
    summarize_tree: bool,
    progress: ProgressFn = None,
) -> BuildOutput:
    """Build a nested tree from uploaded bytes + extracted text."""
    kind = classify_upload(filename)
    input_tokens = count_tokens(text, model=model)
    output_tokens = 0
    semantic_used = False
    skip_thin = kind.is_zip

    async def emit(payload: dict) -> None:
        if progress:
            await progress(payload)

    if kind.is_zip:
        from src.parser.zip import parse_zip

        tmp = temp_upload_file(bid, filename, suffix=".zip")
        try:
            tmp.write_bytes(raw)
            tree = parse_zip(str(tmp), parser=mode)
        finally:
            tmp.unlink(missing_ok=True)
    elif kind.is_html:
        from src.parser.html import parse_html

        tmp = temp_upload_file(bid, filename, suffix=".html")
        try:
            tmp.write_text(text, encoding="utf-8")
            nodes = parse_html(str(tmp))
        finally:
            tmp.unlink(missing_ok=True)
        tree = build_nodes(nodes)
    elif mode == "semantic":
        nodes = await extract_structure(text, model=model)
        attach_text_ranges(nodes, text)
        tree = build_nodes(nodes)
        semantic_used = True
        output_tokens += SEMANTIC_OUTPUT_TOKEN_ESTIMATE
    else:
        use_zero_shot = (
            mode == "auto" and kind.is_pdf and input_tokens <= AUTO_SEMANTIC_MAX_TOKENS
        )
        nodes: list[dict] = []

        if use_zero_shot:
            await emit(
                {
                    "stage": "semantic_zero_shot",
                    "message": "short PDF; extracting structure with LLM",
                    "input_tokens": input_tokens,
                }
            )
            nodes = await extract_structure(text, model=model)
            attach_text_ranges(nodes, text)
            semantic_used = True
            output_tokens += SEMANTIC_OUTPUT_TOKEN_ESTIMATE
        else:
            tmp = temp_upload_file(bid, filename)
            try:
                tmp.write_text(text, encoding="utf-8")
                nodes = parse_md(str(tmp))
            finally:
                tmp.unlink(missing_ok=True)
            await emit({"stage": "md_parsed", "nodes": len(nodes)})

        if not nodes and not semantic_used:
            await emit(
                {
                    "stage": "semantic",
                    "message": "markdown empty; extracting semantic structure",
                }
            )
            nodes = await extract_structure(text, model=model)
            attach_text_ranges(nodes, text)
            semantic_used = True
            output_tokens += SEMANTIC_OUTPUT_TOKEN_ESTIMATE

        tree = build_nodes(nodes)

    await emit(
        {
            "stage": "structure_done",
            "root_nodes": len(tree),
            "node_count": len(flatten_tree(tree)),
            "semantic_used": semantic_used,
        }
    )

    if semantic_used:
        await emit({"stage": "refine", "message": "splitting large sections"})

        async def refine_progress(payload: dict) -> None:
            await emit({"stage": "refine", **payload})

        expanded = await refine_tree_granularity(
            tree, model=model, progress=refine_progress
        )
        logger.info("[build] bid=%s semantic refined leaves=%d", bid, expanded)

    if not skip_thin:
        await emit({"stage": "thin", "message": "thinning tree"})
        thin_tree(tree, threshold=THINNING_THRESHOLD, model=model)
        assign_node_ids(tree)
        await emit(
            {
                "stage": "thin_done",
                "node_count": len(flatten_tree(tree)),
            }
        )

    if summarize_tree:
        await emit({"stage": "summarize", "total": len(flatten_tree(tree))})

        async def summarize_progress(payload: dict) -> None:
            await emit({"stage": "summarize", **payload})

        await summarize_tree(tree, model=model, progress=summarize_progress)
        output_tokens += sum(
            count_tokens(n.get("summary", ""), model=model) for n in flatten_tree(tree)
        )

    verify_result = None
    if tree:
        await emit({"stage": "verify", "message": "verifying tree"})
        try:
            verify_result = await verify_tree(tree, model=model)
            await emit(
                {
                    "stage": "verify_done",
                    "ok": verify_result["ok"],
                    "score": verify_result["score"],
                }
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("[build] bid=%s verify error: %s", bid, exc)
            await emit({"stage": "verify_failed", "message": str(exc)})

    return BuildOutput(
        tree=tree,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        verify_result=verify_result,
        semantic_used=semantic_used,
    )
