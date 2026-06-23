"""Async document → tree pipeline shared by the HTTP server."""

from __future__ import annotations

import logging
import os
import tempfile
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

from src.llm import count_tokens
from src.tree.builder import (
    THINNING_THRESHOLD,
    assign_node_ids,
    build_nodes,
    flatten_tree,
    summarize_tree as run_summarize,
    thin_tree,
)
from src.tree.doc_kind import DocKind
from src.tree.structure import build_tree_structure
from src.tree.verify import verify_tree

logger = logging.getLogger(__name__)

ProgressFn = Callable[[dict], Awaitable[None]] | None


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
    doc_kind: str


def classify_upload(filename: str) -> UploadKind:
    lname = filename.lower()
    return UploadKind(
        is_pdf=lname.endswith(".pdf"),
        is_zip=lname.endswith(".zip"),
        is_html=lname.endswith((".html", ".htm")),
    )


def decode_text_bytes(raw: bytes) -> str:
    """Decode plain-text uploads; try UTF-8 first, then GB18030 for legacy Chinese .txt."""
    if not raw:
        return ""
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


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
    return decode_text_bytes(raw)


def text_from_url(url: str, **kwargs) -> str:
    """Fetch a remote URL and return Markdown or plain text."""
    from src.parser.url import parse_url

    return parse_url(url, **kwargs)


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
    summarize: bool,
    progress: ProgressFn = None,
) -> BuildOutput:
    """Build a nested tree from uploaded bytes + extracted text."""
    kind = classify_upload(filename)
    input_tokens = count_tokens(text, model=model)
    output_tokens = 0
    semantic_used = False
    doc_kind = DocKind.UNSTRUCTURED.value
    skip_thin = kind.is_zip

    async def emit(payload: dict) -> None:
        if progress:
            await progress(payload)

    async def semantic_progress(payload: dict) -> None:
        await emit({"stage": "semantic", **payload})

    if kind.is_zip:
        from src.parser.zip import parse_zip

        tmp = temp_upload_file(bid, filename, suffix=".zip")
        try:
            tmp.write_bytes(raw)
            tree = parse_zip(str(tmp), parser=mode)
        finally:
            tmp.unlink(missing_ok=True)
        doc_kind = DocKind.ZIP.value
    elif kind.is_html:
        from src.parser.html import parse_html

        tmp = temp_upload_file(bid, filename, suffix=".html")
        try:
            tmp.write_text(text, encoding="utf-8")
            nodes = parse_html(str(tmp))
        finally:
            tmp.unlink(missing_ok=True)
        tree = build_nodes(nodes)
        doc_kind = DocKind.HTML.value
    else:
        tmp = temp_upload_file(bid, filename)
        try:
            tmp.write_text(text, encoding="utf-8")
            await emit(
                {
                    "stage": "classify",
                    "message": "detecting document type",
                    "mode": mode,
                }
            )
            tree, detected, semantic_used, est = await build_tree_structure(
                text=text,
                source_path=tmp,
                model=model,
                mode=mode,
                input_tokens=input_tokens,
                is_pdf=kind.is_pdf,
                semantic_progress=semantic_progress,
            )
            doc_kind = detected.value
            output_tokens += est
            await emit(
                {
                    "stage": "structure",
                    "doc_kind": doc_kind,
                    "semantic_used": semantic_used,
                    "root_nodes": len(tree),
                }
            )
        finally:
            tmp.unlink(missing_ok=True)

    await emit(
        {
            "stage": "structure_done",
            "doc_kind": doc_kind,
            "root_nodes": len(tree),
            "node_count": len(flatten_tree(tree)),
            "semantic_used": semantic_used,
        }
    )

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

    if summarize:
        await emit({"stage": "summarize", "total": len(flatten_tree(tree))})

        async def summarize_progress(payload: dict) -> None:
            await emit({"stage": "summarize", **payload})

        await run_summarize(tree, model=model, progress=summarize_progress)
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
        doc_kind=doc_kind,
    )
