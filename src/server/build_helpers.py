"""Shared helpers for build API responses."""

from __future__ import annotations

import time

from src.vis.mermaid import to_mermaid


def attach_file_meta(result: dict, file_meta: dict | None, bid: str) -> None:
    if not file_meta:
        return
    result.update(
        {
            "content_type": file_meta["content_type"],
            "file_size": file_meta["size"],
            "sha256": file_meta["sha256"],
            "storage_key": file_meta["key"],
            "has_original_file": True,
            "original_file_url": f"/api/build/{bid}/file",
        }
    )


def cached_build_result(
    bid: str,
    filename: str,
    cached: dict,
    *,
    file_meta: dict | None,
) -> dict:
    tree = cached["tree"]
    result = {
        "id": bid,
        "filename": filename,
        "raw_text": cached.get("raw_text", ""),
        "mermaid": cached.get("mermaid", to_mermaid(tree)),
        "tree": tree,
        "stats": cached["stats"],
        "created_at": time.strftime("%H:%M:%S"),
        "cached": True,
    }
    attach_file_meta(result, file_meta, bid)
    return result


def build_stats(
    *,
    input_tokens: int,
    output_tokens: int,
    elapsed_sec: float,
    model: str,
    mode: str,
    node_count: int,
    doc_kind: str | None = None,
    verify_result: dict | None = None,
) -> dict:
    stats = {
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "node_count": node_count,
        "elapsed_sec": round(elapsed_sec, 1),
        "model": model,
        "mode": mode,
    }
    if doc_kind:
        stats["doc_kind"] = doc_kind
    if verify_result:
        stats["verify"] = verify_result
    return stats


def success_build_result(
    bid: str,
    filename: str,
    text: str,
    tree: list[dict],
    *,
    stats: dict,
    file_meta: dict | None,
) -> dict:
    result = {
        "id": bid,
        "filename": filename,
        "raw_text": text,
        "mermaid": to_mermaid(tree),
        "tree": tree,
        "stats": stats,
        "created_at": time.strftime("%H:%M:%S"),
    }
    attach_file_meta(result, file_meta, bid)
    return result


def error_build_result(
    bid: str,
    filename: str,
    error: str,
    *,
    elapsed_sec: float,
    model: str,
    mode: str,
    file_meta: dict | None,
) -> dict:
    result = {
        "id": bid,
        "filename": filename,
        "error": error,
        "tree": [],
        "mermaid": "",
        "stats": {
            "input_tokens": 0,
            "output_tokens": 0,
            "node_count": 0,
            "elapsed_sec": round(elapsed_sec, 1),
            "model": model,
            "mode": mode,
        },
        "created_at": time.strftime("%H:%M:%S"),
    }
    attach_file_meta(result, file_meta, bid)
    return result
