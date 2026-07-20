"""Bottom-up tree summarization.

This module walks the legacy nested dict tree after build and fills summaries
from leaves to parents. It delegates the actual text generation to the LLM
summary helper, but keeps traversal and tree-specific policy inside builder.
"""

from __future__ import annotations

from src.llm.summarize import summarize_text


def summarize_legacy_tree(nodes: list[dict]) -> None:
    for node in nodes:
        summarize_node(node)


def summarize_node(node: dict) -> str | None:
    children = node.get("children", [])
    child_summaries: list[str] = []

    for child in children:
        child_summary = summarize_node(child)
        if child_summary:
            child_summaries.append(child_summary)

    title = str(node.get("title", "")).strip()
    text = str(node.get("text", "")).strip()
    if not text and not child_summaries:
        return None

    summary = summarize_text(
        title=title or "Untitled",
        content=text,
        child_summaries=child_summaries,
    ).strip()
    if not summary:
        return None

    node["summary"] = summary
    return summary


__all__ = [
    "summarize_legacy_tree",
    "summarize_node",
]
