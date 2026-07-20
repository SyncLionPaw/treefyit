"""LLM-backed summarization helpers."""

from __future__ import annotations

from typing import Any

from src.llm.client import acomplete, complete
from src.llm.prompts import SUMMARY_SYSTEM_PROMPT, build_summary_prompt


def summarize_text(
    *,
    title: str,
    content: str = "",
    child_summaries: list[str] | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> str:
    prompt = build_summary_prompt(
        title=title,
        content=content,
        child_summaries=child_summaries,
    )
    summary = complete(
        prompt,
        system=SUMMARY_SYSTEM_PROMPT,
        model=model,
        max_tokens=max_tokens,
        **kwargs,
    )
    return summary.strip()


async def asummarize_text(
    *,
    title: str,
    content: str = "",
    child_summaries: list[str] | None = None,
    model: str | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> str:
    prompt = build_summary_prompt(
        title=title,
        content=content,
        child_summaries=child_summaries,
    )
    summary = await acomplete(
        prompt,
        system=SUMMARY_SYSTEM_PROMPT,
        model=model,
        max_tokens=max_tokens,
        **kwargs,
    )
    return summary.strip()


__all__ = [
    "asummarize_text",
    "summarize_text",
]
