"""Public LLM helpers for treefyit.

This package contains the thin model invocation layer used by builder-side LLM
strategies. It should stay independent from tree construction logic.
"""

from src.llm.client import acomplete, complete, count_tokens
from src.llm.prompts import (
    SECTION_REFINER_SYSTEM_PROMPT,
    SECTION_REFINER_USER_PROMPT_TEMPLATE,
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_USER_PROMPT_TEMPLATE,
    build_section_refine_prompt,
    build_summary_prompt,
)
from src.llm.refine import refine_section
from src.llm.summarize import asummarize_text, summarize_text

__all__ = [
    "SECTION_REFINER_SYSTEM_PROMPT",
    "SECTION_REFINER_USER_PROMPT_TEMPLATE",
    "SUMMARY_SYSTEM_PROMPT",
    "SUMMARY_USER_PROMPT_TEMPLATE",
    "acomplete",
    "asummarize_text",
    "build_section_refine_prompt",
    "build_summary_prompt",
    "complete",
    "count_tokens",
    "refine_section",
    "summarize_text",
]
