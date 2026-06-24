from __future__ import annotations

import asyncio

from treefyit.llm import (
    SUMMARY_SYSTEM_PROMPT,
    asummarize_text,
    build_summary_prompt,
    summarize_text,
)


def test_build_summary_prompt_formats_stable_sections():
    prompt = build_summary_prompt(
        title="Overview",
        content="This section explains the architecture.",
        child_summaries=["First child summary", "Second child summary"],
    )

    assert "当前节点标题：\nOverview" in prompt
    assert "当前节点正文：\nThis section explains the architecture." in prompt
    assert "子节点摘要：\n- First child summary\n- Second child summary" in prompt


def test_build_summary_prompt_uses_placeholder_values():
    prompt = build_summary_prompt(title="", content="", child_summaries=[])

    assert "当前节点标题：\nUntitled" in prompt
    assert "当前节点正文：\n无" in prompt
    assert "子节点摘要：\n无" in prompt


def test_summarize_text_calls_complete(monkeypatch):
    captured = {}

    def fake_complete(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return " summary "

    monkeypatch.setattr("treefyit.llm.summarize.complete", fake_complete)

    summary = summarize_text(
        title="Overview",
        content="Main content",
        child_summaries=["Child one"],
    )

    assert summary == "summary"
    assert captured["kwargs"]["system"] == SUMMARY_SYSTEM_PROMPT
    assert "当前节点标题：\nOverview" in captured["prompt"]


def test_asummarize_text_calls_acomplete(monkeypatch):
    captured = {}

    async def fake_acomplete(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return " async summary "

    monkeypatch.setattr("treefyit.llm.summarize.acomplete", fake_acomplete)

    summary = asyncio.run(
        asummarize_text(
            title="Overview",
            content="Main content",
        )
    )

    assert summary == "async summary"
    assert captured["kwargs"]["system"] == SUMMARY_SYSTEM_PROMPT
    assert "当前节点正文：\nMain content" in captured["prompt"]
