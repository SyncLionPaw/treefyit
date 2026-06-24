from __future__ import annotations

from treefyit.llm import build_section_refine_prompt, refine_section
from treefyit.llm.refine import (
    extract_json_text,
    extract_section_blocks,
    parse_refined_section_response,
)


def test_build_section_refine_prompt_formats_stable_sections():
    prompt = build_section_refine_prompt(
        title="Overview",
        content="Long section body.",
        source_kind="markdown",
        max_parts=4,
    )

    assert "source_kind: markdown" in prompt
    assert "title: Overview" in prompt
    assert "Long section body." in prompt
    assert "最多返回 4 个 [SECTION]" in prompt
    assert "[SECTION]" in prompt
    assert "[TEXT]" in prompt


def test_extract_section_blocks_supports_fenced_delimited_text():
    text = """```text
[SECTION]
title: Overview
level_delta: 0
content_kind: text
[TEXT]
Body
[/TEXT]
[/SECTION]
```"""

    assert extract_section_blocks(text) == [
        "title: Overview\nlevel_delta: 0\ncontent_kind: text\n[TEXT]\nBody\n[/TEXT]"
    ]


def test_parse_refined_section_response_parses_delimited_text():
    payload = parse_refined_section_response(
        """[SECTION]
title: Overview
level_delta: 1
content_kind: text
[TEXT]
Body
[/TEXT]
[SUMMARY]
Summary
[/SUMMARY]
[/SECTION]"""
    )

    assert len(payload.sections) == 1
    assert payload.sections[0].title == "Overview"
    assert payload.sections[0].level_delta == 1
    assert payload.sections[0].text == "Body"
    assert payload.sections[0].summary == "Summary"


def test_parse_refined_section_response_keeps_json_fallback():
    payload = parse_refined_section_response(
        '{"sections":[{"title":"Overview","level_delta":1,"text":"Body"}]}'
    )

    assert len(payload.sections) == 1
    assert payload.sections[0].title == "Overview"


def test_refine_section_calls_complete(monkeypatch):
    captured = {}

    def fake_complete(prompt, **kwargs):
        captured["prompt"] = prompt
        captured["kwargs"] = kwargs
        return """[SECTION]
title: Overview
level_delta: 0
content_kind: text
[TEXT]
Body
[/TEXT]
[/SECTION]"""

    monkeypatch.setattr("treefyit.llm.refine.complete", fake_complete)

    sections = refine_section(
        title="Overview",
        content="Body",
        source_kind="markdown",
        max_parts=3,
    )

    assert sections == [
        {"title": "Overview", "level_delta": 0, "text": "Body", "content_kind": "text"}
    ]
    assert "title: Overview" in captured["prompt"]
    assert captured["kwargs"]["system"]
