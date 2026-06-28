from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from treefyit.config import AppSettings, BuilderSettings, LLMSettings, MinerUSettings
from treefyit.llm import acomplete, complete, count_tokens


def make_response(content):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                message=SimpleNamespace(content=content),
            )
        ]
    )


def test_complete_calls_litellm(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return make_response("done")

    monkeypatch.setattr("treefyit.llm.client.litellm.completion", fake_completion)

    result = complete(
        "hello",
        model="litellm/gpt-4o-mini",
        system="reply shortly",
        temperature=0.2,
    )

    assert result == "done"
    assert captured["model"] == "gpt-4o-mini"
    assert captured["messages"] == [
        {"role": "system", "content": "reply shortly"},
        {"role": "user", "content": "hello"},
    ]
    assert captured["temperature"] == 0.2


def test_acomplete_calls_litellm(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return make_response("async done")

    monkeypatch.setattr("treefyit.llm.client.litellm.acompletion", fake_acompletion)

    result = asyncio.run(acomplete("hello async"))

    assert result == "async done"
    assert captured["messages"] == [
        {"role": "user", "content": "hello async"},
    ]


def test_complete_supports_content_blocks(monkeypatch: pytest.MonkeyPatch):
    def fake_completion(**kwargs):
        return make_response(
            [
                {"type": "text", "text": "first"},
                {"type": "tool_use", "name": "noop"},
                {"type": "text", "text": "second"},
            ]
        )

    monkeypatch.setattr("treefyit.llm.client.litellm.completion", fake_completion)

    assert complete("hello") == "first\nsecond"


def test_complete_requires_text_content(monkeypatch: pytest.MonkeyPatch):
    def fake_completion(**kwargs):
        return make_response([{"type": "tool_use", "name": "noop"}])

    monkeypatch.setattr("treefyit.llm.client.litellm.completion", fake_completion)

    with pytest.raises(ValueError):
        complete("hello")


def test_count_tokens_returns_zero_for_empty_text():
    assert count_tokens("") == 0


def test_complete_uses_llm_settings_by_default(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return make_response("done")

    monkeypatch.setattr(
        "treefyit.llm.client.get_settings",
        lambda: AppSettings(
            llm=LLMSettings(
                model="gpt-4o-mini",
                api_key="demo-key",
                base_url="https://example.com/v1",
                temperature=0.3,
                max_tokens=123,
            ),
            mineru=MinerUSettings(),
            builder=BuilderSettings(),
        ),
    )
    monkeypatch.setattr("treefyit.llm.client.litellm.completion", fake_completion)

    result = complete("hello")

    assert result == "done"
    assert captured["model"] == "gpt-4o-mini"
    assert captured["api_key"] == "demo-key"
    assert captured["base_url"] == "https://example.com/v1"
    assert captured["temperature"] == 0.3
    assert captured["max_tokens"] == 123


def test_complete_normalizes_blank_llm_settings(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return make_response("done")

    monkeypatch.setattr(
        "treefyit.llm.client.get_settings",
        lambda: AppSettings(
            llm=LLMSettings(
                model="ollama/qwen3:1.7b",
                api_key="   ",
                base_url=" http://127.0.0.1:11434/ ",
            ),
            mineru=MinerUSettings(),
            builder=BuilderSettings(),
        ),
    )
    monkeypatch.setattr("treefyit.llm.client.litellm.completion", fake_completion)

    result = complete("hello")

    assert result == "done"
    assert captured["api_key"] is None
    assert captured["base_url"] == "http://127.0.0.1:11434/"
