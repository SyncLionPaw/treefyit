"""Tests for src.llm — integration tests require API keys."""

import os

import pytest

from src.llm import LLMError, achat, chat, count_tokens


def test_count_tokens():
    assert count_tokens("hello world") > 0
    assert count_tokens("") == 0


@pytest.mark.integration
def test_chat():
    if not os.getenv("DEEPSEEK_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        pytest.skip("No API key configured")
    resp = chat("Reply with exactly: OK", model="deepseek/deepseek-chat")
    assert isinstance(resp, str) and len(resp) > 0


@pytest.mark.integration
def test_chat_with_system():
    if not os.getenv("DEEPSEEK_API_KEY") and not os.getenv("OPENAI_API_KEY"):
        pytest.skip("No API key configured")
    resp = chat(
        "hello",
        model="deepseek/deepseek-chat",
        system="Reply with exactly one word: bonjour",
    )
    assert isinstance(resp, str) and len(resp) > 0


def test_llm_error_is_exported():
    assert issubclass(LLMError, RuntimeError)
