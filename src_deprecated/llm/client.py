"""Minimal LLM request helpers for treefyit."""

from __future__ import annotations

from typing import Any

import litellm

from src.config import get_settings

litellm.drop_params = True


def normalize_optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return value
    cleaned = value.strip()
    return cleaned or None


def build_messages(prompt: str, system: str | None = None) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    return messages


def get_response_text(response: Any) -> str:
    content = response.choices[0].message.content
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        texts = [
            item["text"]
            for item in content
            if isinstance(item, dict)
            and item.get("type") == "text"
            and isinstance(item.get("text"), str)
        ]
        if texts:
            return "\n".join(texts)

    raise ValueError("LLM response did not contain text content")


def complete(
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> str:
    settings = get_settings().llm
    response = litellm.completion(
        model=str(model or settings.model).removeprefix("litellm/"),
        messages=build_messages(prompt, system),
        temperature=temperature if temperature is not None else settings.temperature,
        max_tokens=max_tokens if max_tokens is not None else settings.max_tokens,
        api_key=normalize_optional_text(kwargs.pop("api_key", None) or settings.api_key),
        base_url=normalize_optional_text(kwargs.pop("base_url", None) or settings.base_url),
        **kwargs,
    )
    return get_response_text(response)


async def acomplete(
    prompt: str,
    *,
    model: str | None = None,
    system: str | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    **kwargs: Any,
) -> str:
    settings = get_settings().llm
    response = await litellm.acompletion(
        model=str(model or settings.model).removeprefix("litellm/"),
        messages=build_messages(prompt, system),
        temperature=temperature if temperature is not None else settings.temperature,
        max_tokens=max_tokens if max_tokens is not None else settings.max_tokens,
        api_key=normalize_optional_text(kwargs.pop("api_key", None) or settings.api_key),
        base_url=normalize_optional_text(kwargs.pop("base_url", None) or settings.base_url),
        **kwargs,
    )
    return get_response_text(response)


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    if not text:
        return 0
    return litellm.token_counter(model=str(model).removeprefix("litellm/"), text=text)


__all__ = [
    "acomplete",
    "complete",
    "count_tokens",
]
