"""LLM client — one function, zero ceremony.

Set ``OPENAI_API_KEY`` via environment or a ``.env`` file.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
from pathlib import Path
from typing import Any

import litellm
from dotenv import load_dotenv

# Auto-load .env from project root and ancestors
_root = Path(__file__).resolve().parent.parent.parent
for d in [_root] + list(_root.parents):
    load_dotenv(d / ".env", override=False)
load_dotenv()  # also try cwd

# Backward-compat: CHATGPT_API_KEY → OPENAI_API_KEY
if not os.getenv("OPENAI_API_KEY") and os.getenv("CHATGPT_API_KEY"):
    os.environ["OPENAI_API_KEY"] = os.getenv("CHATGPT_API_KEY")

logger = logging.getLogger(__name__)
litellm.drop_params = True

MAX_RETRIES = 10
RETRY_DELAY = 1.0


class LLMError(RuntimeError):
    """Raised when all LLM retry attempts are exhausted."""


def chat(prompt: str, **kwargs: Any) -> str:
    """Send a prompt to the LLM and return the response text.

    Args:
        prompt: The user prompt.
        **kwargs: Passed to ``litellm.completion``. Common keys:
            model (str)      — default ``"gpt-4o"``
            system (str)     — system message
            temperature (float) — default 0
            max_tokens (int) — output limit

    Returns:
        Response text.

    Raises:
        LLMError: If all retries fail.
    """
    model = str(kwargs.pop("model", "gpt-4o")).removeprefix("litellm/")
    system = kwargs.pop("system", None)
    temperature = kwargs.pop("temperature", 0.0)
    max_tokens = kwargs.pop("max_tokens", None)

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for i in range(MAX_RETRIES):
        try:
            response = litellm.completion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return response.choices[0].message.content or ""
        except Exception:
            logger.warning("chat attempt %d/%d failed for model=%s", i + 1, MAX_RETRIES, model, exc_info=True)
            if i < MAX_RETRIES - 1:
                time.sleep(RETRY_DELAY)
    raise LLMError(f"chat failed after {MAX_RETRIES} attempts for model={model}")


async def achat(prompt: str, **kwargs: Any) -> str:
    """Async version of :func:`chat`."""
    model = str(kwargs.pop("model", "gpt-4o")).removeprefix("litellm/")
    system = kwargs.pop("system", None)
    temperature = kwargs.pop("temperature", 0.0)
    max_tokens = kwargs.pop("max_tokens", None)

    messages: list[dict[str, str]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    for i in range(MAX_RETRIES):
        try:
            response = await litellm.acompletion(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                **kwargs,
            )
            return response.choices[0].message.content or ""
        except Exception:
            logger.warning("achat attempt %d/%d failed for model=%s", i + 1, MAX_RETRIES, model, exc_info=True)
            if i < MAX_RETRIES - 1:
                await asyncio.sleep(RETRY_DELAY)
    raise LLMError(f"achat failed after {MAX_RETRIES} attempts for model={model}")


def count_tokens(text: str, model: str = "gpt-4o") -> int:
    """Return the token count for *text*."""
    if not text:
        return 0
    return litellm.token_counter(model=model, text=text)
