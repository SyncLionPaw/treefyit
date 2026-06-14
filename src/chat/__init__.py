"""Chat API — a thin wrapper around ``pagent.Agent`` for tree-based Q&A.

Usage::

    from src.chat import build_streamer

    async for line in build_streamer(bid="81d0882e", question="讲一下背景", model="deepseek-chat"):
        # line is bytes of JSON, one event per line
        pass

Or use the FastAPI endpoint in ``src.server.server``: ``POST /api/chat``.
"""

from __future__ import annotations

from src.chat.agent import build_events, build_streamer  # noqa: F401

__all__ = ["build_streamer", "build_events"]
