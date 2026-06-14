"""Input-content hash cache — dedup identical parses.

Each cache file is named ``cache_{sha256_prefix}.json`` in the results
directory.  The prefix is 16 hex chars — enough to avoid accidental collisions
on a single developer machine, short enough for comfortable filenames.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from . import _ROOT


def key_for(text: str, model: str, mode: str, summarize: bool) -> str:
    """Stable hash prefix for an (input, model, mode, summarize) tuple."""
    raw = f"{text}|{model}|{mode}|{summarize}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def _path(key: str) -> Path:
    _ROOT.mkdir(parents=True, exist_ok=True)
    return _ROOT / f"cache_{key}.json"


def cache_get(key: str) -> dict | None:
    p = _path(key)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def cache_put(key: str, data: dict) -> None:
    _path(key).write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )
