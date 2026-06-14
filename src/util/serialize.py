"""Serialize / deserialize / persist the tree structure."""

from __future__ import annotations

import json
from pathlib import Path


# ---------------------------------------------------------------------------
# Serialize / Deserialize
# ---------------------------------------------------------------------------

def to_json(tree: list[dict], indent: int = 2) -> str:
    """Tree → JSON string."""
    return json.dumps(tree, indent=indent, ensure_ascii=False)


def from_json(data: str) -> list[dict]:
    """JSON string → tree."""
    return json.loads(data)


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------

def save(tree: list[dict], path: str | Path) -> None:
    """Save tree to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(tree), encoding="utf-8")


def load(path: str | Path) -> list[dict]:
    """Load tree from a JSON file."""
    return from_json(Path(path).read_text(encoding="utf-8"))
