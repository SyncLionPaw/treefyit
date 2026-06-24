"""Serialize / deserialize / persist the tree structure."""

from __future__ import annotations

import json
from pathlib import Path

from src.tree.model import Tree, from_wire_tree, to_wire_tree


# ---------------------------------------------------------------------------
# Serialize / Deserialize
# ---------------------------------------------------------------------------

def to_json(tree: Tree, indent: int = 2) -> str:
    """Tree → JSON string."""
    return json.dumps(to_wire_tree(tree), indent=indent, ensure_ascii=False)


def from_json(data: str) -> Tree:
    """JSON string → tree."""
    return from_wire_tree(json.loads(data))


# ---------------------------------------------------------------------------
# Persist
# ---------------------------------------------------------------------------

def save(tree: Tree, path: str | Path) -> None:
    """Save tree to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(to_json(tree), encoding="utf-8")


def load(path: str | Path) -> Tree:
    """Load tree from a JSON file."""
    return from_json(Path(path).read_text(encoding="utf-8"))
