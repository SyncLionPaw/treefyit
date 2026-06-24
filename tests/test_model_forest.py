from __future__ import annotations

import pytest
from pydantic import ValidationError

from treefyit.model.forest import Forest
from treefyit.model.tree import LeafType, TextContent, Tree


def test_forest_roundtrip():
    tree = Tree(
        node_id="doc-1",
        title="paper",
        leaf_type=LeafType.TEXT,
        content=TextContent(text="hello"),
        depth=0,
        subtree_size=1,
        leaf_count=1,
    )
    forest = Forest(forest_id="forest-1", trees=[tree])

    data = forest.to_dict()

    assert data == {
        "forest_id": "forest-1",
        "trees": [
            {
                "node_id": "doc-1",
                "title": "paper",
                "leaf_type": "text",
                "content": {"kind": "text", "text": "hello"},
                "children": [],
                "depth": 0,
                "subtree_size": 1,
                "leaf_count": 1,
            }
        ],
    }

    loaded = Forest.from_dict(data)
    assert loaded == forest
    assert loaded.tree_count == 1
    assert loaded.get_tree("doc-1") is not None


def test_forest_rejects_duplicate_ids():
    forest = Forest(forest_id="forest-1")
    tree = Tree(node_id="doc-1", title="paper")
    forest.add_tree(tree)

    with pytest.raises(ValueError):
        forest.add_tree(tree)


def test_forest_validation():
    with pytest.raises(ValidationError):
        Forest.from_dict({"forest_id": "", "trees": []})
