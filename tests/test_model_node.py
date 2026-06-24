from __future__ import annotations

import pytest
from pydantic import ValidationError

from treefyit.model.tree import (
    LeafType,
    Node,
    ResourceContent,
    TextContent,
    Tree,
    UrlContent,
)


def test_node_to_dict_and_from_dict_roundtrip():
    root = Node(
        node_id="root",
        title="doc",
        summary="document root",
        children=[
            Node(
                node_id="0001",
                title="Intro",
                content=TextContent(text="hello"),
                children=[
                    Node(
                        node_id="0001.0001",
                        title="Scope",
                        leaf_type=LeafType.TEXT,
                    )
                ],
            )
        ],
    )

    data = root.to_dict()

    assert data == {
        "node_id": "root",
        "title": "doc",
        "summary": "document root",
        "children": [
            {
                "node_id": "0001",
                "title": "Intro",
                "content": {"kind": "text", "text": "hello"},
                "children": [
                    {
                        "node_id": "0001.0001",
                        "title": "Scope",
                        "leaf_type": "text",
                        "children": [],
                    }
                ],
            }
        ],
    }

    loaded = Node.from_dict(data)
    assert loaded == root
    assert isinstance(loaded.children[0].content, TextContent)
    assert loaded.children[0].children[0].is_leaf is True
    assert loaded.children[0].children[0].leaf_type is LeafType.TEXT


def test_node_from_dict_rejects_invalid_payload():
    with pytest.raises(ValidationError):
        Node.from_dict({"node_id": "", "title": "x", "children": []})

    with pytest.raises(ValidationError):
        Node.from_dict({"node_id": "n1", "title": "x", "children": {}})

    with pytest.raises(ValidationError):
        Node.from_dict(
            {"node_id": "n1", "title": "x", "children": [], "leaf_type": "bad"}
        )

    with pytest.raises(ValidationError):
        Node.from_dict(
            {
                "node_id": "n1",
                "title": "x",
                "children": [],
                "content": {"kind": "bad"},
            }
        )


def test_tree_keeps_stats_and_leaf_semantics():
    node = Tree(
        node_id="n1",
        title="figure",
        leaf_type=LeafType.IMAGE,
        depth=2,
        subtree_size=1,
        leaf_count=1,
    )

    assert node.is_leaf is True
    assert node.child_count == 0
    assert node.to_dict() == {
        "node_id": "n1",
        "title": "figure",
        "leaf_type": "image",
        "children": [],
        "depth": 2,
        "subtree_size": 1,
        "leaf_count": 1,
    }

    loaded = Tree.from_dict(node.to_dict())
    assert isinstance(loaded, Tree)
    assert loaded.depth == 2
    assert loaded.subtree_size == 1
    assert loaded.leaf_count == 1


def test_node_content_supports_url_and_resource():
    url_node = Node(
        node_id="n-url",
        title="homepage",
        leaf_type=LeafType.LINK,
        content=UrlContent(url="https://example.com/docs"),
    )
    assert url_node.to_dict()["content"] == {
        "kind": "url",
        "url": "https://example.com/docs",
    }

    resource_node = Node.from_dict(
        {
            "node_id": "n-res",
            "title": "image",
            "leaf_type": "image",
            "children": [],
            "content": {
                "kind": "resource",
                "uri": "https://example.com/image.png",
                "media_type": "image/png",
            },
        }
    )
    assert isinstance(resource_node.content, ResourceContent)
    assert str(resource_node.content.uri) == "https://example.com/image.png"
