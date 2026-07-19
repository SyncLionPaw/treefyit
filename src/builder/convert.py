"""Model conversion.

This module converts the internal nested dict tree into the typed `treefyit.model.tree.Tree`
model and fills derived structural stats. It also assigns node ids while creating
the final tree model, instead of mutating the legacy intermediate tree.
"""

from __future__ import annotations

from treefyit.model.tree import LeafType, ResourceContent, TextContent, Tree, UrlContent


def build_model_tree(legacy_tree: list[dict], *, root_id: str, root_title: str) -> Tree:
    counter = 0

    def next_node_id() -> str:
        nonlocal counter
        counter += 1
        return f"node-{counter}"

    children = [
        build_model_node(node, depth=1, next_node_id=next_node_id)
        for node in legacy_tree
    ]
    return Tree(
        node_id=root_id,
        title=root_title,
        leaf_type=LeafType.TEXT,
        children=children,
        depth=0,
        subtree_size=1 + sum(child.subtree_size or 1 for child in children),
        leaf_count=sum(child.leaf_count or 1 for child in children),
    )


def build_model_node(legacy_node: dict, *, depth: int, next_node_id) -> Tree:
    child_models = [
        build_model_node(child, depth=depth + 1, next_node_id=next_node_id)
        for child in legacy_node.get("children", [])
    ]
    content_text = str(legacy_node.get("text", "")).strip()
    summary = str(legacy_node.get("summary", "")).strip() or None
    subtree_size = 1 + sum(child.subtree_size or 1 for child in child_models)
    leaf_count = (
        1 if not child_models else sum(child.leaf_count or 1 for child in child_models)
    )
    content = build_node_content(legacy_node, content_text)
    leaf_type = build_leaf_type(legacy_node, content)

    return Tree(
        node_id=next_node_id(),
        title=str(legacy_node.get("title", "")).strip() or "Untitled",
        content=content,
        summary=summary,
        leaf_type=leaf_type,
        children=child_models,
        depth=depth,
        subtree_size=subtree_size,
        leaf_count=leaf_count,
    )


def build_node_content(legacy_node: dict, content_text: str):
    content_kind = legacy_node.get("content_kind")
    if content_kind == "url":
        url = str(legacy_node.get("url", "")).strip()
        if url:
            return UrlContent(url=url)

    if content_kind == "resource":
        uri = str(legacy_node.get("uri", "")).strip()
        if uri:
            media_type = str(legacy_node.get("media_type", "")).strip() or None
            return ResourceContent(uri=uri, media_type=media_type)

    if content_text:
        return TextContent(text=content_text)
    return None


def build_leaf_type(legacy_node: dict, content) -> LeafType | None:
    raw_leaf_type = str(legacy_node.get("leaf_type", "")).strip()
    if raw_leaf_type:
        try:
            return LeafType(raw_leaf_type)
        except ValueError:
            pass
    if content is None:
        return None
    return LeafType.TEXT


__all__ = ["build_model_tree", "build_model_node"]
