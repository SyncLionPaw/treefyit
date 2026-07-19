from __future__ import annotations

from enum import StrEnum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field


class NodeKind(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    LINK = "link"


def quote(value: str) -> str:
    escaped = value.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


class TreeNode(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    kind: NodeKind | None = None
    content: str | None = None
    summary: str | None = None
    children: list[TreeNode] = Field(default_factory=list)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Self:
        return cls.model_validate(data)

    def __repr__(self) -> str:
        parts = [f"id={self.id!r}", f"title={self.title!r}"]
        if self.kind is not None:
            parts.append(f"kind={self.kind.value!r}")
        if self.children:
            parts.append(f"children={len(self.children)}")
        return f"TreeNode({', '.join(parts)})"

    def __str__(self) -> str:
        return self.outline(depth=1)

    def detail(self, *, max_content_chars: int | None = None) -> str:
        """查看当前节点详情（含正文），不含深层子树。"""
        lines = [
            f"id={quote(self.id)}",
            f"title={quote(self.title)}",
        ]
        if self.kind is not None:
            lines.append(f"kind={self.kind.value}")
        if self.summary is not None:
            lines.append(f"summary={quote(self.summary)}")

        if self.content is None:
            lines.append("content=null")
        else:
            text = self.content
            truncated = False
            if max_content_chars is not None and len(text) > max_content_chars:
                text = text[:max_content_chars]
                truncated = True
            lines.append(f"content={quote(text)}")
            lines.append(f"content_chars={len(self.content)}")
            if truncated:
                lines.append("content_truncated=true")

        lines.append(f"children={len(self.children)}")
        for index, child in enumerate(self.children):
            parts = [
                f"path={index}",
                f"id={quote(child.id)}",
                f"title={quote(child.title)}",
            ]
            if child.kind is not None:
                parts.append(f"kind={child.kind.value}")
            parts.append(f"children={len(child.children)}")
            lines.append("child: " + " ".join(parts))
        return "\n".join(lines)

    def outline(self, depth: int = 1, *, indent: str = "  ") -> str:
        """LLM 友好大纲：自身 + 向下最多 ``depth`` 层孩子。

        ``depth=0`` 只显示自身；``depth=1`` 显示自身与直接孩子。
        每行带 ``path``，便于模型按路径导航。不包含正文全文。
        """
        if depth < 0:
            raise ValueError("depth must be >= 0")

        def line(node: TreeNode, path: str) -> str:
            parts = [
                f"path={path}",
                f"id={quote(node.id)}",
                f"title={quote(node.title)}",
            ]
            if node.kind is not None:
                parts.append(f"kind={node.kind.value}")
            if node.summary:
                parts.append(f"summary={quote(node.summary)}")
            if node.content is not None:
                parts.append(f"content_chars={len(node.content)}")
            parts.append(f"children={len(node.children)}")
            return " ".join(parts)

        lines = [line(self, "root")]

        def walk(
            nodes: list[TreeNode],
            parent_path: str,
            level: int,
            remaining: int,
        ) -> None:
            if remaining <= 0:
                return
            prefix = indent * level
            for index, node in enumerate(nodes):
                path = str(index) if parent_path == "root" else f"{parent_path}.{index}"
                lines.append(f"{prefix}{line(node, path)}")
                if node.children:
                    walk(node.children, path, level + 1, remaining - 1)

        walk(self.children, "root", 1, depth)
        return "\n".join(lines)
