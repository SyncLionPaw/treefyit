from __future__ import annotations

from enum import StrEnum
from typing import Any, Annotated, Literal, Self

from pydantic import AnyUrl, BaseModel, ConfigDict, Field


class LeafType(StrEnum):
    TEXT = "text"
    IMAGE = "image"
    TABLE = "table"
    LINK = "link"


class TextContent(BaseModel):
    kind: Literal["text"] = "text"
    text: str


class UrlContent(BaseModel):
    kind: Literal["url"] = "url"
    url: AnyUrl


class ResourceContent(BaseModel):
    kind: Literal["resource"] = "resource"
    uri: AnyUrl
    media_type: str | None = None


type NodeContent = Annotated[
    TextContent | UrlContent | ResourceContent,
    Field(discriminator="kind"),
]


class Node(BaseModel):
    """A structural node inside one document tree."""

    model_config = ConfigDict(extra="forbid")

    node_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    content: NodeContent | None = None
    summary: str | None = None
    leaf_type: LeafType | None = None
    children: list["Node"] = Field(default_factory=list)

    def add_child(self, child: Self) -> None:
        self.children.append(child)

    @property
    def is_leaf(self) -> bool:
        return not self.children

    @property
    def child_count(self) -> int:
        return len(self.children)

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Self:
        return cls.model_validate(data)


class Tree(Node):
    depth: int | None = Field(default=None, ge=0)
    subtree_size: int | None = Field(default=None, ge=0)
    leaf_count: int | None = Field(default=None, ge=0)
    children: list["Tree"] = Field(default_factory=list)


Node.model_rebuild()
Tree.model_rebuild()

__all__ = [
    "LeafType",
    "TextContent",
    "UrlContent",
    "ResourceContent",
    "NodeContent",
    "Node",
    "Tree",
]
