from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from treefyit.model.tree import Tree


class Forest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    forest_id: str = Field(min_length=1)
    trees: list[Tree] = Field(default_factory=list)

    @property
    def tree_count(self) -> int:
        return len(self.trees)

    def add_tree(self, tree: Tree) -> None:
        if self.get_tree(tree.node_id) is not None:
            raise ValueError(f"duplicate tree id: {tree.node_id}")
        self.trees.append(tree)

    def get_tree(self, tree_id: str) -> Tree | None:
        for tree in self.trees:
            if tree.node_id == tree_id:
                return tree
        return None

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> "Forest":
        return cls.model_validate(data)


__all__ = [
    "Forest",
]
