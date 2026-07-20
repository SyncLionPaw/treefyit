"""TreeNode 持久化目录：JSON 落盘，可再次加载。"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field

from .ops import collect_ids
from .tree import TreeNode


def slugify(value: str) -> str:
    text = value.strip().lower()
    text = re.sub(r"\s+", "-", text)
    text = re.sub(r"[^a-z0-9_\u4e00-\u9fff-]+", "", text)
    return text[:48] or "tree"


def utc_now() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def file_sha256(path: Path) -> str:
    digest = sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


class TreeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    tree_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    source_path: str | None = None
    source_sha256: str | None = None
    updated_at: str
    node_count: int = Field(ge=1)
    root: TreeNode

    def summary(self) -> dict[str, object]:
        return {
            "tree_id": self.tree_id,
            "title": self.title,
            "source_path": self.source_path,
            "updated_at": self.updated_at,
            "node_count": self.node_count,
        }


class TreeStore:
    """``{data_dir}/trees/{tree_id}.json`` 一棵树一个文件。"""

    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)

    @property
    def trees_dir(self) -> Path:
        return self.data_dir / "trees"

    def ensure_dirs(self) -> None:
        self.trees_dir.mkdir(parents=True, exist_ok=True)

    def tree_path(self, tree_id: str) -> Path:
        return self.trees_dir / f"{tree_id}.json"

    def exists(self, tree_id: str) -> bool:
        return self.tree_path(tree_id).exists()

    def save(
        self,
        root: TreeNode,
        *,
        tree_id: str | None = None,
        title: str | None = None,
        source_path: str | Path | None = None,
    ) -> TreeRecord:
        self.ensure_dirs()
        resolved_id = tree_id or f"{slugify(title or root.title)}-{uuid4().hex[:8]}"
        source = str(source_path) if source_path is not None else None
        source_hash = None
        if source_path is not None:
            path = Path(source_path)
            if path.is_file():
                source_hash = file_sha256(path)

        record = TreeRecord(
            tree_id=resolved_id,
            title=title or root.title,
            source_path=source,
            source_sha256=source_hash,
            updated_at=utc_now(),
            node_count=len(collect_ids(root)),
            root=root,
        )
        path = self.tree_path(resolved_id)
        path.write_text(
            record.model_dump_json(indent=2, exclude_none=True),
            encoding="utf-8",
        )
        return record

    def load(self, tree_id: str) -> TreeRecord:
        path = self.tree_path(tree_id)
        if not path.exists():
            raise KeyError(f"tree not found: {tree_id}")
        return TreeRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def list(self) -> list[dict[str, object]]:
        if not self.trees_dir.exists():
            return []
        items: list[dict[str, object]] = []
        for path in sorted(self.trees_dir.glob("*.json")):
            data = json.loads(path.read_text(encoding="utf-8"))
            items.append(
                {
                    "tree_id": data["tree_id"],
                    "title": data["title"],
                    "source_path": data.get("source_path"),
                    "updated_at": data["updated_at"],
                    "node_count": data.get("node_count", 1),
                }
            )
        return items

    def delete(self, tree_id: str) -> bool:
        path = self.tree_path(tree_id)
        if not path.exists():
            return False
        path.unlink()
        return True
