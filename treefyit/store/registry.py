from __future__ import annotations

import json
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from treefyit.model.tree import Tree
from treefyit.query.query import TreeIndex


class RegistryStore:
    def __init__(self, data_dir: str | Path) -> None:
        self.data_dir = Path(data_dir)

    @property
    def trees_dir(self) -> Path:
        return self.data_dir / "trees"

    @property
    def indexes_dir(self) -> Path:
        return self.data_dir / "indexes"

    @property
    def builds_dir(self) -> Path:
        return self.data_dir / "builds"

    @property
    def cache_dir(self) -> Path:
        return self.data_dir / "cache"

    @property
    def originals_dir(self) -> Path:
        return self.data_dir / "originals"

    @property
    def queries_path(self) -> Path:
        return self.data_dir / "queries.jsonl"

    def tree_path(self, tree_id: str) -> Path:
        return self.trees_dir / f"{tree_id}.json"

    def index_path(self, tree_id: str) -> Path:
        return self.indexes_dir / f"{tree_id}.json"

    def build_path(self, build_id: str) -> Path:
        return self.builds_dir / f"{build_id}.json"

    def ensure_dirs(self) -> None:
        self.trees_dir.mkdir(parents=True, exist_ok=True)
        self.indexes_dir.mkdir(parents=True, exist_ok=True)
        self.builds_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.originals_dir.mkdir(parents=True, exist_ok=True)

    def save_tree(self, tree: Tree) -> Path:
        self.ensure_dirs()
        path = self.tree_path(tree.node_id)
        write_text_atomically(path, tree.model_dump_json(indent=2, exclude_none=True))
        return path

    def save_index(self, index: TreeIndex) -> Path:
        self.ensure_dirs()
        path = self.index_path(index.tree_id)
        write_text_atomically(path, index.model_dump_json(indent=2, exclude_none=True))
        return path

    def cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def cache_key_for(self, text: str, model: str, mode: str, summarize: bool) -> str:
        digest = sha256()
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
        digest.update(model.encode("utf-8"))
        digest.update(b"\0")
        digest.update(mode.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(summarize).encode("utf-8"))
        return digest.hexdigest()

    def cache_put(self, key: str, payload: dict) -> Path:
        self.ensure_dirs()
        path = self.cache_path(key)
        write_json_atomically(path, payload)
        return path

    def cache_get(self, key: str) -> dict | None:
        path = self.cache_path(key)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def save_build(
        self,
        build_or_id: dict | str,
        result: dict | None = None,
        cache_key: str | None = None,
    ) -> Path:
        self.ensure_dirs()
        build = dict(result or build_or_id)
        if isinstance(build_or_id, str):
            build["id"] = build_or_id
        if cache_key is not None:
            build["cache_key"] = cache_key
        build_id = str(build["id"])
        path = self.build_path(build_id)
        write_json_atomically(path, build)
        return path

    def load_builds(self) -> dict[str, dict]:
        if not self.builds_dir.exists():
            return {}

        builds: dict[str, dict] = {}
        for path in sorted(self.builds_dir.glob("*.json")):
            build = json.loads(path.read_text(encoding="utf-8"))
            builds[str(build["id"])] = build
        return builds

    def load_build(self, build_id: str) -> dict | None:
        path = self.build_path(build_id)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def list_builds(self) -> list[dict]:
        builds = list(self.load_builds().values())
        builds.sort(key=lambda build: str(build.get("created_at", "")), reverse=True)
        return builds

    def delete_build(self, build_id: str) -> None:
        path = self.build_path(build_id)
        if path.exists():
            path.unlink()

    def save_original(self, build_id: str, filename: str, data: bytes) -> dict:
        self.ensure_dirs()
        original_dir = self.originals_dir / build_id
        original_dir.mkdir(parents=True, exist_ok=True)
        safe_name = Path(filename).name or "document.bin"
        path = original_dir / safe_name
        path.write_bytes(data)
        return {
            "storage_key": f"{build_id}/{safe_name}",
            "filename": safe_name,
            "size": len(data),
            "path": str(path),
        }

    def original_path(self, storage_key: str) -> Path:
        return self.originals_dir / storage_key

    def delete_originals(self, build_id: str) -> None:
        original_dir = self.originals_dir / build_id
        if not original_dir.exists():
            return
        for path in original_dir.iterdir():
            if path.is_file():
                path.unlink()
        original_dir.rmdir()

    def append_query(self, query: dict) -> None:
        self.ensure_dirs()
        with self.queries_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(query, ensure_ascii=False) + "\n")

    def load_queries(self, limit: int = 200) -> list[dict]:
        if not self.queries_path.exists():
            return []
        lines = self.queries_path.read_text(encoding="utf-8").splitlines()
        queries = [json.loads(line) for line in lines if line.strip()]
        return list(reversed(queries[-limit:]))

    def load_trees(self) -> dict[str, Tree]:
        if not self.trees_dir.exists():
            return {}

        trees: dict[str, Tree] = {}
        for path in sorted(self.trees_dir.glob("*.json")):
            tree = Tree.model_validate_json(path.read_text(encoding="utf-8"))
            trees[tree.node_id] = tree
        return trees

    def load_indexes(self, *, tree_ids: set[str] | None = None) -> dict[str, TreeIndex]:
        if not self.indexes_dir.exists():
            return {}

        indexes: dict[str, TreeIndex] = {}
        for path in sorted(self.indexes_dir.glob("*.json")):
            index = TreeIndex.model_validate_json(path.read_text(encoding="utf-8"))
            if tree_ids is not None and index.tree_id not in tree_ids:
                continue
            indexes[index.tree_id] = index
        return indexes

    def delete_tree(self, tree_id: str) -> None:
        path = self.tree_path(tree_id)
        if path.exists():
            path.unlink()

    def delete_index(self, tree_id: str) -> None:
        path = self.index_path(tree_id)
        if path.exists():
            path.unlink()

    def delete_bundle(self, tree_id: str) -> None:
        self.delete_tree(tree_id)
        self.delete_index(tree_id)
        self.delete_build(tree_id)
        self.delete_originals(tree_id)


def write_json_atomically(path: Path, data: dict) -> None:
    write_text_atomically(path, json.dumps(data, ensure_ascii=False, indent=2))


def write_text_atomically(path: Path, text: str) -> None:
    tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    tmp_path.write_text(text, encoding="utf-8")
    tmp_path.replace(path)


__all__ = ["RegistryStore"]
