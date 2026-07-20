"""Agent-facing TreefyIt tool client.

This module provides a tiny, stable tool surface for agents that need to
build and inspect TreefyIt knowledge bases over HTTP.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

DEFAULT_BASE_URL = "http://127.0.0.1:8765"
BASE_URL_ENV = "TREEFYIT_BASE_URL"


class TreefyitToolError(RuntimeError):
    pass


def normalize_base_url(base_url: str) -> str:
    return base_url.rstrip("/")


def resolve_base_url(base_url: str | None) -> str:
    explicit = (base_url or "").strip()
    if explicit:
        return normalize_base_url(explicit)

    from_env = os.getenv(BASE_URL_ENV, "").strip()
    if from_env:
        return normalize_base_url(from_env)

    return DEFAULT_BASE_URL


def parse_json_response(response: requests.Response) -> Any:
    try:
        return response.json()
    except ValueError as exc:
        raise TreefyitToolError(
            f"TreefyIt returned non-JSON response: {response.text[:400]}"
        ) from exc


def ensure_success(response: requests.Response) -> Any:
    if response.ok:
        return parse_json_response(response)
    payload = response.text[:400]
    raise TreefyitToolError(
        f"TreefyIt request failed: {response.status_code} {response.reason}; body={payload}"
    )


@dataclass
class TreefyitClient:
    base_url: str | None = None
    timeout_sec: float = 60.0

    def __post_init__(self) -> None:
        self.base_url = resolve_base_url(self.base_url)

    def build_knowledge(
        self, file: str | Path, *, summarize: bool = True
    ) -> dict[str, Any]:
        file_path = Path(file)
        if not file_path.exists():
            raise FileNotFoundError(file_path)
        if not file_path.is_file():
            raise TreefyitToolError(f"Expected a file path, got: {file_path}")

        with file_path.open("rb") as handle:
            response = requests.post(
                f"{self.base_url}/api/build",
                files={"file": (file_path.name, handle)},
                data={"summarize": str(summarize).lower()},
                timeout=self.timeout_sec,
            )
        return ensure_success(response)

    def overview_forest(self) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/api/forest",
            timeout=self.timeout_sec,
        )
        return ensure_success(response)

    def search_trees(self, query: str, *, limit: int = 5) -> list[dict[str, Any]]:
        if not query.strip():
            return []
        response = requests.get(
            f"{self.base_url}/api/forest/search/trees",
            params={"q": query, "limit": limit},
            timeout=self.timeout_sec,
        )
        return ensure_success(response)

    def overview(self, tree_id: str) -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/api/trees/{tree_id}",
            timeout=self.timeout_sec,
        )
        return ensure_success(response)

    def children(self, tree_id: str, node_id: str = "0") -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/api/trees/{tree_id}/children/{node_id}",
            timeout=self.timeout_sec,
        )
        return ensure_success(response)

    def inspect(self, tree_id: str, node_id: str = "0") -> dict[str, Any]:
        response = requests.get(
            f"{self.base_url}/api/trees/{tree_id}/nodes/{node_id}",
            timeout=self.timeout_sec,
        )
        return ensure_success(response)


default_client = TreefyitClient()


def build_knowledge(file: str | Path, *, summarize: bool = True) -> dict[str, Any]:
    return default_client.build_knowledge(file, summarize=summarize)


def overview_forest() -> dict[str, Any]:
    return default_client.overview_forest()


def search_trees(query: str, *, limit: int = 5) -> list[dict[str, Any]]:
    return default_client.search_trees(query, limit=limit)


def overview(tree_id: str) -> dict[str, Any]:
    return default_client.overview(tree_id)


def children(tree_id: str, node_id: str = "0") -> dict[str, Any]:
    return default_client.children(tree_id, node_id=node_id)


def inspect(tree_id: str, node_id: str = "0") -> dict[str, Any]:
    return default_client.inspect(tree_id, node_id=node_id)


__all__ = [
    "TreefyitClient",
    "TreefyitToolError",
    "build_knowledge",
    "overview_forest",
    "search_trees",
    "overview",
    "children",
    "inspect",
]
