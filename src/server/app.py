from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from mimetypes import guess_type
from pathlib import Path
from tempfile import NamedTemporaryFile
from urllib.parse import quote
from uuid import uuid4

import yaml
from fastapi import FastAPI, File, Form, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from src.builder import (
    BuildOptions,
    RuleBasedSectionRefiner,
    build_tree_from_file,
    build_tree_from_text,
)
from src.builder.parse import parse_file_text
from src.chat import build_pagent_events
from src.chat.pagent import event_to_ndjson
from src.chat.session import (
    ChatSessionService,
    InMemoryChatSessionStorage,
    JsonChatSessionStorage,
    SqliteChatSessionStorage,
)
from src.config import ChatSettings, get_settings
from src.logging_config import configure_treefyit_logging
from src.model.forest import Forest
from src.model.tree import Tree
from src.query.query import (
    NodeQueryHit,
    TreeQueryHit,
    TreeIndex,
    InMemoryForestQuery,
    build_tree_index,
    content_to_search_text,
    score_nodes_bm25,
)
from src.server.build_tasks import (
    BuildTask,
    BuildTaskExecutionError,
    BuildTaskManager,
)
from src.store import RegistryStore

configure_treefyit_logging()
logger = logging.getLogger("treefyit.server")

LOCAL_DEV_ORIGIN_REGEX = r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$"


class BuildTreeRequest(BaseModel):
    text: str = Field(min_length=1)
    filename: str = Field(default="document.md", min_length=1)
    summarize: bool = False
    refine_split_threshold: int | None = Field(default=None, ge=1)
    refine_max_parts: int | None = Field(default=None, ge=1)


class RootNodePreview(BaseModel):
    path: str
    title: str
    summary: str | None = None
    children_count: int = Field(ge=0)


class TreeBuildResponse(BaseModel):
    tree_id: str
    filename: str
    title: str
    node_count: int = Field(ge=1)
    max_depth: int = Field(ge=0)
    root_count: int = Field(ge=0)
    roots: list[RootNodePreview] = Field(default_factory=list)


class FileBuildForm(BaseModel):
    filename: str = Field(min_length=1)
    summarize: bool = False
    refine_split_threshold: int | None = Field(default=None, ge=1)
    refine_max_parts: int | None = Field(default=None, ge=1)


class TreeIndexMetaResponse(BaseModel):
    tree_id: str
    tree_title: str
    document_count: int = Field(ge=0)
    average_document_length: float = Field(ge=0.0)
    term_count: int = Field(ge=0)
    tree_document_length: int = Field(ge=0)


class TreeSummaryResponse(BaseModel):
    tree_id: str
    title: str
    node_count: int = Field(ge=1)
    max_depth: int = Field(ge=0)
    root_count: int = Field(ge=0)


class TreeOverviewResponse(BaseModel):
    tree_id: str
    title: str
    node_count: int = Field(ge=1)
    max_depth: int = Field(ge=0)
    root_count: int = Field(ge=0)
    roots: list[RootNodePreview] = Field(default_factory=list)


class NodeDetailResponse(BaseModel):
    tree_id: str
    path: str
    title: str
    summary: str | None = None
    text: str = ""
    children_count: int = Field(ge=0)
    children: list[str] = Field(default_factory=list)


class NodeChildrenResponse(BaseModel):
    tree_id: str
    path: str
    children: list[RootNodePreview] = Field(default_factory=list)


class ForestSummaryResponse(BaseModel):
    forest_id: str
    tree_count: int = Field(ge=0)
    trees: list[TreeSummaryResponse] = Field(default_factory=list)


class ChatRequest(BaseModel):
    bid: str | None = None
    tree_id: str | None = None
    question: str | None = Field(default=None, min_length=1)
    session_id: str | None = None


def build_root_previews(tree: Tree) -> list[RootNodePreview]:
    previews: list[RootNodePreview] = []
    for index, child in enumerate(tree.children):
        previews.append(
            RootNodePreview(
                path=str(index),
                title=child.title,
                summary=child.summary,
                children_count=len(child.children),
            )
        )
    return previews


def build_tree_response(tree: Tree, *, filename: str) -> TreeBuildResponse:
    return TreeBuildResponse(
        tree_id=tree.node_id,
        filename=filename,
        title=tree.title,
        node_count=tree.subtree_size or 1,
        max_depth=max_tree_depth(tree),
        root_count=len(tree.children),
        roots=build_root_previews(tree),
    )


def max_tree_depth(tree: Tree) -> int:
    if not tree.children:
        return tree.depth or 0
    return max(max_tree_depth(child) for child in tree.children)


def build_tree_with_text(request: BuildTreeRequest) -> tuple[Tree, str]:
    section_refiner = RuleBasedSectionRefiner(
        split_threshold=request.refine_split_threshold,
        max_parts=request.refine_max_parts,
    )
    tree = build_tree_from_text(
        request.text,
        filename=request.filename,
        options=BuildOptions(summarize=request.summarize),
        section_refiner=section_refiner,
    )
    return tree, request.filename


def build_tree_with_file(file_bytes: bytes, form: FileBuildForm) -> tuple[Tree, str]:
    section_refiner = RuleBasedSectionRefiner(
        split_threshold=form.refine_split_threshold,
        max_parts=form.refine_max_parts,
    )
    suffix = Path(form.filename).suffix or ".txt"
    with NamedTemporaryFile(suffix=suffix, delete=True) as temp_file:
        temp_file.write(file_bytes)
        temp_file.flush()
        tree = build_tree_from_file(
            temp_file.name,
            options=BuildOptions(summarize=form.summarize),
            section_refiner=section_refiner,
        )
    tree.title = form.filename
    return tree, form.filename


def now_iso() -> str:
    return datetime.now(UTC).isoformat()


def upload_log_fields(
    *,
    endpoint: str,
    request_id: str,
    filename: str,
    content_type: str | None,
    file_size: int,
) -> dict[str, str | int | None]:
    return {
        "endpoint": endpoint,
        "request_id": request_id,
        "filename": filename,
        "content_type": content_type or "application/octet-stream",
        "file_size": file_size,
    }


def build_content_disposition(filename: str, *, disposition: str = "inline") -> str:
    normalized = Path(filename).name or "download"
    stem = Path(normalized).stem
    suffix = Path(normalized).suffix
    fallback_stem = "".join(
        char
        for char in stem
        if ord(char) < 128 and char not in {'"', "\\", ";", "\r", "\n"}
    ).strip(" .")
    fallback_suffix = "".join(
        char
        for char in suffix
        if ord(char) < 128 and char not in {'"', "\\", ";", "\r", "\n"}
    )
    fallback = f"{fallback_stem or 'download'}{fallback_suffix}"
    encoded = quote(normalized, safe="")
    return f"{disposition}; filename=\"{fallback}\"; filename*=UTF-8''{encoded}"


def build_file_sha256(data: bytes) -> str:
    return sha256(data).hexdigest()


def file_parsed_text_preview(
    data: bytes,
    *,
    filename: str,
    content_type: str | None,
) -> str | None:
    media_type = (content_type or guess_type(filename)[0] or "").lower()
    suffix = Path(filename).suffix.lower()
    text_suffixes = {
        ".css",
        ".csv",
        ".htm",
        ".html",
        ".json",
        ".md",
        ".mdx",
        ".svg",
        ".text",
        ".toml",
        ".tsv",
        ".txt",
        ".xml",
        ".yaml",
        ".yml",
    }
    if not (
        media_type.startswith("text/")
        or media_type in {"application/json", "application/xml", "application/x-yaml"}
        or suffix in text_suffixes
    ):
        return None
    with NamedTemporaryFile(suffix=suffix or ".txt", delete=True) as temp_file:
        temp_file.write(data)
        temp_file.flush()
        return parse_file_text(Path(temp_file.name))


def build_sha256_index(
    build_history: dict[str, dict],
    tree_registry: dict[str, Tree],
) -> dict[str, str]:
    index: dict[str, str] = {}
    for build in sorted(
        build_history.values(),
        key=lambda item: str(item.get("created_at", "")),
    ):
        build_id = str(build.get("id") or build.get("tree_id") or "")
        file_sha256 = str(build.get("sha256") or "").strip()
        if not build_id or not file_sha256:
            continue
        if build_id not in tree_registry:
            continue
        index[file_sha256] = build_id
    return index


def register_tree(app: FastAPI, tree: Tree) -> str:
    tree_id = uuid4().hex
    tree.node_id = tree_id
    app.state.tree_registry[tree_id] = tree
    store = get_registry_store(app)
    if store is not None:
        store.save_tree(tree)
    sync_default_forest(app)
    return tree_id


def finalize_tree_build(
    app: FastAPI,
    tree: Tree,
    *,
    filename: str,
    raw_text: str | None = None,
    parsed_text: str | None = None,
    original_file: bytes | None = None,
    content_type: str | None = None,
    file_sha256: str | None = None,
) -> TreeBuildResponse:
    tree_id = register_tree(app, tree)
    original_meta = save_original_file(
        app,
        tree_id,
        filename=filename,
        data=original_file,
        content_type=content_type,
    )
    response = build_tree_response(tree, filename=filename)
    build = build_history_record(
        response,
        raw_text=raw_text,
        parsed_text=parsed_text,
        original_meta=original_meta,
        file_sha256=file_sha256,
    )
    app.state.build_history[tree_id] = build
    if file_sha256:
        app.state.build_sha256_index[file_sha256] = tree_id
    store = get_registry_store(app)
    if store is not None:
        store.save_build(build)
    logger.info(
        "build persisted tree_id=%s filename=%s has_original=%s storage_key=%s sha256=%s",
        tree_id,
        filename,
        original_meta is not None,
        build.get("storage_key"),
        file_sha256,
    )
    return response


def save_original_file(
    app: FastAPI,
    build_id: str,
    *,
    filename: str,
    data: bytes | None,
    content_type: str | None,
) -> dict | None:
    if data is None:
        return None

    meta = {
        "filename": filename,
        "size": len(data),
        "content_type": content_type
        or guess_type(filename)[0]
        or "application/octet-stream",
    }
    app.state.original_registry[build_id] = {
        **meta,
        "data": data,
    }
    store = get_registry_store(app)
    if store is None:
        return meta

    stored = store.save_original(build_id, filename, data)
    return {
        **meta,
        "storage_key": stored["storage_key"],
    }


def build_history_record(
    response: TreeBuildResponse,
    *,
    raw_text: str | None,
    parsed_text: str | None,
    original_meta: dict | None,
    file_sha256: str | None,
) -> dict:
    stats = {
        "node_count": response.node_count,
        "max_depth": response.max_depth,
        "root_count": response.root_count,
    }
    build = {
        "id": response.tree_id,
        "tree_id": response.tree_id,
        "filename": response.filename,
        "title": response.title,
        "created_at": now_iso(),
        "stats": stats,
        "cached": False,
        "error": None,
        "has_original_file": original_meta is not None,
        "original_file_url": (
            f"/api/build/{response.tree_id}/file" if original_meta is not None else None
        ),
        "storage_key": (original_meta or {}).get("storage_key"),
        "content_type": (original_meta or {}).get("content_type"),
        "file_size": (original_meta or {}).get("size"),
        "sha256": file_sha256,
    }
    if raw_text is not None:
        build["raw_text"] = raw_text
    if parsed_text is not None:
        build["parsed_text"] = parsed_text
    return build


def build_compat_response(
    response: TreeBuildResponse,
    tree: Tree,
    *,
    raw_text: str | None = None,
    parsed_text: str | None = None,
    error: str | None = None,
) -> dict:
    payload = response.model_dump()
    payload.update(
        {
            "id": response.tree_id,
            "bid": response.tree_id,
            "error": error,
            "tree": [
                child.model_dump(mode="json", exclude_none=True)
                for child in tree.children
            ],
            "stats": {
                "node_count": response.node_count,
                "max_depth": response.max_depth,
                "root_count": response.root_count,
            },
            "cached": False,
        }
    )
    if raw_text is not None:
        payload["raw_text"] = raw_text
    if parsed_text is not None:
        payload["parsed_text"] = parsed_text
    return payload


def get_registered_tree(app: FastAPI, tree_id: str) -> Tree:
    tree = app.state.tree_registry.get(tree_id)
    if tree is None:
        raise HTTPException(status_code=404, detail=f"unknown tree_id: {tree_id}")
    return tree


def get_registry_store(app: FastAPI) -> RegistryStore | None:
    return app.state.registry_store


def get_build_record(app: FastAPI, build_id: str) -> dict:
    build = app.state.build_history.get(build_id)
    if build is None:
        raise HTTPException(status_code=404, detail=f"unknown build id: {build_id}")
    return build


def get_deduplicated_build(app: FastAPI, file_sha256: str) -> tuple[dict, Tree] | None:
    build_id = app.state.build_sha256_index.get(file_sha256)
    if not build_id:
        return None
    build = app.state.build_history.get(build_id)
    tree = app.state.tree_registry.get(build_id)
    if build is None or tree is None:
        return None
    return build, tree


def remove_build_sha256_index(app: FastAPI, build_id: str) -> None:
    build = app.state.build_history.get(build_id)
    if build is None:
        return
    file_sha256 = str(build.get("sha256") or "").strip()
    if not file_sha256:
        return
    if app.state.build_sha256_index.get(file_sha256) != build_id:
        return
    for candidate_id, candidate in app.state.build_history.items():
        if candidate_id == build_id:
            continue
        if str(candidate.get("sha256") or "").strip() != file_sha256:
            continue
        if candidate_id not in app.state.tree_registry:
            continue
        app.state.build_sha256_index[file_sha256] = candidate_id
        return
    app.state.build_sha256_index.pop(file_sha256, None)


def build_cached_tree_response(build: dict, tree: Tree) -> TreeBuildResponse:
    return build_tree_response(tree, filename=str(build.get("filename") or tree.title))


def build_cached_compat_response(build: dict, tree: Tree) -> dict:
    response = build_tree_response(
        tree, filename=str(build.get("filename") or tree.title)
    )
    payload = build_compat_response(
        response,
        tree,
        raw_text=build.get("raw_text"),
        parsed_text=build.get("parsed_text"),
        error=build.get("error"),
    )
    payload["cached"] = True
    payload["sha256"] = build.get("sha256")
    payload["content_type"] = build.get("content_type")
    payload["file_size"] = build.get("file_size")
    payload["original_file_url"] = build.get("original_file_url")
    payload["has_original_file"] = build.get("has_original_file")
    payload["storage_key"] = build.get("storage_key")
    return payload


def build_default_forest(app: FastAPI) -> Forest:
    trees = list(app.state.tree_registry.values())
    trees.sort(key=lambda tree: tree.node_id)
    return Forest(forest_id="default", trees=trees)


def sync_default_forest(app: FastAPI) -> None:
    app.state.forest_registry["default"] = build_default_forest(app)


def build_index_meta_response(index: TreeIndex) -> TreeIndexMetaResponse:
    return TreeIndexMetaResponse(
        tree_id=index.tree_id,
        tree_title=index.tree_title,
        document_count=index.corpus.document_count,
        average_document_length=index.corpus.average_document_length,
        term_count=len(index.postings),
        tree_document_length=index.tree_document_length,
    )


def build_tree_summary(tree: Tree) -> TreeSummaryResponse:
    return TreeSummaryResponse(
        tree_id=tree.node_id,
        title=tree.title,
        node_count=tree.subtree_size or 1,
        max_depth=max_tree_depth(tree),
        root_count=len(tree.children),
    )


def build_forest_summary(forest: Forest) -> ForestSummaryResponse:
    return ForestSummaryResponse(
        forest_id=forest.forest_id,
        tree_count=forest.tree_count,
        trees=[build_tree_summary(tree) for tree in forest.trees],
    )


def get_default_forest(app: FastAPI) -> Forest:
    forest = app.state.forest_registry.get("default")
    if forest is None:
        sync_default_forest(app)
        forest = app.state.forest_registry["default"]
    return forest


def build_tree_overview(tree: Tree) -> TreeOverviewResponse:
    summary = build_tree_summary(tree)
    return TreeOverviewResponse(
        tree_id=summary.tree_id,
        title=summary.title,
        node_count=summary.node_count,
        max_depth=summary.max_depth,
        root_count=summary.root_count,
        roots=build_root_previews(tree),
    )


def get_registered_index(app: FastAPI, tree_id: str) -> TreeIndex:
    get_registered_tree(app, tree_id)
    index = app.state.index_registry.get(tree_id)
    if index is None:
        raise HTTPException(
            status_code=404, detail=f"index not found for tree_id: {tree_id}"
        )
    return index


def resolve_tree_path(tree: Tree, path: str) -> Tree:
    if path == "root":
        return tree

    current = tree
    for segment in path.split("."):
        if not segment.isdigit():
            raise HTTPException(status_code=404, detail=f"invalid path: {path}")
        index = int(segment)
        if index < 0 or index >= len(current.children):
            raise HTTPException(status_code=404, detail=f"invalid path: {path}")
        current = current.children[index]
    return current


def build_child_path(parent_path: str, index: int) -> str:
    if parent_path == "root":
        return str(index)
    return f"{parent_path}.{index}"


def build_node_detail(tree_id: str, path: str, node: Tree) -> NodeDetailResponse:
    child_paths = [
        build_child_path(path, index) for index, _ in enumerate(node.children)
    ]
    return NodeDetailResponse(
        tree_id=tree_id,
        path=path,
        title=node.title,
        summary=node.summary,
        text=content_to_search_text(node.content),
        children_count=len(node.children),
        children=child_paths,
    )


def build_node_children(tree_id: str, path: str, node: Tree) -> NodeChildrenResponse:
    children = [
        RootNodePreview(
            path=build_child_path(path, index),
            title=child.title,
            summary=child.summary,
            children_count=len(child.children),
        )
        for index, child in enumerate(node.children)
    ]
    return NodeChildrenResponse(tree_id=tree_id, path=path, children=children)


def log_query(app: FastAPI, tool: str, tree_id: str, path: str, result: object) -> None:
    query = {
        "id": uuid4().hex,
        "tool": tool,
        "tree_id": tree_id,
        "path": path,
        "created_at": now_iso(),
        "result_size": len(result) if isinstance(result, list) else 1,
    }
    app.state.query_history.insert(0, query)
    app.state.query_history = app.state.query_history[:200]
    store = get_registry_store(app)
    if store is not None:
        store.append_query(query)


def create_chat_session_service(
    store: RegistryStore | None,
    settings: ChatSettings,
) -> ChatSessionService:
    if settings.session_backend == "memory" or store is None:
        return ChatSessionService(InMemoryChatSessionStorage())
    if settings.session_backend == "sqlite":
        database_path = settings.session_sqlite_path or store.data_dir / "chat.sqlite3"
        return ChatSessionService(SqliteChatSessionStorage(database_path))
    return ChatSessionService(JsonChatSessionStorage(store.data_dir / "sessions"))


def build_chat_answer(tree: Tree, question: str, hits: list[NodeQueryHit]) -> str:
    if not hits:
        return f"未在 `{tree.title}` 中找到和问题相关的节点。"

    lines = [f"在 `{tree.title}` 中找到 {len(hits)} 个相关节点："]
    for index, hit in enumerate(hits[:5], start=1):
        lines.append(f"{index}. {hit.title} ({hit.path}): {hit.snippet}")
    lines.append(f"问题：{question}")
    return "\n".join(lines)


def create_app(*, store: RegistryStore | None = None) -> FastAPI:
    settings = get_settings()
    app = FastAPI(title="treefyit")
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=LOCAL_DEV_ORIGIN_REGEX,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.state.registry_store = store
    app.state.tree_registry = store.load_trees() if store is not None else {}
    app.state.index_registry = (
        store.load_indexes(tree_ids=set(app.state.tree_registry))
        if store is not None
        else {}
    )
    app.state.build_history = store.load_builds() if store is not None else {}
    app.state.build_sha256_index = build_sha256_index(
        app.state.build_history,
        app.state.tree_registry,
    )
    app.state.query_history = store.load_queries() if store is not None else []
    app.state.chat_sessions = create_chat_session_service(store, settings.chat)
    app.state.original_registry = {}
    app.state.forest_registry = {"default": Forest(forest_id="default", trees=[])}
    sync_default_forest(app)

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {
            "status": "ok",
            "service": "treefyit",
        }

    @app.get("/openapi.yaml", include_in_schema=False)
    async def openapi_yaml() -> Response:
        return Response(
            content=yaml.safe_dump(app.openapi(), sort_keys=False, allow_unicode=True),
            media_type="application/x-yaml",
        )

    @app.post("/api/trees", response_model=TreeBuildResponse)
    async def create_tree(request: BuildTreeRequest) -> TreeBuildResponse:
        endpoint = "/api/trees"
        request_id = uuid4().hex[:8]
        started_at = time.perf_counter()
        logger.info(
            "build received endpoint=%s request_id=%s filename=%s text_chars=%s summarize=%s refine_split_threshold=%s refine_max_parts=%s",
            endpoint,
            request_id,
            request.filename,
            len(request.text),
            request.summarize,
            request.refine_split_threshold,
            request.refine_max_parts,
        )
        try:
            build_started_at = time.perf_counter()
            tree, filename = build_tree_with_text(request)
            build_ms = (time.perf_counter() - build_started_at) * 1000
            finalize_started_at = time.perf_counter()
            response = finalize_tree_build(
                app,
                tree,
                filename=filename,
                raw_text=request.text,
                parsed_text=request.text,
            )
            finalize_ms = (time.perf_counter() - finalize_started_at) * 1000
            total_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "build completed endpoint=%s request_id=%s filename=%s tree_id=%s node_count=%s build_ms=%.2f finalize_ms=%.2f total_ms=%.2f",
                endpoint,
                request_id,
                filename,
                response.tree_id,
                response.node_count,
                build_ms,
                finalize_ms,
                total_ms,
            )
            return response
        except Exception:
            total_ms = (time.perf_counter() - started_at) * 1000
            logger.exception(
                "build failed endpoint=%s request_id=%s filename=%s total_ms=%.2f",
                endpoint,
                request_id,
                request.filename,
                total_ms,
            )
            raise

    @app.post("/api/trees/from-file", response_model=TreeBuildResponse)
    async def create_tree_from_file(
        file: UploadFile = File(...),
        summarize: bool = Form(False),
        refine_split_threshold: int | None = Form(default=None),
        refine_max_parts: int | None = Form(default=None),
    ) -> TreeBuildResponse:
        endpoint = "/api/trees/from-file"
        request_id = uuid4().hex[:8]
        filename = file.filename or "document.txt"
        content_type = file.content_type
        started_at = time.perf_counter()
        read_started_at = time.perf_counter()

        try:
            file_bytes = await file.read()
            file_sha256 = build_file_sha256(file_bytes)
            read_ms = (time.perf_counter() - read_started_at) * 1000
            log_fields = upload_log_fields(
                endpoint=endpoint,
                request_id=request_id,
                filename=filename,
                content_type=content_type,
                file_size=len(file_bytes),
            )
            logger.info(
                "upload received endpoint=%s request_id=%s filename=%s content_type=%s file_size=%s summarize=%s refine_split_threshold=%s refine_max_parts=%s read_ms=%.2f",
                log_fields["endpoint"],
                log_fields["request_id"],
                log_fields["filename"],
                log_fields["content_type"],
                log_fields["file_size"],
                summarize,
                refine_split_threshold,
                refine_max_parts,
                read_ms,
            )
            deduplicated = get_deduplicated_build(app, file_sha256)
            if deduplicated is not None:
                build, tree = deduplicated
                total_ms = (time.perf_counter() - started_at) * 1000
                logger.info(
                    "upload deduplicated endpoint=%s request_id=%s filename=%s tree_id=%s file_sha256=%s total_ms=%.2f",
                    endpoint,
                    request_id,
                    filename,
                    build["id"],
                    file_sha256,
                    total_ms,
                )
                return build_cached_tree_response(build, tree)
            form = FileBuildForm(
                filename=filename,
                summarize=summarize,
                refine_split_threshold=refine_split_threshold,
                refine_max_parts=refine_max_parts,
            )
            build_started_at = time.perf_counter()
            tree, built_filename = build_tree_with_file(file_bytes, form)
            build_ms = (time.perf_counter() - build_started_at) * 1000
            finalize_started_at = time.perf_counter()
            response = finalize_tree_build(
                app,
                tree,
                filename=built_filename,
                parsed_text=file_parsed_text_preview(
                    file_bytes,
                    filename=built_filename,
                    content_type=content_type,
                ),
                original_file=file_bytes,
                content_type=content_type,
                file_sha256=file_sha256,
            )
            finalize_ms = (time.perf_counter() - finalize_started_at) * 1000
            total_ms = (time.perf_counter() - started_at) * 1000
            logger.info(
                "upload completed endpoint=%s request_id=%s filename=%s tree_id=%s node_count=%s read_ms=%.2f build_ms=%.2f finalize_ms=%.2f total_ms=%.2f",
                endpoint,
                request_id,
                built_filename,
                response.tree_id,
                response.node_count,
                read_ms,
                build_ms,
                finalize_ms,
                total_ms,
            )
            return response
        except Exception:
            total_ms = (time.perf_counter() - started_at) * 1000
            logger.exception(
                "upload failed endpoint=%s request_id=%s filename=%s content_type=%s total_ms=%.2f",
                endpoint,
                request_id,
                filename,
                content_type or "application/octet-stream",
                total_ms,
            )
            raise

    @app.post("/api/build")
    async def build_compat(
        request: Request,
        file: UploadFile | None = File(default=None),
        summarize: bool = Form(False),
        mode: str = Form("md"),
        refine_split_threshold: int | None = Form(default=None),
        refine_max_parts: int | None = Form(default=None),
    ) -> dict:
        content_type = request.headers.get("content-type", "")
        if "application/json" in content_type:
            endpoint = "/api/build"
            request_id = uuid4().hex[:8]
            started_at = time.perf_counter()
            payload = await request.json()
            build_request = BuildTreeRequest.model_validate(payload)
            logger.info(
                "build received endpoint=%s request_id=%s filename=%s text_chars=%s summarize=%s refine_split_threshold=%s refine_max_parts=%s compat_mode=json",
                endpoint,
                request_id,
                build_request.filename,
                len(build_request.text),
                build_request.summarize,
                build_request.refine_split_threshold,
                build_request.refine_max_parts,
            )
            try:
                build_started_at = time.perf_counter()
                tree, filename = build_tree_with_text(build_request)
                build_ms = (time.perf_counter() - build_started_at) * 1000
                finalize_started_at = time.perf_counter()
                response = finalize_tree_build(
                    app,
                    tree,
                    filename=filename,
                    raw_text=build_request.text,
                    parsed_text=build_request.text,
                )
                finalize_ms = (time.perf_counter() - finalize_started_at) * 1000
                total_ms = (time.perf_counter() - started_at) * 1000
                logger.info(
                    "build completed endpoint=%s request_id=%s filename=%s tree_id=%s node_count=%s build_ms=%.2f finalize_ms=%.2f total_ms=%.2f compat_mode=json",
                    endpoint,
                    request_id,
                    filename,
                    response.tree_id,
                    response.node_count,
                    build_ms,
                    finalize_ms,
                    total_ms,
                )
                return build_compat_response(
                    response,
                    tree,
                    raw_text=build_request.text,
                    parsed_text=build_request.text,
                )
            except Exception:
                total_ms = (time.perf_counter() - started_at) * 1000
                logger.exception(
                    "build failed endpoint=%s request_id=%s filename=%s total_ms=%.2f compat_mode=json",
                    endpoint,
                    request_id,
                    build_request.filename,
                    total_ms,
                )
                raise

        if "multipart/form-data" in content_type:
            if file is None:
                raise HTTPException(status_code=400, detail="file is required")
            endpoint = "/api/build"
            request_id = uuid4().hex[:8]
            filename = file.filename or "document.txt"
            content_type_header = file.content_type
            started_at = time.perf_counter()
            read_started_at = time.perf_counter()
            try:
                file_bytes = await file.read()
                file_sha256 = build_file_sha256(file_bytes)
                read_ms = (time.perf_counter() - read_started_at) * 1000
                logger.info(
                    "upload received endpoint=%s request_id=%s filename=%s content_type=%s file_size=%s summarize=%s mode=%s refine_split_threshold=%s refine_max_parts=%s read_ms=%.2f",
                    endpoint,
                    request_id,
                    filename,
                    content_type_header or "application/octet-stream",
                    len(file_bytes),
                    summarize,
                    mode,
                    refine_split_threshold,
                    refine_max_parts,
                    read_ms,
                )
                if Path(filename).suffix.lower() == ".pdf" and mode == "md":
                    total_ms = (time.perf_counter() - started_at) * 1000
                    logger.warning(
                        "upload rejected endpoint=%s request_id=%s filename=%s content_type=%s file_size=%s reason=%s total_ms=%.2f",
                        endpoint,
                        request_id,
                        filename,
                        content_type_header or "application/octet-stream",
                        len(file_bytes),
                        "unsupported file type for md mode: .pdf",
                        total_ms,
                    )
                    return {
                        "id": uuid4().hex,
                        "bid": None,
                        "filename": filename,
                        "error": "unsupported file type for md mode: .pdf",
                        "tree": [],
                        "stats": {},
                        "cached": False,
                    }
                deduplicated = get_deduplicated_build(app, file_sha256)
                if deduplicated is not None:
                    build, tree = deduplicated
                    total_ms = (time.perf_counter() - started_at) * 1000
                    logger.info(
                        "upload deduplicated endpoint=%s request_id=%s filename=%s tree_id=%s file_sha256=%s total_ms=%.2f",
                        endpoint,
                        request_id,
                        filename,
                        build["id"],
                        file_sha256,
                        total_ms,
                    )
                    return build_cached_compat_response(build, tree)
                form = FileBuildForm(
                    filename=filename,
                    summarize=summarize,
                    refine_split_threshold=refine_split_threshold,
                    refine_max_parts=refine_max_parts,
                )
                build_started_at = time.perf_counter()
                tree, built_filename = build_tree_with_file(file_bytes, form)
                build_ms = (time.perf_counter() - build_started_at) * 1000
                parsed_text = file_parsed_text_preview(
                    file_bytes,
                    filename=built_filename,
                    content_type=content_type_header,
                )
                finalize_started_at = time.perf_counter()
                response = finalize_tree_build(
                    app,
                    tree,
                    filename=built_filename,
                    parsed_text=parsed_text,
                    original_file=file_bytes,
                    content_type=content_type_header,
                    file_sha256=file_sha256,
                )
                finalize_ms = (time.perf_counter() - finalize_started_at) * 1000
                total_ms = (time.perf_counter() - started_at) * 1000
                logger.info(
                    "upload completed endpoint=%s request_id=%s filename=%s tree_id=%s node_count=%s read_ms=%.2f build_ms=%.2f finalize_ms=%.2f total_ms=%.2f",
                    endpoint,
                    request_id,
                    built_filename,
                    response.tree_id,
                    response.node_count,
                    read_ms,
                    build_ms,
                    finalize_ms,
                    total_ms,
                )
                return build_compat_response(
                    response,
                    tree,
                    parsed_text=parsed_text,
                )
            except Exception:
                total_ms = (time.perf_counter() - started_at) * 1000
                logger.exception(
                    "upload failed endpoint=%s request_id=%s filename=%s content_type=%s total_ms=%.2f",
                    endpoint,
                    request_id,
                    filename,
                    content_type_header or "application/octet-stream",
                    total_ms,
                )
                raise

        raise HTTPException(status_code=415, detail="unsupported content type")

    @app.post("/api/build/stream")
    async def build_stream(
        file: UploadFile = File(...),
        summarize: bool = Form(False),
        refine_split_threshold: int | None = Form(default=None),
        refine_max_parts: int | None = Form(default=None),
    ) -> StreamingResponse:
        endpoint = "/api/build/stream"
        request_id = uuid4().hex[:8]
        filename = file.filename or "document.txt"
        content_type = file.content_type
        started_at = time.perf_counter()
        read_started_at = time.perf_counter()
        file_bytes = await file.read()
        file_sha256 = build_file_sha256(file_bytes)
        read_ms = (time.perf_counter() - read_started_at) * 1000
        logger.info(
            "upload received endpoint=%s request_id=%s filename=%s content_type=%s file_size=%s summarize=%s refine_split_threshold=%s refine_max_parts=%s read_ms=%.2f",
            endpoint,
            request_id,
            filename,
            content_type or "application/octet-stream",
            len(file_bytes),
            summarize,
            refine_split_threshold,
            refine_max_parts,
            read_ms,
        )

        async def events():
            stream_started_at = time.time()
            trace_seq = 0

            def line(payload: dict) -> str:
                nonlocal trace_seq
                trace_seq += 1
                payload.setdefault("request_id", request_id)
                payload.setdefault("trace_seq", trace_seq)
                payload.setdefault("timestamp", datetime.now(UTC).isoformat())
                payload.setdefault(
                    "elapsed_sec", round(time.time() - stream_started_at, 3)
                )
                return json.dumps(payload, ensure_ascii=False) + "\n"

            async def yield_task_events(
                task: asyncio.Task[dict],
                queue: asyncio.Queue[dict],
            ):
                while not task.done() or not queue.empty():
                    try:
                        payload = await asyncio.wait_for(queue.get(), timeout=0.1)
                    except TimeoutError:
                        continue
                    yield line(payload)

            yield line(
                {
                    "type": "start",
                    "stage": "start",
                    "filename": filename,
                    "file_size": len(file_bytes),
                }
            )
            try:
                deduplicated = get_deduplicated_build(app, file_sha256)
                if deduplicated is not None:
                    build, _tree = deduplicated
                    total_ms = (time.perf_counter() - started_at) * 1000
                    logger.info(
                        "upload deduplicated endpoint=%s request_id=%s filename=%s tree_id=%s file_sha256=%s total_ms=%.2f",
                        endpoint,
                        request_id,
                        filename,
                        build["id"],
                        file_sha256,
                        total_ms,
                    )
                    result = dict(build)
                    result["cached"] = True
                    yield line(
                        {
                            "type": "done",
                            "stage": "done",
                            "bid": build["id"],
                            "cached": True,
                            "result": result,
                        }
                    )
                    return

                task_queue: asyncio.Queue[dict] = asyncio.Queue()

                async def emit_task_event(payload: dict) -> None:
                    event_type = str(payload.get("type") or "")
                    stage = str(payload.get("stage") or "build")
                    message = str(payload.get("description") or stage)
                    await task_queue.put(
                        {
                            **payload,
                            "type": "progress"
                            if event_type in {"task_start", "task_done"}
                            else event_type,
                            "stage": stage,
                            "message": message,
                        }
                    )

                def prepare_form(_context: dict) -> FileBuildForm:
                    return FileBuildForm(
                        filename=filename,
                        summarize=summarize,
                        refine_split_threshold=refine_split_threshold,
                        refine_max_parts=refine_max_parts,
                    )

                def run_builder(context: dict) -> tuple[Tree, str]:
                    return build_tree_with_file(file_bytes, context["prepare"])

                def run_finalize(context: dict) -> TreeBuildResponse:
                    tree, built_filename = context["build"]
                    return finalize_tree_build(
                        app,
                        tree,
                        filename=built_filename,
                        parsed_text=file_parsed_text_preview(
                            file_bytes,
                            filename=built_filename,
                            content_type=content_type,
                        ),
                        original_file=file_bytes,
                        content_type=content_type,
                        file_sha256=file_sha256,
                    )

                tasks = [
                    BuildTask(
                        "prepare",
                        prepare_form,
                        description="准备构建参数",
                    ),
                    BuildTask(
                        "build",
                        run_builder,
                        depends_on=("prepare",),
                        description="解析文档并生成知识树",
                    ),
                    BuildTask(
                        "finalize",
                        run_finalize,
                        depends_on=("build",),
                        description="保存构建结果和原始文件",
                    ),
                ]
                yield line(
                    {
                        "type": "task_plan",
                        "stage": "task_plan",
                        "tasks": [
                            {
                                "task": task.name,
                                "stage": task.name,
                                "description": task.description,
                                "depends_on": list(task.depends_on),
                                "status": "pending",
                            }
                            for task in tasks
                        ],
                    }
                )

                manager = BuildTaskManager(logger=logger)
                managed_build = asyncio.create_task(
                    manager.run(tasks, on_event=emit_task_event)
                )
                async for task_event in yield_task_events(managed_build, task_queue):
                    yield task_event

                context = await managed_build
                result = context["finalize"]
                _tree, built_filename = context["build"]
                total_ms = (time.perf_counter() - started_at) * 1000
                logger.info(
                    "upload completed endpoint=%s request_id=%s filename=%s tree_id=%s node_count=%s read_ms=%.2f total_ms=%.2f",
                    endpoint,
                    request_id,
                    built_filename,
                    result.tree_id,
                    result.node_count,
                    read_ms,
                    total_ms,
                )
                yield line(
                    {
                        "type": "done",
                        "stage": "done",
                        "bid": result.tree_id,
                        "cached": False,
                        "result": get_build_record(app, result.tree_id),
                    }
                )
            except BuildTaskExecutionError as exc:
                total_ms = (time.perf_counter() - started_at) * 1000
                logger.exception(
                    "upload task failed endpoint=%s request_id=%s filename=%s task=%s content_type=%s total_ms=%.2f",
                    endpoint,
                    request_id,
                    filename,
                    exc.task_name,
                    content_type or "application/octet-stream",
                    total_ms,
                )
                yield line(
                    {
                        "type": "error",
                        "stage": "error",
                        "message": str(exc),
                    }
                )
            except Exception as exc:
                total_ms = (time.perf_counter() - started_at) * 1000
                logger.exception(
                    "upload failed endpoint=%s request_id=%s filename=%s content_type=%s total_ms=%.2f",
                    endpoint,
                    request_id,
                    filename,
                    content_type or "application/octet-stream",
                    total_ms,
                )
                yield line(
                    {
                        "type": "error",
                        "stage": "error",
                        "message": str(exc),
                    }
                )

        return StreamingResponse(events(), media_type="application/x-ndjson")

    @app.get("/api/history")
    async def list_history() -> list[dict]:
        builds = list(app.state.build_history.values())
        builds.sort(key=lambda build: build["created_at"], reverse=True)
        return [
            {
                key: value
                for key, value in build.items()
                if key not in {"raw_text", "parsed_text"}
            }
            for build in builds
        ]

    @app.get("/api/build/{bid}")
    async def get_build(bid: str) -> dict:
        build = dict(get_build_record(app, bid))
        tree = get_registered_tree(app, bid)
        build["tree"] = tree.model_dump(mode="json", exclude_none=True)
        return build

    @app.get("/api/build/{bid}/file")
    async def get_build_file(bid: str) -> Response:
        build = get_build_record(app, bid)
        if not build.get("has_original_file"):
            raise HTTPException(
                status_code=404, detail="original file is not available"
            )

        original = app.state.original_registry.get(bid)
        if original is not None:
            data = original["data"]
            filename = original["filename"]
            content_type = original["content_type"]
        else:
            storage_key = build.get("storage_key")
            if not storage_key:
                raise HTTPException(
                    status_code=404, detail="original file is not available"
                )
            store = get_registry_store(app)
            if store is None:
                raise HTTPException(
                    status_code=404, detail="original file is not available"
                )
            path = store.original_path(storage_key)
            if not path.exists():
                raise HTTPException(
                    status_code=404, detail="original file is not available"
                )
            data = path.read_bytes()
            filename = build["filename"]
            content_type = build.get("content_type") or "application/octet-stream"

        return Response(
            content=data,
            media_type=content_type,
            headers={"Content-Disposition": build_content_disposition(filename)},
        )

    @app.delete("/api/build/{bid}")
    async def delete_build(bid: str) -> dict[str, object]:
        if bid not in app.state.build_history and bid not in app.state.tree_registry:
            raise HTTPException(status_code=404, detail=f"unknown build id: {bid}")
        remove_build_sha256_index(app, bid)
        app.state.build_history.pop(bid, None)
        app.state.original_registry.pop(bid, None)
        app.state.tree_registry.pop(bid, None)
        app.state.index_registry.pop(bid, None)
        store = get_registry_store(app)
        if store is not None:
            store.delete_bundle(bid)
        sync_default_forest(app)
        return {"ok": True, "id": bid}

    @app.post("/api/trees/{tree_id}/index", response_model=TreeIndexMetaResponse)
    @app.post("/api/tree/{tree_id}/index", response_model=TreeIndexMetaResponse)
    async def create_tree_index(tree_id: str) -> TreeIndexMetaResponse:
        tree = get_registered_tree(app, tree_id)
        index = build_tree_index(tree)
        app.state.index_registry[tree_id] = index
        store = get_registry_store(app)
        if store is not None:
            store.save_index(index)
        return build_index_meta_response(index)

    @app.get("/api/trees/{tree_id}/index/meta", response_model=TreeIndexMetaResponse)
    @app.get("/api/tree/{tree_id}/index/meta", response_model=TreeIndexMetaResponse)
    async def get_tree_index_meta(tree_id: str) -> TreeIndexMetaResponse:
        index = get_registered_index(app, tree_id)
        return build_index_meta_response(index)

    @app.get("/api/trees/{tree_id}/search/nodes", response_model=list[NodeQueryHit])
    @app.get("/api/tree/{tree_id}/search/nodes", response_model=list[NodeQueryHit])
    async def search_tree_nodes(
        tree_id: str,
        q: str,
        limit: int = 8,
    ) -> list[NodeQueryHit]:
        index = get_registered_index(app, tree_id)
        hits = score_nodes_bm25(index, q, limit=limit)
        log_query(app, "search_nodes", tree_id, q, hits)
        return hits

    @app.get("/api/trees", response_model=list[TreeSummaryResponse])
    async def list_trees() -> list[TreeSummaryResponse]:
        trees = list(app.state.tree_registry.values())
        trees.sort(key=lambda tree: tree.node_id)
        return [build_tree_summary(tree) for tree in trees]

    @app.get("/api/forest", response_model=ForestSummaryResponse)
    async def get_forest() -> ForestSummaryResponse:
        return build_forest_summary(get_default_forest(app))

    @app.get("/api/forest/search/trees", response_model=list[TreeQueryHit])
    async def search_forest_trees(
        q: str,
        limit: int = 5,
    ) -> list[TreeQueryHit]:
        query = InMemoryForestQuery(get_default_forest(app))
        hits = query.find_trees(q, limit=limit)
        log_query(app, "forest_search_trees", "default", q, hits)
        return hits

    @app.get("/api/forest/search/nodes", response_model=list[NodeQueryHit])
    async def search_forest_nodes(
        q: str,
        limit: int = 8,
    ) -> list[NodeQueryHit]:
        query = InMemoryForestQuery(get_default_forest(app))
        hits = query.find_nodes(q, limit=limit)
        log_query(app, "forest_search_nodes", "default", q, hits)
        return hits

    @app.get("/api/forest/search")
    async def search_forest(q: str, limit: int = 8) -> dict[str, object]:
        if not q.strip():
            return {
                "trees": {"error": "empty query"},
                "sections": {"error": "empty query"},
            }
        query = InMemoryForestQuery(get_default_forest(app))
        trees = query.find_trees(q, limit=min(limit, 20))
        nodes = query.find_nodes(q, limit=min(limit, 20))
        log_query(app, "forest_search", "default", q, [*trees, *nodes])
        return {"trees": trees, "sections": nodes}

    @app.get("/api/trees/{tree_id}", response_model=TreeOverviewResponse)
    @app.get("/api/tree/{tree_id}", response_model=TreeOverviewResponse)
    async def get_tree_overview(tree_id: str) -> TreeOverviewResponse:
        tree = get_registered_tree(app, tree_id)
        overview = build_tree_overview(tree)
        log_query(app, "overview", tree_id, "", overview)
        return overview

    @app.get(
        "/api/trees/{tree_id}/nodes/{path:path}", response_model=NodeDetailResponse
    )
    @app.get("/api/tree/{tree_id}/nodes/{path:path}", response_model=NodeDetailResponse)
    async def get_tree_node(tree_id: str, path: str) -> NodeDetailResponse:
        tree = get_registered_tree(app, tree_id)
        node = resolve_tree_path(tree, path)
        detail = build_node_detail(tree_id, path, node)
        log_query(app, "inspect", tree_id, path, detail)
        return detail

    @app.get(
        "/api/trees/{tree_id}/children/{path:path}",
        response_model=NodeChildrenResponse,
    )
    @app.get(
        "/api/tree/{tree_id}/children/{path:path}",
        response_model=NodeChildrenResponse,
    )
    async def get_tree_children(tree_id: str, path: str) -> NodeChildrenResponse:
        tree = get_registered_tree(app, tree_id)
        node = resolve_tree_path(tree, path)
        children = build_node_children(tree_id, path, node)
        log_query(app, "get_children", tree_id, path, children)
        return children

    @app.delete("/api/trees/{tree_id}")
    @app.delete("/api/tree/{tree_id}")
    async def delete_tree(tree_id: str) -> dict[str, object]:
        tree = app.state.tree_registry.pop(tree_id, None)
        if tree is None:
            raise HTTPException(status_code=404, detail=f"unknown tree_id: {tree_id}")
        app.state.index_registry.pop(tree_id, None)
        store = get_registry_store(app)
        if store is not None:
            store.delete_bundle(tree_id)
        sync_default_forest(app)
        return {
            "ok": True,
            "tree_id": tree_id,
        }

    @app.get("/api/queries")
    async def list_queries() -> list[dict]:
        return app.state.query_history[:200]

    @app.get("/api/queries/stats")
    async def query_stats() -> dict:
        items = app.state.query_history[:200]
        tools = Counter(item["tool"] for item in items)
        trees = Counter(item["tree_id"] for item in items)
        return {
            "total": len(items),
            "by_tool": dict(tools.most_common()),
            "by_tree": dict(trees.most_common()),
            "recent": items[:20],
        }

    @app.post("/api/chat")
    async def chat(payload: ChatRequest) -> StreamingResponse:
        tree_id = payload.tree_id or payload.bid
        if not payload.question:
            raise HTTPException(status_code=400, detail="question is required")
        question = payload.question

        async def events():
            async for event in build_pagent_events(
                app,
                tree_id=tree_id,
                question=question,
                session_id=payload.session_id,
            ):
                yield event_to_ndjson(event)

        return StreamingResponse(events(), media_type="application/x-ndjson")

    @app.get("/api/sessions")
    async def list_sessions(limit: int = 100) -> dict:
        sessions = app.state.chat_sessions.list(limit=limit)
        return {"sessions": [session.summary() for session in sessions]}

    @app.get("/api/sessions/{sid}/turns")
    async def session_turns(sid: str, limit: int = 200) -> dict:
        turns = app.state.chat_sessions.get_turns(sid, limit=limit)
        if turns is None:
            raise HTTPException(status_code=404, detail=f"unknown session_id: {sid}")
        return {"session_id": sid, "turns": [turn.model_dump() for turn in turns]}

    @app.delete("/api/sessions/{sid}")
    async def delete_session(sid: str) -> dict:
        deleted = app.state.chat_sessions.delete(sid)
        return {"deleted": deleted, "session_id": sid}

    return app


app = create_app(store=RegistryStore(get_settings().store.data_dir))


__all__ = [
    "BuildTreeRequest",
    "FileBuildForm",
    "ForestSummaryResponse",
    "NodeChildrenResponse",
    "NodeDetailResponse",
    "RootNodePreview",
    "TreeBuildResponse",
    "TreeIndexMetaResponse",
    "TreeOverviewResponse",
    "TreeSummaryResponse",
    "app",
    "build_child_path",
    "create_chat_session_service",
    "build_default_forest",
    "build_forest_summary",
    "build_index_meta_response",
    "build_node_children",
    "build_node_detail",
    "build_tree_with_file",
    "build_tree_with_text",
    "build_root_previews",
    "build_tree_response",
    "build_tree_overview",
    "build_tree_summary",
    "create_app",
    "finalize_tree_build",
    "get_default_forest",
    "get_registry_store",
    "get_registered_index",
    "get_registered_tree",
    "max_tree_depth",
    "register_tree",
    "resolve_tree_path",
    "sync_default_forest",
]
