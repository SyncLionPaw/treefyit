use std::collections::HashMap;
use std::net::SocketAddr;
use std::sync::{Arc, Mutex};
use std::time::{SystemTime, UNIX_EPOCH};

use anyhow::Result;
use axum::body::{Body, Bytes, to_bytes};
use axum::extract::{FromRequest, Multipart, Path, Query, Request, State};
use axum::http::{StatusCode, header};
use axum::response::{IntoResponse, Response};
use axum::routing::{delete, get, post};
use axum::{Json, Router};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use uuid::Uuid;

use crate::builder::{BuildOptions, RuleBasedSectionRefiner, build_tree_from_text_with_refiner};
use crate::config::get_settings;
use crate::llm::summarize_tree;
use crate::model::{Forest, Tree};
use crate::query::{
    InMemoryForestQuery, NodeQueryHit, TreeIndex, TreeQueryHit, build_tree_index,
    content_to_search_text, score_nodes_bm25,
};
use crate::store::RegistryStore;

pub type SharedState = Arc<Mutex<AppState>>;

#[derive(Debug)]
pub struct AppState {
    pub tree_registry: HashMap<String, Tree>,
    pub index_registry: HashMap<String, TreeIndex>,
    pub forest_registry: HashMap<String, Forest>,
    pub build_history: HashMap<String, BuildRecord>,
    pub original_registry: HashMap<String, OriginalFile>,
    pub query_history: Vec<QueryLogItem>,
    pub sessions: HashMap<String, ChatSession>,
    pub store: Option<RegistryStore>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildTreeRequest {
    pub text: String,
    #[serde(default = "default_filename")]
    pub filename: String,
    #[serde(default)]
    pub summarize: bool,
    pub refine_split_threshold: Option<usize>,
    pub refine_max_parts: Option<usize>,
}

#[derive(Debug, Clone)]
struct MultipartBuildInput {
    request: BuildTreeRequest,
    original_file: Bytes,
    content_type: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RootNodePreview {
    pub path: String,
    pub title: String,
    pub summary: Option<String>,
    pub children_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeBuildResponse {
    pub tree_id: String,
    pub filename: String,
    pub title: String,
    pub node_count: usize,
    pub max_depth: usize,
    pub root_count: usize,
    pub roots: Vec<RootNodePreview>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LegacyBuildResponse {
    pub tree_id: String,
    pub filename: String,
    pub title: String,
    pub node_count: usize,
    pub max_depth: usize,
    pub root_count: usize,
    pub roots: Vec<RootNodePreview>,
    pub id: String,
    pub bid: String,
    pub error: Option<String>,
    pub tree: Vec<Tree>,
    pub stats: BuildStats,
    pub cached: bool,
    pub raw_text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildStats {
    pub node_count: usize,
    pub max_depth: usize,
    pub root_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeSummaryResponse {
    pub tree_id: String,
    pub title: String,
    pub node_count: usize,
    pub max_depth: usize,
    pub root_count: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeOverviewResponse {
    pub tree_id: String,
    pub title: String,
    pub node_count: usize,
    pub max_depth: usize,
    pub root_count: usize,
    pub roots: Vec<RootNodePreview>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TreeIndexMetaResponse {
    pub tree_id: String,
    pub tree_title: String,
    pub document_count: usize,
    pub average_document_length: f64,
    pub term_count: usize,
    pub tree_document_length: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeDetailResponse {
    pub tree_id: String,
    pub path: String,
    pub title: String,
    pub summary: Option<String>,
    pub text: String,
    pub children_count: usize,
    pub children: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct NodeChildrenResponse {
    pub tree_id: String,
    pub path: String,
    pub children: Vec<RootNodePreview>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ForestSummaryResponse {
    pub forest_id: String,
    pub tree_count: usize,
    pub trees: Vec<TreeSummaryResponse>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BuildRecord {
    pub id: String,
    pub tree_id: String,
    pub filename: String,
    pub title: String,
    pub created_at: String,
    pub stats: BuildStats,
    pub cached: bool,
    pub error: Option<String>,
    pub has_original_file: bool,
    pub original_file_url: Option<String>,
    pub storage_key: Option<String>,
    pub content_type: Option<String>,
    pub file_size: Option<usize>,
    pub raw_text: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OriginalFile {
    pub filename: String,
    pub content_type: String,
    pub data: Vec<u8>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryLogItem {
    pub id: String,
    pub tool: String,
    pub tree_id: String,
    pub path: String,
    pub created_at: String,
    pub result_size: usize,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatRequest {
    pub bid: Option<String>,
    pub tree_id: Option<String>,
    pub question: Option<String>,
    pub model: Option<String>,
    pub session_id: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSession {
    pub session_id: String,
    pub tree_id: String,
    pub model: Option<String>,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
    pub turns: Vec<ChatTurn>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatTurn {
    pub role: String,
    pub content: String,
    pub created_at: String,
}

#[derive(Debug, Clone)]
struct AgentChatResult {
    events: Vec<Value>,
    answer: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteBuildResponse {
    pub ok: bool,
    pub id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionListResponse {
    pub sessions: Vec<ChatSessionSummary>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatSessionSummary {
    pub session_id: String,
    pub tree_id: String,
    pub model: Option<String>,
    pub title: String,
    pub created_at: String,
    pub updated_at: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionTurnsResponse {
    pub session_id: String,
    pub turns: Vec<ChatTurn>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteSessionResponse {
    pub deleted: bool,
    pub session_id: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QueryStatsResponse {
    pub total: usize,
    pub by_tool: HashMap<String, usize>,
    pub by_tree: HashMap<String, usize>,
    pub recent: Vec<QueryLogItem>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DeleteTreeResponse {
    pub ok: bool,
    pub tree_id: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SearchParams {
    pub q: String,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SessionListParams {
    pub bid: Option<String>,
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SessionTurnsParams {
    pub limit: Option<usize>,
}

#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub service: String,
}

#[derive(Debug, Clone, Serialize)]
pub struct ErrorDetail {
    pub detail: String,
}

#[derive(Debug, thiserror::Error)]
pub enum ApiError {
    #[error("{0}")]
    NotFound(String),
    #[error("{0}")]
    BadRequest(String),
    #[error("{0}")]
    Internal(String),
}

impl IntoResponse for ApiError {
    fn into_response(self) -> Response {
        let status = match self {
            ApiError::NotFound(_) => StatusCode::NOT_FOUND,
            ApiError::BadRequest(_) => StatusCode::BAD_REQUEST,
            ApiError::Internal(_) => StatusCode::INTERNAL_SERVER_ERROR,
        };
        let detail = self.to_string();
        (status, Json(ErrorDetail { detail })).into_response()
    }
}

pub fn create_app(store: Option<RegistryStore>) -> Result<Router> {
    let tree_registry = if let Some(store) = &store {
        store.load_trees()?
    } else {
        HashMap::new()
    };
    let tree_ids = tree_registry.keys().cloned().collect();
    let index_registry = if let Some(store) = &store {
        store.load_indexes(Some(&tree_ids))?
    } else {
        HashMap::new()
    };
    let build_history = if let Some(store) = &store {
        store
            .load_builds()?
            .into_iter()
            .filter_map(|(id, value)| {
                serde_json::from_value(value)
                    .ok()
                    .map(|record| (id, record))
            })
            .collect()
    } else {
        HashMap::new()
    };
    let query_history = if let Some(store) = &store {
        store
            .load_queries(200)?
            .into_iter()
            .filter_map(|value| serde_json::from_value(value).ok())
            .collect()
    } else {
        Vec::new()
    };
    let sessions = if let Some(store) = &store {
        store
            .load_sessions()?
            .into_iter()
            .filter_map(|(id, value)| {
                serde_json::from_value(value)
                    .ok()
                    .map(|session| (id, session))
            })
            .collect()
    } else {
        HashMap::new()
    };
    let state = Arc::new(Mutex::new(AppState {
        tree_registry,
        index_registry,
        forest_registry: HashMap::new(),
        build_history,
        original_registry: HashMap::new(),
        query_history,
        sessions,
        store,
    }));
    sync_default_forest(&state)?;

    Ok(Router::new()
        .route("/health", get(health))
        .route("/api/trees", get(list_trees).post(create_tree))
        .route("/api/trees/from-file", post(create_tree_from_file))
        .route("/api/build", post(legacy_build))
        .route("/api/build/stream", post(build_stream))
        .route("/api/history", get(list_history))
        .route("/api/build/{bid}", get(get_build).delete(delete_build))
        .route("/api/build/{bid}/file", get(get_build_file))
        .route(
            "/api/trees/{tree_id}",
            get(get_tree_overview).delete(delete_tree),
        )
        .route(
            "/api/tree/{tree_id}",
            get(get_tree_overview).delete(delete_tree),
        )
        .route("/api/trees/{tree_id}/index", post(create_tree_index))
        .route("/api/tree/{tree_id}/index", post(create_tree_index))
        .route("/api/trees/{tree_id}/index/meta", get(get_tree_index_meta))
        .route("/api/tree/{tree_id}/index/meta", get(get_tree_index_meta))
        .route("/api/trees/{tree_id}/search/nodes", get(search_tree_nodes))
        .route("/api/tree/{tree_id}/search/nodes", get(search_tree_nodes))
        .route("/api/trees/{tree_id}/nodes/{*path}", get(get_tree_node))
        .route("/api/tree/{tree_id}/nodes/{*path}", get(get_tree_node))
        .route(
            "/api/trees/{tree_id}/children/{*path}",
            get(get_tree_children),
        )
        .route(
            "/api/tree/{tree_id}/children/{*path}",
            get(get_tree_children),
        )
        .route("/api/forest", get(get_forest))
        .route("/api/forest/search/trees", get(search_forest_trees))
        .route("/api/forest/search/nodes", get(search_forest_nodes))
        .route("/api/forest/search", get(search_forest))
        .route("/api/queries", get(list_queries))
        .route("/api/queries/stats", get(query_stats))
        .route("/api/chat", post(chat))
        .route("/api/sessions", get(list_sessions))
        .route("/api/sessions/{sid}/turns", get(session_turns))
        .route("/api/sessions/{sid}", delete(delete_session))
        .with_state(state))
}

pub async fn serve(addr: impl Into<SocketAddr>, store: Option<RegistryStore>) -> Result<()> {
    let listener = tokio::net::TcpListener::bind(addr.into()).await?;
    axum::serve(listener, create_app(store)?).await?;
    Ok(())
}

async fn health() -> Json<HealthResponse> {
    Json(HealthResponse {
        status: "ok".to_string(),
        service: "treefyust".to_string(),
    })
}

async fn create_tree(
    State(state): State<SharedState>,
    Json(request): Json<BuildTreeRequest>,
) -> Result<Json<TreeBuildResponse>, ApiError> {
    let tree = build_tree_request(&request).await?;
    let response = finalize_tree_build(
        &state,
        tree,
        &request.filename,
        Some(request.text.clone()),
        None,
        None,
    )?;
    Ok(Json(response))
}

async fn create_tree_from_file(
    State(state): State<SharedState>,
    multipart: Multipart,
) -> Result<Json<TreeBuildResponse>, ApiError> {
    let input = build_request_from_multipart(multipart).await?;
    let tree = build_tree_request(&input.request).await?;
    Ok(Json(finalize_tree_build(
        &state,
        tree,
        &input.request.filename,
        None,
        Some(input.original_file),
        input.content_type,
    )?))
}

async fn legacy_build(
    State(state): State<SharedState>,
    request: Request,
) -> Result<Json<LegacyBuildResponse>, ApiError> {
    let content_type = request
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("");
    if content_type.starts_with("multipart/form-data") {
        let multipart = Multipart::from_request(request, &())
            .await
            .map_err(|err| ApiError::BadRequest(err.to_string()))?;
        let input = build_request_from_multipart(multipart).await?;
        let tree = build_tree_request(&input.request).await?;
        let response = finalize_tree_build(
            &state,
            tree.clone(),
            &input.request.filename,
            None,
            Some(input.original_file),
            input.content_type,
        )?;
        Ok(Json(build_legacy_response(response, &tree, None)))
    } else {
        let bytes = to_bytes(request.into_body(), usize::MAX)
            .await
            .map_err(|err| ApiError::BadRequest(err.to_string()))?;
        let request = serde_json::from_slice::<BuildTreeRequest>(&bytes)
            .map_err(|err| ApiError::BadRequest(err.to_string()))?;
        let tree = build_tree_request(&request).await?;
        let response = finalize_tree_build(
            &state,
            tree.clone(),
            &request.filename,
            Some(request.text.clone()),
            None,
            None,
        )?;
        Ok(Json(build_legacy_response(
            response,
            &tree,
            Some(request.text),
        )))
    }
}

async fn build_request_from_multipart(
    mut multipart: Multipart,
) -> Result<MultipartBuildInput, ApiError> {
    let mut file_name = "document.md".to_string();
    let mut file_bytes: Option<Bytes> = None;
    let mut content_type: Option<String> = None;
    let mut summarize = false;
    let mut refine_split_threshold: Option<usize> = None;
    let mut refine_max_parts: Option<usize> = None;

    while let Some(field) = multipart
        .next_field()
        .await
        .map_err(|err| ApiError::BadRequest(err.to_string()))?
    {
        let name = field.name().unwrap_or("").to_string();
        if name == "file" {
            file_name = field.file_name().unwrap_or("document.md").to_string();
            content_type = field.content_type().map(ToString::to_string);
            file_bytes = Some(
                field
                    .bytes()
                    .await
                    .map_err(|err| ApiError::BadRequest(err.to_string()))?,
            );
            continue;
        }
        if name == "summarize" {
            let value = field
                .text()
                .await
                .map_err(|err| ApiError::BadRequest(err.to_string()))?;
            summarize = value == "true";
            continue;
        }
        if name == "refine_split_threshold" {
            let value = field
                .text()
                .await
                .map_err(|err| ApiError::BadRequest(err.to_string()))?;
            refine_split_threshold =
                parse_optional_usize_form_field("refine_split_threshold", &value)?;
            continue;
        }
        if name == "refine_max_parts" {
            let value = field
                .text()
                .await
                .map_err(|err| ApiError::BadRequest(err.to_string()))?;
            refine_max_parts = parse_optional_usize_form_field("refine_max_parts", &value)?;
        }
    }

    let bytes = file_bytes.ok_or_else(|| ApiError::BadRequest("file is required".to_string()))?;
    let text = String::from_utf8(bytes.to_vec())
        .map_err(|err| ApiError::BadRequest(format!("file is not valid utf-8: {err}")))?;
    let request = BuildTreeRequest {
        text,
        filename: file_name,
        summarize,
        refine_split_threshold,
        refine_max_parts,
    };
    Ok(MultipartBuildInput {
        request,
        original_file: bytes,
        content_type,
    })
}

fn parse_optional_usize_form_field(name: &str, value: &str) -> Result<Option<usize>, ApiError> {
    let value = value.trim();
    if value.is_empty() {
        return Ok(None);
    }
    let parsed = value
        .parse::<usize>()
        .map_err(|_| ApiError::BadRequest(format!("{name} must be a positive integer")))?;
    if parsed == 0 {
        return Err(ApiError::BadRequest(format!(
            "{name} must be a positive integer"
        )));
    }
    Ok(Some(parsed))
}

async fn list_trees(
    State(state): State<SharedState>,
) -> Result<Json<Vec<TreeSummaryResponse>>, ApiError> {
    let state = lock_state(&state)?;
    let mut trees: Vec<&Tree> = state.tree_registry.values().collect();
    trees.sort_by(|left, right| left.node_id.cmp(&right.node_id));
    Ok(Json(trees.into_iter().map(build_tree_summary).collect()))
}

async fn create_tree_index(
    State(state): State<SharedState>,
    Path(tree_id): Path<String>,
) -> Result<Json<TreeIndexMetaResponse>, ApiError> {
    let tree = {
        let state = lock_state(&state)?;
        state
            .tree_registry
            .get(&tree_id)
            .cloned()
            .ok_or_else(|| ApiError::NotFound(format!("unknown tree_id: {tree_id}")))?
    };
    let index = build_tree_index(&tree);
    let response = build_index_meta_response(&index);
    let mut state = lock_state(&state)?;
    if let Some(store) = &state.store {
        store
            .save_index(&index)
            .map_err(|err| ApiError::Internal(err.to_string()))?;
    }
    state.index_registry.insert(tree_id, index);
    Ok(Json(response))
}

async fn get_tree_index_meta(
    State(state): State<SharedState>,
    Path(tree_id): Path<String>,
) -> Result<Json<TreeIndexMetaResponse>, ApiError> {
    let state = lock_state(&state)?;
    let index = get_registered_index(&state, &tree_id)?;
    Ok(Json(build_index_meta_response(index)))
}

async fn search_tree_nodes(
    State(state): State<SharedState>,
    Path(tree_id): Path<String>,
    Query(params): Query<SearchParams>,
) -> Result<Json<Vec<NodeQueryHit>>, ApiError> {
    let index = {
        let state_guard = lock_state(&state)?;
        get_registered_index(&state_guard, &tree_id)?.clone()
    };
    let hits = score_nodes_bm25(&index, &params.q, params.limit.unwrap_or(8));
    log_query(&state, "search_nodes", &tree_id, &params.q, hits.len())?;
    Ok(Json(hits))
}

async fn get_tree_overview(
    State(state): State<SharedState>,
    Path(tree_id): Path<String>,
) -> Result<Json<TreeOverviewResponse>, ApiError> {
    let overview = {
        let state_guard = lock_state(&state)?;
        let tree = get_registered_tree(&state_guard, &tree_id)?;
        build_tree_overview(tree)
    };
    log_query(&state, "overview", &tree_id, "", 1)?;
    Ok(Json(overview))
}

async fn get_tree_node(
    State(state): State<SharedState>,
    Path((tree_id, path)): Path<(String, String)>,
) -> Result<Json<NodeDetailResponse>, ApiError> {
    let detail = {
        let state_guard = lock_state(&state)?;
        let tree = get_registered_tree(&state_guard, &tree_id)?;
        let node = resolve_tree_path(tree, &path)?;
        build_node_detail(&tree_id, &path, node)
    };
    log_query(&state, "inspect", &tree_id, &path, 1)?;
    Ok(Json(detail))
}

async fn get_tree_children(
    State(state): State<SharedState>,
    Path((tree_id, path)): Path<(String, String)>,
) -> Result<Json<NodeChildrenResponse>, ApiError> {
    let children = {
        let state_guard = lock_state(&state)?;
        let tree = get_registered_tree(&state_guard, &tree_id)?;
        let node = resolve_tree_path(tree, &path)?;
        build_node_children(&tree_id, &path, node)
    };
    log_query(
        &state,
        "get_children",
        &tree_id,
        &path,
        children.children.len(),
    )?;
    Ok(Json(children))
}

async fn delete_tree(
    State(state): State<SharedState>,
    Path(tree_id): Path<String>,
) -> Result<Json<DeleteTreeResponse>, ApiError> {
    let mut state_guard = lock_state(&state)?;
    if state_guard.tree_registry.remove(&tree_id).is_none() {
        return Err(ApiError::NotFound(format!("unknown tree_id: {tree_id}")));
    }
    state_guard.index_registry.remove(&tree_id);
    state_guard.build_history.remove(&tree_id);
    state_guard.original_registry.remove(&tree_id);
    if let Some(store) = &state_guard.store {
        store
            .delete_bundle(&tree_id)
            .map_err(|err| ApiError::Internal(err.to_string()))?;
    }
    drop(state_guard);
    sync_default_forest(&state)?;
    Ok(Json(DeleteTreeResponse { ok: true, tree_id }))
}

async fn get_forest(
    State(state): State<SharedState>,
) -> Result<Json<ForestSummaryResponse>, ApiError> {
    let state = lock_state(&state)?;
    let forest = state
        .forest_registry
        .get("default")
        .ok_or_else(|| ApiError::Internal("default forest missing".to_string()))?;
    Ok(Json(build_forest_summary(forest)))
}

async fn search_forest_trees(
    State(state): State<SharedState>,
    Query(params): Query<SearchParams>,
) -> Result<Json<Vec<TreeQueryHit>>, ApiError> {
    let forest = {
        let state_guard = lock_state(&state)?;
        state_guard
            .forest_registry
            .get("default")
            .cloned()
            .unwrap_or_else(|| Forest::new("default"))
    };
    let query = InMemoryForestQuery::new(forest);
    let hits = query.find_trees(&params.q, params.limit.unwrap_or(5));
    log_query(
        &state,
        "forest_search_trees",
        "default",
        &params.q,
        hits.len(),
    )?;
    Ok(Json(hits))
}

async fn search_forest_nodes(
    State(state): State<SharedState>,
    Query(params): Query<SearchParams>,
) -> Result<Json<Vec<NodeQueryHit>>, ApiError> {
    let forest = {
        let state_guard = lock_state(&state)?;
        state_guard
            .forest_registry
            .get("default")
            .cloned()
            .unwrap_or_else(|| Forest::new("default"))
    };
    let query = InMemoryForestQuery::new(forest);
    let hits = query.find_nodes(&params.q, params.limit.unwrap_or(8));
    log_query(
        &state,
        "forest_search_nodes",
        "default",
        &params.q,
        hits.len(),
    )?;
    Ok(Json(hits))
}

async fn build_stream(
    State(state): State<SharedState>,
    multipart: Multipart,
) -> Result<Response, ApiError> {
    let input = build_request_from_multipart(multipart).await?;
    let mut events = Vec::new();
    events.push(json!({
        "type": "start",
        "stage": "start",
        "filename": input.request.filename,
        "file_size": input.original_file.len(),
    }));
    events.push(json!({"type": "progress", "stage": "build"}));
    let tree = build_tree_request(&input.request).await?;
    let response = finalize_tree_build(
        &state,
        tree,
        &input.request.filename,
        None,
        Some(input.original_file),
        input.content_type,
    )?;
    events.push(json!({
        "type": "done",
        "stage": "done",
        "bid": response.tree_id,
        "result": response,
    }));
    Ok(ndjson_response(events, "application/x-ndjson"))
}

async fn list_history(
    State(state): State<SharedState>,
) -> Result<Json<Vec<BuildRecord>>, ApiError> {
    let state = lock_state(&state)?;
    let mut builds: Vec<BuildRecord> = state
        .build_history
        .values()
        .cloned()
        .map(|mut build| {
            build.raw_text = None;
            build
        })
        .collect();
    builds.sort_by(|left, right| right.created_at.cmp(&left.created_at));
    Ok(Json(builds))
}

async fn get_build(
    State(state): State<SharedState>,
    Path(bid): Path<String>,
) -> Result<Json<Value>, ApiError> {
    let state = lock_state(&state)?;
    let build = state
        .build_history
        .get(&bid)
        .cloned()
        .ok_or_else(|| ApiError::NotFound(format!("unknown build id: {bid}")))?;
    let tree = state
        .tree_registry
        .get(&bid)
        .ok_or_else(|| ApiError::NotFound(format!("unknown build id: {bid}")))?;
    let mut value =
        serde_json::to_value(build).map_err(|err| ApiError::Internal(err.to_string()))?;
    if let Value::Object(map) = &mut value {
        map.insert(
            "tree".to_string(),
            serde_json::to_value(tree).map_err(|err| ApiError::Internal(err.to_string()))?,
        );
    }
    Ok(Json(value))
}

async fn get_build_file(
    State(state): State<SharedState>,
    Path(bid): Path<String>,
) -> Result<Response, ApiError> {
    let (build, original, store) = {
        let state = lock_state(&state)?;
        let build = state
            .build_history
            .get(&bid)
            .cloned()
            .ok_or_else(|| ApiError::NotFound(format!("unknown build id: {bid}")))?;
        let original = state.original_registry.get(&bid).cloned();
        (build, original, state.store.clone())
    };
    if !build.has_original_file {
        return Err(ApiError::NotFound(
            "original file is not available".to_string(),
        ));
    }
    let original = if let Some(original) = original {
        original
    } else {
        let store = store
            .ok_or_else(|| ApiError::NotFound("original file is not available".to_string()))?;
        let storage_key = build
            .storage_key
            .clone()
            .ok_or_else(|| ApiError::NotFound("original file is not available".to_string()))?;
        let path = store.original_path(&storage_key);
        let data = std::fs::read(path)
            .map_err(|_| ApiError::NotFound("original file is not available".to_string()))?;
        OriginalFile {
            filename: build.filename.clone(),
            content_type: build
                .content_type
                .clone()
                .unwrap_or_else(|| "application/octet-stream".to_string()),
            data,
        }
    };
    Ok((
        [
            (header::CONTENT_TYPE, original.content_type),
            (
                header::CONTENT_DISPOSITION,
                format!("inline; filename=\"{}\"", original.filename),
            ),
        ],
        Body::from(original.data),
    )
        .into_response())
}

async fn delete_build(
    State(state): State<SharedState>,
    Path(bid): Path<String>,
) -> Result<Json<DeleteBuildResponse>, ApiError> {
    let mut state_guard = lock_state(&state)?;
    if !state_guard.build_history.contains_key(&bid)
        && !state_guard.tree_registry.contains_key(&bid)
    {
        return Err(ApiError::NotFound(format!("unknown build id: {bid}")));
    }
    state_guard.build_history.remove(&bid);
    state_guard.original_registry.remove(&bid);
    state_guard.tree_registry.remove(&bid);
    state_guard.index_registry.remove(&bid);
    if let Some(store) = &state_guard.store {
        store
            .delete_bundle(&bid)
            .map_err(|err| ApiError::Internal(err.to_string()))?;
    }
    drop(state_guard);
    sync_default_forest(&state)?;
    Ok(Json(DeleteBuildResponse { ok: true, id: bid }))
}

async fn search_forest(
    State(state): State<SharedState>,
    Query(params): Query<SearchParams>,
) -> Result<Json<Value>, ApiError> {
    if params.q.trim().is_empty() {
        return Ok(Json(json!({
            "trees": {"error": "empty query"},
            "sections": {"error": "empty query"},
        })));
    }
    let forest = {
        let state_guard = lock_state(&state)?;
        state_guard
            .forest_registry
            .get("default")
            .cloned()
            .unwrap_or_else(|| Forest::new("default"))
    };
    let query = InMemoryForestQuery::new(forest);
    let trees = query.find_trees(&params.q, params.limit.unwrap_or(8));
    let nodes = query.find_nodes(&params.q, params.limit.unwrap_or(8));
    log_query(
        &state,
        "forest_search",
        "default",
        &params.q,
        trees.len() + nodes.len(),
    )?;
    Ok(Json(json!({"trees": trees, "sections": nodes})))
}

async fn list_queries(
    State(state): State<SharedState>,
) -> Result<Json<Vec<QueryLogItem>>, ApiError> {
    let state = lock_state(&state)?;
    Ok(Json(
        state.query_history.iter().take(200).cloned().collect(),
    ))
}

async fn query_stats(
    State(state): State<SharedState>,
) -> Result<Json<QueryStatsResponse>, ApiError> {
    let state = lock_state(&state)?;
    let mut by_tool = HashMap::new();
    let mut by_tree = HashMap::new();
    for item in &state.query_history {
        *by_tool.entry(item.tool.clone()).or_insert(0) += 1;
        *by_tree.entry(item.tree_id.clone()).or_insert(0) += 1;
    }
    Ok(Json(QueryStatsResponse {
        total: state.query_history.len(),
        by_tool,
        by_tree,
        recent: state.query_history.iter().take(20).cloned().collect(),
    }))
}

async fn chat(
    State(state): State<SharedState>,
    Json(request): Json<ChatRequest>,
) -> Result<Response, ApiError> {
    let tree_id = request
        .tree_id
        .clone()
        .or_else(|| request.bid.clone())
        .ok_or_else(|| ApiError::BadRequest("bid or tree_id is required".to_string()))?;
    let question = request
        .question
        .clone()
        .filter(|value| !value.trim().is_empty())
        .ok_or_else(|| ApiError::BadRequest("question is required".to_string()))?;
    let tree = {
        let state_guard = lock_state(&state)?;
        get_registered_tree(&state_guard, &tree_id)?.clone()
    };
    let mut result =
        run_agent_chat(&state, &tree, &tree_id, &question, request.model.as_deref()).await?;
    let session = append_chat_turn(
        &state,
        request.session_id,
        tree_id.clone(),
        request.model.clone(),
        question.clone(),
        result.answer.clone(),
    )?;
    result.events.insert(
        0,
        json!({
            "type": "start",
            "session_id": session.session_id,
            "bid": tree_id,
            "tree_id": tree_id,
            "model": request.model,
        }),
    );
    result.events.push(json!({
        "type": "done",
        "answer": result.answer,
        "turns": session.turns.len(),
    }));
    Ok(ndjson_response(result.events, "text/event-stream"))
}

async fn list_sessions(
    State(state): State<SharedState>,
    Query(params): Query<SessionListParams>,
) -> Result<Json<SessionListResponse>, ApiError> {
    let state = lock_state(&state)?;
    let mut sessions: Vec<&ChatSession> = state.sessions.values().collect();
    if let Some(bid) = params.bid {
        sessions.retain(|session| session.tree_id == bid);
    }
    sessions.sort_by(|left, right| right.updated_at.cmp(&left.updated_at));
    let limit = params.limit.unwrap_or(100);
    Ok(Json(SessionListResponse {
        sessions: sessions
            .into_iter()
            .take(limit)
            .map(|session| ChatSessionSummary {
                session_id: session.session_id.clone(),
                tree_id: session.tree_id.clone(),
                model: session.model.clone(),
                title: session.title.clone(),
                created_at: session.created_at.clone(),
                updated_at: session.updated_at.clone(),
            })
            .collect(),
    }))
}

async fn session_turns(
    State(state): State<SharedState>,
    Path(sid): Path<String>,
    Query(params): Query<SessionTurnsParams>,
) -> Result<Json<SessionTurnsResponse>, ApiError> {
    let state = lock_state(&state)?;
    let session = state
        .sessions
        .get(&sid)
        .ok_or_else(|| ApiError::NotFound(format!("unknown session_id: {sid}")))?;
    let limit = params.limit.unwrap_or(200);
    let start = session.turns.len().saturating_sub(limit);
    Ok(Json(SessionTurnsResponse {
        session_id: sid,
        turns: session.turns[start..].to_vec(),
    }))
}

async fn delete_session(
    State(state): State<SharedState>,
    Path(sid): Path<String>,
) -> Result<Json<DeleteSessionResponse>, ApiError> {
    let mut state = lock_state(&state)?;
    let mut deleted = state.sessions.remove(&sid).is_some();
    if let Some(store) = &state.store {
        deleted = store
            .delete_session(&sid)
            .map_err(|err| ApiError::Internal(err.to_string()))?
            || deleted;
    }
    Ok(Json(DeleteSessionResponse {
        deleted,
        session_id: sid,
    }))
}

async fn build_tree_request(request: &BuildTreeRequest) -> Result<Tree, ApiError> {
    let refiner =
        RuleBasedSectionRefiner::new(request.refine_split_threshold, request.refine_max_parts);
    let mut tree = build_tree_from_text_with_refiner(
        &request.text,
        &request.filename,
        BuildOptions { summarize: false },
        refiner,
    );
    if request.summarize {
        summarize_tree(&mut tree)
            .await
            .map_err(|err| ApiError::Internal(err.to_string()))?;
    }
    Ok(tree)
}

fn finalize_tree_build(
    state: &SharedState,
    mut tree: Tree,
    filename: &str,
    raw_text: Option<String>,
    original_file: Option<Bytes>,
    content_type: Option<String>,
) -> Result<TreeBuildResponse, ApiError> {
    let tree_id = Uuid::new_v4().simple().to_string();
    tree.node_id = tree_id.clone();
    let response = build_tree_response(&tree, filename);
    let mut state_guard = lock_state(state)?;
    let original_meta = original_file
        .map(|bytes| {
            save_original_file(
                &mut state_guard,
                &tree_id,
                filename,
                bytes,
                content_type.unwrap_or_else(|| "application/octet-stream".to_string()),
            )
        })
        .transpose()?;
    let build = build_record(&response, raw_text, original_meta);
    if let Some(store) = &state_guard.store {
        store
            .save_tree(&tree)
            .map_err(|err| ApiError::Internal(err.to_string()))?;
        store
            .save_build(&tree_id, &build)
            .map_err(|err| ApiError::Internal(err.to_string()))?;
    }
    state_guard.build_history.insert(tree_id.clone(), build);
    state_guard.tree_registry.insert(tree_id, tree);
    drop(state_guard);
    sync_default_forest(state)?;
    Ok(response)
}

fn save_original_file(
    state: &mut AppState,
    tree_id: &str,
    filename: &str,
    data: Bytes,
    content_type: String,
) -> Result<(String, String, usize), ApiError> {
    let size = data.len();
    state.original_registry.insert(
        tree_id.to_string(),
        OriginalFile {
            filename: filename.to_string(),
            content_type: content_type.clone(),
            data: data.to_vec(),
        },
    );
    if let Some(store) = &state.store {
        let storage_key = store
            .save_original(tree_id, filename, &data)
            .map_err(|err| ApiError::Internal(err.to_string()))?;
        return Ok((storage_key, content_type, size));
    }
    Ok((String::new(), content_type, size))
}

fn build_record(
    response: &TreeBuildResponse,
    raw_text: Option<String>,
    original_meta: Option<(String, String, usize)>,
) -> BuildRecord {
    let stats = BuildStats {
        node_count: response.node_count,
        max_depth: response.max_depth,
        root_count: response.root_count,
    };
    let has_original_file = original_meta.is_some();
    let (storage_key, content_type, file_size) = original_meta
        .map(|(storage_key, content_type, size)| {
            (
                if storage_key.is_empty() {
                    None
                } else {
                    Some(storage_key)
                },
                Some(content_type),
                Some(size),
            )
        })
        .unwrap_or((None, None, None));
    BuildRecord {
        id: response.tree_id.clone(),
        tree_id: response.tree_id.clone(),
        filename: response.filename.clone(),
        title: response.title.clone(),
        created_at: now_string(),
        stats,
        cached: false,
        error: None,
        has_original_file,
        original_file_url: has_original_file
            .then(|| format!("/api/build/{}/file", response.tree_id)),
        storage_key,
        content_type,
        file_size,
        raw_text,
    }
}

fn build_legacy_response(
    response: TreeBuildResponse,
    tree: &Tree,
    raw_text: Option<String>,
) -> LegacyBuildResponse {
    LegacyBuildResponse {
        id: response.tree_id.clone(),
        bid: response.tree_id.clone(),
        error: None,
        tree: tree.children.clone(),
        stats: BuildStats {
            node_count: response.node_count,
            max_depth: response.max_depth,
            root_count: response.root_count,
        },
        cached: false,
        raw_text,
        tree_id: response.tree_id,
        filename: response.filename,
        title: response.title,
        node_count: response.node_count,
        max_depth: response.max_depth,
        root_count: response.root_count,
        roots: response.roots,
    }
}

fn log_query(
    state: &SharedState,
    tool: &str,
    tree_id: &str,
    path: &str,
    result_size: usize,
) -> Result<(), ApiError> {
    let query = QueryLogItem {
        id: Uuid::new_v4().simple().to_string(),
        tool: tool.to_string(),
        tree_id: tree_id.to_string(),
        path: path.to_string(),
        created_at: now_string(),
        result_size,
    };
    let mut state = lock_state(state)?;
    state.query_history.insert(0, query.clone());
    state.query_history.truncate(200);
    if let Some(store) = &state.store {
        store
            .append_query(&query)
            .map_err(|err| ApiError::Internal(err.to_string()))?;
    }
    Ok(())
}

fn ensure_index(state: &SharedState, tree_id: &str) -> Result<(Tree, TreeIndex), ApiError> {
    let mut state = lock_state(state)?;
    let tree = state
        .tree_registry
        .get(tree_id)
        .cloned()
        .ok_or_else(|| ApiError::NotFound(format!("unknown tree_id: {tree_id}")))?;
    if let Some(index) = state.index_registry.get(tree_id) {
        return Ok((tree, index.clone()));
    }
    let index = build_tree_index(&tree);
    if let Some(store) = &state.store {
        store
            .save_index(&index)
            .map_err(|err| ApiError::Internal(err.to_string()))?;
    }
    state
        .index_registry
        .insert(tree_id.to_string(), index.clone());
    Ok((tree, index))
}

async fn run_agent_chat(
    state: &SharedState,
    tree: &Tree,
    tree_id: &str,
    question: &str,
    model: Option<&str>,
) -> Result<AgentChatResult, ApiError> {
    if model == Some("mock") {
        let (_, index) = ensure_index(state, tree_id)?;
        let hits = score_nodes_bm25(&index, question, 5);
        let answer = build_chat_answer(tree, question, &hits);
        return Ok(AgentChatResult {
            events: vec![
                json!({
                    "type": "tool_call",
                    "id": "tc-0",
                    "name": "find_sections",
                    "arguments": json!({"query": question}).to_string(),
                }),
                json!({
                    "type": "tool_result",
                    "id": "tc-0",
                    "name": "find_sections",
                    "ok": true,
                    "content": hits,
                }),
                json!({"type": "text", "text": answer}),
            ],
            answer,
        });
    }

    let settings = get_settings().map_err(|err| ApiError::Internal(err.to_string()))?;
    let llm = settings.llm;
    let model_name = model.unwrap_or(&llm.model);
    let base_url = llm
        .base_url
        .as_deref()
        .unwrap_or("http://127.0.0.1:11434")
        .trim_end_matches('/')
        .to_string();
    let model_id = model_name.strip_prefix("ollama/").unwrap_or(model_name);
    let client = Client::new();
    let mut messages = vec![
        json!({
            "role": "system",
            "content": build_agent_system_prompt(tree, tree_id),
        }),
        json!({
            "role": "user",
            "content": question,
        }),
    ];
    let tools = vec![find_sections_tool_schema()];
    let first = client
        .post(format!("{base_url}/api/chat"))
        .json(&json!({
            "model": model_id,
            "messages": messages,
            "stream": false,
            "tools": tools,
            "options": {"temperature": llm.temperature},
        }))
        .send()
        .await
        .map_err(|err| ApiError::Internal(err.to_string()))?
        .error_for_status()
        .map_err(|err| ApiError::Internal(err.to_string()))?
        .json::<Value>()
        .await
        .map_err(|err| ApiError::Internal(err.to_string()))?;

    let message = first.get("message").cloned().unwrap_or_else(|| json!({}));
    let tool_calls = message
        .get("tool_calls")
        .and_then(Value::as_array)
        .cloned()
        .unwrap_or_default();
    if tool_calls.is_empty() {
        let answer = message
            .get("content")
            .and_then(Value::as_str)
            .unwrap_or("")
            .to_string();
        return Ok(AgentChatResult {
            events: vec![json!({"type": "text", "text": answer})],
            answer,
        });
    }

    messages.push(message);
    let mut events = Vec::new();
    for (index, tool_call) in tool_calls.iter().enumerate() {
        let id = format!("tc-{index}");
        let name = tool_call
            .pointer("/function/name")
            .and_then(Value::as_str)
            .unwrap_or("");
        let arguments = tool_call
            .pointer("/function/arguments")
            .cloned()
            .unwrap_or_else(|| json!({}));
        events.push(json!({
            "type": "tool_call",
            "id": id,
            "name": name,
            "arguments": arguments_to_string(&arguments),
        }));
        let content = execute_agent_tool(state, name, &arguments)?;
        events.push(json!({
            "type": "tool_result",
            "id": id,
            "name": name,
            "ok": true,
            "content": content,
        }));
        messages.push(json!({
            "role": "tool",
            "content": content,
        }));
    }

    let second = client
        .post(format!("{base_url}/api/chat"))
        .json(&json!({
            "model": model_id,
            "messages": messages,
            "stream": false,
            "options": {"temperature": llm.temperature},
        }))
        .send()
        .await
        .map_err(|err| ApiError::Internal(err.to_string()))?
        .error_for_status()
        .map_err(|err| ApiError::Internal(err.to_string()))?
        .json::<Value>()
        .await
        .map_err(|err| ApiError::Internal(err.to_string()))?;
    let answer = second
        .pointer("/message/content")
        .and_then(Value::as_str)
        .unwrap_or("")
        .to_string();
    events.push(json!({"type": "text", "text": answer}));
    Ok(AgentChatResult { events, answer })
}

fn build_agent_system_prompt(tree: &Tree, tree_id: &str) -> String {
    format!(
        "You are a document assistant for `{}` (tree_id={}). Use tools for document-content questions. Answer in the user's language.",
        tree.title, tree_id
    )
}

fn find_sections_tool_schema() -> Value {
    json!({
        "type": "function",
        "function": {
            "name": "find_sections",
            "description": "Search all document trees for sections matching a topic.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query"
                    },
                    "limit": {
                        "type": "integer",
                        "description": "Maximum result count",
                        "default": 8
                    }
                },
                "required": ["query"]
            }
        }
    })
}

fn execute_agent_tool(
    state: &SharedState,
    name: &str,
    arguments: &Value,
) -> Result<String, ApiError> {
    if name != "find_sections" {
        return Ok(format!("error: unknown tool {name}"));
    }
    let query = tool_argument(arguments, "query").unwrap_or_default();
    let limit = tool_argument(arguments, "limit")
        .and_then(|value| value.parse::<usize>().ok())
        .unwrap_or(8);
    let trees: Vec<Tree> = {
        let state = lock_state(state)?;
        state.tree_registry.values().cloned().collect()
    };
    let mut hits = Vec::new();
    for tree in &trees {
        let index = build_tree_index(tree);
        hits.extend(score_nodes_bm25(&index, &query, limit));
    }
    hits.sort_by(|left, right| right.score.total_cmp(&left.score));
    if hits.is_empty() {
        return Ok(format!("no sections matched: {query:?}"));
    }
    let mut lines = vec![format!("sections matching {query:?}:")];
    for hit in hits.into_iter().take(limit) {
        lines.push(format!(
            "- tree={} path={} {}: {}",
            hit.tree_id, hit.path, hit.title, hit.snippet
        ));
    }
    Ok(lines.join("\n"))
}

fn tool_argument(arguments: &Value, key: &str) -> Option<String> {
    if let Some(value) = arguments.get(key) {
        if let Some(text) = value.as_str() {
            return Some(text.to_string());
        }
        return Some(value.to_string());
    }
    if let Some(text) = arguments.as_str()
        && let Ok(value) = serde_json::from_str::<Value>(text)
    {
        return tool_argument(&value, key);
    }
    None
}

fn arguments_to_string(arguments: &Value) -> String {
    if let Some(text) = arguments.as_str() {
        return text.to_string();
    }
    arguments.to_string()
}

fn build_chat_answer(tree: &Tree, question: &str, hits: &[NodeQueryHit]) -> String {
    if hits.is_empty() {
        return format!("未在 `{}` 中找到和问题相关的节点。", tree.title);
    }
    let mut lines = vec![format!(
        "在 `{}` 中找到 {} 个相关节点：",
        tree.title,
        hits.len()
    )];
    for (index, hit) in hits.iter().take(5).enumerate() {
        lines.push(format!(
            "{}. {} ({}): {}",
            index + 1,
            hit.title,
            hit.path,
            hit.snippet
        ));
    }
    lines.push(format!("问题：{question}"));
    lines.join("\n")
}

fn append_chat_turn(
    state: &SharedState,
    session_id: Option<String>,
    tree_id: String,
    model: Option<String>,
    question: String,
    answer: String,
) -> Result<ChatSession, ApiError> {
    let mut state = lock_state(state)?;
    let sid = session_id.unwrap_or_else(|| Uuid::new_v4().simple().to_string());
    let now = now_string();
    let session = state.sessions.entry(sid.clone()).or_insert(ChatSession {
        session_id: sid.clone(),
        tree_id,
        model,
        title: question.chars().take(120).collect(),
        created_at: now.clone(),
        updated_at: now.clone(),
        turns: Vec::new(),
    });
    session.updated_at = now_string();
    session.turns.push(ChatTurn {
        role: "user".to_string(),
        content: question,
        created_at: now_string(),
    });
    session.turns.push(ChatTurn {
        role: "assistant".to_string(),
        content: answer,
        created_at: now_string(),
    });
    let session = session.clone();
    if let Some(store) = &state.store {
        store
            .save_session(&sid, &session)
            .map_err(|err| ApiError::Internal(err.to_string()))?;
    }
    Ok(session)
}

fn ndjson_response(events: Vec<Value>, content_type: &'static str) -> Response {
    let body = events
        .into_iter()
        .map(|event| serde_json::to_string(&event).unwrap_or_else(|_| "{}".to_string()))
        .collect::<Vec<_>>()
        .join("\n")
        + "\n";
    ([(header::CONTENT_TYPE, content_type)], Body::from(body)).into_response()
}

fn now_string() -> String {
    let seconds = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_secs())
        .unwrap_or(0);
    seconds.to_string()
}

fn sync_default_forest(state: &SharedState) -> Result<(), ApiError> {
    let mut state = lock_state(state)?;
    let mut trees: Vec<Tree> = state.tree_registry.values().cloned().collect();
    trees.sort_by(|left, right| left.node_id.cmp(&right.node_id));
    state.forest_registry.insert(
        "default".to_string(),
        Forest {
            forest_id: "default".to_string(),
            trees,
        },
    );
    Ok(())
}

fn lock_state(state: &SharedState) -> Result<std::sync::MutexGuard<'_, AppState>, ApiError> {
    state
        .lock()
        .map_err(|_| ApiError::Internal("state lock poisoned".to_string()))
}

fn get_registered_tree<'a>(state: &'a AppState, tree_id: &str) -> Result<&'a Tree, ApiError> {
    state
        .tree_registry
        .get(tree_id)
        .ok_or_else(|| ApiError::NotFound(format!("unknown tree_id: {tree_id}")))
}

fn get_registered_index<'a>(state: &'a AppState, tree_id: &str) -> Result<&'a TreeIndex, ApiError> {
    get_registered_tree(state, tree_id)?;
    state
        .index_registry
        .get(tree_id)
        .ok_or_else(|| ApiError::NotFound(format!("index not found for tree_id: {tree_id}")))
}

fn build_root_previews(tree: &Tree) -> Vec<RootNodePreview> {
    tree.children
        .iter()
        .enumerate()
        .map(|(index, child)| RootNodePreview {
            path: index.to_string(),
            title: child.title.clone(),
            summary: child.summary.clone(),
            children_count: child.children.len(),
        })
        .collect()
}

fn build_tree_response(tree: &Tree, filename: &str) -> TreeBuildResponse {
    TreeBuildResponse {
        tree_id: tree.node_id.clone(),
        filename: filename.to_string(),
        title: tree.title.clone(),
        node_count: tree.subtree_size.unwrap_or(1),
        max_depth: max_tree_depth(tree),
        root_count: tree.children.len(),
        roots: build_root_previews(tree),
    }
}

fn build_tree_summary(tree: &Tree) -> TreeSummaryResponse {
    TreeSummaryResponse {
        tree_id: tree.node_id.clone(),
        title: tree.title.clone(),
        node_count: tree.subtree_size.unwrap_or(1),
        max_depth: max_tree_depth(tree),
        root_count: tree.children.len(),
    }
}

fn build_tree_overview(tree: &Tree) -> TreeOverviewResponse {
    TreeOverviewResponse {
        tree_id: tree.node_id.clone(),
        title: tree.title.clone(),
        node_count: tree.subtree_size.unwrap_or(1),
        max_depth: max_tree_depth(tree),
        root_count: tree.children.len(),
        roots: build_root_previews(tree),
    }
}

fn build_index_meta_response(index: &TreeIndex) -> TreeIndexMetaResponse {
    TreeIndexMetaResponse {
        tree_id: index.tree_id.clone(),
        tree_title: index.tree_title.clone(),
        document_count: index.corpus.document_count,
        average_document_length: index.corpus.average_document_length,
        term_count: index.postings.len(),
        tree_document_length: index.tree_document_length,
    }
}

fn build_forest_summary(forest: &Forest) -> ForestSummaryResponse {
    ForestSummaryResponse {
        forest_id: forest.forest_id.clone(),
        tree_count: forest.tree_count(),
        trees: forest.trees.iter().map(build_tree_summary).collect(),
    }
}

fn max_tree_depth(tree: &Tree) -> usize {
    tree.children
        .iter()
        .map(max_tree_depth)
        .max()
        .unwrap_or_else(|| tree.depth.unwrap_or(0))
        .max(tree.depth.unwrap_or(0))
}

fn resolve_tree_path<'a>(tree: &'a Tree, path: &str) -> Result<&'a Tree, ApiError> {
    if path == "root" {
        return Ok(tree);
    }
    let mut current = tree;
    for segment in path.split('.') {
        let index = segment
            .parse::<usize>()
            .map_err(|_| ApiError::NotFound(format!("invalid path: {path}")))?;
        current = current
            .children
            .get(index)
            .ok_or_else(|| ApiError::NotFound(format!("invalid path: {path}")))?;
    }
    Ok(current)
}

fn build_child_path(parent_path: &str, index: usize) -> String {
    if parent_path == "root" {
        index.to_string()
    } else {
        format!("{parent_path}.{index}")
    }
}

fn build_node_detail(tree_id: &str, path: &str, node: &Tree) -> NodeDetailResponse {
    NodeDetailResponse {
        tree_id: tree_id.to_string(),
        path: path.to_string(),
        title: node.title.clone(),
        summary: node.summary.clone(),
        text: content_to_search_text(node.content.as_ref()),
        children_count: node.children.len(),
        children: node
            .children
            .iter()
            .enumerate()
            .map(|(index, _)| build_child_path(path, index))
            .collect(),
    }
}

fn build_node_children(tree_id: &str, path: &str, node: &Tree) -> NodeChildrenResponse {
    NodeChildrenResponse {
        tree_id: tree_id.to_string(),
        path: path.to_string(),
        children: node
            .children
            .iter()
            .enumerate()
            .map(|(index, child)| RootNodePreview {
                path: build_child_path(path, index),
                title: child.title.clone(),
                summary: child.summary.clone(),
                children_count: child.children.len(),
            })
            .collect(),
    }
}

fn default_filename() -> String {
    "document.md".to_string()
}

#[cfg(test)]
mod tests {
    use axum::body::Body;
    use axum::http::{Request, StatusCode};
    use tempfile::tempdir;
    use tower::ServiceExt;

    use super::*;

    #[tokio::test]
    async fn service_builds_indexes_searches_and_deletes_tree() {
        let app = create_app(None).unwrap();
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/trees")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r##"{"text":"# Intro\n\nHello world.\n\n## Detail\n\nBM25 reranking section.","filename":"sample.md"}"##,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: TreeBuildResponse = serde_json::from_slice(&body).unwrap();
        let tree_id = payload.tree_id;

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/trees/{tree_id}/index"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!(
                        "/api/trees/{tree_id}/search/nodes?q=reranking&limit=1"
                    ))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let response = app
            .oneshot(
                Request::builder()
                    .method("DELETE")
                    .uri(format!("/api/trees/{tree_id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn legacy_build_accepts_multipart_file() {
        let app = create_app(None).unwrap();
        let boundary = "TREEFYUST_BOUNDARY";
        let body = format!(
            "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"file\"; filename=\"legacy.md\"\r\n\
             Content-Type: text/markdown\r\n\r\n\
             # Legacy\n\nUploaded content.\r\n\
             --{boundary}--\r\n"
        );

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/build")
                    .header(
                        "content-type",
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: TreeBuildResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload.filename, "legacy.md");
        assert_eq!(payload.title, "legacy.md");
    }

    #[tokio::test]
    async fn multipart_build_accepts_refine_options() {
        let app = create_app(None).unwrap();
        let boundary = "TREEFYUST_REFINE_BOUNDARY";
        let body = format!(
            "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"refine_split_threshold\"\r\n\r\n\
             10\r\n\
             --{boundary}\r\n\
             Content-Disposition: form-data; name=\"refine_max_parts\"\r\n\r\n\
             2\r\n\
             --{boundary}\r\n\
             Content-Disposition: form-data; name=\"file\"; filename=\"refine.md\"\r\n\
             Content-Type: text/markdown\r\n\r\n\
             # Long\n\nalpha beta gamma delta epsilon zeta eta theta.\r\n\
             --{boundary}--\r\n"
        );

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/trees/from-file")
                    .header(
                        "content-type",
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: TreeBuildResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(payload.filename, "refine.md");
        assert!(payload.node_count > 2);
        assert_eq!(payload.roots[0].children_count, 2);
    }

    #[tokio::test]
    async fn multipart_build_rejects_invalid_refine_options() {
        let app = create_app(None).unwrap();
        let boundary = "TREEFYUST_BAD_REFINE_BOUNDARY";
        let body = format!(
            "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"refine_max_parts\"\r\n\r\n\
             nope\r\n\
             --{boundary}\r\n\
             Content-Disposition: form-data; name=\"file\"; filename=\"bad.md\"\r\n\
             Content-Type: text/markdown\r\n\r\n\
             # Bad\n\ncontent.\r\n\
             --{boundary}--\r\n"
        );

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/trees/from-file")
                    .header(
                        "content-type",
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::BAD_REQUEST);
    }

    #[tokio::test]
    async fn service_restores_tree_index_and_forest_from_store() {
        let tempdir = tempdir().unwrap();
        let store = RegistryStore::new(tempdir.path());
        let app = create_app(Some(store.clone())).unwrap();
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/trees")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r##"{"text":"# Persisted\n\nStored retrieval content.","filename":"persisted.md"}"##,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: TreeBuildResponse = serde_json::from_slice(&body).unwrap();
        let tree_id = payload.tree_id;

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(format!("/api/trees/{tree_id}/index"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let restored = create_app(Some(store)).unwrap();
        let response = restored
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/api/trees/{tree_id}/index/meta"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let response = restored
            .oneshot(
                Request::builder()
                    .uri("/api/forest")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let forest: ForestSummaryResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(forest.tree_count, 1);
    }

    #[tokio::test]
    async fn app_layer_history_file_stream_queries_and_chat_work() {
        let app = create_app(None).unwrap();
        let boundary = "TREEFYUST_APP_BOUNDARY";
        let body = format!(
            "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"file\"; filename=\"app.md\"\r\n\
             Content-Type: text/markdown\r\n\r\n\
             # App\n\nQueryable content.\r\n\
             --{boundary}--\r\n"
        );

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/trees/from-file")
                    .header(
                        "content-type",
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: TreeBuildResponse = serde_json::from_slice(&body).unwrap();
        let tree_id = payload.tree_id;

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/history")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let history: Vec<BuildRecord> = serde_json::from_slice(&body).unwrap();
        assert_eq!(history[0].id, tree_id);
        assert!(history[0].raw_text.is_none());

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/api/build/{tree_id}/file"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        assert_eq!(&body[..], b"# App\n\nQueryable content.");

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/api/trees/{tree_id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri("/api/queries/stats")
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let stats: QueryStatsResponse = serde_json::from_slice(&body).unwrap();
        assert!(stats.by_tool.contains_key("overview"));

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat")
                    .header("content-type", "application/json")
                    .body(Body::from(format!(
                        r#"{{"bid":"{tree_id}","question":"content","model":"mock"}}"#
                    )))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("\"type\":\"tool_call\""));
        assert!(text.contains("\"type\":\"tool_result\""));
        assert!(text.contains("\"id\":\"tc-0\""));
        assert!(text.contains("\"type\":\"done\""));

        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .uri(format!("/api/sessions?bid={tree_id}"))
                    .body(Body::empty())
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let sessions: SessionListResponse = serde_json::from_slice(&body).unwrap();
        assert_eq!(sessions.sessions.len(), 1);
    }

    #[tokio::test]
    async fn build_stream_returns_ndjson_events() {
        let app = create_app(None).unwrap();
        let boundary = "TREEFYUST_STREAM_BOUNDARY";
        let body = format!(
            "--{boundary}\r\n\
             Content-Disposition: form-data; name=\"file\"; filename=\"stream.md\"\r\n\
             Content-Type: text/markdown\r\n\r\n\
             # Stream\n\nBody.\r\n\
             --{boundary}--\r\n"
        );

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/build/stream")
                    .header(
                        "content-type",
                        format!("multipart/form-data; boundary={boundary}"),
                    )
                    .body(Body::from(body))
                    .unwrap(),
            )
            .await
            .unwrap();

        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("\"type\":\"start\""));
        assert!(text.contains("\"type\":\"done\""));
    }

    #[tokio::test]
    #[ignore = "requires local Ollama gemma4:latest"]
    async fn ollama_chat_uses_tool_call() {
        let app = create_app(None).unwrap();
        let response = app
            .clone()
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/trees")
                    .header("content-type", "application/json")
                    .body(Body::from(
                        r##"{
                            "filename": "ollama-chat.md",
                            "text": "# Intro\n\nTreefyit builds document trees.\n\n## Search\n\nThe search layer uses BM25 indexes for nodes."
                        }"##,
                    ))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let payload: TreeBuildResponse = serde_json::from_slice(&body).unwrap();

        let response = app
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri("/api/chat")
                    .header("content-type", "application/json")
                    .body(Body::from(format!(
                        r#"{{"bid":"{}","question":"Treefyit 的 search layer 使用什么索引？请简短回答。","model":"ollama/gemma4:latest"}}"#,
                        payload.tree_id
                    )))
                    .unwrap(),
            )
            .await
            .unwrap();
        assert_eq!(response.status(), StatusCode::OK);
        let body = axum::body::to_bytes(response.into_body(), usize::MAX)
            .await
            .unwrap();
        let text = String::from_utf8(body.to_vec()).unwrap();
        assert!(text.contains("\"type\":\"tool_call\""), "{text}");
        assert!(text.contains("\"type\":\"tool_result\""), "{text}");
        assert!(text.contains("\"id\":\"tc-0\""), "{text}");
        assert!(text.to_lowercase().contains("bm25"), "{text}");
    }
}
