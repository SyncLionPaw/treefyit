# treefyust

Rust implementation of the current `treefyit/` typed service.

## Run

```bash
cargo run
```

The server listens on `0.0.0.0:8765`.

## Implemented Capabilities

- Typed `Tree` / `Forest` models
- TOML settings for `llm`, `builder`, and `store`
- Markdown/text builder
- Rule-based long section refine
- Parent section preserved as a structural node
- Tree index with node term statistics, postings, corpus stats, and tree aggregation
- BM25 node search
- In-memory forest query
- JSON registry store for `Tree` and `TreeIndex`
- Application-layer JSON store for build history, original uploads, query logs, and chat sessions
- Axum HTTP service
- Minimal async LLM client for Ollama `/api/chat` and OpenAI-compatible `/chat/completions`
- Agent-style `/api/chat` backed by local Ollama tools. The current tool set includes `find_sections`, and tool call/result ids are normalized as `tc-0`, `tc-1`, ...
- `summarize=true` builds the tree first, then runs bottom-up LLM summarization from leaves to parents

## API

Main endpoints:

- `GET /health`
- `POST /api/trees`
- `POST /api/trees/from-file`
- `POST /api/build`
- `POST /api/build/stream`
- `GET /api/history`
- `GET /api/build/{bid}`
- `GET /api/build/{bid}/file`
- `DELETE /api/build/{bid}`
- `GET /api/trees`
- `GET /api/trees/{tree_id}`
- `DELETE /api/trees/{tree_id}`
- `POST /api/trees/{tree_id}/index`
- `GET /api/trees/{tree_id}/index/meta`
- `GET /api/trees/{tree_id}/search/nodes`
- `GET /api/trees/{tree_id}/nodes/{path}`
- `GET /api/trees/{tree_id}/children/{path}`
- `GET /api/forest`
- `GET /api/forest/search`
- `GET /api/forest/search/trees`
- `GET /api/forest/search/nodes`
- `GET /api/queries`
- `GET /api/queries/stats`
- `POST /api/chat`
- `GET /api/sessions`
- `GET /api/sessions/{sid}/turns`
- `DELETE /api/sessions/{sid}`

Compatibility aliases are also provided:

- `POST /api/build` accepts either JSON text build or multipart file upload
- `/api/tree/{tree_id}/...`

Settings are read from both `config/settings*.toml` and `treefyust/config/settings*.toml`, so `cargo run` works from either this crate directory or the parent repository directory.

The default LLM settings target the local Ollama service:

```toml
[llm]
model = "ollama/gemma4:latest"
base_url = "http://127.0.0.1:11434"
temperature = 0.0
max_tokens = 2048
```

## Example

```bash
curl -X POST http://127.0.0.1:8765/api/trees \
  -H 'Content-Type: application/json' \
  -d '{"text":"# Intro\n\nHello world.\n\n## Detail\n\nMore text.","filename":"sample.md"}'
```

Then build an index:

```bash
curl -X POST http://127.0.0.1:8765/api/trees/{tree_id}/index
```

Search nodes:

```bash
curl 'http://127.0.0.1:8765/api/trees/{tree_id}/search/nodes?q=detail&limit=5'
```

## Verification

The full delivery status is tracked in [DELIVERY_REPORT.md](file:///Users/bytedance/docs/treefyit/treefyust/DELIVERY_REPORT.md).

```bash
cargo fmt
cargo test
cargo check
cargo clippy --all-targets -- -D warnings
```
