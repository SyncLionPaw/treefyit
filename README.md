# treefyit

Tree-based document indexing for Markdown, HTML, and PDF documents.

## Configure API keys

`treefyit` reads runtime settings from
[src/config/settings.local.toml](src/config/settings.local.toml).
Start from
[src/config/settings.toml.example](src/config/settings.toml.example)
and override locally as needed.

```toml
[llm]
model = "deepseek-chat"
api_key = "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

[mineru]
api_key = "..."
```

## Quick Start

### System dependency

`src.builder` uses `libmagic` through `python-magic` to detect file types
by content.

- macOS

```bash
brew install libmagic
```

- Debian / Ubuntu

```bash
sudo apt-get install libmagic1 libmagic-dev
```

### CLI

```bash
# Install dependencies
uv sync

# Start the HTTP service
uv run python main.py -p 8765

# Start via installed script entrypoint
uv run treefyit -p 8765

# Clear the local store, then start
uv run python main.py --clear

# Or run the module directly
uv run python -m src.server
```

### Python — build a tree programmatically

```python
from src.builder import BuildOptions, build_tree_from_file

tree = build_tree_from_file(
    "paper.md",
    options=BuildOptions(summarize=True),
)

print(tree.title)
print(tree.children[0].title)
```

### Python — builder flow

```text
source -> parse -> infer levels -> refine sections -> build tree -> summarize -> Tree model
```

- `source`: detect input kind (`pdf`, `html`, `text`, `zip`)
- `parse`: normalize source content into flat sections
- `infer levels`: repair section hierarchy
- `refine sections`: split or normalize oversized sections
- `build tree`: assemble the nested structure
- `summarize`: optionally summarize nodes
- `Tree model`: convert to `src.model.tree.Tree`

### Python — LLM client

```python
from src.llm import acomplete, complete, count_tokens

text = complete("Summarize in one sentence: ...")
resp = await acomplete("...")
n = count_tokens("long text")
```

## Project layout

```text
src/
  builder/       typed build pipeline
  chat/          chat event building
  config/        TOML settings
  llm/           LiteLLM wrapper
  model/         Tree / Forest models
  query/         tree / forest query and BM25 index
  server/        FastAPI service
  store/         JSON registry persistence
main.py          thin CLI for `src.server`
openapi.yaml     `src.server` API spec
```

## HTTP API

The current service implementation lives in `src.server`.

```bash
uv run python -m src.server
```

Default configuration:

- LLM settings: [src/config/settings.local.toml](src/config/settings.local.toml)
- Store directory: `.treefyit-store/`
- OpenAPI spec: [openapi.yaml](openapi.yaml)

### Endpoints

| Method | Path | Description |
|------|------|------|
| GET | `/health` | Health check |
| POST | `/api/trees` | Build a tree from text |
| POST | `/api/trees/from-file` | Build a tree from file upload |
| POST | `/api/build` | Compatibility build endpoint |
| POST | `/api/build/stream` | NDJSON build progress stream |
| GET | `/api/history` | Build history |
| GET | `/api/build/{bid}` | Build detail and tree |
| GET | `/api/build/{bid}/file` | Download uploaded original |
| DELETE | `/api/build/{bid}` | Delete build, tree, index, and original |
| GET | `/api/trees` | List registered trees |
| GET | `/api/trees/{tree_id}` | Tree overview |
| DELETE | `/api/trees/{tree_id}` | Delete tree and index |
| POST | `/api/trees/{tree_id}/index` | Build or rebuild index |
| GET | `/api/trees/{tree_id}/index/meta` | Index summary |
| GET | `/api/trees/{tree_id}/search/nodes` | Single-tree node search |
| GET | `/api/trees/{tree_id}/nodes/{path}` | Node detail |
| GET | `/api/trees/{tree_id}/children/{path}` | Child node list |
| GET | `/api/forest` | Default forest overview |
| GET | `/api/forest/search` | Compatibility forest search |
| GET | `/api/forest/search/trees` | Forest tree search |
| GET | `/api/forest/search/nodes` | Forest node search |
| GET | `/api/queries` | Query log |
| GET | `/api/queries/stats` | Query stats |
| POST | `/api/chat` | Streaming chat events with optional tree or forest context |
| GET | `/api/sessions` | Session list |
| GET | `/api/sessions/{sid}/turns` | Session turns |
| DELETE | `/api/sessions/{sid}` | Delete session |

### Minimal examples

Build from text:

```bash
curl -X POST http://127.0.0.1:8765/api/trees \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "# Intro\n\nHello world.\n\n## Detail\n\nMore text.",
    "filename": "sample.md"
  }'
```

Build an index:

```bash
curl -X POST http://127.0.0.1:8765/api/trees/{tree_id}/index
```

Search nodes:

```bash
curl 'http://127.0.0.1:8765/api/trees/{tree_id}/search/nodes?q=detail&limit=5'
```

Search forest trees:

```bash
curl 'http://127.0.0.1:8765/api/forest/search/trees?q=reranking&limit=5'
```

Chat directly without a knowledge base:

```bash
curl -X POST http://127.0.0.1:8765/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "question": "用一句话解释什么是 BM25"
  }'
```

Chat with an optional tree context:

```bash
curl -X POST http://127.0.0.1:8765/api/chat \
  -H 'Content-Type: application/json' \
  -d '{
    "tree_id": "{tree_id}",
    "question": "总结这份资料"
  }'
```
