# treefyit

Tree-based document indexing — extract hierarchical tree structures from PDF and Markdown documents.

## Configure API keys

Both the build pipeline (`src/llm`) and the chat agent (`src/chat`) read their API keys from a
`.env` file in the project root. `python-dotenv` is already loaded by both modules, so you
only need to create the file once.

```
# .env (project root — same directory as pyproject.toml)

# Required for `POST /api/build` with mode=semantic / default, and for `POST /api/chat`
# when the requested model starts with "deepseek/" or equals "deepseek-chat" / "deepseek-reasoner".
DEEPSEEK_API_KEY=sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# (Optional) For any other model name passed to `POST /api/chat` (pagent.LLM).
# OPENAI_API_KEY=sk-...

# (Optional) MinerU Precision API token — required for large PDFs (>10 MB / >20 pages).
# MINERU_TOKEN=...
```

The chat agent at `POST /api/chat` auto-picks the pagent provider from the `model` field:

- `model = "deepseek-chat"`, `"deepseek-reasoner"`, or anything starting with `"deepseek/..."`
  → reads `DEEPSEEK_API_KEY` and uses `pagent.DeepSeek`.
- Any other value → reads `OPENAI_API_KEY` and uses `pagent.LLM`.

If the required env var is missing, the endpoint returns a JSON `{"type": "error", "message": "..."}`
event and stops — no silent fallback.

## Quick Start

### System dependency

`treefyit.builder` now uses `libmagic` through `python-magic` to detect file types by content, not only by filename suffix.

- macOS:

```bash
brew install libmagic
```

- Debian / Ubuntu:

```bash
sudo apt-get install libmagic1 libmagic-dev
```

### CLI

```bash
# Install dependencies (requires uv)
uv sync

# Build a tree from a local Markdown file (prints tree to terminal)
uv run python main.py build path/to/doc.md --mode auto

# Export HTML visualization
uv run python main.py build path/to/doc.md -o out.html

# Start the HTTP API server
uv run python main.py serve -p 8765

# Clear persisted builds/caches/uploads, then serve
uv run python main.py serve --clear
```

### Python — build a tree programmatically

```python
from src.tree import build_tree
from src.vis import save_html

tree = build_tree("paper.md", model="deepseek/deepseek-chat", mode="auto", summarize=True)
save_html(tree, "paper.html")
```

### Python — `treefyit.builder` flow

`treefyit/*` is intended to be an independent typed implementation, separate from the legacy `src/*` pipeline.

The builder flow is:

```text
source -> parse -> infer levels -> build tree -> finalize -> Tree model
```

- `source`: detect the input kind (`pdf`, `html`, `text`, `zip`)
- `parse`: normalize multi-type documents into Markdown or flat sections
- `infer levels`: fix section hierarchy before tree assembly
- `build tree`: turn flat sections into a nested tree
- `finalize`: assign node ids and optionally summarize
- `Tree model`: convert the nested tree into `treefyit.model.tree.Tree`

`parse` is the normalization stage. It is responsible for turning different source formats into a common intermediate form, usually Markdown or flat sections. Different parse backends can be plugged in here, for example:

- `markdown-it-py` for Markdown / text structure extraction
- `MinerU` for PDF to Markdown conversion
- `MarkItDown` for HTML / PDF to Markdown conversion

`infer levels` is intentionally designed as a pluggable stage and should live in `treefyit/builder/infer.py`. The parser only provides raw headings / sections; hierarchy quality comes from the inferer. Different inferers can be plugged in here, for example:

- rule-based inference for numbering patterns like `1`, `1.1`, `1.1.1`, `一、`, `（一）`
- LLM-based inference for messy or weakly structured documents

The important boundary is that `infer` should take flat sections as input and return flat sections with corrected levels, so the later tree-building step stays uniform.

### Python — LLM client

`src/llm` wraps [LiteLLM](https://github.com/BerriAI/litellm) with retries and automatic
`.env` loading from the project root:

```python
from src.llm import chat, achat, count_tokens

# Sync
text = chat("Summarize in one sentence: ...", model="deepseek/deepseek-chat", system="Be concise.")

# Async (used by the build pipeline and server)
resp = await achat("...", model="gpt-4o")

# Token estimate (no API call)
n = count_tokens(long_text, model="deepseek/deepseek-chat")
```

| Function | Returns | Notes |
|----------|---------|-------|
| `chat(prompt, **kwargs)` | `str` | Sync; kwargs forwarded to `litellm.completion` (`model`, `system`, `temperature`, …) |
| `await achat(prompt, **kwargs)` | `str` | Async via `litellm.acompletion` |
| `count_tokens(text, model=...)` | `int` | Local estimate via LiteLLM |

Set `OPENAI_API_KEY` (or legacy alias `CHATGPT_API_KEY`) for OpenAI-compatible models.
Both `chat` and `achat` retry up to 10 times on transient failures and raise
`LLMError` if all attempts fail.

## Project layout

```
treefyit/
  builder/       typed build pipeline
  config/        TOML settings
  llm/           LiteLLM wrapper
  model/         Tree / Forest models
  query/         tree / forest query and BM25 index
  server/        new FastAPI service
  store/         JSON registry persistence
src/
  tree/          legacy build pipeline
  server/        legacy FastAPI app
  parser/        md, pdf, html, zip
  store/         legacy SQLite + JSON persistence under results/
  chat/          pagent agent + streaming chat
  tools/         tree navigation for the agent
  llm/           chat / achat / count_tokens
main.py          CLI entry (build | serve)
openapi.yaml     new `treefyit.server` API spec
```

## treefyit.server API

`treefyit.server` 是当前正在推进的独立服务实现，只依赖 `treefyit/*`。

启动方式：

```bash
uv run python -m treefyit.server
```

默认配置：

- LLM 配置见 [treefyit/config/settings.local.toml](treefyit/config/settings.local.toml)
- store 目录默认是 `.treefyit-store/`
- OpenAPI 规范见 [openapi.yaml](openapi.yaml)

### 接口概览

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/health` | 健康检查 |
| POST | `/api/trees` | 从 text 构建 tree |
| POST | `/api/trees/from-file` | 从文件构建 tree |
| POST | `/api/build` | 兼容旧 build 路径 |
| POST | `/api/build/stream` | 文件构建 NDJSON 进度流 |
| GET | `/api/history` | 构建历史 |
| GET | `/api/build/{bid}` | 构建详情和 tree |
| GET | `/api/build/{bid}/file` | 下载上传原件 |
| DELETE | `/api/build/{bid}` | 删除 build、tree、index 和原件 |
| GET | `/api/trees` | 列出所有已注册 tree |
| GET | `/api/trees/{tree_id}` | tree 概览 |
| DELETE | `/api/trees/{tree_id}` | 删除 tree 和 index |
| POST | `/api/trees/{tree_id}/index` | 构建或重建索引 |
| GET | `/api/trees/{tree_id}/index/meta` | 读取索引摘要 |
| GET | `/api/trees/{tree_id}/search/nodes` | 单树节点搜索 |
| GET | `/api/trees/{tree_id}/nodes/{path}` | 节点详情 |
| GET | `/api/trees/{tree_id}/children/{path}` | 子节点列表 |
| GET | `/api/forest` | 默认 forest 概览 |
| GET | `/api/forest/search` | 兼容旧版 forest 综合搜索 |
| GET | `/api/forest/search/trees` | forest tree 搜索 |
| GET | `/api/forest/search/nodes` | forest node 搜索 |
| GET | `/api/queries` | 查询日志 |
| GET | `/api/queries/stats` | 查询统计 |
| POST | `/api/chat` | 基于 tree/index 的流式检索问答事件 |
| GET | `/api/sessions` | 会话列表 |
| GET | `/api/sessions/{sid}/turns` | 会话 turns |
| DELETE | `/api/sessions/{sid}` | 删除会话 |

### 最小样例

#### 1. 从 text 构建 tree

```bash
curl -X POST http://127.0.0.1:8765/api/trees \
  -H 'Content-Type: application/json' \
  -d '{
    "text": "# Intro\n\nHello world.\n\n## Detail\n\nMore text.",
    "filename": "sample.md"
  }'
```

#### 2. 构建索引

```bash
curl -X POST http://127.0.0.1:8765/api/trees/{tree_id}/index
```

响应示例：

```json
{
  "tree_id": "a1b2c3",
  "tree_title": "sample.md",
  "document_count": 3,
  "average_document_length": 6.0,
  "term_count": 12,
  "tree_document_length": 18
}
```

#### 3. 单树搜索

```bash
curl 'http://127.0.0.1:8765/api/trees/{tree_id}/search/nodes?q=detail&limit=5'
```

#### 4. forest 搜索

```bash
curl 'http://127.0.0.1:8765/api/forest/search/trees?q=reranking&limit=5'
```

说明：

- 对外主路径使用 RESTful 命名
- 兼容别名仍然保留，例如 `/api/build` 与 `/api/tree/{tree_id}/...`
- build / index / delete 会同步写入 JSON store，并在服务启动时自动恢复

## Legacy HTTP API

所有接口均返回 JSON（流式接口除外）。路径前缀均为 `/api`。源实现见 [src/server/server.py](src/server/server.py)。

### 1. Build API — 构建文档树

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/build` | 上传文件构建树；支持 `.md / .pdf / .html / .zip`，命中缓存时直接返回 |
| POST | `/api/build/stream` | 与 `/api/build` 相同参数，NDJSON 流式返回各阶段进度 |
| GET  | `/api/history` | 列出所有构建记录（精简列表，无 raw_text/tree） |
| GET  | `/api/build/{bid}` | 读取某次构建的完整结果 |
| GET  | `/api/build/{bid}/file` | 获取该次构建的原始上传文件（二进制流） |
| DELETE | `/api/build/{bid}` | 删除某次构建（内存 + SQLite + 磁盘缓存 + 原件） |

#### POST /api/build

- **请求**: `multipart/form-data`
  - `file` (文件, 必需) – 要处理的文档；支持 `.md` / `.pdf` / `.html` / `.htm` / `.zip`
  - `model` (string) – 大模型标识，默认 `deepseek/deepseek-chat`
  - `mode` (string) – 解析模式，`auto`（默认，按标题 + 编号）/ `md`（纯 Markdown 标题）/ `semantic`（LLM 驱动）
  - `summarize` (boolean) – 是否为节点生成摘要，默认 `true`

- **成功响应** (`200 application/json`)

```json
{
  "id": "1902897a3b4c0d1e",
  "filename": "README.md",
  "raw_text": "原始文档文本字符串...",
  "mermaid": "flowchart TD\n  0[\"标题 A\"]\n  0 --> 1[\"标题 A.1\"]\n...",
  "tree": [
    {
      "title": "1. Introduction",
      "text": "This section introduces...",
      "summary": "介绍项目背景与动机。",
      "children": [
        {
          "title": "1.1 Motivation",
          "text": "...",
          "summary": "...",
          "children": []
        }
      ]
    }
  ],
  "stats": {
    "input_tokens": 1200,
    "output_tokens": 320,
    "node_count": 42,
    "elapsed_sec": 3.1,
    "model": "deepseek/deepseek-chat",
    "mode": "auto"
  },
  "created_at": "14:30:05",
  "cached": true
}
```

字段说明：

| 字段 | 类型 | 含义 |
|------|------|------|
| `id` | string | 构建 ID；由 `int(now * 1000)` 的十六进制前缀生成，作为持久化 key |
| `filename` | string | 上传时的原始文件名 |
| `content_type` | string | 原始文件的 MIME 类型（如 `application/pdf`） |
| `file_size` | integer | 原始文件大小（字节） |
| `sha256` | string | 原始文件的 SHA-256 哈希 |
| `has_original_file` | boolean | 是否成功保存了原件 |
| `original_file_url` | string? | 获取原件的 URL（如 `/api/build/{bid}/file`）；缺失时表示无原件 |
| `raw_text` | string | 文档提取的纯文本（PDF/HTML 会先转成文本）；可能很长 |
| `mermaid` | string | 该树对应的 Mermaid 流程图源码 |
| `tree` | `list[Node]` | 递归树；每个节点含 `title`, `text`, `summary`, `children: list[Node]` |
| `stats` | object | `{input_tokens, output_tokens, node_count, elapsed_sec, model, mode}`；可选 `verify` 子对象（LLM 结构校验结果） |
| `created_at` | string | `HH:MM:SS` 时间戳（服务端本地） |
| `cached` | boolean | 本次是否命中输入内容缓存（`true` 表示未重新解析） |
| `error` | string? | 仅在失败时出现，内容为错误消息；此时 `tree`/`mermaid` 通常为空 |

- **失败响应** (`200 application/json`)

```json
{
  "id": "1902897a3b4c0d1e",
  "filename": "broken.pdf",
  "error": "....",
  "tree": [],
  "mermaid": "",
  "stats": {
    "input_tokens": 0,
    "output_tokens": 0,
    "node_count": 0,
    "elapsed_sec": 0.1,
    "model": "deepseek/deepseek-chat",
    "mode": "auto"
  },
  "created_at": "14:30:05"
}
```

缓存 key 由 `sha256(text|model|mode|summarize)[:16]` 计算，命中时直接跳过解析。

#### POST /api/build/stream

- **请求**: 与 `POST /api/build` 相同的 `multipart/form-data` 字段
- **响应**: `application/x-ndjson`（每行一条 JSON 事件）

常见事件类型：

| type | 说明 |
|------|------|
| `start` | 构建开始，含 `filename`, `model`, `mode`, `file_size` |
| `progress` | 阶段进度；`stage` 如 `parse`, `structure_done`, `refine`, `thin`, `summarize`, `verify` |
| `warning` | 非致命告警（如原件保存失败、verify 失败） |
| `done` | 构建完成；`cached: true/false`，`result` 为完整 build 对象 |
| `error` | 构建失败；含 `message` 与失败时的 `result` |

`POST /api/build` 保持为稳定的一次性 JSON 接口；需要 UI 进度条时使用 stream 版本。

#### GET /api/history

- **请求**: 无参数
- **响应**: `200 application/json`

```json
[
  {
    "id": "1902897a3b4c0d1e",
    "filename": "paper.pdf",
    "content_type": "application/pdf",
    "file_size": 1234567,
    "has_original_file": true,
    "original_file_url": "/api/build/1902897a3b4c0d1e/file",
    "stats": { "node_count": 42, "elapsed_sec": 3.1, "model": "deepseek-chat", "mode": "auto" },
    "created_at": "14:30:05",
    "cached": false,
    "error": null
  }
]
```

返回最近构建的精简列表，按构建 `id` 降序（`id` 为毫秒时间戳的十六进制，可反映创建先后）；不含 `raw_text`、`tree`、`mermaid` 等大字段，但包含原件状态，方便前端判断预览能力。

#### GET /api/build/{bid}

- **路径参数**: `bid` (string) – 构建 ID
- **成功响应** (`200 application/json`): 与 `POST /api/build` 相同的完整 build 对象
- **未找到** (`404 application/json`):

```json
{ "error": "not found" }
```

#### GET /api/build/{bid}/file

- **路径参数**: `bid` (string) – 构建 ID
- **成功响应** (`200`): 原始文件二进制流
  - `Content-Type`: 根据文件类型自动推断（如 `application/pdf`、`text/html`、`text/markdown`）
  - `Content-Disposition`: `inline; filename="..."`
  - `Cache-Control`: `private, max-age=300`
  - `ETag`: `"sha256-<hash>"`
- **原件不存在** (`404 application/json`):

```json
{ "error": { "code": "ORIGINAL_FILE_NOT_FOUND", "message": "Original file is not available for this build." } }
```

#### DELETE /api/build/{bid}

- **成功响应** (`200 application/json`):

```json
{ "ok": true }
```

同时删除内存缓存、SQLite 记录、磁盘 JSON 文件以及原始上传文件。

删除操作会同时清理：进程内的 `history` 字典、`builds` SQLite 行、磁盘上的 `build_{bid}.json` 文件，以及 agent 工具注册表中的对应 tree_id。

---

### 2. Agent Tools API — 检索已构建的树

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/api/trees` | 列出当前注册过的 tree_id（与 build id 一致） |
| GET | `/api/forest` | 林场目录：所有已注册文档的概览（含根节点标题） |
| GET | `/api/forest/search` | 跨文档 BM25 搜索（`q` 查询词，`limit` 条数上限） |
| GET | `/api/trees/{tree_id}` | 某树的概览（overview） |
| GET | `/api/trees/{tree_id}/nodes/{path}` | inspect 指定路径节点的详细内容 |
| GET | `/api/trees/{tree_id}/children/{path}` | 获取指定路径节点的子节点列表 |
| GET | `/api/queries` | 最近 200 条工具调用记录 |
| GET | `/api/queries/stats` | 查询统计：总数、按工具/按树聚合、最近 20 条 |

路径 `{path}` 为点分索引路径，例如 `0`、`0.1`、`0.1.2`。

调用 `/api/trees/*` 系列接口会同时向 `queries` 表追加一行日志，供 `/api/queries` 使用。

#### GET /api/trees

- **响应** (`200 application/json`):

```json
[
  { "tree_id": "1902897a3b4c0d1e", "node_count": 42, "max_depth": 4 },
  "..."
]
```

#### GET /api/forest

- **响应** (`200 application/json`):

```json
{
  "tree_count": 2,
  "trees": [
    {
      "tree_id": "1902897a3b4c0d1e",
      "filename": "paper.pdf",
      "doc_kind": "structured",
      "node_count": 42,
      "max_depth": 4,
      "roots": [
        { "path": "0", "title": "1. Introduction", "summary": "...", "children_count": 2 }
      ]
    }
  ]
}
```

#### GET /api/forest/search

- **参数**: `q` (string, 必需) 查询词；`limit` (integer, 可选, 默认 8, 最大 20)
- **响应** (`200 application/json`):

```json
{
  "trees": {
    "query": "白茶",
    "hits": [
      {
        "tree_id": "1902897a3b4c0d1e",
        "filename": "tea.html",
        "doc_kind": "html",
        "score": 1.23,
        "node_count": 5,
        "root_titles": ["白茶"]
      }
    ]
  },
  "sections": {
    "query": "白茶",
    "hits": [
      {
        "tree_id": "1902897a3b4c0d1e",
        "filename": "tea.html",
        "path": "0",
        "title": "白茶",
        "summary": "...",
        "score": 2.1
      }
    ]
  }
}
```

#### GET /api/trees/{tree_id}

- **成功响应** (`200 application/json`):

```json
{
  "tree_id": "1902897a3b4c0d1e",
  "node_count": 42,
  "max_depth": 4,
  "roots": [
    { "path": "0", "title": "1. Introduction", "summary": "...", "children_count": 2 },
    { "path": "1", "title": "2. Design",       "summary": "...", "children_count": 0 }
  ]
}
```

- **未知 tree_id** (`200 application/json`):

```json
{ "error": "unknown tree_id: <id>" }
```

#### GET /api/trees/{tree_id}/nodes/{path}

- **成功响应** (`200 application/json`):

```json
{
  "tree_id": "1902897a3b4c0d1e",
  "path": "0.1",
  "title": "1.1 Motivation",
  "text": "本小节的完整正文...",
  "summary": "该节点的 LLM 摘要，若未开启 summarize 则为空字符串。",
  "children_count": 2,
  "children": ["0.1.0", "0.1.1"]
}
```

- **未知 tree_id / 路径不合法**:

```json
{ "error": "unknown tree_id: <id>" }
{ "error": "invalid path: <path>" }
```

#### GET /api/trees/{tree_id}/children/{path}

- **成功响应** (`200 application/json`):

```json
{
  "tree_id": "1902897a3b4c0d1e",
  "path": "0",
  "title": "1. Introduction",
  "children_count": 2,
  "children": [
    { "path": "0.0", "title": "1.1 Motivation", "summary": "...", "children_count": 0 },
    { "path": "0.1", "title": "1.2 Background", "summary": "...", "children_count": 1 }
  ]
}
```

#### GET /api/queries

- **响应** (`200 application/json`, 最多 200 条，最新在前):

```json
[
  {
    "tool": "inspect",
    "tree_id": "1902897a3b4c0d1e",
    "path": "0.1",
    "summary": "node: 1.1 Motivation (0 children)",
    "timestamp": "2026-06-12 14:31:08"
  },
  "..."
]
```

`summary` 规则：若结果包含 `error` 字段 → `error: <msg>`；若包含 `title` → `node: <title> (N children)`；若包含 `roots` → `tree: N nodes, depth D`；否则为 `ok`。

#### GET /api/queries/stats

- **响应** (`200 application/json`):

```json
{
  "total": 128,
  "by_tool": { "inspect": 50, "overview": 40, "get_children": 38 },
  "by_tree": { "1902897a3b4c0d1e": 118, "1902a9c37bb88d11": 10 },
  "recent": [ "<同 GET /api/queries 中每条的结构，最多 20 条>" ]
}
```

---

### 3. Chat API — 基于文档树的问答（流式）

| 方法 | 路径 | 说明 |
|------|------|------|
| POST | `/api/chat` | 向已构建的文档提问；支持 session 记忆，流式返回 NDJSON 事件 |
| GET | `/api/sessions` | 列出聊天会话（可按 `bid` 过滤） |
| GET | `/api/sessions/{sid}/turns` | 获取某会话的所有轮次 |
| DELETE | `/api/sessions/{sid}` | 删除会话及其轮次 |

#### POST /api/chat

- **请求**: `application/json`

```json
{
  "bid": "1902897a3b4c0d1e",
  "question": "这篇论文的核心结论是什么？",
  "model": "deepseek-chat",
  "session_id": "s_abc123"
}
```

字段说明：

| 字段 | 类型 | 必需 | 说明 |
|------|------|------|------|
| `bid` | string | 是 | 构建 ID |
| `question` | string | 是 | 用户问题 |
| `model` | string | 否 | 默认 `deepseek-chat`；`deepseek/...` 前缀走 DeepSeek provider，其余走 OpenAI 兼容 provider |
| `session_id` | string | 否 | 可选。提供时复用该 session 的历史上下文；省略时自动创建新 session |

- **响应**: `text/event-stream`（NDJSON，每行一条 JSON）

```
{"type": "start", "bid": "1902897a3b4c0d1e", "filename": "paper.pdf", "model": "deepseek-chat", "session_id": "s_abc123"}
{"type": "text", "text": "该论文提出了"}
{"type": "text", "text": "一种新的树结构索引方法"}
{"type": "tool_call", "id": "call_1", "name": "node_content", "arguments": "{\"path\": \"0\"}"}
{"type": "tool_result", "id": "call_1", "name": "node_content", "ok": true, "content": "..."}
{"type": "done", "answer": "该论文提出了一种新的树结构索引方法...", "turns": 2, "prompt_tokens": 1200, "completion_tokens": 340, "total_tokens": 1540}
```

事件类型：

| 类型 | 说明 |
|------|------|
| `start` | 会话开始；含 `bid`, `filename`, `model`, `session_id` |
| `text` | 答案文本增量 |
| `reasoning` | 模型内部思考（仅 reasoning 模型） |
| `tool_call` | 模型调用工具；含 `id`, `name`, `arguments` |
| `tool_result` | 工具执行结果；含 `id`, `name`, `ok`, `content` |
| `done` | 最终答案与用量统计 |
| `error` | 执行中断错误 |

**Session 机制**：首次提问时不填 `session_id`，后端会自动创建并在 `start` 事件中返回 `session_id`。后续追问带上同一个 `session_id`，agent 会自动加载之前的对话历史，实现多轮上下文记忆。历史存储在 SQLite 的 `chat_sessions` / `chat_turns` 表中。

#### GET /api/sessions

- **参数**: `bid` (string, 可选) 过滤；`limit` (integer, 可选, 默认 100)
- **响应** (`200 application/json`):

```json
{
  "sessions": [
    {
      "id": "s_abc123",
      "bid": "1902897a3b4c0d1e",
      "model": "deepseek-chat",
      "title": "这篇论文的核心结论是什么？",
      "turn_count": 4,
      "created_at": "2026-06-13T10:30:00Z",
      "updated_at": "2026-06-13T10:35:00Z"
    }
  ]
}
```

#### GET /api/sessions/{sid}/turns

- **参数**: `sid` (path, 必需); `limit` (query, 可选, 默认 200)
- **响应** (`200 application/json`):

```json
{
  "session_id": "s_abc123",
  "turns": [
    { "session_id": "s_abc123", "turn_idx": 0, "role": "user", "text": "...", "tool_calls": null, "tool_results": null, "created_at": "..." },
    { "session_id": "s_abc123", "turn_idx": 1, "role": "assistant", "text": "...", "tool_calls": "[{...}]", "tool_results": "[{...}]", "created_at": "..." }
  ]
}
```

#### DELETE /api/sessions/{sid}

- **响应** (`200 application/json`):

```json
{ "deleted": true, "session_id": "s_abc123" }
```

---

### 4. Base URL / CORS

- 默认监听 `0.0.0.0:8765`
- 统一前缀 `/api`；所有响应为 `application/json`
- CORS 默认允许任意源；生产环境可通过环境变量限制：
  `TREEFYIT_CORS_ORIGINS=http://localhost:3000,https://app.example.com`

### 5. OpenAPI / 文档界面

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/openapi.yaml` | 机器可读的 OpenAPI 3.0.3 规范（YAML） |
| GET | `/docs`         | 交互式 Swagger UI（jsDelivr CDN） |
| GET | `/redoc`        | 静态文档视图 Redoc（jsDelivr CDN） |

首次访问 `/docs` 或 `/redoc` 时，页面会请求 `/openapi.yaml` 加载定义，
因此 `openapi.yaml` 需要位于项目根目录（与 `main.py` 同级）。
Swagger UI 支持直接在浏览器里调用所有接口；`/redoc` 是只读文档视图。

embedding检索，在模糊的，语义等价场景很多的场景上面有效，
grep find 等等检索方法，在需求准确变量名，sdk的场景生效
然而在：文档多，但是信息杂乱，但是层级清楚，主要面向人类的场景，这些并不完全合适，这就是本项目的目标。我们希望做一个非RAG的知识库，受到 PageIndex 的启发，我们希望做一个人类和agent都友好的，检索增强生成框架。
