# Treefyit API

HTTP endpoints for building document trees and chatting with agents.

## Authentication

No auth required for local development. Production deployments should add a reverse proxy.

## Build

### POST /api/build

Upload a file and receive a nested tree with optional LLM summaries.

Request fields:

- `file` — PDF, Markdown, HTML, or plain text
- `mode` — `auto`, `md`, or `semantic`
- `summarize` — generate per-node summaries

### POST /api/build/stream

Same as `/api/build`, but streams NDJSON progress events.

## Chat

### POST /api/chat

Ask questions against a completed build using agent tools.

```json
{"bid": "abc123", "question": "Summarize chapter 3"}
```

## Errors

| Code | Meaning |
|------|---------|
| 400 | Invalid upload or parameters |
| 404 | Unknown build id |
| 500 | Parser or LLM failure |
