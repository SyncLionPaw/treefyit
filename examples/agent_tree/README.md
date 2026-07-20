# Agent tree demo

1. **Builder agent** turns `white_tea.md` into a persisted tree.
2. **Q&A agents** use search tools over that library to answer questions.

## Files

- `white_tea.md` — source document for the builder agent
- `demo.py` — offline index + search tools, optional live agents

## Offline (no API key)

```bash
uv run python examples/agent_tree/demo.py
```

This indexes the markdown into `.tree-library/trees/white-tea.json`, then runs the same search tools a Q&A agent would use.

## Proof scripts (tool-level end-to-end)

```bash
uv run python examples/agent_tree/prove_flow.py
uv run python examples/agent_tree/prove_harness.py
```

These execute the real builder/search `FunctionTool`s (and local-backend `Runner.create`) without an LLM key, and write artifacts under `/opt/cursor/artifacts/agent_tree_proof/`.

## Live agents

```bash
export DEEPSEEK_API_KEY=...
uv run python examples/agent_tree/demo.py --live
```

- Builder: `open_runner(..., md_path=white_tea.md)` + tree CRUD tools
- Searcher: `open_search_runner(...)` with `search_document` / `view_detail`
