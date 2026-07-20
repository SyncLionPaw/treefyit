# examples

## build_tree_demo.py

用 `ChatRunner` + DeepSeek 让 agent 把一篇 markdown 整理成文档树，并演示对话历史跨进程续接。

一次跑两轮：

1. 带建树工具，agent 自主 `seed_from_markdown` → 看内容 → 补 summary / 调结构 → `save_tree` 落盘。
2. 换一个新 `ChatRunner`，同一个 `thread_id`，不给工具，仅凭上一轮落盘的对话历史回答问题。

产物落在仓库根的 `.pagent/`（已 gitignore）：`store/` 存树库，`threads/` 存对话 jsonl。重复跑会续接同一对话。

### 运行

`pagentv4` 只读进程环境变量，先把 key export 进去：

```bash
export DEEPSEEK_API_KEY=sk-...
uv run python examples/build_tree_demo.py
```

想验证续接，跑第二次即可——第二轮会打印加载到的历史消息条数。
