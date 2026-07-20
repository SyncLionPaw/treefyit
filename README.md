# treefyit

把 markdown 文档整理成一棵可检索的节点树。

`core` 提供数据结构、确定性建树、以及给 LLM agent 用的工具，让 agent 自主地把长文档拆解成结构清晰、带摘要的树，落盘后可再次加载和检索。

## 安装

用 [uv](https://github.com/astral-sh/uv) 管理依赖：

```bash
uv sync
```

## core 三层

依赖单向，上层依赖下层：

- `core.model`：数据结构与持久化。`TreeNode`（id / title / kind / content / summary / children）、纯函数节点操作（`create_node` / `mount_node` / `move_node` / `update_node` …）、`TreeStore`（一棵树一个 JSON 文件）。
- `core.build`：确定性构建。`markdown_to_tree` 按标题层级搭骨架；`search_tree` / `search_store` 做关键字检索。不需要 LLM。
- `core.agent`：给 LLM 的工具。`build_tree_tools` 返回一套建树 / 编辑 / 持久化工具（11 个），`build_search_tools` 返回只读问答工具（3 个）。

顶层 `from core import ...` 重导出常用 API，保持一处引用。

## 建树工具

`build_tree_tools` 返回 `(session, tools)`：`tools` 交给 agent，`session.root` 是随工具调用实时增长的活树。

```python
from core.agent import build_tree_tools
from core.model import TreeStore

store = TreeStore(".pagent/store")
session, tools = build_tree_tools(
    store=store,
    tree_id="domain-auth",
    source_md_path="域名授权体系讲座方案.md",
)
```

不传 `root` 时会造一个空根，agent 用 `seed_from_markdown` 从文档起骨架，或用 `create_child` 逐节点手工建。工具覆盖查看（`view_outline` / `view_detail`）、编辑（`create_child` / `update_fields` / `relocate_node` / `delete_node`）、检索（`search_working_tree`）、持久化（`save_tree` / `load_tree` / `list_saved_trees`）。

## 检索工具

`build_search_tools` 吃一棵 `root`，返回只读的问答工具，agent 只能查不能改：

```python
from core.agent import build_search_tools
from core.model import TreeStore

record = TreeStore(".pagent/store").load("domain-auth")
tools = build_search_tools(record.root, tree_id=record.tree_id)
```

## 不用 LLM 建树

只想拿确定性骨架时，直接调 `markdown_to_tree`：

```python
from pathlib import Path
from core.build import markdown_to_tree

root = markdown_to_tree(Path("域名授权体系讲座方案.md").read_text(encoding="utf-8"))
print(root.outline(depth=2))
```

## 示例

见 [examples/](examples/)。`build_tree_demo.py` 用 `ChatRunner` + DeepSeek 演示 agent 自主建树，并验证对话历史跨进程续接。
