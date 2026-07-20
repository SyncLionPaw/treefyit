"""Example: 用 ChatRunner 让 DeepSeek agent 建树，并演示对话持久化。

ChatRunner 的定位：持久化对话（jsonl 落盘）+ 自定义工具 + 不开沙箱。

运行前确保环境变量里有 key：
    export DEEPSEEK_API_KEY=sk-...
然后：
    uv run python examples/build_tree_demo.py
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path

from pagentv4 import Agent, ChatRunner, DeepSeek

from core.agent import build_tree_tools
from core.model import TreeStore, view_node

SAMPLE_MD = """# TreefyIt 快速上手

TreefyIt 把 markdown 文档转成可检索的节点树。

## 安装

用 uv 安装依赖：

    uv sync

## 建树

调用 build_tree_tools 拿到工具，交给 Runner，agent 会自己建树。

### 从 markdown 起步

先用 seed_from_markdown 生成骨架，再细化。

### 手工建

也可以用 create_child 一个个挂节点。

## 检索

用 build_search_tools 在树里做只读问答。
"""

EXTRA_SYSTEM = (
    "你用工具把 markdown 建成文档树。工具在本机进程内直接读写文件，"
    "路径就是本机真实路径，无需复制或上传。\n"
    "流程：1) seed_from_markdown 生成骨架；"
    "2) update_fields 给节点补 summary；3) save_tree 落盘。"
)


async def run_turn(runner: ChatRunner, prompt: str) -> None:
    async for chunk in runner.run(prompt, return_type="text"):
        print(chunk, end="", flush=True)
    print()


async def main() -> None:
    workdir = Path(tempfile.mkdtemp(prefix="treefyit-demo-"))
    thread_root = workdir / "threads"
    md_path = workdir / "guide.md"
    md_path.write_text(SAMPLE_MD, encoding="utf-8")
    store = TreeStore(workdir / "store")

    provider = DeepSeek("deepseek-v4-flash")
    session, tools = build_tree_tools(
        store=store,
        tree_id="guide",
        source_md_path=md_path,
    )

    # 第一次对话：建树。thread_id + root 固定，对话落盘到 thread_root。
    runner = ChatRunner(
        Agent(provider, system=EXTRA_SYSTEM, tools=tools, max_turns=12),
        thread_id="guide-chat",
        root=thread_root,
    )
    print("=== 第一轮：建树 ===")
    await run_turn(
        runner,
        f"源 markdown 在 {md_path}。"
        "请调用 seed_from_markdown 建骨架，为每个二级标题补一句 summary，"
        "最后调用 save_tree 落盘。完成后简述你建了什么。",
    )
    await runner.close()

    # 第二次对话：全新 runner，同一个 thread_id，验证历史被持久化并加载。
    # 只问不给建树工具，答案只能来自上一轮的对话记忆。
    resumed = ChatRunner(
        Agent(provider, system=EXTRA_SYSTEM, max_turns=4),
        thread_id="guide-chat",
        root=thread_root,
    )
    print("\n=== 第二轮：新 runner 续接同一对话 ===")
    print(f"（加载到 {len(resumed.messages)} 条历史消息）")
    await run_turn(
        resumed,
        "不要调用任何工具，仅凭我们刚才的对话回答：你建的树有哪几个二级标题？",
    )
    await resumed.close()

    print("\n=== 最终树 ===")
    print(view_node(session.root, session.root.id, depth=3))
    print("\n=== 已保存 ===")
    for item in store.list():
        print(f"- {item['tree_id']}: {item['title']} (nodes={item['node_count']})")
    print("\n=== 对话落盘位置 ===")
    for path in sorted(thread_root.rglob("*")):
        if path.is_file():
            print(f"- {path.relative_to(workdir)}")


if __name__ == "__main__":
    asyncio.run(main())
