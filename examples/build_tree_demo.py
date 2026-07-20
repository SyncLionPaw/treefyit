"""Example: 用 ChatRunner 让 DeepSeek agent 建树，并演示对话持久化。

ChatRunner 的定位：持久化对话（jsonl 落盘）+ 自定义工具 + 不开沙箱。

运行前确保环境变量里有 key：
    export DEEPSEEK_API_KEY=sk-...
然后：
    uv run python examples/build_tree_demo.py
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from pagentv4 import Agent, ChatRunner, DeepSeek

from core.agent import build_tree_tools
from core.model import TreeStore, view_node

REPO_ROOT = Path(__file__).resolve().parent.parent
# 真实样例文档：一篇讲座方案 markdown。
MD_PATH = REPO_ROOT / "域名授权体系讲座方案.md"
# 产物固定落在工作目录的 .pagent/ 下（已被 gitignore），重复跑可续接对话。
PAGENT_DIR = REPO_ROOT / ".pagent"
THREAD_ROOT = PAGENT_DIR / "threads"  # 对话历史 jsonl
STORE_DIR = PAGENT_DIR / "store"  # 树库

EXTRA_SYSTEM = (
    "你用工具把 markdown 整理成一棵文档树。工具在本机进程内直接读写文件，"
    "路径就是本机真实路径，无需复制或上传。\n"
    "你可以自主决定怎么建树：先看内容、再定结构，"
    "该拆的拆、该并的并、该补 summary 的补，最后落盘。"
)


async def main() -> None:
    PAGENT_DIR.mkdir(parents=True, exist_ok=True)
    store = TreeStore(STORE_DIR)

    provider = DeepSeek("deepseek-v4-flash")
    session, tools = build_tree_tools(
        store=store,
        tree_id="domain-auth",
        source_md_path=MD_PATH,
    )

    # 第一次对话：建树。thread_id + root 固定，对话落盘到 THREAD_ROOT。
    runner = ChatRunner(
        Agent(provider, system=EXTRA_SYSTEM, tools=tools, max_turns=20),
        thread_id="domain-auth-chat",
        root=THREAD_ROOT,
    )
    print("=== 第一轮：建树 ===")
    prompt = (
        f"源 markdown 在 {MD_PATH}，请把它整理成一棵好用的文档树。\n"
        "目标：结构清晰、层级合理，每个节点都能一眼看懂讲什么。\n"
        "工具随你调度——seed_from_markdown 起骨架，view_outline / view_detail "
        "看内容，create_child / update_fields / relocate_node / delete_node "
        "调整结构和补 summary，search_working_tree 定位节点。\n"
        "满意后 save_tree 落盘，并说说你做了哪些整理。"
    )
    async for chunk in runner.run(prompt, return_type="text"):
        print(chunk, end="", flush=True)
    print()
    await runner.close()

    # 第二次对话：全新 runner，同一个 thread_id，验证历史被持久化并加载。
    # 只问不给建树工具，答案只能来自上一轮的对话记忆。
    resumed = ChatRunner(
        Agent(provider, system=EXTRA_SYSTEM, max_turns=4),
        thread_id="domain-auth-chat",
        root=THREAD_ROOT,
    )
    print("\n=== 第二轮：新 runner 续接同一对话 ===")
    print(f"（加载到 {len(resumed.messages)} 条历史消息）")
    async for chunk in resumed.run(
        "不要调用任何工具，仅凭我们刚才的对话回答：你建的树有哪几个二级标题？",
        return_type="text",
    ):
        print(chunk, end="", flush=True)
    print()
    await resumed.close()

    print("\n=== 最终树 ===")
    print(view_node(session.root, session.root.id, depth=3))
    print("\n=== 已保存 ===")
    for item in store.list():
        print(f"- {item['tree_id']}: {item['title']} (nodes={item['node_count']})")
    print("\n=== 对话落盘位置 ===")
    for path in sorted(THREAD_ROOT.rglob("*")):
        if path.is_file():
            print(f"- {path.relative_to(PAGENT_DIR)}")


if __name__ == "__main__":
    asyncio.run(main())
