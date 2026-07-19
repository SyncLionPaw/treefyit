"""Builder agent indexes a markdown file; Q&A agents search that tree.

Offline path (no API key):
  uv run python examples/agent_tree/demo.py

With DeepSeek (optional live agents):
  export DEEPSEEK_API_KEY=...
  uv run python examples/agent_tree/demo.py --live
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
from pathlib import Path

from core import (
    TreeStore,
    build_library_search_tools,
    index_markdown,
    open_runner,
    open_search_runner,
)

ROOT = Path(__file__).resolve().parent
MD_PATH = ROOT / "white_tea.md"
DEFAULT_STORE = ROOT / ".tree-library"
TREE_ID = "white-tea"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--store-dir",
        type=Path,
        default=DEFAULT_STORE,
        help="Persistent tree library directory",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Run builder + search agents with DeepSeek (needs DEEPSEEK_API_KEY)",
    )
    parser.add_argument(
        "--question",
        default="What water temperature should I use for white tea?",
        help="Question for the search/Q&A agent",
    )
    return parser.parse_args()


def offline_demo(store_dir: Path, question: str) -> None:
    record = index_markdown(MD_PATH, store_dir, tree_id=TREE_ID, title="White Tea Guide")
    print(f"indexed markdown -> tree_id={record.tree_id} nodes={record.node_count}")
    print(f"store: {store_dir / 'trees' / (record.tree_id + '.json')}")

    tools = {
        tool.name: tool
        for tool in build_library_search_tools(TreeStore(store_dir), tree_id=TREE_ID)
    }
    print("\nQ&A agent tools:", ", ".join(sorted(tools)))

    hits = tools["search_document"].call({"query": question, "limit": 3})
    print(f"\nquestion: {question}")
    print(hits.content)
    if hits.ok and "no matches" not in hits.content:
        # pull first node id from "id=..."
        for line in hits.content.splitlines():
            if " id=" not in line:
                continue
            node_id = line.split(" id=", 1)[1].split(" ", 1)[0]
            detail = tools["view_detail"].call(
                {"node_id": node_id, "max_content_chars": 400}
            )
            print("\nevidence:")
            print(detail.content)
            break


async def live_demo(store_dir: Path, question: str) -> None:
    from pagentv4 import DeepSeek, TextDelta, ToolCallBegin, ToolResult

    if not os.getenv("DEEPSEEK_API_KEY"):
        raise SystemExit("Set DEEPSEEK_API_KEY for --live")

    provider = DeepSeek("deepseek-chat")

    print("== builder agent: create tree from markdown ==")
    _session, builder = await open_runner(
        "builder-white-tea",
        provider,
        store_dir=store_dir,
        md_path=MD_PATH,
        tree_id=TREE_ID,
    )
    try:
        async for event in builder.run(
            "Read /home/agent/source.md, call seed_from_markdown, "
            "then save_tree with tree_id='white-tea' and title='White Tea Guide'."
        ):
            if isinstance(event, ToolCallBegin):
                print(f"tool → {event.name}")
            elif isinstance(event, ToolResult):
                preview = event.content.replace("\n", " ")
                print(f"  {'ok' if event.ok else 'fail'}: {preview[:160]}")
            elif isinstance(event, TextDelta):
                sys.stdout.write(event.text)
                sys.stdout.flush()
        print()
    finally:
        await builder.close()

    print("\n== search agent: answer from persisted tree ==")
    searcher = await open_search_runner(
        "qa-white-tea",
        provider,
        store_dir=store_dir,
        tree_id=TREE_ID,
    )
    try:
        async for event in searcher.run(question):
            if isinstance(event, ToolCallBegin):
                print(f"tool → {event.name}({event.arguments})")
            elif isinstance(event, ToolResult):
                preview = event.content.replace("\n", " ")
                print(f"  {'ok' if event.ok else 'fail'}: {preview[:200]}")
            elif isinstance(event, TextDelta):
                sys.stdout.write(event.text)
                sys.stdout.flush()
        print()
    finally:
        await searcher.close()


def main() -> None:
    args = parse_args()
    args.store_dir.mkdir(parents=True, exist_ok=True)
    if args.live:
        asyncio.run(live_demo(args.store_dir, args.question))
    else:
        offline_demo(args.store_dir, args.question)


if __name__ == "__main__":
    main()
