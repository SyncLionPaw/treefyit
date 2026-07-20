"""Prove desktop-aligned Runner harness stages MD and exposes agent tools."""

from __future__ import annotations

import asyncio
import json
import shutil
from pathlib import Path

from core import open_runner, open_search_runner
from pagentv4 import DeepSeek

ROOT = Path(__file__).resolve().parent
MD_PATH = ROOT / "white_tea.md"
STORE_DIR = ROOT / ".proof-harness-library"
ARTIFACT_DIR = Path("/opt/cursor/artifacts/agent_tree_proof")


async def main() -> None:
    if STORE_DIR.exists():
        shutil.rmtree(STORE_DIR)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    # Never call the model; only open runners and invoke tools directly.
    provider = DeepSeek("deepseek-chat", apikey="proof-unused")

    print("=== open builder runner (local backend) ===")
    session, builder = await open_runner(
        "proof-builder",
        provider,
        store_dir=STORE_DIR,
        md_path=MD_PATH,
    )
    builder_tool_names: list[str] = []
    try:
        source = await builder.sandbox.files.read_text("source.md")
        assert "White Tea" in source or "white tea" in source.lower()
        print(f"sandbox source.md chars={len(source)}")
        builder_tool_names = [tool.name for tool in builder.agent.tools]
        print(f"builder tools={builder_tool_names}")
        assert "seed_from_markdown" in builder_tool_names
        assert "save_tree" in builder_tool_names

        tools = {tool.name: tool for tool in builder.agent.tools}
        seed = await tools["seed_from_markdown"].acall({})
        assert seed.ok, seed.content
        print("seed_from_markdown ok")
        saved = await tools["save_tree"].acall(
            {"tree_id": "white-tea", "title": "White Tea Guide"}
        )
        assert saved.ok, saved.content
        print(saved.content)
        assert session.tree_id == "white-tea"
    finally:
        await builder.close()

    tree_path = STORE_DIR / "trees" / "white-tea.json"
    assert tree_path.is_file()
    print(f"persisted via harness: {tree_path}")

    print("\n=== open search runner (separate agent) ===")
    searcher = await open_search_runner(
        "proof-qa",
        provider,
        store_dir=STORE_DIR,
        tree_id="white-tea",
    )
    try:
        qa_tools = {tool.name: tool for tool in searcher.agent.tools}
        print(f"qa tools={sorted(qa_tools)}")
        assert "create_child" not in qa_tools
        assert "search_document" in qa_tools

        hits = await qa_tools["search_document"].acall(
            {
                "query": "What water temperature should I use for white tea?",
                "limit": 5,
            }
        )
        assert hits.ok, hits.content
        assert "Brewing Advice" in hits.content or "85" in hits.content
        print(hits.content)

        proof = {
            "builder_tools": builder_tool_names,
            "qa_tools": sorted(qa_tools),
            "search_hits": hits.content,
            "tree_path": str(tree_path),
            "source_md_staged": True,
            "builder_session_tree_id": "white-tea",
        }
        (ARTIFACT_DIR / "harness_proof.json").write_text(
            json.dumps(proof, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print("\nPASS harness builder + search runners")
    finally:
        await searcher.close()


if __name__ == "__main__":
    asyncio.run(main())
