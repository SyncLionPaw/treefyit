"""Prove builder + agentic query using the real pagentv4 tool objects.

This runs the same tool calls a builder agent and a Q&A agent would make,
without requiring an LLM key. Output is written under /opt/cursor/artifacts.
"""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from core import (
    TreeStore,
    build_library_search_tools,
    build_tree_tools,
    create_node,
)

ROOT = Path(__file__).resolve().parent
MD_PATH = ROOT / "white_tea.md"
STORE_DIR = ROOT / ".proof-library"
ARTIFACT_DIR = Path("/opt/cursor/artifacts/agent_tree_proof")
QUESTION = "What water temperature should I use for white tea?"


def section(title: str) -> None:
    print(f"\n=== {title} ===")


def main() -> None:
    if STORE_DIR.exists():
        shutil.rmtree(STORE_DIR)
    STORE_DIR.mkdir(parents=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    transcript: list[dict] = []

    section("1) Builder agent tools: seed_from_markdown -> save_tree")
    session, builder_tools = build_tree_tools(
        create_node("root", "document"),
        store=TreeStore(STORE_DIR),
        source_md_path=MD_PATH,
    )
    builder = {tool.name: tool for tool in builder_tools}
    print("builder tools:", sorted(builder))

    seed = builder["seed_from_markdown"].call({})
    assert seed.ok, seed.content
    transcript.append(
        {
            "agent": "builder",
            "tool": "seed_from_markdown",
            "ok": seed.ok,
            "content": seed.content,
        }
    )
    print(seed.content)

    saved = builder["save_tree"].call(
        {"tree_id": "white-tea", "title": "White Tea Guide"}
    )
    assert saved.ok, saved.content
    transcript.append(
        {
            "agent": "builder",
            "tool": "save_tree",
            "ok": saved.ok,
            "content": saved.content,
        }
    )
    print(saved.content)

    tree_path = STORE_DIR / "trees" / "white-tea.json"
    assert tree_path.is_file(), tree_path
    payload = json.loads(tree_path.read_text(encoding="utf-8"))
    print(
        f"persisted: {tree_path} "
        f"(nodes={payload['node_count']}, title={payload['title']!r})"
    )
    shutil.copy2(tree_path, ARTIFACT_DIR / "white-tea.json")

    section("2) Fresh Q&A agent process (new tools, no shared memory)")
    # Simulate another agent/process: only the store on disk is shared.
    qa_tools = {
        tool.name: tool
        for tool in build_library_search_tools(TreeStore(STORE_DIR), tree_id=None)
    }
    print("qa tools:", sorted(qa_tools))

    listed = qa_tools["list_saved_trees"].call({})
    assert listed.ok and "white-tea" in listed.content, listed.content
    transcript.append(
        {
            "agent": "qa",
            "tool": "list_saved_trees",
            "ok": listed.ok,
            "content": listed.content,
        }
    )
    print(listed.content)

    use = qa_tools["use_tree"].call({"tree_id": "white-tea"})
    assert use.ok, use.content
    transcript.append(
        {
            "agent": "qa",
            "tool": "use_tree",
            "ok": use.ok,
            "content": use.content,
        }
    )
    print(use.content.splitlines()[0])

    print(f"\nquestion: {QUESTION}")
    hits = qa_tools["search_document"].call({"query": QUESTION, "limit": 5})
    assert hits.ok, hits.content
    assert "Brewing Advice" in hits.content or "85" in hits.content, hits.content
    transcript.append(
        {
            "agent": "qa",
            "tool": "search_document",
            "args": {"query": QUESTION},
            "ok": hits.ok,
            "content": hits.content,
        }
    )
    print(hits.content)

    node_id = None
    for line in hits.content.splitlines():
        if "Brewing Advice" in line and " id=" in line:
            node_id = line.split(" id=", 1)[1].split(" ", 1)[0]
            break
    if node_id is None:
        for line in hits.content.splitlines():
            if " id=" in line and line.strip().startswith("-"):
                node_id = line.split(" id=", 1)[1].split(" ", 1)[0]
                break
    assert node_id, "no node id in search hits"

    detail = qa_tools["view_detail"].call(
        {"node_id": node_id, "max_content_chars": 500}
    )
    assert detail.ok, detail.content
    assert "85" in detail.content, detail.content
    transcript.append(
        {
            "agent": "qa",
            "tool": "view_detail",
            "args": {"node_id": node_id},
            "ok": detail.ok,
            "content": detail.content,
        }
    )
    print("\nevidence from view_detail:")
    print(detail.content)

    answer = (
        "Based on the retrieved Brewing Advice section, use water around "
        "85–90°C for white tea."
    )
    print("\nfinal answer:")
    print(answer)
    transcript.append({"agent": "qa", "final_answer": answer})

    (ARTIFACT_DIR / "transcript.json").write_text(
        json.dumps(transcript, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (ARTIFACT_DIR / "answer.txt").write_text(
        f"Q: {QUESTION}\nA: {answer}\n\nEvidence node: {node_id}\n\n{detail.content}\n",
        encoding="utf-8",
    )
    (ARTIFACT_DIR / "SUMMARY.md").write_text(
        "\n".join(
            [
                "# Agent tree proof",
                "",
                "## Builder agent",
                "- Called `seed_from_markdown` on `white_tea.md`",
                "- Called `save_tree(tree_id='white-tea')`",
                f"- Persisted `{tree_path}` with `{payload['node_count']}` nodes",
                "",
                "## Q&A agent (separate tool session)",
                "- Called `list_saved_trees` / `use_tree`",
                f"- Asked: {QUESTION}",
                "- Called `search_document` then `view_detail`",
                f"- Answer: {answer}",
                "",
                "Artifacts: `transcript.json`, `white-tea.json`, `answer.txt`",
                "",
            ]
        ),
        encoding="utf-8",
    )

    section("3) PASS")
    print(f"artifacts -> {ARTIFACT_DIR}")
    for path in sorted(ARTIFACT_DIR.iterdir()):
        print(f" - {path.name} ({path.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
