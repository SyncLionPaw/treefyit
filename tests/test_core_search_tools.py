from __future__ import annotations

from pathlib import Path

from core import TreeStore, build_library_search_tools, index_markdown


def test_index_markdown_and_search_tools_answer_question(tmp_path: Path):
    md = Path("examples/agent_tree/white_tea.md").resolve()
    store_dir = tmp_path / "lib"
    record = index_markdown(md, store_dir, tree_id="white-tea", title="White Tea Guide")
    assert record.tree_id == "white-tea"
    assert record.node_count >= 5

    tools = {
        tool.name: tool
        for tool in build_library_search_tools(TreeStore(store_dir), tree_id="white-tea")
    }

    listed = tools["list_saved_trees"].call({})
    assert listed.ok
    assert "white-tea" in listed.content

    hits = tools["search_document"].call(
        {"query": "water temperature brewing", "limit": 5}
    )
    assert hits.ok
    assert "Brewing" in hits.content or "85" in hits.content

    # Find a hit node id and inspect detail as a Q&A agent would.
    node_id = None
    for line in hits.content.splitlines():
        if " id=" in line:
            node_id = line.split(" id=", 1)[1].split(" ", 1)[0]
            break
    assert node_id
    detail = tools["view_detail"].call(
        {"node_id": node_id, "max_content_chars": 400}
    )
    assert detail.ok
    assert detail.content.strip()
