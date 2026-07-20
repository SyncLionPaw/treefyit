from __future__ import annotations

from pathlib import Path

from experiments.tree_vs_flat.flat import chunk_markdown, search_flat
from experiments.tree_vs_flat.run import run


def test_flat_chunker_strips_heading_markers_and_returns_chunks():
    text = Path("examples/agent_tree/white_tea.md").read_text(encoding="utf-8")
    chunks = chunk_markdown(text, max_chars=220, overlap=30)
    assert len(chunks) >= 3
    assert all(not line.startswith("#") for c in chunks for line in c.text.splitlines())


def test_tree_beats_flat_on_qa_unlock_benchmark():
    report = run()
    metrics = report["metrics"]
    assert report["winner"] == "tree"
    assert metrics["tree_recall_at_1"] >= metrics["flat_recall_at_1"]
    assert metrics["tree_noise_rate"] < metrics["flat_noise_rate"]
    assert metrics["tree_section_precision"] > metrics["flat_section_precision"]

    # Spot-check temperature question: tree should land on Brewing Advice.
    temp = next(case for case in report["cases"] if case["id"] == "q_temp")
    assert temp["tree_recall@1"] is True
    assert "Brewing Advice" in temp["tree_top"]


def test_search_flat_finds_temperature_keywords():
    text = Path("examples/agent_tree/white_tea.md").read_text(encoding="utf-8")
    chunks = chunk_markdown(text, max_chars=420, overlap=80)
    hits = search_flat(chunks, "85 90 brewing water", limit=5)
    assert hits
    assert any("85" in hit.text for hit in hits)
