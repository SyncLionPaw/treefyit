"""Experiment 1: tree structure vs flat chunks for Q&A unlocking.

Hypothesis: hierarchical tree nodes + branch unlock help retrieve cleaner
answer evidence than flat overlapping chunks of the same markdown.

Offline metrics (no LLM required):
- recall@1: top evidence contains all gold answer keys
- section precision: tree top title matches gold section
- noise rate: top evidence contains distractor phrases

Usage:
  uv run python experiments/tree_vs_flat/run.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core import index_markdown  # noqa: E402
from experiments.tree_vs_flat.eval import (  # noqa: E402
    CaseResult,
    contains_all,
    contains_any,
    summarize,
)
from experiments.tree_vs_flat.flat import chunk_markdown, search_flat  # noqa: E402
from experiments.tree_vs_flat.tree_rank import unlock_tree_evidence  # noqa: E402

HERE = Path(__file__).resolve().parent
MD_PATH = ROOT / "examples" / "agent_tree" / "white_tea.md"
CASES_PATH = HERE / "cases.json"
ARTIFACT_DIR = Path("/opt/cursor/artifacts/exp_tree_vs_flat")


def load_cases() -> list[dict]:
    return json.loads(CASES_PATH.read_text(encoding="utf-8"))


def run() -> dict:
    cases = load_cases()
    store_dir = HERE / ".exp-store"
    record = index_markdown(
        MD_PATH,
        store_dir,
        tree_id="white-tea",
        title="White Tea Guide",
    )
    root = record.root
    md_text = MD_PATH.read_text(encoding="utf-8")
    # Larger flat windows intentionally mix adjacent sections (common RAG baseline).
    chunks = chunk_markdown(md_text, max_chars=420, overlap=80)

    results: list[CaseResult] = []
    rows: list[dict] = []

    for case in cases:
        question = case["question"]
        must = case["must_contain"]
        gold_titles = case.get("gold_tree_titles", [])
        avoid = case.get("avoid_in_top1_if_possible", [])

        tree_best = unlock_tree_evidence(root, question)
        flat_hits = search_flat(chunks, question, limit=5)

        if tree_best is not None:
            tree_evidence = tree_best.evidence
            tree_title = tree_best.title
            tree_hit = f"{tree_best.node_id}:{tree_best.title}"
        else:
            tree_evidence = ""
            tree_title = ""
            tree_hit = "none"

        flat_evidence = flat_hits[0].text if flat_hits else ""
        flat_id = flat_hits[0].chunk_id if flat_hits else "none"

        tree_recall = contains_all(tree_evidence, must)
        flat_recall = contains_all(flat_evidence, must)
        tree_precision = bool(gold_titles) and tree_title in gold_titles
        flat_precision = flat_recall and not contains_any(flat_evidence, avoid)
        tree_noise = contains_any(tree_evidence, avoid)
        flat_noise = contains_any(flat_evidence, avoid)

        result = CaseResult(
            case_id=case["id"],
            question=question,
            tree_hit=tree_hit,
            flat_hit=flat_id,
            tree_evidence=tree_evidence,
            flat_evidence=flat_evidence,
            tree_recall=tree_recall,
            flat_recall=flat_recall,
            tree_precision_proxy=tree_precision,
            flat_precision_proxy=flat_precision,
            tree_noise=tree_noise,
            flat_noise=flat_noise,
        )
        results.append(result)
        rows.append(
            {
                "id": case["id"],
                "question": question,
                "tree_top": result.tree_hit,
                "flat_top": result.flat_hit,
                "tree_recall@1": tree_recall,
                "flat_recall@1": flat_recall,
                "tree_section_precision": tree_precision,
                "flat_clean_precision": flat_precision,
                "tree_noise": tree_noise,
                "flat_noise": flat_noise,
                "must_contain": must,
                "tree_evidence_preview": " ".join(tree_evidence.split())[:220],
                "flat_evidence_preview": " ".join(flat_evidence.split())[:220],
            }
        )

    metrics = summarize(results)
    winner = (
        "tree"
        if (
            metrics["tree_recall_at_1"],
            -metrics["tree_noise_rate"],
            metrics["tree_section_precision"],
        )
        > (
            metrics["flat_recall_at_1"],
            -metrics["flat_noise_rate"],
            metrics["flat_section_precision"],
        )
        else "flat"
    )

    report = {
        "experiment": "tree_vs_flat",
        "document": str(MD_PATH),
        "tree_nodes": record.node_count,
        "flat_chunks": len(chunks),
        "method": {
            "tree": "content-first ranking + hierarchical branch unlock",
            "flat": "heading-stripped overlapping paragraph windows",
        },
        "metrics": metrics,
        "winner": winner,
        "cases": rows,
    }
    return report


def render_markdown(report: dict) -> str:
    m = report["metrics"]
    lines = [
        "# Experiment 1: Tree vs Flat for Q&A unlocking",
        "",
        f"Document: `{report['document']}`",
        f"Tree nodes: **{report['tree_nodes']}** · Flat chunks: **{report['flat_chunks']}**",
        "",
        f"- Tree method: {report['method']['tree']}",
        f"- Flat method: {report['method']['flat']}",
        "",
        "## Aggregate metrics",
        "",
        "| metric | tree | flat |",
        "|---|---:|---:|",
        f"| recall@1 (answer keys in top evidence) | {m['tree_recall_at_1']:.0%} | {m['flat_recall_at_1']:.0%} |",
        f"| section/clean precision proxy | {m['tree_section_precision']:.0%} | {m['flat_section_precision']:.0%} |",
        f"| noise rate (distractors in top evidence) | {m['tree_noise_rate']:.0%} | {m['flat_noise_rate']:.0%} |",
        "",
        f"**Winner:** `{report['winner']}`",
        "",
        "## Per-question",
        "",
    ]
    for case in report["cases"]:
        lines.extend(
            [
                f"### {case['id']}",
                f"- Q: {case['question']}",
                f"- Tree top: `{case['tree_top']}` · recall={case['tree_recall@1']} · "
                f"section_precision={case['tree_section_precision']} · noise={case['tree_noise']}",
                f"- Flat top: `{case['flat_top']}` · recall={case['flat_recall@1']} · "
                f"clean_precision={case['flat_clean_precision']} · noise={case['flat_noise']}",
                f"- Tree evidence: {case['tree_evidence_preview']}",
                f"- Flat evidence: {case['flat_evidence_preview']}",
                "",
            ]
        )
    lines.extend(
        [
            "## Interpretation",
            "",
            "Tree Q&A unlocking ranks content-bearing sections and can refine inside a",
            "branch, so the top evidence is a titled unit (e.g. Brewing Advice / Storage).",
            "Flat windows often glue neighboring sections together, so even when answer",
            "keys appear, distractors remain in the same evidence block.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    report = run()
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    (ARTIFACT_DIR / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    md = render_markdown(report)
    (ARTIFACT_DIR / "REPORT.md").write_text(md, encoding="utf-8")
    (HERE / "REPORT.md").write_text(md, encoding="utf-8")

    print(md)
    print(f"\nartifacts -> {ARTIFACT_DIR}")

    metrics = report["metrics"]
    assert report["winner"] == "tree", metrics
    assert metrics["tree_recall_at_1"] >= metrics["flat_recall_at_1"], metrics
    assert metrics["tree_noise_rate"] < metrics["flat_noise_rate"], metrics
    assert metrics["tree_section_precision"] > metrics["flat_section_precision"], metrics
    print("\nPASS: tree unlocks cleaner Q&A evidence than flat chunks")


if __name__ == "__main__":
    main()
