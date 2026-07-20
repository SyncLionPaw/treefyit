# Experiment 1 — Tree vs Flat for Q&A unlocking

Compare hierarchical **tree nodes** against **flat chunks** on the same markdown.

## Hypothesis

Tree structure helps Q&A because retrieval units are titled, coherent sections.
Flat windows mix neighboring content and introduce distractors.

## Run

```bash
uv run python experiments/tree_vs_flat/run.py
```

## Outputs

- `REPORT.md` — human-readable comparison
- `/opt/cursor/artifacts/exp_tree_vs_flat/report.json` — machine metrics
