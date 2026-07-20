"""Shared metrics for tree vs flat retrieval."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CaseResult:
    case_id: str
    question: str
    tree_hit: str
    flat_hit: str
    tree_evidence: str
    flat_evidence: str
    tree_recall: bool
    flat_recall: bool
    tree_precision_proxy: bool
    flat_precision_proxy: bool
    tree_noise: bool
    flat_noise: bool


def contains_all(text: str, needles: list[str]) -> bool:
    blob = text.casefold()
    return all(needle.casefold() in blob for needle in needles)


def contains_any(text: str, needles: list[str]) -> bool:
    blob = text.casefold()
    return any(needle.casefold() in blob for needle in needles)


def summarize(results: list[CaseResult]) -> dict[str, float]:
    n = len(results) or 1
    return {
        "n": float(len(results)),
        "tree_recall_at_1": sum(r.tree_recall for r in results) / n,
        "flat_recall_at_1": sum(r.flat_recall for r in results) / n,
        "tree_section_precision": sum(r.tree_precision_proxy for r in results) / n,
        "flat_section_precision": sum(r.flat_precision_proxy for r in results) / n,
        "tree_noise_rate": sum(r.tree_noise for r in results) / n,
        "flat_noise_rate": sum(r.flat_noise for r in results) / n,
    }
