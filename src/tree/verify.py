"""LLM-based tree verification — port of PageIndex's verify_toc layer.

The idea: after the mechanical build (headers + numbering + thinning) we ask an
LLM to sanity-check the tree structure.  If the LLM says "this looks wrong" we
can fall back to semantic extraction instead of serving a broken tree to the
agent tools.
"""

from __future__ import annotations

import json
import logging

from src.tree.builder import flatten_tree
from src.llm import achat

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Rule-based pre-checks (cheap, no LLM call)
# ---------------------------------------------------------------------------


def _rule_precheck(tree: list[dict]) -> tuple[bool, str]:
    """Return (ok, reason) after cheap heuristic checks.

    If any check fails we can skip the LLM call and immediately flag the tree
    as suspicious.
    """
    if not tree:
        return False, "empty tree"

    flat = flatten_tree(tree)

    # All nodes are level-1 (flat) → suspicious unless the doc is tiny.
    if len(flat) > 3 and all("children" not in n for n in flat):
        return False, "all nodes are flat (no hierarchy inferred)"

    # Too many root nodes for a normal document.
    if len(tree) > 20:
        return False, f"too many root nodes ({len(tree)})"

    # Too many total nodes → probably over-fragmented.
    if len(flat) > 200:
        return False, f"too many total nodes ({len(flat)})"

    return True, ""


# ---------------------------------------------------------------------------
# LLM structure validation
# ---------------------------------------------------------------------------


async def verify_tree(
    tree: list[dict],
    model: str = "gpt-4o-mini",
    sample_nodes: int = 10,
) -> dict:
    """Validate the built tree with an LLM.

    Returns a dict:
        {
            "ok": bool,
            "score": float,        # 0.0 – 1.0  estimated quality
            "issues": [str],       # human-readable problems
            "suspicious_nodes": [   # nodes whose title does not match text
                {"path": str, "title": str, "reason": str}
            ],
        }

    The validation is two-phase:
    1. Structural sanity — does the outline make sense as a document?
    2. Title-matching spot-check — do a sample of node titles actually appear
       near the start of their text blocks?
    """
    # Phase 0 — cheap rules
    ok, reason = _rule_precheck(tree)
    if not ok:
        return {
            "ok": False,
            "score": 0.0,
            "issues": [reason],
            "suspicious_nodes": [],
        }

    flat = flatten_tree(tree)

    # Phase 1 — structural sanity (single LLM call)
    outline = _render_outline(tree)
    struct_prompt = (
        "You are reviewing the automatically-extracted outline of a document.\n\n"
        "Outline:\n" + outline + "\n\n"
        "Evaluate the outline and return JSON:\n"
        '{"valid": bool, "issues": ["..."], "score": float 0-1}\n'
        "Score 1.0 = perfect academic/technical document structure. "
        "Score 0.0 = completely broken / nonsensical.\n"
        "Return only the JSON."
    )

    try:
        raw = await achat(struct_prompt, model=model)
        struct_result = _extract_json(raw)
    except Exception as exc:  # noqa: BLE001
        logger.warning("LLM structure validation failed: %s", exc)
        struct_result = {"valid": True, "issues": [], "score": 1.0}

    # Phase 2 — spot-check titles against text (sample nodes, concurrent)
    sample = _pick_sample(flat, sample_nodes)
    suspicious: list[dict] = []
    for node in sample:
        match = _title_appears_near_start(node)
        if not match:
            suspicious.append(
                {
                    "path": _node_path(node),
                    "title": node["title"],
                    "reason": "title not found near start of text",
                }
            )

    issues: list[str] = struct_result.get("issues", [])
    if suspicious:
        issues.append(
            f"{len(suspicious)} sampled nodes have titles that do not match their text"
        )

    score = float(struct_result.get("score", 1.0))
    if suspicious:
        score = max(0.0, score - len(suspicious) * 0.1)

    ok = bool(struct_result.get("valid", True)) and len(suspicious) <= sample_nodes // 2

    return {
        "ok": ok,
        "score": round(score, 2),
        "issues": issues,
        "suspicious_nodes": suspicious,
    }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _render_outline(nodes: list[dict], depth: int = 0) -> str:
    """Render a compact outline for the LLM prompt."""
    lines: list[str] = []
    indent = "  " * depth
    for n in nodes:
        title = n.get("title", "")
        child_count = len(n.get("children", []))
        lines.append(f"{indent}- {title} ({child_count} sub-sections)")
        lines.append(_render_outline(n.get("children", []), depth + 1))
    return "\n".join(lines)


def _pick_sample(flat: list[dict], k: int) -> list[dict]:
    """Pick up to *k* nodes that have non-empty text for spot-checking."""
    candidates = [n for n in flat if n.get("text", "").strip()]
    if len(candidates) <= k:
        return candidates
    # Spread the sample across the document: pick first, last, and evenly
    # spaced in between.
    step = len(candidates) // k
    return [candidates[i * step] for i in range(k)]


def _title_appears_near_start(node: dict, max_chars: int = 500) -> bool:
    """Cheap fuzzy check: does the node title appear in the first *max_chars*
    characters of its text?"""
    title = node.get("title", "").strip()
    text = node.get("text", "").strip()
    if not title or not text:
        return True  # nothing to check
    head = text[:max_chars].lower()
    # Strip common prefixes like "1.1 " or "Appendix A: "
    clean_title = title.lower()
    for prefix in ("appendix ", "chapter ", "section ", "part "):
        if clean_title.startswith(prefix):
            clean_title = clean_title[len(prefix) :]
    # Simple substring check — good enough for a spot-check.
    return clean_title in head or any(
        part in head for part in clean_title.split() if len(part) > 3
    )


def _node_path(node: dict) -> str:
    """Return a human-readable path using node_id if present."""
    return node.get("node_id", node.get("title", "?"))


def _extract_json(text: str) -> dict:
    """Robust JSON extraction from an LLM response."""
    text = text.strip()
    if text.startswith("```json"):
        text = text[7:]
    if text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Fallback: find the first { and last }
        start = text.find("{")
        end = text.rfind("}")
        if start != -1 and end != -1:
            return json.loads(text[start : end + 1])
        raise
