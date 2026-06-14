"""Tree builder — one call to parse, build, summarize, and visualize.

Tree-thinning logic ported from PageIndex: small leaf nodes (below a token
threshold) are merged into their parent so the agent does not drown in
micro-sections.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

from src.llm import achat, count_tokens
from src.parser.md import parse_md
from src.vis.tree_view import show as _show

SUMMARY_THRESHOLD = 200  # tokens: if text is shorter, use text as summary
THINNING_THRESHOLD = 5000  # tokens: merge children into parent when smaller
VERIFY_SAMPLE_NODES = 10  # how many nodes to spot-check per build
AUTO_SEMANTIC_MAX_TOKENS = 40_000


def build_tree(
    source: str | Path | list[dict],
    *,
    model: str = "gpt-4o",
    summarize: bool = True,
    mode: str = "auto",
    **kwargs,
) -> list[dict]:
    """Build a nested tree from a Markdown or PDF file.

    Args:
        source: Path to a ``.md`` / ``.pdf`` file, OR a list of flat nodes.
        model: LLM model for summary / semantic extraction.
        summarize: If True, generate LLM summaries for every node.
        mode: Hierarchy detection strategy:
            - ``"auto"`` (default) — MD headers + numbering inference.
            - ``"semantic"`` — LLM reads the full document and extracts
              structure semantically (best for complex/unstructured docs).
            - ``"md"`` — strict ``#`` header levels only, no inference.
        **kwargs: Passed to the PDF parser (language, page_range, etc.).

    Returns:
        Nested tree: ``[{"title": ..., "summary": ..., "children": [...]}, ...]``
    """
    path = Path(source) if isinstance(source, (str, Path)) else None

    if path:
        suffix = path.suffix.lower()

        # PDF → MD via MinerU
        if suffix == ".pdf":
            from src.parser.pdf import parse_pdf

            md_text = parse_pdf(str(path), **kwargs)
            tmp_md = path.with_suffix(".tmp.md")
            tmp_md.write_text(md_text, encoding="utf-8")
            try:
                use_semantic = mode == "semantic" or (
                    mode == "auto"
                    and count_tokens(md_text, model=model) <= AUTO_SEMANTIC_MAX_TOKENS
                )
                if use_semantic:
                    tree = asyncio.run(_semantic_build(tmp_md, model=model))
                else:
                    nodes = parse_md(str(tmp_md))
                    tree = build_nodes(nodes)
            finally:
                tmp_md.unlink()

        # HTML
        elif suffix in (".html", ".htm"):
            from src.parser.html import parse_html

            nodes = parse_html(str(path))
            tree = build_nodes(nodes)

        # ZIP — multiple documents
        elif suffix == ".zip":
            from src.parser.zip import parse_zip

            tree = parse_zip(str(path), parser=mode)

        # Semantic mode
        elif mode == "semantic":
            tree = asyncio.run(_semantic_build(path, model=model))

        # Markdown (default) — falls back to semantic extraction when no
        # Markdown headers are found (common for plain-text .txt files).
        else:
            nodes = parse_md(str(path))
            if not nodes:
                tree = asyncio.run(_semantic_build(path, model=model))
            else:
                tree = build_nodes(nodes)

    else:
        tree = build_nodes(source)

    # Apply tree thinning and assign internal node IDs (not exposed to users)
    if isinstance(tree, list) and tree:
        thin_tree(tree, threshold=THINNING_THRESHOLD, model=model)
        assign_node_ids(tree)

        # LLM validation layer (ported from PageIndex verify_toc)
        from .verify import verify_tree

        result = asyncio.run(
            verify_tree(tree, model=model, sample_nodes=VERIFY_SAMPLE_NODES)
        )
        log = logging.getLogger(__name__)
        log.info(
            "[build] tree verify: ok=%s score=%.2f issues=%s",
            result["ok"],
            result["score"],
            result["issues"],
        )
        if not result["ok"]:
            log.warning(
                "[build] tree verify FAILED — issues: %s",
                result["issues"],
            )
            for root in tree:
                root["_verify"] = result

    if summarize:
        asyncio.run(summarize_tree(tree, model=model))

    # Auto-register so agents can find it
    if path:
        from src.tools import register

        tree_id = path.stem  # filename without extension
        register(tree_id, tree)

    _show(tree)
    return tree


# ---------------------------------------------------------------------------
# Semantic build (LLM-driven)
# ---------------------------------------------------------------------------


async def _semantic_build(path: Path, model: str) -> list[dict]:
    """Use LLM to extract the document hierarchy semantically."""
    from .semantic import attach_text_ranges, extract_structure, refine_tree_granularity

    text = path.read_text(encoding="utf-8")
    nodes = await extract_structure(text, model=model)
    attach_text_ranges(nodes, text)
    tree = build_nodes(nodes)
    await refine_tree_granularity(tree, model=model)

    return tree


# ---------------------------------------------------------------------------
# Mechanical build
# ---------------------------------------------------------------------------


def build_nodes(nodes: list[dict]) -> list[dict]:
    stack: list[tuple[dict, int]] = []
    roots: list[dict] = []

    for node in nodes:
        tree_node = {
            "title": node["title"],
            "line_num": node.get("line_num", 0),
            "text": node.get("text", ""),
            "children": [],
        }
        level = node["level"]

        while stack and stack[-1][1] >= level:
            stack.pop()

        if not stack:
            roots.append(tree_node)
        else:
            stack[-1][0]["children"].append(tree_node)

        stack.append((tree_node, level))

    _clean(roots)
    return roots


def _clean(nodes: list[dict]) -> None:
    for node in nodes:
        if not node["children"]:
            del node["children"]
        else:
            _clean(node["children"])


# ---------------------------------------------------------------------------
# Tree thinning (ported from PageIndex)
# ---------------------------------------------------------------------------


def thin_tree(
    nodes: list[dict], threshold: int = THINNING_THRESHOLD, model: str | None = None
) -> None:
    """Post-order traversal: merge small subtrees into their parent.

    For every node we compute the total token count of the node plus all
    descendants.  If the total is below *threshold* we delete the children
    and append their text to the parent's text.
    """
    for node in nodes:
        if node.get("children"):
            thin_tree(node["children"], threshold, model)

    # Process from end to beginning so that when a child is merged into
    # its parent the parent's text_token_count can be recomputed cleanly.
    i = len(nodes) - 1
    while i >= 0:
        node = nodes[i]
        total_text = _node_total_text(node)
        total_tokens = count_tokens(total_text, model=model)

        if (
            total_tokens < threshold
            and node.get("children")
            and not node.get("_preserve_children")
        ):
            # Merge all children text into this node and drop the children.
            child_texts = []
            for child in node["children"]:
                child_total = _node_total_text(child)
                if child_total.strip():
                    child_texts.append(child_total)
            if child_texts:
                parent_text = node.get("text", "")
                merged = parent_text
                for ct in child_texts:
                    if merged and not merged.endswith("\n"):
                        merged += "\n\n"
                    merged += ct
                node["text"] = merged
            del node["children"]
        i -= 1


def _node_total_text(node: dict) -> str:
    """Return the full text of a node including all descendants."""
    parts = [node.get("text", "")]
    for child in node.get("children", []):
        parts.append(_node_total_text(child))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Node IDs (internal only — not exposed to agent tool text)
# ---------------------------------------------------------------------------


def assign_node_ids(nodes: list[dict], counter: int = 1) -> int:
    """Assign sequential 4-digit node_ids to every node.  Returns next counter."""
    for node in nodes:
        node["node_id"] = str(counter).zfill(4)
        counter += 1
        if "children" in node:
            counter = assign_node_ids(node["children"], counter)
    return counter


# ---------------------------------------------------------------------------
# Summarize
# ---------------------------------------------------------------------------


async def summarize_tree(tree: list[dict], model: str, progress=None) -> None:
    """Generate summaries for all nodes concurrently."""
    nodes = flatten_tree(tree)
    if not nodes:
        if progress:
            await progress({"done": 0, "total": 0})
        return

    tasks = [asyncio.create_task(_summarize_one(n, model=model)) for n in nodes]
    done = 0
    for task in asyncio.as_completed(tasks):
        await task
        done += 1
        if progress:
            await progress({"done": done, "total": len(nodes)})


async def _summarize_one(node: dict, model: str) -> None:
    if node.get("summary"):
        return

    text = node.get("text", "")
    if not text:
        node["summary"] = ""
        return

    if count_tokens(text, model=model) < SUMMARY_THRESHOLD:
        node["summary"] = text.strip()
        return

    resp = await achat(
        f"Generate a concise description of the main points covered in this document section.\n\n"
        f"Section text:\n{text}",
        model=model,
        system="You are a technical document summarizer. Be concise.",
    )
    node["summary"] = resp.strip()


def flatten_tree(tree: list[dict]) -> list[dict]:
    result: list[dict] = []
    for node in tree:
        result.append(node)
        if "children" in node:
            result.extend(flatten_tree(node["children"]))
    return result
