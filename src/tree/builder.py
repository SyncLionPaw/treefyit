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
from src.tree.model import FlatNode, Tree, TreeNode, to_wire_tree
from src.vis.tree_view import show as _show

SUMMARY_THRESHOLD = 200  # tokens: if text is shorter, use text as summary
THINNING_THRESHOLD = 5000  # tokens: merge children into parent when smaller
VERIFY_SAMPLE_NODES = 10  # how many nodes to spot-check per build


def build_tree(
    source: str | Path | list[FlatNode],
    *,
    model: str = "gpt-4o",
    summarize: bool = True,
    mode: str = "auto",
    **kwargs,
) -> Tree:
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
                from src.tree.structure import build_tree_structure

                tree, _, _, _ = asyncio.run(
                    build_tree_structure(
                        text=md_text,
                        source_path=tmp_md,
                        model=model,
                        mode=mode,
                        input_tokens=count_tokens(md_text, model=model),
                        is_pdf=True,
                    )
                )
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

        # Semantic / Markdown / plain text — unified type-aware path
        else:
            from src.tree.structure import build_tree_structure

            text = path.read_text(encoding="utf-8")
            tree, _, _, _ = asyncio.run(
                build_tree_structure(
                    text=text,
                    source_path=path,
                    model=model,
                    mode=mode,
                    input_tokens=count_tokens(text, model=model),
                    is_pdf=False,
                )
            )

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

    public_tree = to_wire_tree(tree)

    # Auto-register so agents can find it
    if path:
        from src.tools import register

        tree_id = path.stem  # filename without extension
        register(tree_id, public_tree, filename=path.name)

    _show(public_tree, max_text=48 if summarize else 0)
    return public_tree


# ---------------------------------------------------------------------------
# Mechanical build
# ---------------------------------------------------------------------------


def build_nodes(nodes: list[FlatNode]) -> Tree:
    stack: list[tuple[TreeNode, int]] = []
    roots: Tree = []

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


def _clean(nodes: Tree) -> None:
    for node in nodes:
        if not node["children"]:
            del node["children"]
        else:
            _clean(node["children"])


# ---------------------------------------------------------------------------
# Tree thinning (ported from PageIndex)
# ---------------------------------------------------------------------------


def thin_tree(
    nodes: Tree, threshold: int = THINNING_THRESHOLD, model: str | None = None
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


def _node_total_text(node: TreeNode) -> str:
    """Return the full text of a node including all descendants."""
    parts = [node.get("text", "")]
    for child in node.get("children", []):
        parts.append(_node_total_text(child))
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Node IDs (internal only — not exposed to agent tool text)
# ---------------------------------------------------------------------------


def assign_node_ids(nodes: Tree, counter: int = 1) -> int:
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


async def summarize_tree(tree: Tree, model: str, progress=None) -> None:
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


async def _summarize_one(node: TreeNode, model: str) -> None:
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


def flatten_tree(tree: Tree) -> list[TreeNode]:
    result: list[TreeNode] = []
    for node in tree:
        result.append(node)
        if "children" in node:
            result.extend(flatten_tree(node["children"]))
    return result
