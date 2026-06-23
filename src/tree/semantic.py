"""Semantic tree extraction — LLM reads the document and infers hierarchy.

This mirrors PageIndex's ``process_no_toc`` path: chunk the document, have the
LLM generate a structured TOC, then merge the results.
"""

from __future__ import annotations

import asyncio
import logging
import re

from src.llm import achat, count_tokens

logger = logging.getLogger(__name__)

SPLIT_MIN_TOKENS = 1200
SPLIT_MAX_CHILDREN = 8
SPLIT_CONCURRENCY = 4

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


async def extract_structure(
    text: str,
    model: str = "gpt-4o",
    progress=None,
) -> list[dict]:
    """Have the LLM read *text* and extract a hierarchical section list.

    Returns a flat list of {title, level, line_num} that can be fed to
    :func:`build_tree`.
    """
    chunks = _chunk(text, model=model)
    if not chunks:
        return []

    total = len(chunks)
    logger.info(
        "[semantic] extract_structure start: %d chunks, %d chars",
        total,
        len(text),
    )

    async def report(done: int, sections: int) -> None:
        if progress:
            await progress(
                {
                    "done": done,
                    "total": total,
                    "sections": sections,
                    "message": f"LLM structure extraction {done}/{total}",
                }
            )

    await report(0, 0)

    result = await _extract_init(chunks[0], model=model)
    logger.info("[semantic] chunk 1/%d done → %d sections", total, len(result))
    await report(1, len(result))

    for i, chunk in enumerate(chunks[1:], start=2):
        continuation = await _extract_continue(result, chunk, model=model)
        result.extend(continuation)
        logger.info("[semantic] chunk %d/%d done → %d sections", i, total, len(result))
        await report(i, len(result))

    parsed = _parse_structure(result)
    logger.info("[semantic] extract_structure done: %d nodes", len(parsed))
    return parsed


def attach_text_ranges(nodes: list[dict], text: str) -> None:
    """Attach line numbers and text ranges to LLM-extracted structure nodes.

    Semantic extraction returns titles and levels, but agent tools need the
    original section text.  We map titles back to the source in document order,
    then fill each node's text up to the next matched title.
    """
    if not nodes or not text:
        return

    lines = text.split("\n")
    cursor = 0
    matched = 0
    for node in nodes:
        title = str(node.get("title", "")).strip()
        idx = _find_title_line(title, lines, cursor)
        if idx is None and cursor:
            idx = _find_title_line(title, lines, 0)
        if idx is None:
            node["line_num"] = 0
            continue
        node["line_num"] = idx + 1
        cursor = idx + 1
        matched += 1

    nodes.sort(key=lambda n: n.get("line_num") or 10**12)
    for i, node in enumerate(nodes):
        start = int(node.get("line_num") or 0) - 1
        if start < 0:
            node.setdefault("text", "")
            continue

        end = len(lines)
        for later in nodes[i + 1 :]:
            later_line = int(later.get("line_num") or 0)
            if later_line > start + 1:
                end = later_line - 1
                break
        node["text"] = "\n".join(lines[start:end]).strip()

    logger.info("[semantic] attached text ranges for %d/%d nodes", matched, len(nodes))


async def refine_tree_granularity(
    tree: list[dict],
    *,
    model: str = "gpt-4o",
    min_tokens: int = SPLIT_MIN_TOKENS,
    max_children: int = SPLIT_MAX_CHILDREN,
    max_leaf_depth: int | None = None,
    progress=None,
) -> int:
    """Split oversized leaf nodes into semantic sub-sections with an LLM.

    Initial structure extraction often stops at chapter-level for novels or
    other prose.  This pass asks the model to identify meaningful scenes /
    topic shifts inside each large leaf, then anchors those segments back to
    the original text using short quotes supplied by the model.

    When *max_leaf_depth* is set (e.g. ``1`` for 章回体), only leaves at that
    depth from the root are split — sub-segments are never subdivided again.

    Returns the number of leaf nodes that were expanded.
    """
    leaves = [
        node
        for node in _walk_nodes(tree)
        if not node.get("children")
        and count_tokens(node.get("text", ""), model=model) >= min_tokens
        and (
            max_leaf_depth is None
            or _node_depth_in_tree(tree, node) == max_leaf_depth
        )
    ]
    if not leaves:
        if progress:
            await progress({"done": 0, "total": 0, "expanded": 0})
        return 0

    logger.info("[semantic] refining %d large leaf nodes", len(leaves))
    semaphore = asyncio.Semaphore(SPLIT_CONCURRENCY)
    done = 0
    expanded = 0

    async def _run(node: dict) -> bool:
        async with semaphore:
            return await _split_leaf_node(
                node,
                model=model,
                max_children=max_children,
            )

    tasks = [asyncio.create_task(_run(node)) for node in leaves]
    for task in asyncio.as_completed(tasks):
        ok = await task
        done += 1
        if ok:
            expanded += 1
        if progress:
            await progress({"done": done, "total": len(leaves), "expanded": expanded})

    logger.info("[semantic] refined %d/%d large leaf nodes", expanded, len(leaves))
    return expanded


def _walk_nodes(nodes: list[dict]):
    for node in nodes:
        yield node
        children = node.get("children") or []
        yield from _walk_nodes(children)


def _node_depth_in_tree(tree: list[dict], target: dict) -> int | None:
    def walk(nodes: list[dict], depth: int) -> int | None:
        for node in nodes:
            if node is target:
                return depth
            children = node.get("children") or []
            found = walk(children, depth + 1)
            if found is not None:
                return found
        return None

    return walk(tree, 1)


def deepest_leaf_depth(tree: list[dict]) -> int:
    """Maximum depth among leaf nodes (nodes without children)."""
    depth = 0
    for node in _walk_nodes(tree):
        if node.get("children"):
            continue
        node_depth = _node_depth_in_tree(tree, node)
        if node_depth is not None:
            depth = max(depth, node_depth)
    return depth


async def _split_leaf_node(
    node: dict,
    *,
    model: str,
    max_children: int,
) -> bool:
    text = str(node.get("text", "")).strip()
    if not text:
        return False

    segments = await _extract_segments(
        title=str(node.get("title", "")),
        text=text,
        model=model,
        max_children=max_children,
    )
    children = _segments_to_children(
        segments,
        parent_text=text,
        parent_line_num=int(node.get("line_num") or 1),
    )
    if len(children) < 2:
        return False

    node["children"] = children
    node["_preserve_children"] = True
    logger.info(
        "[semantic] split leaf %r into %d children",
        node.get("title", ""),
        len(children),
    )
    return True


async def _extract_segments(
    *,
    title: str,
    text: str,
    model: str,
    max_children: int,
) -> list[dict]:
    resp = await achat(
        f"""You are segmenting a long document section into smaller semantic units.

Section title:
{title}

Task:
- Split the section into 3 to {max_children} meaningful sub-sections.
- Prefer plot beats, topic shifts, argument stages, or scene changes.
- Do NOT rely on formal headings; infer the structure semantically.
- Preserve the original order.
- For each segment, provide exact short quotes copied from the source text:
  - start_quote: 8-30 original characters near the segment beginning
  - end_quote: 8-30 original characters near the segment ending

Return ONLY a JSON array:
[
  {{
    "title": "short semantic title",
    "summary": "one sentence summary",
    "start_quote": "exact quote from the source",
    "end_quote": "exact quote from the source"
  }}
]

Section text:
{text}""",
        model=model,
        temperature=0,
    )
    parsed = _parse_llm_json(resp)
    return parsed if isinstance(parsed, list) else []


def _segments_to_children(
    segments: list[dict],
    *,
    parent_text: str,
    parent_line_num: int,
) -> list[dict]:
    starts: list[tuple[dict, int]] = []
    cursor = 0
    for seg in segments:
        start_quote = str(seg.get("start_quote", "")).strip()
        start = _find_snippet(parent_text, start_quote, cursor)
        if start is None:
            continue
        starts.append((seg, start))
        cursor = start + max(1, len(start_quote))

    if len(starts) < 2:
        return []

    children: list[dict] = []
    for idx, (seg, start) in enumerate(starts):
        next_start = starts[idx + 1][1] if idx + 1 < len(starts) else len(parent_text)
        end_quote = str(seg.get("end_quote", "")).strip()
        end = _find_snippet_end(parent_text, end_quote, start, next_start)
        if end is None:
            end = next_start
        if end <= start:
            continue

        child_text = parent_text[start:end].strip()
        if not child_text:
            continue

        title = str(seg.get("title", "")).strip() or f"Segment {idx + 1}"
        summary = str(seg.get("summary", "")).strip()
        children.append(
            {
                "title": title,
                "summary": summary,
                "line_num": parent_line_num + parent_text.count("\n", 0, start),
                "text": child_text,
            }
        )

    return children


def _find_snippet(text: str, snippet: str, start: int = 0) -> int | None:
    if not snippet:
        return None
    exact = text.find(snippet, start)
    if exact != -1:
        return exact
    return _find_snippet_normalized(text, snippet, start)


def _find_snippet_end(
    text: str,
    snippet: str,
    start: int,
    limit: int,
) -> int | None:
    if not snippet:
        return None
    pos = _find_snippet(text, snippet, start)
    if pos is None or pos >= limit:
        return None
    return pos + len(snippet)


def _find_snippet_normalized(text: str, snippet: str, start: int) -> int | None:
    normalized_text, index_map = _normalize_with_index(text[start:])
    normalized_snippet = _normalize_title(snippet)
    if not normalized_text or not normalized_snippet:
        return None
    pos = normalized_text.find(normalized_snippet)
    if pos == -1:
        return None
    return start + index_map[pos]


def _normalize_with_index(value: str) -> tuple[str, list[int]]:
    chars: list[str] = []
    index_map: list[int] = []
    for idx, ch in enumerate(value):
        norm = _normalize_title(ch)
        if not norm:
            continue
        chars.append(norm)
        index_map.append(idx)
    return "".join(chars), index_map


def _find_title_line(title: str, lines: list[str], start: int) -> int | None:
    if not title:
        return None
    title_norm = _normalize_title(title)
    if not title_norm:
        return None
    for idx in range(start, len(lines)):
        line_norm = _normalize_title(lines[idx])
        if not line_norm:
            continue
        if title_norm in line_norm or line_norm in title_norm:
            return idx
    return None


def _normalize_title(value: str) -> str:
    value = value.strip().lower()
    return re.sub(r"[\s:：,，。\.、\-—_《》「」“”\"'()\[\]（）]+", "", value)


# ---------------------------------------------------------------------------
# Chunking
# ---------------------------------------------------------------------------


def _chunk(text: str, model: str = "gpt-4o", max_tokens: int = 8000) -> list[str]:
    """Split text into chunks that fit within the LLM context window."""
    paragraphs = text.split("\n\n")
    chunks: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in paragraphs:
        t = count_tokens(para, model=model)
        if current_tokens + t > max_tokens and current:
            chunks.append("\n\n".join(current))
            current = []
            current_tokens = 0
        current.append(para)
        current_tokens += t

    if current:
        chunks.append("\n\n".join(current))

    return chunks


# ---------------------------------------------------------------------------
# LLM prompts (mirrors page_index.py's generate_toc_init / generate_toc_continue)
# ---------------------------------------------------------------------------


async def _extract_init(chunk: str, model: str) -> list[dict]:
    """First chunk: generate initial tree structure."""
    resp = await achat(
        f"""You are an expert in extracting hierarchical tree structure from documents.

The structure uses numeric indices to represent hierarchy:
  - "1" = first top-level section
  - "1.1" = first subsection under section 1
  - "1.1.1" = first sub-subsection
  etc.

For each section, extract the original title from the text (fix only space inconsistencies).

Return ONLY a JSON array:
[
    {{"structure": "1",      "title": "Introduction"}},
    {{"structure": "1.1",    "title": "Background"}},
    {{"structure": "2",      "title": "Methods"}},
    ...
]

Document text:
{chunk}""",
        model=model,
    )
    return _parse_llm_json(resp)


async def _extract_continue(previous: list[dict], chunk: str, model: str) -> list[dict]:
    """Subsequent chunks: continue the tree structure."""
    resp = await achat(
        f"""Continue the hierarchical tree structure from the previous part.

Previous structure:
{_format_previous(previous)}

Current document text:
{chunk}

Return ONLY the additional sections as a JSON array (same format as before):
[
    {{"structure": "x.x.x", "title": "..."}},
    ...
]""",
        model=model,
    )
    return _parse_llm_json(resp)


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------


def _parse_structure(structured: list[dict]) -> list[dict]:
    """Convert LLM output to the internal node format."""
    nodes: list[dict] = []
    for item in structured:
        structure = item.get("structure", "")
        depth = structure.count(".") + 1 if structure else 1
        nodes.append(
            {
                "title": item.get("title", ""),
                "level": depth,
                "line_num": 0,  # semantic extraction doesn't have line numbers
            }
        )
    return nodes


def _format_previous(items: list[dict]) -> str:
    """Format previous structure for the LLM prompt."""
    lines: list[str] = []
    for item in items:
        lines.append(f"  {item.get('structure', '?')}: {item.get('title', '')}")
    return "\n".join(lines)


def _parse_llm_json(resp: str) -> list[dict]:
    """Best-effort JSON extraction from LLM response."""
    import json
    import re

    # Try fenced JSON
    start = resp.find("```json")
    if start == -1:
        start = resp.find("```")
    if start != -1:
        start = (
            resp.find("\n", start) + 1 if resp.find("\n", start) != -1 else start + 3
        )
        end = resp.rfind("```")
        if end != -1:
            resp = resp[start:end]

    # Clean and parse
    resp = resp.strip()
    resp = resp.replace("None", "null")
    resp = re.sub(r",\s*]", "]", resp)
    resp = re.sub(r",\s*}", "}", resp)

    try:
        return json.loads(resp)
    except json.JSONDecodeError:
        # Try to find array bounds
        s = resp.find("[")
        e = resp.rfind("]")
        if s != -1 and e != -1:
            try:
                return json.loads(resp[s : e + 1])
            except json.JSONDecodeError:
                pass
        return []
