"""Flat chunk index: same document, no hierarchy."""

from __future__ import annotations

import re
from dataclasses import dataclass

from core.build.query import tokenize

paragraph_split = re.compile(r"\n\s*\n+")


@dataclass(frozen=True)
class FlatChunk:
    chunk_id: str
    text: str


@dataclass(frozen=True)
class FlatHit:
    chunk_id: str
    score: float
    snippet: str
    text: str


def strip_heading_markers(text: str) -> str:
    """Remove markdown heading syntax so flat baseline gets no free structure."""
    lines: list[str] = []
    for line in text.splitlines():
        if line.startswith("#"):
            lines.append(line.lstrip("#").strip())
        else:
            lines.append(line)
    return "\n".join(lines).strip()


def chunk_markdown(
    text: str,
    *,
    max_chars: int = 280,
    overlap: int = 40,
) -> list[FlatChunk]:
    """Split markdown into overlapping flat windows (no heading hierarchy)."""
    cleaned = strip_heading_markers(text)
    if not cleaned:
        return []

    # Prefer paragraph packing; fall back to hard windows for long paragraphs.
    paragraphs = [p.strip() for p in paragraph_split.split(cleaned) if p.strip()]
    packed: list[str] = []
    buf = ""
    for para in paragraphs:
        candidate = para if not buf else f"{buf}\n\n{para}"
        if len(candidate) <= max_chars:
            buf = candidate
            continue
        if buf:
            packed.append(buf)
        if len(para) <= max_chars:
            buf = para
            continue
        packed.extend(_hard_windows(para, max_chars=max_chars, overlap=overlap))
        buf = ""
    if buf:
        packed.append(buf)

    # Add modest overlap between adjacent packed chunks.
    chunks: list[FlatChunk] = []
    for index, body in enumerate(packed):
        if index > 0 and overlap > 0:
            prev_tail = packed[index - 1][-overlap:]
            body = f"{prev_tail}\n\n{body}"
        chunks.append(FlatChunk(chunk_id=f"c{index}", text=body))
    return chunks


def _hard_windows(text: str, *, max_chars: int, overlap: int) -> list[str]:
    step = max(max_chars - overlap, 1)
    return [text[i : i + max_chars] for i in range(0, len(text), step)]


def score_chunk(text: str, terms: list[str]) -> float:
    """Content-only scoring (no title boost), fair flat baseline."""
    if not terms:
        return 0.0
    tokens = tokenize(text)
    blob = text.casefold()
    score = 0.0
    for term in terms:
        if term in tokens:
            score += 1.0
        if term in blob:
            score += 0.5
    return score


def search_flat(
    chunks: list[FlatChunk],
    query: str,
    *,
    limit: int = 5,
) -> list[FlatHit]:
    terms = tokenize(query)
    if not terms:
        return []
    hits: list[FlatHit] = []
    for chunk in chunks:
        score = score_chunk(chunk.text, terms)
        if score <= 0:
            continue
        snippet = " ".join(chunk.text.split())
        if len(snippet) > 160:
            snippet = snippet[:159] + "…"
        hits.append(
            FlatHit(
                chunk_id=chunk.chunk_id,
                score=score,
                snippet=snippet,
                text=chunk.text,
            )
        )
    hits.sort(key=lambda hit: (-hit.score, hit.chunk_id))
    return hits[: max(limit, 0)]
