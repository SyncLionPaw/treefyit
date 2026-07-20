"""Rule-based multi-step build: parse → infer → refine → assemble."""

from __future__ import annotations

from pathlib import Path

from core.tree import TreeNode

from .assemble import assemble_tree
from .infer import infer_levels
from .parse import parse_text
from .refine import refine_sections
from .types import Section


def build_sections_from_text(
    text: str,
    *,
    format: str = "md",
    refine: bool = True,
    refine_split_threshold: int = 400,
    refine_max_parts: int = 4,
) -> list[Section]:
    """Run parse → infer → refine and return flat sections."""
    sections = parse_text(text, format=format)
    sections = infer_levels(sections)
    if refine:
        sections = refine_sections(
            sections,
            split_threshold=refine_split_threshold,
            max_parts=refine_max_parts,
        )
    return sections


def build_tree_from_text(
    text: str,
    *,
    format: str = "md",
    root_id: str = "root",
    root_title: str = "Document",
    refine: bool = True,
    refine_split_threshold: int = 400,
    refine_max_parts: int = 4,
) -> TreeNode:
    """Build a TreeNode from markdown/plaintext via the rule-based pipeline."""
    sections = build_sections_from_text(
        text,
        format=format,
        refine=refine,
        refine_split_threshold=refine_split_threshold,
        refine_max_parts=refine_max_parts,
    )
    return assemble_tree(sections, root_id=root_id, root_title=root_title)


def build_tree_from_file(
    path: str | Path,
    *,
    root_id: str | None = None,
    root_title: str | None = None,
    format: str | None = None,
    refine: bool = True,
    refine_split_threshold: int = 400,
    refine_max_parts: int = 4,
) -> TreeNode:
    """Load a .md/.txt file and build a TreeNode."""
    p = Path(path)
    text = p.read_text(encoding="utf-8")
    fmt = format or ("txt" if p.suffix.lower() == ".txt" else "md")
    title = root_title or p.stem
    return build_tree_from_text(
        text,
        format=fmt,
        root_id=root_id or p.stem or "root",
        root_title=title,
        refine=refine,
        refine_split_threshold=refine_split_threshold,
        refine_max_parts=refine_max_parts,
    )
