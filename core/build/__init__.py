"""Rule-based heuristic tree build (no LLM).

Pipeline: parse → infer → refine → assemble.
"""

from .assemble import assemble_tree
from .infer import infer_levels
from .parse import parse_text, parse_text_sections
from .pipeline import (
    build_sections_from_text,
    build_tree_from_file,
    build_tree_from_text,
)
from .refine import refine_sections
from .types import Section

__all__ = [
    "Section",
    "assemble_tree",
    "build_sections_from_text",
    "build_tree_from_file",
    "build_tree_from_text",
    "infer_levels",
    "parse_text",
    "parse_text_sections",
    "refine_sections",
]
