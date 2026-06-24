"""Builder orchestration.

This module wires the end-to-end tree build flow:
source detection -> parse -> infer levels -> refine sections -> build nested tree
-> summarize tree -> model conversion.
It coordinates the stages, but it does not own parsing rules or infer strategies.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel

from treefyit.builder.convert import build_model_tree
from treefyit.builder.infer import LevelInferer, RuleBasedLevelInferer
from treefyit.builder.parse import (
    build_legacy_tree_from_sections,
    parse_html_sections,
    parse_pdf_sections,
    parse_text_sections,
)
from treefyit.builder.refine import RuleBasedSectionRefiner, SectionRefiner
from treefyit.builder.source import detect_source_kind
from treefyit.builder.summarize import summarize_legacy_tree
from treefyit.model.tree import Tree


class BuildOptions(BaseModel):
    summarize: bool = False


def build_tree_from_file(
    path: str | Path,
    options: BuildOptions | None = None,
    level_inferer: LevelInferer | None = None,
    section_refiner: SectionRefiner | None = None,
) -> Tree:
    source_path = Path(path)
    opts = options or BuildOptions()
    kind = detect_source_kind(source_path)
    if kind == "zip":
        raise NotImplementedError("ZIP build is not supported in treefyit.builder yet")

    legacy_tree = build_legacy_tree_from_file(
        source_path,
        opts,
        level_inferer,
        section_refiner,
    )
    root_id = source_path.stem or "document"
    root_title = source_path.name or root_id
    return build_model_tree(legacy_tree, root_id=root_id, root_title=root_title)


def build_tree_from_text(
    text: str,
    filename: str = "document.md",
    options: BuildOptions | None = None,
    level_inferer: LevelInferer | None = None,
    section_refiner: SectionRefiner | None = None,
) -> Tree:
    opts = options or BuildOptions()
    source_name = Path(filename)
    if source_name.suffix.lower() == ".zip":
        raise NotImplementedError("ZIP build is not supported in treefyit.builder yet")
    if source_name.suffix.lower() == ".pdf":
        raise ValueError("build_tree_from_text does not support pdf input")

    legacy_tree = build_legacy_tree_from_text(
        text, opts, level_inferer, section_refiner
    )
    root_id = source_name.stem or "document"
    root_title = source_name.name or root_id
    return build_model_tree(legacy_tree, root_id=root_id, root_title=root_title)


def build_tree_from_sections(
    sections: list[dict],
    options: BuildOptions | None = None,
    level_inferer: LevelInferer | None = None,
    section_refiner: SectionRefiner | None = None,
) -> Tree:
    opts = options or BuildOptions()
    inferred_sections = infer_sections(sections, level_inferer=level_inferer)
    refined_sections = refine_sections(
        inferred_sections,
        section_refiner=section_refiner,
    )
    legacy_tree = build_legacy_tree_from_sections(refined_sections)
    summarize_legacy_tree_if_enabled(legacy_tree, opts)
    return build_model_tree(legacy_tree, root_id="document", root_title="document")


def build_legacy_tree_from_file(
    path: Path,
    options: BuildOptions,
    level_inferer: LevelInferer | None = None,
    section_refiner: SectionRefiner | None = None,
) -> list[dict]:
    kind = detect_source_kind(path)

    if kind == "pdf":
        sections = parse_pdf_sections(path)
        inferred_sections = infer_sections(
            sections,
            level_inferer=level_inferer,
            source_kind=kind,
        )
        refined_sections = refine_sections(
            inferred_sections,
            source_kind=kind,
            section_refiner=section_refiner,
        )
        legacy_tree = build_legacy_tree_from_sections(refined_sections)
        summarize_legacy_tree_if_enabled(legacy_tree, options)
        return legacy_tree
    if kind == "html":
        sections = parse_html_sections(path)
        inferred_sections = infer_sections(
            sections,
            level_inferer=level_inferer,
            source_kind=kind,
        )
        refined_sections = refine_sections(
            inferred_sections,
            source_kind=kind,
            section_refiner=section_refiner,
        )
        legacy_tree = build_legacy_tree_from_sections(refined_sections)
        summarize_legacy_tree_if_enabled(legacy_tree, options)
        return legacy_tree
    if kind == "text":
        return build_legacy_tree_from_text(
            path.read_text(encoding="utf-8"),
            options,
            level_inferer,
            section_refiner,
        )
    raise NotImplementedError("ZIP build is not supported in treefyit.builder yet")


def build_legacy_tree_from_text(
    text: str,
    options: BuildOptions,
    level_inferer: LevelInferer | None = None,
    section_refiner: SectionRefiner | None = None,
) -> list[dict]:
    sections = parse_text_sections(text)
    inferred_sections = infer_sections(sections, text=text, level_inferer=level_inferer)
    refined_sections = refine_sections(
        inferred_sections,
        text=text,
        section_refiner=section_refiner,
    )
    legacy_tree = build_legacy_tree_from_sections(refined_sections)
    summarize_legacy_tree_if_enabled(legacy_tree, options)
    return legacy_tree


def infer_sections(
    sections: list[dict],
    *,
    text: str | None = None,
    source_kind: str | None = None,
    level_inferer: LevelInferer | None = None,
) -> list[dict]:
    inferer = level_inferer or RuleBasedLevelInferer()
    return inferer.infer(sections, text=text, source_kind=source_kind)


def refine_sections(
    sections: list[dict],
    *,
    text: str | None = None,
    source_kind: str | None = None,
    section_refiner: SectionRefiner | None = None,
) -> list[dict]:
    refiner = section_refiner or RuleBasedSectionRefiner()
    return refiner.refine(sections, text=text, source_kind=source_kind)


def summarize_legacy_tree_if_enabled(
    legacy_tree: list[dict],
    options: BuildOptions,
) -> None:
    if not options.summarize:
        return
    summarize_legacy_tree(legacy_tree)


__all__ = [
    "BuildOptions",
    "build_tree_from_file",
    "build_tree_from_text",
    "build_tree_from_sections",
]
