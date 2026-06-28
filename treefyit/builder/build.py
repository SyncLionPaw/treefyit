"""Builder orchestration.

This module wires the end-to-end tree build flow:
source detection -> parse -> infer levels -> refine sections -> build nested tree
-> summarize tree -> model conversion.
It coordinates the stages, but it does not own parsing rules or infer strategies.
"""

from __future__ import annotations

import logging
import time
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
from treefyit.builder.refine import (
    LLMSectionRefiner,
    RuleBasedSectionRefiner,
    SectionRefiner,
)
from treefyit.builder.source import detect_source_kind
from treefyit.builder.summarize import summarize_legacy_tree
from treefyit.model.tree import Tree

logger = logging.getLogger("treefyit.builder")


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
    started_at = time.perf_counter()
    kind = detect_source_kind(source_path)
    logger.info(
        "builder start source=file path=%s source_kind=%s summarize=%s",
        source_path,
        kind,
        opts.summarize,
    )
    if kind == "zip":
        raise NotImplementedError("ZIP build is not supported in treefyit.builder yet")

    legacy_tree = build_legacy_tree_from_file(
        source_path,
        opts,
        level_inferer,
        section_refiner,
    )
    model_started_at = time.perf_counter()
    root_id = source_path.stem or "document"
    root_title = source_path.name or root_id
    tree = build_model_tree(legacy_tree, root_id=root_id, root_title=root_title)
    model_ms = (time.perf_counter() - model_started_at) * 1000
    total_ms = (time.perf_counter() - started_at) * 1000
    logger.info(
        "builder completed source=file path=%s source_kind=%s node_count=%s max_depth=%s root_count=%s model_ms=%.2f total_ms=%.2f",
        source_path,
        kind,
        tree.subtree_size or 1,
        max_tree_depth(tree),
        len(tree.children),
        model_ms,
        total_ms,
    )
    return tree


def build_tree_from_text(
    text: str,
    filename: str = "document.md",
    options: BuildOptions | None = None,
    level_inferer: LevelInferer | None = None,
    section_refiner: SectionRefiner | None = None,
) -> Tree:
    opts = options or BuildOptions()
    source_name = Path(filename)
    started_at = time.perf_counter()
    logger.info(
        "builder start source=text filename=%s summarize=%s text_chars=%s",
        filename,
        opts.summarize,
        len(text),
    )
    if source_name.suffix.lower() == ".zip":
        raise NotImplementedError("ZIP build is not supported in treefyit.builder yet")
    if source_name.suffix.lower() == ".pdf":
        raise ValueError("build_tree_from_text does not support pdf input")

    legacy_tree = build_legacy_tree_from_text(
        text, opts, level_inferer, section_refiner
    )
    model_started_at = time.perf_counter()
    root_id = source_name.stem or "document"
    root_title = source_name.name or root_id
    tree = build_model_tree(legacy_tree, root_id=root_id, root_title=root_title)
    model_ms = (time.perf_counter() - model_started_at) * 1000
    total_ms = (time.perf_counter() - started_at) * 1000
    logger.info(
        "builder completed source=text filename=%s node_count=%s max_depth=%s root_count=%s model_ms=%.2f total_ms=%.2f",
        filename,
        tree.subtree_size or 1,
        max_tree_depth(tree),
        len(tree.children),
        model_ms,
        total_ms,
    )
    return tree


def build_tree_from_sections(
    sections: list[dict],
    options: BuildOptions | None = None,
    level_inferer: LevelInferer | None = None,
    section_refiner: SectionRefiner | None = None,
) -> Tree:
    opts = options or BuildOptions()
    started_at = time.perf_counter()
    logger.info(
        "builder start source=sections summarize=%s section_count=%s",
        opts.summarize,
        len(sections),
    )
    inferred_sections = infer_sections(sections, level_inferer=level_inferer)
    refined_sections = refine_sections(
        inferred_sections,
        options=opts,
        section_refiner=section_refiner,
    )
    legacy_tree = build_legacy_tree_from_sections(refined_sections)
    summarize_legacy_tree_if_enabled(legacy_tree, opts)
    tree = build_model_tree(legacy_tree, root_id="document", root_title="document")
    total_ms = (time.perf_counter() - started_at) * 1000
    logger.info(
        "builder completed source=sections node_count=%s max_depth=%s root_count=%s total_ms=%.2f",
        tree.subtree_size or 1,
        max_tree_depth(tree),
        len(tree.children),
        total_ms,
    )
    return tree


def build_legacy_tree_from_file(
    path: Path,
    options: BuildOptions,
    level_inferer: LevelInferer | None = None,
    section_refiner: SectionRefiner | None = None,
) -> list[dict]:
    kind = detect_source_kind(path)

    if kind == "pdf":
        return build_legacy_tree_from_section_source(
            parse_sections=lambda: parse_pdf_sections(path),
            source_label=str(path),
            source_kind=kind,
            options=options,
            level_inferer=level_inferer,
            section_refiner=section_refiner,
        )
    if kind == "html":
        return build_legacy_tree_from_section_source(
            parse_sections=lambda: parse_html_sections(path),
            source_label=str(path),
            source_kind=kind,
            options=options,
            level_inferer=level_inferer,
            section_refiner=section_refiner,
        )
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
    return build_legacy_tree_from_section_source(
        parse_sections=lambda: parse_text_sections(text),
        source_label="inline_text",
        source_kind="text",
        options=options,
        level_inferer=level_inferer,
        section_refiner=section_refiner,
        text=text,
    )


def build_legacy_tree_from_section_source(
    *,
    parse_sections,
    source_label: str,
    source_kind: str | None,
    options: BuildOptions,
    level_inferer: LevelInferer | None = None,
    section_refiner: SectionRefiner | None = None,
    text: str | None = None,
) -> list[dict]:
    parse_started_at = time.perf_counter()
    sections = parse_sections()
    parse_ms = (time.perf_counter() - parse_started_at) * 1000
    logger.info(
        "builder parse completed source=%s source_kind=%s section_count=%s parse_ms=%.2f",
        source_label,
        source_kind,
        len(sections),
        parse_ms,
    )

    inferer = resolve_level_inferer(level_inferer)
    infer_started_at = time.perf_counter()
    inferred_sections = inferer.infer(sections, text=text, source_kind=source_kind)
    infer_ms = (time.perf_counter() - infer_started_at) * 1000
    logger.info(
        "builder infer completed source=%s source_kind=%s inferer=%s sections_in=%s sections_out=%s infer_ms=%.2f",
        source_label,
        source_kind,
        inferer.__class__.__name__,
        len(sections),
        len(inferred_sections),
        infer_ms,
    )

    refiner = resolve_section_refiner(options=options, section_refiner=section_refiner)
    refine_started_at = time.perf_counter()
    refined_sections = refiner.refine(
        inferred_sections,
        text=text,
        source_kind=source_kind,
    )
    refine_ms = (time.perf_counter() - refine_started_at) * 1000
    logger.info(
        "builder refine completed source=%s source_kind=%s refiner=%s sections_in=%s sections_out=%s refine_ms=%.2f",
        source_label,
        source_kind,
        refiner.__class__.__name__,
        len(inferred_sections),
        len(refined_sections),
        refine_ms,
    )

    assemble_started_at = time.perf_counter()
    legacy_tree = build_legacy_tree_from_sections(refined_sections)
    assemble_ms = (time.perf_counter() - assemble_started_at) * 1000
    logger.info(
        "builder assemble completed source=%s source_kind=%s root_count=%s assemble_ms=%.2f",
        source_label,
        source_kind,
        len(legacy_tree),
        assemble_ms,
    )

    summary_started_at = time.perf_counter()
    summarize_legacy_tree_if_enabled(legacy_tree, options)
    summary_ms = (time.perf_counter() - summary_started_at) * 1000
    logger.info(
        "builder summary completed source=%s source_kind=%s summarize=%s summary_ms=%.2f",
        source_label,
        source_kind,
        options.summarize,
        summary_ms,
    )
    return legacy_tree


def infer_sections(
    sections: list[dict],
    *,
    text: str | None = None,
    source_kind: str | None = None,
    level_inferer: LevelInferer | None = None,
) -> list[dict]:
    inferer = resolve_level_inferer(level_inferer)
    return inferer.infer(sections, text=text, source_kind=source_kind)


def refine_sections(
    sections: list[dict],
    *,
    text: str | None = None,
    source_kind: str | None = None,
    options: BuildOptions | None = None,
    section_refiner: SectionRefiner | None = None,
) -> list[dict]:
    refiner = resolve_section_refiner(
        options=options,
        section_refiner=section_refiner,
    )
    return refiner.refine(sections, text=text, source_kind=source_kind)


def resolve_level_inferer(level_inferer: LevelInferer | None) -> LevelInferer:
    return level_inferer or RuleBasedLevelInferer()


def resolve_section_refiner(
    *,
    options: BuildOptions | None,
    section_refiner: SectionRefiner | None,
) -> SectionRefiner:
    if section_refiner is not None:
        return section_refiner
    if options is not None and options.summarize:
        return LLMSectionRefiner()
    return RuleBasedSectionRefiner()


def summarize_legacy_tree_if_enabled(
    legacy_tree: list[dict],
    options: BuildOptions,
) -> None:
    if not options.summarize:
        return
    summarize_legacy_tree(legacy_tree)


def max_tree_depth(tree: Tree) -> int:
    if not tree.children:
        return tree.depth or 0
    return max(max_tree_depth(child) for child in tree.children)


__all__ = [
    "BuildOptions",
    "build_tree_from_file",
    "build_tree_from_text",
    "build_tree_from_sections",
]
