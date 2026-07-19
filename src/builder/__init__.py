"""Public builder exports.

This module only re-exports the stable builder entrypoints and available inferer
and refiner strategies. It does not contain build logic itself.
"""

from treefyit.builder.build import (
    BuildOptions,
    build_tree_from_file,
    build_tree_from_sections,
    build_tree_from_text,
)
from treefyit.builder.infer import (
    LLMLevelInferer,
    NoopLevelInferer,
    RuleBasedLevelInferer,
)
from treefyit.builder.refine import (
    LLMSectionRefiner,
    NoopSectionRefiner,
    RuleBasedSectionRefiner,
)

__all__ = [
    "BuildOptions",
    "LLMLevelInferer",
    "LLMSectionRefiner",
    "NoopLevelInferer",
    "NoopSectionRefiner",
    "RuleBasedLevelInferer",
    "RuleBasedSectionRefiner",
    "build_tree_from_file",
    "build_tree_from_sections",
    "build_tree_from_text",
]
