"""Source-kind detection.

This module identifies the coarse input kind for the builder (`pdf`, `html`, `zip`, `text`)
using libmagic first and filename suffix as fallback. It only classifies sources and does not
parse document contents.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

type SourceKind = Literal["pdf", "html", "zip", "text"]


def detect_source_kind(path: Path) -> SourceKind:
    mime_type = detect_mime_type(path)
    if mime_type == "application/pdf":
        return "pdf"
    if mime_type in {"text/html", "application/xhtml+xml"}:
        return "html"
    if mime_type == "application/zip":
        return "zip"

    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in {".html", ".htm"}:
        return "html"
    if suffix == ".zip":
        return "zip"
    return "text"


def detect_mime_type(path: Path) -> str | None:
    try:
        import magic
    except ImportError:
        return None

    try:
        return magic.from_file(str(path), mime=True)
    except OSError:
        return None


__all__ = ["SourceKind", "detect_source_kind", "detect_mime_type"]
