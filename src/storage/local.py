"""Local filesystem storage provider for original build files.

Layout:
    data/uploads/builds/{bid}/original/{sha256}-{safe_filename}

Security:
    - Path-traversal guard: resolved path must stay under root_dir.
    - Filename sanitisation: strips directory components and unsafe chars.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from src.store import _ROOT

logger = logging.getLogger(__name__)

_ROOT_DIR = _ROOT / "uploads"

# File size limits (bytes)
_MAX_FILE_SIZE = 50 * 1024 * 1024  # 50 MB
_MAX_ZIP_SIZE = 100 * 1024 * 1024  # 100 MB

# MIME type whitelist for inline preview
_PREVIEW_TYPES = {
    "application/pdf",
    "text/html",
    "text/htm",
    "text/markdown",
    "text/plain",
    "application/json",
    "application/zip",
}


def _safe_filename(name: str) -> str:
    """Sanitise a user-supplied filename for safe disk storage."""
    # Strip path components
    name = Path(name).name
    # Replace unsafe characters with underscore
    name = re.sub(r"[^\w.\-]", "_", name)
    # Prevent hidden files and empty names
    name = name.lstrip(".")
    if not name:
        name = "unnamed"
    return name


def _guess_content_type(filename: str) -> str:
    """Guess MIME type from filename extension."""
    ext = Path(filename).suffix.lower()
    mapping = {
        ".pdf": "application/pdf",
        ".html": "text/html",
        ".htm": "text/html",
        ".md": "text/markdown",
        ".txt": "text/plain",
        ".json": "application/json",
        ".zip": "application/zip",
    }
    return mapping.get(ext, "application/octet-stream")


class LocalStorageProvider:
    """Save and retrieve original build files on local disk."""

    def __init__(self, root_dir: Path | str | None = None) -> None:
        self.root_dir = Path(root_dir) if root_dir else _ROOT_DIR
        self.root_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def save_original(
        self,
        bid: str,
        filename: str,
        data: bytes,
        content_type: str | None = None,
    ) -> dict:
        """Persist an original file and return its metadata.

        Returns:
            dict with keys: provider, key, size, sha256, content_type
        """
        size = len(data)
        if size == 0:
            raise ValueError("empty file")

        ext = Path(filename).suffix.lower()
        max_size = _MAX_ZIP_SIZE if ext == ".zip" else _MAX_FILE_SIZE
        if size > max_size:
            raise ValueError(
                f"file too large: {size} bytes (max {max_size} bytes)"
            )

        sha256 = hashlib.sha256(data).hexdigest()
        safe_name = _safe_filename(filename)
        key = f"builds/{bid}/original/{sha256[:16]}-{safe_name}"
        dest = self.root_dir / key
        dest.parent.mkdir(parents=True, exist_ok=True)

        dest.write_bytes(data)
        logger.info("[storage] saved original bid=%s key=%s size=%d", bid, key, size)

        ct = content_type or _guess_content_type(filename)
        if ct not in _PREVIEW_TYPES and ct != "application/octet-stream":
            logger.warning("[storage] unknown content_type %s for %s", ct, filename)

        return {
            "provider": "local",
            "key": key,
            "size": size,
            "sha256": sha256,
            "content_type": ct,
        }

    def open_original(self, key: str) -> tuple[bytes, dict]:
        """Read an original file back.

        Returns:
            (data, metadata) where metadata has content_type and size.
        """
        path = self._resolve(key)
        if not path.exists():
            raise FileNotFoundError(f"original file not found: {key}")
        data = path.read_bytes()
        ct = _guess_content_type(path.name)
        return data, {"content_type": ct, "size": len(data)}

    def delete_original(self, key: str) -> bool:
        """Remove an original file. Returns True if it existed."""
        path = self._resolve(key)
        if path.exists():
            path.unlink()
            logger.info("[storage] deleted original key=%s", key)
            return True
        return False

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _resolve(self, key: str) -> Path:
        """Resolve a storage key to an absolute path, guarding traversal."""
        # Reject absolute paths and parent references
        if key.startswith("/") or ".." in key:
            raise ValueError(f"invalid storage key: {key}")
        dest = (self.root_dir / key).resolve()
        if not str(dest).startswith(str(self.root_dir.resolve())):
            raise ValueError(f"storage key escapes root: {key}")
        return dest


# Module-level singleton
storage = LocalStorageProvider()
