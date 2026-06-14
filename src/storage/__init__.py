"""Storage abstraction for original file persistence."""

from .local import LocalStorageProvider, storage as default_storage

__all__ = ["LocalStorageProvider", "default_storage"]
