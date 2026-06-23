"""Shared pytest fixtures."""

from __future__ import annotations

import pytest


@pytest.fixture
def isolated_store(tmp_path, monkeypatch):
    """Point store + uploads at a temp directory and re-init SQLite."""
    import src.store as store
    import src.store.builds as builds
    import src.store.cache as cache
    import src.store.sqlite as sqlite
    import src.storage.local as local

    monkeypatch.setattr(store, "_ROOT", tmp_path)
    monkeypatch.setattr(builds, "_ROOT", tmp_path)
    monkeypatch.setattr(cache, "_ROOT", tmp_path)
    monkeypatch.setattr(sqlite, "_ROOT", tmp_path)
    monkeypatch.setattr(sqlite, "_DB_PATH", tmp_path / "data.sqlite")
    monkeypatch.setattr(sqlite, "_initialized", False)
    monkeypatch.setattr(local, "_ROOT", tmp_path)

    store.history.clear()
    store.init()
    yield store
    store.history.clear()


@pytest.fixture
def api_client():
    from fastapi.testclient import TestClient

    from src.server.server import app

    return TestClient(app)
