from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from treefyit.config import reload_settings
from treefyit.server import create_app
from treefyit.store import RegistryStore


@pytest.fixture(autouse=True)
def reset_treefyit_settings_cache():
    reload_settings()
    yield
    reload_settings()


@pytest.fixture
def api_client():
    return TestClient(create_app())


@pytest.fixture
def isolated_store(tmp_path):
    return RegistryStore(tmp_path)
