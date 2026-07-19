"""Read typed settings from TOML files."""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator


class LLMSettings(BaseModel):
    model: str = "gpt-4o"
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None


class MinerUSettings(BaseModel):
    api_key: str | None = None
    base_url: str | None = None
    model: str | None = None


class BuilderSettings(BaseModel):
    refine_split_threshold: int = Field(default=400, ge=1)
    refine_max_parts: int = Field(default=4, ge=1)


class StoreSettings(BaseModel):
    data_dir: Path = Path(".treefyit-store")


class ChatSettings(BaseModel):
    session_backend: Literal["memory", "json", "sqlite"] = "json"
    session_sqlite_path: Path | None = None

    @field_validator("session_sqlite_path", mode="before")
    @classmethod
    def empty_sqlite_path_as_none(cls, value: object) -> object:
        if isinstance(value, str) and not value.strip():
            return None
        return value


class AppSettings(BaseModel):
    llm: LLMSettings = Field(default_factory=LLMSettings)
    mineru: MinerUSettings = Field(default_factory=MinerUSettings)
    builder: BuilderSettings = Field(default_factory=BuilderSettings)
    store: StoreSettings = Field(default_factory=StoreSettings)
    chat: ChatSettings = Field(default_factory=ChatSettings)


def get_config_paths() -> list[Path]:
    config_dir = Path(__file__).resolve().parent
    return [
        config_dir / "settings.toml",
        config_dir / "settings.local.toml",
    ]


def read_toml_file(path: Path) -> dict:
    if not path.exists():
        return {}

    with path.open("rb") as file:
        data = tomllib.load(file)
    if isinstance(data, dict):
        return data
    return {}


def merge_dict(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = merge_dict(base[key], value)
            continue
        merged[key] = value
    return merged


def read_config_file() -> dict:
    config: dict = {}
    for path in get_config_paths():
        config = merge_dict(config, read_toml_file(path))
    return config


def build_settings() -> AppSettings:
    return AppSettings.model_validate(read_config_file())


def get_settings() -> AppSettings:
    return build_settings()


def reload_settings() -> AppSettings:
    return build_settings()


__all__ = [
    "AppSettings",
    "BuilderSettings",
    "ChatSettings",
    "LLMSettings",
    "MinerUSettings",
    "StoreSettings",
    "build_settings",
    "get_settings",
    "reload_settings",
]
