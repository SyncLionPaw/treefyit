from __future__ import annotations

from pathlib import Path

from treefyit.builder.refine import RuleBasedSectionRefiner
from treefyit.config import reload_settings
from treefyit.config.settings import BuilderSettings, LLMSettings, StoreSettings


def test_reload_settings_reads_toml_config(monkeypatch, tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "settings.toml").write_text(
        "\n".join(
            [
                "[llm]",
                'model = "ollama/qwen3:1.7b"',
                'api_key = "demo-key"',
                'base_url = "http://127.0.0.1:11434"',
                "max_tokens = 1024",
                "",
                "[builder]",
                "refine_split_threshold = 256",
                "refine_max_parts = 6",
                "",
                "[store]",
                'data_dir = ".treefyit-test-store"',
            ]
        )
    )
    monkeypatch.setattr(
        "treefyit.config.settings.get_config_paths",
        lambda: [config_dir / "settings.toml"],
    )

    settings = reload_settings()

    assert isinstance(settings.llm, LLMSettings)
    assert settings.llm.model == "ollama/qwen3:1.7b"
    assert settings.llm.api_key == "demo-key"
    assert settings.llm.base_url == "http://127.0.0.1:11434"
    assert settings.llm.max_tokens == 1024
    assert isinstance(settings.builder, BuilderSettings)
    assert settings.builder.refine_split_threshold == 256
    assert settings.builder.refine_max_parts == 6
    assert isinstance(settings.store, StoreSettings)
    assert settings.store.data_dir == Path(".treefyit-test-store")


def test_rule_based_section_refiner_uses_builder_settings(monkeypatch):
    monkeypatch.setattr(
        "treefyit.config.settings.get_config_paths",
        lambda: [],
    )
    reload_settings()

    refiner = RuleBasedSectionRefiner()

    assert refiner.split_threshold == 400
    assert refiner.max_parts == 4


def test_settings_local_toml_overrides_base(monkeypatch, tmp_path: Path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "settings.toml").write_text(
        "\n".join(
            [
                "[llm]",
                'model = "gpt-4.1-mini"',
                'api_key = "toml-key"',
                "",
                "[builder]",
                "refine_split_threshold = 222",
            ]
        )
    )
    (config_dir / "settings.local.toml").write_text(
        "\n".join(
            [
                "[llm]",
                'api_key = "local-key"',
                "",
                "[builder]",
                "refine_max_parts = 5",
            ]
        )
    )
    monkeypatch.setattr(
        "treefyit.config.settings.get_config_paths",
        lambda: [
            config_dir / "settings.toml",
            config_dir / "settings.local.toml",
        ],
    )

    settings = reload_settings()

    assert settings.llm.model == "gpt-4.1-mini"
    assert settings.llm.api_key == "local-key"
    assert settings.builder.refine_split_threshold == 222
    assert settings.builder.refine_max_parts == 5
