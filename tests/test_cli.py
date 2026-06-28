from __future__ import annotations

from pathlib import Path

from treefyit import cli


def test_clear_store_dir_removes_configured_store(
    monkeypatch,
    tmp_path: Path,
):
    store_dir = tmp_path / "store"
    store_dir.mkdir()
    (store_dir / "builds.json").write_text("{}", encoding="utf-8")

    class StoreSettings:
        data_dir = store_dir

    class Settings:
        store = StoreSettings()

    monkeypatch.setattr(cli, "get_settings", lambda: Settings())

    cli.clear_store_dir()

    assert not store_dir.exists()


def test_clear_store_dir_is_noop_when_missing(
    monkeypatch,
    tmp_path: Path,
    capsys,
):
    missing_dir = tmp_path / "missing"

    class StoreSettings:
        data_dir = missing_dir

    class Settings:
        store = StoreSettings()

    monkeypatch.setattr(cli, "get_settings", lambda: Settings())

    cli.clear_store_dir()

    assert f"Store directory does not exist: {missing_dir}" in capsys.readouterr().out


def test_main_runs_server_with_cli_args(monkeypatch):
    captured: dict[str, object] = {}

    monkeypatch.setattr(
        cli,
        "run_server",
        lambda *, host, port: captured.update({"host": host, "port": port}),
    )

    cli.main(["--host", "127.0.0.1", "--port", "9000"])

    assert captured == {
        "host": "127.0.0.1",
        "port": 9000,
    }


def test_main_clears_before_starting(monkeypatch):
    calls: list[str] = []

    monkeypatch.setattr(cli, "clear_store_dir", lambda: calls.append("clear"))
    monkeypatch.setattr(
        cli,
        "run_server",
        lambda *, host, port: calls.append(f"run:{host}:{port}"),
    )

    cli.main(["--clear"])

    assert calls == [
        "clear",
        "run:0.0.0.0:8765",
    ]
