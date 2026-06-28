from __future__ import annotations

import argparse
import shutil
from collections.abc import Sequence

import uvicorn

from treefyit.config import get_settings
from treefyit.logging_config import configure_treefyit_logging
from treefyit.server import app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the treefyit HTTP service.")
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        default=8765,
        help="Port to listen on (default: 8765)",
    )
    parser.add_argument(
        "-H",
        "--host",
        default="0.0.0.0",
        help="Host to bind (default: 0.0.0.0)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="Clear the configured store directory before starting",
    )
    return parser


def clear_store_dir() -> None:
    data_dir = get_settings().store.data_dir
    if not data_dir.exists():
        print(f"Store directory does not exist: {data_dir}")
        return

    shutil.rmtree(data_dir)
    print(f"Cleared treefyit state under {data_dir}")


def run_server(*, host: str, port: int) -> None:
    print(f"treefyit service -> http://{host}:{port}")
    uvicorn.run(app, host=host, port=port, log_level="info")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    configure_treefyit_logging()
    if args.clear:
        clear_store_dir()

    run_server(host=args.host, port=args.port)
