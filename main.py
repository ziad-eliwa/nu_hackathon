"""Small CLI shim for common project actions.

This file remains intentionally lightweight. Core logic lives in `src/absa`.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure local `src` package path is importable when running `python main.py`.
PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="NU Arabic ABSA utility CLI")
    subparsers = parser.add_subparsers(dest="command")

    # This subcommand delegates argument parsing to the training module.
    subparsers.add_parser(
        "train-sentiment",
        help="Train Path B aspect-conditioned sentiment model",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args, remaining = parser.parse_known_args(argv)

    if args.command == "train-sentiment":
        # Dynamic import avoids static path assumptions in editor diagnostics.
        import importlib

        train_module = importlib.import_module("absa.training.train_sentiment")
        return train_module.main(remaining)

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
