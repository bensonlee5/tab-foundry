"""CLI parser assembly."""

from __future__ import annotations

import argparse
from typing import Sequence

from .commands import build_manifest, build_preprocessor_state, evaluate, export, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="tab-foundry tooling")
    sub = parser.add_subparsers(dest="command", required=True)

    build_manifest.register(sub)
    build_preprocessor_state.register(sub)
    train.register(sub)
    evaluate.register(sub)
    export.register(sub)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))
