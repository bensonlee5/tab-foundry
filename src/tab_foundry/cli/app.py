"""CLI parser assembly."""

from __future__ import annotations

import argparse
from typing import Sequence

from .groups import bench, data, eval_, export, research, train


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="tab-foundry tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    data.register(subparsers)
    train.register(subparsers)
    eval_.register(subparsers)
    export.register(subparsers)
    bench.register(subparsers)
    research.register(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, extras = parser.parse_known_args(argv)
    if extras:
        if getattr(args, "allow_unknown_args", False):
            args.argv = extras
        else:
            parser.error(f"unrecognized arguments: {' '.join(extras)}")
    return int(args.func(args))
