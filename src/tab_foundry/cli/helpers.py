"""Shared CLI registration helpers."""

from __future__ import annotations

import argparse
from typing import Callable, Sequence


DelegateHandler = Callable[[Sequence[str] | None], int]


def _run_delegate(args: argparse.Namespace) -> int:
    argv = [str(value) for value in getattr(args, "argv", [])]
    delegate = args.delegate
    return int(delegate(None if not argv else argv))


def register_delegate_leaf(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    *,
    help: str,
    delegate: DelegateHandler,
    description: str | None = None,
) -> argparse.ArgumentParser:
    parser = subparsers.add_parser(
        name,
        help=help,
        description=description,
        add_help=False,
    )
    parser.set_defaults(func=_run_delegate, delegate=delegate, allow_unknown_args=True)
    return parser
