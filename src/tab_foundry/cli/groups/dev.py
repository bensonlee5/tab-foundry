"""Developer tooling CLI group."""

from __future__ import annotations

import argparse
from typing import Sequence

from ..helpers import register_delegate_leaf


def _prepend_command(command: str, argv: Sequence[str] | None = None) -> list[str]:
    return [command, *(list(argv) if argv is not None else [])]


def _run_resolve_config(argv: Sequence[str] | None = None) -> int:
    from tab_foundry.cli.dev import main as dev_main

    return dev_main(_prepend_command("resolve-config", argv))


def _run_forward_check(argv: Sequence[str] | None = None) -> int:
    from tab_foundry.cli.dev import main as dev_main

    return dev_main(_prepend_command("forward-check", argv))


def _run_health_check(argv: Sequence[str] | None = None) -> int:
    from tab_foundry.cli.dev import main as dev_main

    return dev_main(_prepend_command("health-check", argv))


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("dev", help="Developer-focused inspection tools")
    nested = parser.add_subparsers(dest="dev_command", required=True)
    register_delegate_leaf(
        nested,
        "resolve-config",
        help="Compose one config and print the resolved build surface",
        delegate=_run_resolve_config,
    )
    register_delegate_leaf(
        nested,
        "forward-check",
        help="Build one model and run a synthetic forward-only smoke check",
        delegate=_run_forward_check,
    )
    register_delegate_leaf(
        nested,
        "health-check",
        help="Summarize run telemetry and instability signals",
        delegate=_run_health_check,
    )
