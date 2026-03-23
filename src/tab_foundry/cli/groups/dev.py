"""Developer tooling CLI group."""

from __future__ import annotations

import argparse

from .. import dev as dev_cli


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("dev", help="Developer-focused inspection tools")
    dev_cli.register_subparsers(parser.add_subparsers(dest="dev_command", required=True))
