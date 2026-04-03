"""Packaged click CLI assembly."""

from __future__ import annotations

from typing import Sequence

import click

from .groups import bench, data, dev, eval_, export, research, train
from .click_utils import GROUP_KWARGS, run_click_command


@click.group(name="tab-foundry", help="tab-foundry tooling", **GROUP_KWARGS)
def cli() -> None:
    """Root click command."""


cli.add_command(data.GROUP)
cli.add_command(dev.GROUP)
cli.add_command(train.GROUP)
cli.add_command(eval_.GROUP)
cli.add_command(export.GROUP)
cli.add_command(bench.GROUP)
cli.add_command(research.GROUP)


def main(argv: Sequence[str] | None = None) -> int:
    return run_click_command(cli, argv, prog_name="tab-foundry")
