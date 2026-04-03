"""CLI wiring for `tab-foundry research sweep inspect`."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from tab_foundry.cli.click_utils import (
    emit_payload,
    json_output_option,
    run_click_command,
    sweep_path_options,
)
from tab_foundry.research.sweep.inspect import inspect_sweep_row, render_sweep_row_text


def _inspect_command(
    *,
    order: int,
    sweep_id: str,
    json_mode: bool,
    catalog_path: Path,
    index_path: Path,
    sweeps_root: Path,
    registry_path: Path,
) -> int:
    payload = inspect_sweep_row(
        order=order,
        sweep_id=sweep_id,
        index_path=index_path.expanduser().resolve(),
        catalog_path=catalog_path.expanduser().resolve(),
        sweeps_root=sweeps_root.expanduser().resolve(),
        registry_path=registry_path.expanduser().resolve(),
    )
    emit_payload(payload, json_mode=json_mode, render_text=render_sweep_row_text)
    return 0


@click.command(name="inspect", help="Inspect one materialized sweep row and its resolved surfaces")
@click.option("--order", required=True, type=int, help="Row order to inspect")
@click.option("--sweep-id", required=True, help="Sweep id to inspect")
@json_output_option
@sweep_path_options(include_registry=True, include_sweeps_root=True)
def COMMAND(
    order: int,
    sweep_id: str,
    json_mode: bool,
    catalog_path: Path,
    index_path: Path,
    sweeps_root: Path,
    registry_path: Path,
) -> int:
    return _inspect_command(
        order=order,
        sweep_id=sweep_id,
        json_mode=json_mode,
        catalog_path=catalog_path,
        index_path=index_path,
        sweeps_root=sweeps_root,
        registry_path=registry_path,
    )


def main(argv: list[str] | None = None) -> int:
    return run_click_command(COMMAND, argv, prog_name="tab-foundry research sweep inspect")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
