"""CLI wiring for `tab-foundry research sweep diff`."""

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
from tab_foundry.research.sweep.diff import diff_sweep_row, render_sweep_diff_text


def _diff_command(
    *,
    order: int,
    sweep_id: str,
    json_mode: bool,
    against: str,
    against_order: int | None,
    catalog_path: Path,
    index_path: Path,
    sweeps_root: Path,
    registry_path: Path,
) -> int:
    payload = diff_sweep_row(
        order=order,
        sweep_id=sweep_id,
        against=against,
        against_order=against_order,
        index_path=index_path.expanduser().resolve(),
        catalog_path=catalog_path.expanduser().resolve(),
        sweeps_root=sweeps_root.expanduser().resolve(),
        registry_path=registry_path.expanduser().resolve(),
    )
    emit_payload(payload, json_mode=json_mode, render_text=render_sweep_diff_text)
    return 0


@click.command(name="diff", help="Diff one materialized sweep row against the anchor or another row")
@click.option("--order", required=True, type=int, help="Row order to diff")
@click.option("--sweep-id", required=True, help="Sweep id to inspect")
@json_output_option
@click.option("--against", default="anchor", show_default=True, help="Baseline target; only 'anchor' is supported when --against-order is omitted")
@click.option("--against-order", default=None, type=int, help="Compare against another sweep row order instead of the anchor")
@sweep_path_options(include_registry=True, include_sweeps_root=True)
def COMMAND(
    order: int,
    sweep_id: str,
    json_mode: bool,
    against: str,
    against_order: int | None,
    catalog_path: Path,
    index_path: Path,
    sweeps_root: Path,
    registry_path: Path,
) -> int:
    return _diff_command(
        order=order,
        sweep_id=sweep_id,
        json_mode=json_mode,
        against=against,
        against_order=against_order,
        catalog_path=catalog_path,
        index_path=index_path,
        sweeps_root=sweeps_root,
        registry_path=registry_path,
    )


def main(argv: list[str] | None = None) -> int:
    return run_click_command(COMMAND, argv, prog_name="tab-foundry research sweep diff")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
