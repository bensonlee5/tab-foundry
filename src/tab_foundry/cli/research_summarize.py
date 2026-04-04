"""CLI wiring for `tab-foundry research sweep summarize`."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from tab_foundry.cli.click_utils import emit_payload, json_output_option, run_click_command, sweep_path_options
from tab_foundry.research.sweep.summarize import render_sweep_summary_table, summarize_sweep


def _summarize_command(
    *,
    sweep_id: str,
    json_mode: bool,
    include_screened: bool,
    catalog_path: Path,
    index_path: Path,
    sweeps_root: Path,
) -> int:
    payload = summarize_sweep(
        sweep_id=sweep_id,
        include_screened=include_screened,
        index_path=index_path.expanduser().resolve(),
        catalog_path=catalog_path.expanduser().resolve(),
        sweeps_root=sweeps_root.expanduser().resolve(),
    )
    emit_payload(payload, json_mode=json_mode, render_text=render_sweep_summary_table)
    return 0


@click.command(name="summarize", help="Summarize local sweep results into one compact table")
@click.option("--sweep-id", required=True, help="Sweep id to inspect")
@json_output_option
@click.option("--include-screened", is_flag=True, help="Include screened rows alongside completed or blocked rows")
@sweep_path_options(include_registry=False, include_sweeps_root=True)
def COMMAND(
    sweep_id: str,
    json_mode: bool,
    include_screened: bool,
    catalog_path: Path,
    index_path: Path,
    sweeps_root: Path,
) -> int:
    return _summarize_command(
        sweep_id=sweep_id,
        json_mode=json_mode,
        include_screened=include_screened,
        catalog_path=catalog_path,
        index_path=index_path,
        sweeps_root=sweeps_root,
    )


def main(argv: list[str] | None = None) -> int:
    return run_click_command(COMMAND, argv, prog_name="tab-foundry research sweep summarize")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
