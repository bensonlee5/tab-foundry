"""CLI wiring for `tab-foundry research sweep graph`."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from tab_foundry.cli.click_utils import run_click_command, sweep_path_options
from tab_foundry.research.sweep.graph import GraphPaths, render_sweep_graphs


def _graph_command(
    *,
    sweep_id: str,
    anchor: bool,
    all_rows: bool,
    order: tuple[int, ...],
    delta_ref: tuple[str, ...],
    out_dir: Path | None,
    catalog_path: Path,
    index_path: Path,
    registry_path: Path,
    sweeps_root: Path,
) -> int:
    result = render_sweep_graphs(
        sweep_id=sweep_id,
        anchor=anchor,
        all_rows=all_rows,
        orders=list(order),
        delta_refs=list(delta_ref),
        out_dir=out_dir,
        paths=GraphPaths(
            index_path=index_path.expanduser().resolve(),
            catalog_path=catalog_path.expanduser().resolve(),
            sweeps_root=sweeps_root.expanduser().resolve(),
            registry_path=registry_path.expanduser().resolve(),
        ),
    )
    print(
        "Sweep graph render complete.",
        f"sweep_id={result['sweep_id']}",
        f"graphs={len(result['graphs'])}",
        f"index={result['index_path']}",
        flush=True,
    )
    return 0


@click.command(name="graph", help="Render torchview architecture graphs for sweep targets")
@click.option("--sweep-id", required=True, help="Sweep id to inspect")
@click.option("--anchor", is_flag=True, help="Render the selected sweep anchor graph")
@click.option("--all-rows", is_flag=True, help="Render graphs for every row in the sweep")
@click.option("--order", multiple=True, type=int, help="Specific queue order to render")
@click.option("--delta-ref", multiple=True, help="Specific delta_ref / materialized delta_id to render; repeatable")
@click.option("--out-dir", default=None, type=click.Path(path_type=Path), help="Optional output directory; defaults to outputs/staged_ladder/research/<sweep_id>/architecture_graphs")
@sweep_path_options(include_registry=True, include_sweeps_root=True)
def COMMAND(
    sweep_id: str,
    anchor: bool,
    all_rows: bool,
    order: tuple[int, ...],
    delta_ref: tuple[str, ...],
    out_dir: Path | None,
    catalog_path: Path,
    index_path: Path,
    registry_path: Path,
    sweeps_root: Path,
) -> int:
    return _graph_command(
        sweep_id=sweep_id,
        anchor=anchor,
        all_rows=all_rows,
        order=order,
        delta_ref=delta_ref,
        out_dir=out_dir,
        catalog_path=catalog_path,
        index_path=index_path,
        registry_path=registry_path,
        sweeps_root=sweeps_root,
    )


def main(argv: list[str] | None = None) -> int:
    return run_click_command(COMMAND, argv, prog_name="tab-foundry research sweep graph")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
