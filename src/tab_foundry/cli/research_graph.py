"""CLI wiring for `tab-foundry research sweep graph`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tab_foundry.research.sweep.graph import GraphPaths, render_sweep_graphs
from tab_foundry.research.sweep.paths_io import (
    default_catalog_path,
    default_registry_path,
    default_sweep_index_path,
    default_sweeps_root,
)


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--sweep-id", default=None, help="Sweep id to inspect; defaults to the active sweep")
    parser.add_argument("--anchor", action="store_true", help="Render the selected sweep anchor graph")
    parser.add_argument("--all-rows", action="store_true", help="Render graphs for every row in the sweep")
    parser.add_argument("--order", type=int, action="append", default=[], help="Specific queue order to render")
    parser.add_argument(
        "--delta-ref",
        action="append",
        default=[],
        help="Specific delta_ref / materialized delta_id to render; repeatable",
    )
    parser.add_argument(
        "--out-dir",
        default=None,
        help="Optional output directory; defaults to outputs/staged_ladder/research/<sweep_id>/architecture_graphs",
    )
    parser.add_argument(
        "--catalog-path",
        default=str(default_catalog_path()),
        help="Path to reference/system_delta_catalog.yaml",
    )
    parser.add_argument(
        "--index-path",
        default=str(default_sweep_index_path()),
        help="Path to reference/system_delta_sweeps/index.yaml",
    )
    parser.add_argument(
        "--registry-path",
        default=str(default_registry_path()),
        help="Path to benchmark_run_registry_v1.json",
    )
    parser.add_argument(
        "--sweeps-root",
        default=str(default_sweeps_root()),
        help="Path to reference/system_delta_sweeps/",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    return configure_parser(
        argparse.ArgumentParser(description="Render torchview architecture graphs for sweep targets")
    )


def run_from_args(args: argparse.Namespace) -> int:
    result = render_sweep_graphs(
        sweep_id=None if args.sweep_id is None else str(args.sweep_id),
        anchor=bool(args.anchor),
        all_rows=bool(args.all_rows),
        orders=[int(value) for value in args.order],
        delta_refs=[str(value) for value in args.delta_ref],
        out_dir=None if args.out_dir is None else Path(str(args.out_dir)),
        paths=GraphPaths(
            index_path=Path(str(args.index_path)).expanduser().resolve(),
            catalog_path=Path(str(args.catalog_path)).expanduser().resolve(),
            sweeps_root=Path(str(args.sweeps_root)).expanduser().resolve(),
            registry_path=Path(str(args.registry_path)).expanduser().resolve(),
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


def main(argv: list[str] | None = None) -> int:
    return run_from_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
