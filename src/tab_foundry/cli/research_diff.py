"""CLI wiring for `tab-foundry research sweep diff`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tab_foundry.research.sweep.diff import diff_sweep_row, render_sweep_diff_text
from tab_foundry.research.sweep.paths_io import (
    default_catalog_path,
    default_registry_path,
    default_sweep_index_path,
    default_sweeps_root,
)


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--order", type=int, required=True, help="Row order to diff")
    parser.add_argument("--sweep-id", required=True, help="Sweep id to inspect")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument(
        "--against",
        default="anchor",
        help="Baseline target; only 'anchor' is supported when --against-order is omitted",
    )
    parser.add_argument(
        "--against-order",
        type=int,
        default=None,
        help="Compare against another sweep row order instead of the anchor",
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
        "--sweeps-root",
        default=str(default_sweeps_root()),
        help="Path to reference/system_delta_sweeps/",
    )
    parser.add_argument(
        "--registry-path",
        default=str(default_registry_path()),
        help="Path to the benchmark run registry",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    return configure_parser(
        argparse.ArgumentParser(description="Diff one system-delta sweep row against anchor or another row")
    )


def run_from_args(args: argparse.Namespace) -> int:
    payload = diff_sweep_row(
        order=int(args.order),
        sweep_id=str(args.sweep_id),
        against=str(args.against),
        against_order=None if args.against_order is None else int(args.against_order),
        index_path=Path(str(args.index_path)).expanduser().resolve(),
        catalog_path=Path(str(args.catalog_path)).expanduser().resolve(),
        sweeps_root=Path(str(args.sweeps_root)).expanduser().resolve(),
        registry_path=Path(str(args.registry_path)).expanduser().resolve(),
    )
    if bool(args.json):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_sweep_diff_text(payload))
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_from_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
