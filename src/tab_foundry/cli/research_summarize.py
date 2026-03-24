"""CLI wiring for `tab-foundry research sweep summarize`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from tab_foundry.research.sweep.paths_io import (
    default_catalog_path,
    default_sweep_index_path,
    default_sweeps_root,
)
from tab_foundry.research.sweep.summarize import render_sweep_summary_table, summarize_sweep


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument("--sweep-id", default=None, help="Sweep id to inspect; defaults to the active sweep")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument(
        "--include-screened",
        action="store_true",
        help="Include screened rows alongside completed or blocked rows",
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
    return parser


def build_parser() -> argparse.ArgumentParser:
    return configure_parser(
        argparse.ArgumentParser(description="Summarize local system-delta sweep results")
    )


def run_from_args(args: argparse.Namespace) -> int:
    payload = summarize_sweep(
        sweep_id=None if args.sweep_id is None else str(args.sweep_id),
        include_screened=bool(args.include_screened),
        index_path=Path(str(args.index_path)).expanduser().resolve(),
        catalog_path=Path(str(args.catalog_path)).expanduser().resolve(),
        sweeps_root=Path(str(args.sweeps_root)).expanduser().resolve(),
    )
    if bool(args.json):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_sweep_summary_table(payload))
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_from_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
