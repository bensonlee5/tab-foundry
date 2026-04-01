"""CLI wiring for `tab-foundry research adequacy pilot`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tab_foundry.research.adequacy.pilot import run_adequacy_pilot


def configure_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--adequacy-id",
        required=True,
        help="Synthetic adequacy spec id to execute",
    )
    parser.add_argument(
        "--dagzoo-root",
        required=True,
        help="Path to the sibling dagzoo checkout used for corpus materialization",
    )
    parser.add_argument(
        "--device",
        choices=("cpu",),
        default="cpu",
        help="Pilot execution device. The lean adequacy pilot supports CPU only.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force corpus rematerialization and overwrite pilot-local outputs",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="Optional output root override for pilot artifacts",
    )
    return parser


def build_parser() -> argparse.ArgumentParser:
    return configure_parser(argparse.ArgumentParser(description="Run the lean synthetic adequacy pilot"))


def run_from_args(args: argparse.Namespace) -> int:
    summary = run_adequacy_pilot(
        adequacy_id=str(args.adequacy_id),
        dagzoo_root=Path(str(args.dagzoo_root)).expanduser().resolve(),
        device=str(args.device),
        force=bool(args.force),
        out_root=(
            None
            if args.out_root is None
            else Path(str(args.out_root)).expanduser().resolve()
        ),
    )
    summary_paths = summary.get("summary_paths", {})
    print(
        "Adequacy pilot complete.",
        f"adequacy_id={summary['adequacy_id']}",
        f"interpretation={summary['provisional_interpretation']['bucket']}",
        f"summary={summary_paths.get('summary_md')}",
        flush=True,
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_from_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
