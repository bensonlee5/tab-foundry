"""CLI wiring for `tab-foundry research adequacy` commands."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tab_foundry.data.corpus_materialization import default_materialize_processes
from tab_foundry.research.adequacy.pilot import (
    finalize_adequacy_pilot,
    run_adequacy_pilot,
)

_CONTRACT_CHECK_CHOICES = ("fast", "full")


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {raw}.")
    return value


def _positive_int_or_auto(raw: str) -> int | None:
    if str(raw).strip().lower() == "auto":
        return None
    return _positive_int(raw)


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
    parser.add_argument(
        "--materialize-processes",
        type=_positive_int,
        default=default_materialize_processes(),
        help="Maximum concurrent invocation subprocesses to use while materializing pilot corpora",
    )
    parser.add_argument(
        "--materialize-worker-threads",
        type=_positive_int_or_auto,
        default=None,
        help="Per-dagzoo subprocess CPU thread budget. Use 'auto' for the balanced default.",
    )
    parser.add_argument(
        "--contract-check",
        choices=_CONTRACT_CHECK_CHOICES,
        default="fast",
        help="Latent-target contract verification level",
    )
    return parser


def configure_finalize_parser(parser: argparse.ArgumentParser) -> argparse.ArgumentParser:
    parser.add_argument(
        "--adequacy-id",
        required=True,
        help="Synthetic adequacy spec id to finalize",
    )
    parser.add_argument(
        "--dagzoo-root",
        required=True,
        help="Path to the sibling dagzoo checkout used to resolve staged corpus previews",
    )
    parser.add_argument(
        "--out-root",
        default=None,
        help="Optional output root override for pilot artifacts",
    )
    parser.add_argument(
        "--contract-check",
        choices=_CONTRACT_CHECK_CHOICES,
        default="fast",
        help="Latent-target contract verification level",
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
        materialize_processes=int(args.materialize_processes),
        materialize_worker_threads=(
            None
            if args.materialize_worker_threads is None
            else int(args.materialize_worker_threads)
        ),
        contract_check=str(args.contract_check),
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


def run_finalize_from_args(args: argparse.Namespace) -> int:
    summary = finalize_adequacy_pilot(
        adequacy_id=str(args.adequacy_id),
        dagzoo_root=Path(str(args.dagzoo_root)).expanduser().resolve(),
        out_root=(
            None
            if args.out_root is None
            else Path(str(args.out_root)).expanduser().resolve()
        ),
        contract_check=str(args.contract_check),
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
