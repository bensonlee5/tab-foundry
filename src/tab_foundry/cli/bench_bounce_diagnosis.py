"""CLI wiring for `tab-foundry bench diagnose bounce`."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import cast

from tab_foundry.benchmark_registry import default_benchmark_run_registry_path
from tab_foundry.bench.bounce.config import (
    DEFAULT_BOUNCE_BOOTSTRAP_CONFIDENCE,
    DEFAULT_BOUNCE_BOOTSTRAP_SAMPLES,
    RerunMode,
    default_out_root,
)
from tab_foundry.bench.bounce.rerun import resolve_run_dir_from_registry
from tab_foundry.bench.bounce_diagnosis import BenchmarkBounceDiagnosisConfig, run_benchmark_bounce_diagnosis


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", default=None, help="Completed run directory to diagnose")
    parser.add_argument("--run-id", default=None, help="Benchmark registry run id to diagnose")
    parser.add_argument(
        "--registry-path",
        default=str(default_benchmark_run_registry_path()),
        help="Benchmark run registry used with --run-id",
    )
    parser.add_argument("--out-root", default=None, help="Output directory root")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("cpu", "cuda", "mps", "auto"),
        help="Benchmark device",
    )
    parser.add_argument(
        "--benchmark-manifest-path",
        default=None,
        help="Primary benchmark manifest path; defaults to the canonical medium manifest location",
    )
    parser.add_argument(
        "--confirmation-benchmark-manifest-path",
        default=None,
        help="Optional confirmation benchmark manifest path; omit to stay on the primary no-missing surface only",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOUNCE_BOOTSTRAP_SAMPLES,
        help="Task-bootstrap samples per checkpoint",
    )
    parser.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=DEFAULT_BOUNCE_BOOTSTRAP_CONFIDENCE,
        help="Task-bootstrap confidence level",
    )
    parser.add_argument(
        "--dense-checkpoint-every",
        type=int,
        default=None,
        help="Optional diagnosis-only rerun checkpoint cadence",
    )
    parser.add_argument(
        "--dense-run-dir",
        default=None,
        help="Optional precomputed dense-checkpoint run directory",
    )
    parser.add_argument(
        "--rerun-mode",
        default="none",
        choices=("auto", "prior", "train", "none"),
        help="How to rerun when --dense-checkpoint-every is set",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose checkpoint-level benchmark bounce for one run")
    configure_parser(parser)
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    if bool(args.run_dir) == bool(args.run_id):
        raise SystemExit("exactly one of --run-dir or --run-id must be provided")
    run_id = None if args.run_id is None else str(args.run_id)
    run_dir = (
        Path(str(args.run_dir))
        if args.run_dir is not None
        else resolve_run_dir_from_registry(
            str(args.run_id),
            registry_path=Path(str(args.registry_path)) if args.registry_path else None,
        )
    )
    summary = run_benchmark_bounce_diagnosis(
        BenchmarkBounceDiagnosisConfig(
            run_dir=run_dir,
            out_root=default_out_root(run_dir)
            if args.out_root is None
            else Path(str(args.out_root)),
            device=str(args.device),
            benchmark_manifest_path=(
                None if args.benchmark_manifest_path is None else Path(str(args.benchmark_manifest_path))
            ),
            confirmation_benchmark_manifest_path=(
                None
                if args.confirmation_benchmark_manifest_path is None
                else Path(str(args.confirmation_benchmark_manifest_path))
            ),
            bootstrap_samples=int(args.bootstrap_samples),
            bootstrap_confidence=float(args.bootstrap_confidence),
            dense_checkpoint_every=(
                None if args.dense_checkpoint_every is None else int(args.dense_checkpoint_every)
            ),
            dense_run_dir=None if args.dense_run_dir is None else Path(str(args.dense_run_dir)),
            rerun_mode=cast(RerunMode, str(args.rerun_mode)),
            run_id=run_id,
        )
    )
    print("benchmark bounce diagnosis complete:")
    print(f"  run_dir={summary['run_dir']}")
    print(f"  causes={summary['classification']['primary_causes']}")
    print(f"  artifacts={summary['artifacts']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_from_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
