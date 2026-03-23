"""Manual benchmark comparison CLI against external baselines."""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys
from typing import Any, Sequence

from tab_foundry.bench.comparison_runtime import (
    DEFAULT_NANOTABPFN_SEEDS,
    DEFAULT_NANOTABPFN_STEPS,
    DEFAULT_TABICL_CLASSIFIER_CHECKPOINT_VERSION,
    DEFAULT_TABICL_REGRESSOR_CHECKPOINT_VERSION,
    NanoTabPFNBenchmarkConfig,
    run_nanotabpfn_benchmark,
)
from tab_foundry.control_baseline_registry import default_control_baseline_registry_path
from tab_foundry.external_benchmarks import (
    ALLOWED_EXTERNAL_BENCHMARKS,
    DEFAULT_CLI_EXTERNAL_BENCHMARKS,
    EXTERNAL_BENCHMARK_TABICLV2,
    normalize_external_benchmarks,
)


def _default_out_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return Path("/tmp") / f"tab_foundry_benchmark_{stamp}"


def _optional_non_empty_string(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return str(value).strip()


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--tab-foundry-run-dir",
        required=True,
        help="Completed tab-foundry run directory with checkpoint snapshots",
    )
    parser.add_argument("--nanotabpfn-root", default="~/dev/nanoTabPFN", help="Local nanoTabPFN checkout")
    parser.add_argument("--nanotab-prior-dump", default=None, help="Path to nanoTabPFN prior dump (.h5)")
    parser.add_argument(
        "--external-benchmark",
        action="append",
        choices=ALLOWED_EXTERNAL_BENCHMARKS,
        default=None,
        help="Ordered external benchmark to run; repeat to add a secondary comparator. Defaults to tabiclv2.",
    )
    parser.add_argument(
        "--with-tabiclv2",
        action="store_true",
        help="Deprecated alias that appends tabiclv2 to --external-benchmark for one release.",
    )
    parser.add_argument(
        "--tabicl-root",
        default="~/dev/tabicl",
        help="Local TabICLv2 checkout used when tabiclv2 is selected",
    )
    parser.add_argument(
        "--tabicl-classifier-checkpoint-version",
        default=DEFAULT_TABICL_CLASSIFIER_CHECKPOINT_VERSION,
        help="TabICLv2 classifier checkpoint version used when tabiclv2 is selected",
    )
    parser.add_argument(
        "--tabicl-regressor-checkpoint-version",
        default=DEFAULT_TABICL_REGRESSOR_CHECKPOINT_VERSION,
        help="TabICLv2 regressor checkpoint version used when tabiclv2 is selected",
    )
    parser.add_argument("--out-root", default=None, help="Output directory root")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("cpu", "cuda", "mps", "auto"),
        help="Benchmark device",
    )
    parser.add_argument(
        "--nanotabpfn-steps",
        type=int,
        default=DEFAULT_NANOTABPFN_STEPS,
        help="nanoTabPFN training steps",
    )
    parser.add_argument(
        "--nanotabpfn-seeds",
        type=int,
        default=DEFAULT_NANOTABPFN_SEEDS,
        help="Number of nanoTabPFN random seeds",
    )
    parser.add_argument(
        "--control-baseline-id",
        default=None,
        help="Optional frozen control baseline id to copy into comparison_summary.json",
    )
    parser.add_argument(
        "--control-baseline-registry",
        default=str(default_control_baseline_registry_path()),
        help="Control baseline registry JSON path used with --control-baseline-id",
    )
    parser.add_argument(
        "--benchmark-bundle-path",
        default=None,
        help="Optional repo-tracked OpenML benchmark bundle JSON path",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Compare a completed tab-foundry run against external baselines"
    )
    configure_parser(parser)
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    external_benchmarks = (
        []
        if args.external_benchmark is None
        else [str(value) for value in args.external_benchmark]
    )
    if bool(args.with_tabiclv2):
        print(
            "warning: --with-tabiclv2 is deprecated; use --external-benchmark tabiclv2 instead",
            file=sys.stderr,
        )
        if EXTERNAL_BENCHMARK_TABICLV2 not in external_benchmarks:
            external_benchmarks.append(EXTERNAL_BENCHMARK_TABICLV2)
    summary = run_nanotabpfn_benchmark(
        NanoTabPFNBenchmarkConfig(
            tab_foundry_run_dir=Path(str(args.tab_foundry_run_dir)),
            out_root=_default_out_root() if args.out_root is None else Path(str(args.out_root)),
            nanotabpfn_root=Path(str(args.nanotabpfn_root)),
            nanotab_prior_dump=(Path(str(args.nanotab_prior_dump)) if args.nanotab_prior_dump else None),
            device=str(args.device),
            nanotabpfn_steps=int(args.nanotabpfn_steps),
            nanotabpfn_seeds=int(args.nanotabpfn_seeds),
            control_baseline_id=(
                str(args.control_baseline_id) if args.control_baseline_id else None
            ),
            control_baseline_registry=(
                Path(str(args.control_baseline_registry))
                if args.control_baseline_registry
                else None
            ),
            benchmark_bundle_path=(
                Path(str(args.benchmark_bundle_path))
                if args.benchmark_bundle_path
                else None
            ),
            external_benchmarks=normalize_external_benchmarks(
                external_benchmarks,
                default=DEFAULT_CLI_EXTERNAL_BENCHMARKS,
                context="CLI external benchmarks",
            ),
            with_tabiclv2=bool(args.with_tabiclv2),
            tabicl_root=Path(str(args.tabicl_root)),
            tabicl_classifier_checkpoint_version=str(args.tabicl_classifier_checkpoint_version),
            tabicl_regressor_checkpoint_version=str(args.tabicl_regressor_checkpoint_version),
        )
    )
    print("benchmark comparison complete:")
    print(f"  dataset_count={summary['dataset_count']}")
    print(f"  tab_foundry={summary['tab_foundry']}")
    primary_external_benchmark = _optional_non_empty_string(summary.get("primary_external_benchmark"))
    if primary_external_benchmark is not None:
        print(f"  primary_external_benchmark={primary_external_benchmark}")
    if "nanotabpfn" in summary:
        print(f"  nanotabpfn={summary['nanotabpfn']}")
    if "tabiclv2" in summary:
        print(f"  tabiclv2={summary['tabiclv2']}")
    print(f"  artifacts={summary['artifacts']}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    return run_from_args(parser.parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
