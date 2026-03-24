"""CLI wiring for `tab-foundry bench registry register-run`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tab_foundry.benchmark_registry import default_benchmark_run_registry_path
from tab_foundry.bench.run_registration import DEFAULT_BUDGET_CLASS, ALLOWED_DECISIONS, register_benchmark_run


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", required=True, help="Canonical registry id for the run")
    parser.add_argument(
        "--track",
        required=True,
        help="Logical track label, e.g. binary_ladder or many_class_branch",
    )
    parser.add_argument("--run-dir", required=True, help="Completed tab-foundry run directory")
    parser.add_argument(
        "--comparison-summary",
        required=True,
        help="Benchmark comparison_summary.json for the same run",
    )
    parser.add_argument("--experiment", required=True, help="Logical experiment name stored in the registry")
    parser.add_argument(
        "--config-profile",
        default=None,
        help="Config profile stored in the registry entry; defaults to --experiment",
    )
    parser.add_argument(
        "--budget-class",
        default=DEFAULT_BUDGET_CLASS,
        help="Budget class label stored in the registry entry",
    )
    parser.add_argument(
        "--decision",
        required=True,
        choices=ALLOWED_DECISIONS,
        help="Human review decision stored with the run",
    )
    parser.add_argument("--conclusion", required=True, help="One-line keep/reject/defer conclusion")
    parser.add_argument("--parent-run-id", default=None, help="Optional previous-stage benchmark run id")
    parser.add_argument("--anchor-run-id", default=None, help="Optional frozen anchor run id")
    parser.add_argument("--prior-dir", default=None, help="Optional prior-training artifact directory")
    parser.add_argument(
        "--control-baseline-id",
        default=None,
        help="Optional frozen control baseline id associated with the run",
    )
    parser.add_argument("--sweep-id", default=None, help="Optional sweep id associated with the run")
    parser.add_argument("--delta-id", default=None, help="Optional delta id associated with the run")
    parser.add_argument(
        "--parent-sweep-id",
        default=None,
        help="Optional parent sweep id associated with the run",
    )
    parser.add_argument(
        "--queue-order",
        default=None,
        type=int,
        help="Optional positive queue order within the sweep",
    )
    parser.add_argument(
        "--run-kind",
        default=None,
        choices=("primary", "followup"),
        help="Optional sweep-local run kind",
    )
    parser.add_argument(
        "--registry-path",
        default=str(default_benchmark_run_registry_path()),
        help="Benchmark run registry JSON path",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register one completed benchmark-facing tab-foundry run"
    )
    configure_parser(parser)
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    result = register_benchmark_run(
        run_id=str(args.run_id),
        track=str(args.track),
        experiment=str(args.experiment),
        config_profile=str(args.config_profile or args.experiment),
        budget_class=str(args.budget_class),
        run_dir=Path(str(args.run_dir)),
        comparison_summary_path=Path(str(args.comparison_summary)),
        decision=str(args.decision),
        conclusion=str(args.conclusion),
        parent_run_id=None if args.parent_run_id is None else str(args.parent_run_id),
        anchor_run_id=None if args.anchor_run_id is None else str(args.anchor_run_id),
        prior_dir=None if args.prior_dir is None else Path(str(args.prior_dir)),
        control_baseline_id=(
            None if args.control_baseline_id is None else str(args.control_baseline_id)
        ),
        sweep_id=None if args.sweep_id is None else str(args.sweep_id),
        delta_id=None if args.delta_id is None else str(args.delta_id),
        parent_sweep_id=None if args.parent_sweep_id is None else str(args.parent_sweep_id),
        queue_order=None if args.queue_order is None else int(args.queue_order),
        run_kind=None if args.run_kind is None else str(args.run_kind),
        registry_path=Path(str(args.registry_path)),
    )
    print("Benchmark run registered:")
    print(f"  registry_path={result['registry_path']}")
    print(f"  run={result['run']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_from_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
