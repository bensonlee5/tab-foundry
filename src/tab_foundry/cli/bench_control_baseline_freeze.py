"""CLI wiring for `tab-foundry bench registry freeze-baseline`."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tab_foundry.bench.control_baseline_freeze import (
    DEFAULT_BASELINE_ID,
    DEFAULT_BUDGET_CLASS,
    DEFAULT_CONFIG_PROFILE,
    DEFAULT_EXPERIMENT,
    freeze_control_baseline,
)
from tab_foundry.control_baseline_registry import default_control_baseline_registry_path


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-dir", required=True, help="Completed tab-foundry run directory")
    parser.add_argument(
        "--comparison-summary",
        required=True,
        help="Benchmark comparison_summary.json for the same run",
    )
    parser.add_argument(
        "--baseline-id",
        default=DEFAULT_BASELINE_ID,
        help="Registry id for the frozen baseline",
    )
    parser.add_argument(
        "--experiment",
        default=DEFAULT_EXPERIMENT,
        help="Logical experiment name stored in the registry entry",
    )
    parser.add_argument(
        "--config-profile",
        default=DEFAULT_CONFIG_PROFILE,
        help="Config profile name stored in the registry entry",
    )
    parser.add_argument(
        "--budget-class",
        default=DEFAULT_BUDGET_CLASS,
        help="Budget class label stored in the registry entry",
    )
    parser.add_argument(
        "--registry-path",
        default=str(default_control_baseline_registry_path()),
        help="Control baseline registry JSON path",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Freeze one canonical tab-foundry control baseline")
    configure_parser(parser)
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    result = freeze_control_baseline(
        baseline_id=str(args.baseline_id),
        experiment=str(args.experiment),
        config_profile=str(args.config_profile),
        budget_class=str(args.budget_class),
        run_dir=Path(str(args.run_dir)),
        comparison_summary_path=Path(str(args.comparison_summary)),
        registry_path=Path(str(args.registry_path)),
    )
    print("Control baseline frozen:")
    print(f"  registry_path={result['registry_path']}")
    print(f"  baseline={result['baseline']}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_from_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
