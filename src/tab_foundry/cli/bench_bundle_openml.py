"""CLI wiring for `tab-foundry bench bundle build-openml`."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import tab_foundry.bench.openml_benchmark_bundle as bundle_module
from tab_foundry.bench.openml_bundle.config import parse_max_classes_arg
from tab_foundry.bench.openml_task_source_registry import (
    DEFAULT_OPENML_TASK_SOURCE,
    task_source_names,
)


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--out-path", required=True, help="JSON output path for the bundle")
    parser.add_argument("--bundle-name", required=True, help="Bundle name persisted in the JSON payload")
    parser.add_argument("--version", type=int, required=True, help="Bundle version persisted in the JSON payload")
    parser.add_argument(
        "--task-source",
        default=DEFAULT_OPENML_TASK_SOURCE,
        choices=task_source_names(),
        help="Named pinned OpenML task-id source pool used before applying bundle filters",
    )
    parser.add_argument(
        "--discover-from-openml",
        action="store_true",
        help="Query OpenML task metadata directly instead of using a pinned task-source registry",
    )
    parser.add_argument("--new-instances", type=int, default=200, help="Subsampled row count used by the benchmark")
    parser.add_argument(
        "--min-instances",
        type=int,
        default=1,
        help="Minimum raw dataset row count required during OpenML discovery",
    )
    parser.add_argument(
        "--min-task-count",
        type=int,
        default=1,
        help="Minimum validated task count required after OpenML discovery",
    )
    parser.add_argument(
        "--task-type",
        default="supervised_classification",
        choices=("supervised_classification", "supervised_regression"),
        help="OpenML task type used when building the benchmark bundle",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=10,
        help="Maximum raw OpenML feature count allowed by the bundle filter",
    )
    parser.add_argument(
        "--max-classes",
        default="2",
        help="Maximum class count filter, or 'auto' to widen to the highest eligible class count",
    )
    parser.add_argument(
        "--max-missing-pct",
        type=float,
        default=0.0,
        help="Maximum allowed percentage of instances with missing values",
    )
    parser.add_argument(
        "--min-minority-class-pct",
        type=float,
        default=2.5,
        help="Minimum required minority class percentage",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a pinned OpenML benchmark bundle")
    configure_parser(parser)
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    config = bundle_module.OpenMLBenchmarkBundleConfig(
        bundle_name=str(args.bundle_name),
        version=int(args.version),
        task_source=str(args.task_source),
        task_type=str(args.task_type),
        new_instances=int(args.new_instances),
        max_features=int(args.max_features),
        max_classes=parse_max_classes_arg(str(args.max_classes)),
        max_missing_pct=float(args.max_missing_pct),
        min_minority_class_pct=float(args.min_minority_class_pct),
        discover_from_openml=bool(args.discover_from_openml),
        min_instances=int(args.min_instances),
        min_task_count=int(args.min_task_count),
    )
    if config.min_instances <= 0:
        raise ValueError("min_instances must be a positive int")
    if config.min_task_count <= 0:
        raise ValueError("min_task_count must be a positive int")
    if config.discover_from_openml:
        build_result = bundle_module.build_openml_benchmark_bundle_result(config)
        report = bundle_module.render_openml_benchmark_candidate_report(build_result.report_entries)
        if report:
            print(report)
        out_path = bundle_module.write_openml_benchmark_bundle(
            Path(str(args.out_path)),
            config,
            bundle=build_result.bundle,
        )
    else:
        out_path = bundle_module.write_openml_benchmark_bundle(Path(str(args.out_path)), config)
    print(f"wrote benchmark bundle: {out_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_from_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
