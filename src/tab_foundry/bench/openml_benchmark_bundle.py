"""Helpers for building pinned OpenML benchmark bundles."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

import openml

from tab_foundry.bench.artifacts import write_json
from tab_foundry.bench.nanotabpfn import (
    PreparedOpenMLBenchmarkTask,
    normalize_benchmark_bundle,
    prepare_openml_benchmark_task,
    read_required_openml_quality,
)
from tab_foundry.bench.openml_bundle.config import (
    OpenMLBenchmarkBundleBuildResult,
    OpenMLBenchmarkBundleConfig,
    OpenMLBenchmarkCandidateReportEntry,
    OpenMLBenchmarkTaskCandidate,
    parse_max_classes_arg as _parse_max_classes_arg_impl,
)
from tab_foundry.bench.openml_bundle.discovery import (
    collect_discovered_task_candidates as _collect_discovered_task_candidates_impl,
    task_listing_rows_for_config as _task_listing_rows_for_config_impl,
)
from tab_foundry.bench.openml_bundle.reporting import (
    build_openml_benchmark_bundle as _build_openml_benchmark_bundle_impl,
    build_openml_benchmark_bundle_result as _build_openml_benchmark_bundle_result_impl,
    render_openml_benchmark_candidate_report as _render_openml_benchmark_candidate_report_impl,
    write_openml_benchmark_bundle as _write_openml_benchmark_bundle_impl,
)
from tab_foundry.bench.openml_bundle.selection import (
    collect_task_candidates as _collect_task_candidates_impl,
    resolve_selected_tasks as _resolve_selected_tasks_impl,
)
from tab_foundry.bench.openml_task_source_registry import (
    DEFAULT_OPENML_TASK_SOURCE,
    task_ids_for_source,
    task_source_names,
)


def _task_listing_rows_for_config(config: OpenMLBenchmarkBundleConfig) -> list[Mapping[str, Any]]:
    return _task_listing_rows_for_config_impl(
        config,
        list_tasks_fn=openml.tasks.list_tasks,
    )


def _collect_discovered_task_candidates(
    config: OpenMLBenchmarkBundleConfig,
) -> tuple[list[OpenMLBenchmarkTaskCandidate], list[OpenMLBenchmarkCandidateReportEntry]]:
    return _collect_discovered_task_candidates_impl(
        config,
        task_listing_rows_fn=_task_listing_rows_for_config,
    )


def _collect_task_candidates(config: OpenMLBenchmarkBundleConfig) -> list[OpenMLBenchmarkTaskCandidate]:
    return _collect_task_candidates_impl(
        config,
        get_task_fn=openml.tasks.get_task,
        task_ids_for_source_fn=task_ids_for_source,
        read_required_openml_quality_fn=read_required_openml_quality,
    )


def _resolve_selected_tasks(
    config: OpenMLBenchmarkBundleConfig,
) -> tuple[list[PreparedOpenMLBenchmarkTask], int, tuple[OpenMLBenchmarkCandidateReportEntry, ...]]:
    return _resolve_selected_tasks_impl(
        config,
        prepare_openml_benchmark_task_fn=prepare_openml_benchmark_task,
        get_task_fn=openml.tasks.get_task,
        task_ids_for_source_fn=task_ids_for_source,
        read_required_openml_quality_fn=read_required_openml_quality,
        collect_discovered_task_candidates_fn=_collect_discovered_task_candidates,
    )


def build_openml_benchmark_bundle_result(
    config: OpenMLBenchmarkBundleConfig,
) -> OpenMLBenchmarkBundleBuildResult:
    return _build_openml_benchmark_bundle_result_impl(
        config,
        resolve_selected_tasks_fn=_resolve_selected_tasks,
        normalize_benchmark_bundle_fn=normalize_benchmark_bundle,
    )


def render_openml_benchmark_candidate_report(
    entries: Sequence[OpenMLBenchmarkCandidateReportEntry],
) -> str:
    return _render_openml_benchmark_candidate_report_impl(entries)


def build_openml_benchmark_bundle(config: OpenMLBenchmarkBundleConfig) -> dict[str, Any]:
    """Build one normalized benchmark bundle from the notebook task set."""

    return _build_openml_benchmark_bundle_impl(
        config,
        build_openml_benchmark_bundle_result_fn=build_openml_benchmark_bundle_result,
    )


def write_openml_benchmark_bundle(
    path: Path,
    config: OpenMLBenchmarkBundleConfig,
    *,
    bundle: Mapping[str, Any] | None = None,
) -> Path:
    """Write one normalized benchmark bundle to disk."""

    return _write_openml_benchmark_bundle_impl(
        path,
        config,
        bundle=bundle,
        build_openml_benchmark_bundle_fn=build_openml_benchmark_bundle,
        normalize_benchmark_bundle_fn=normalize_benchmark_bundle,
        write_json_fn=write_json,
    )


def _parse_max_classes_arg(raw_value: str) -> int | None:
    return _parse_max_classes_arg_impl(raw_value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a pinned OpenML benchmark bundle")
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
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    config = OpenMLBenchmarkBundleConfig(
        bundle_name=str(args.bundle_name),
        version=int(args.version),
        task_source=str(args.task_source),
        task_type=str(args.task_type),
        new_instances=int(args.new_instances),
        max_features=int(args.max_features),
        max_classes=_parse_max_classes_arg(str(args.max_classes)),
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
        build_result = build_openml_benchmark_bundle_result(config)
        report = render_openml_benchmark_candidate_report(build_result.report_entries)
        if report:
            print(report)
        out_path = write_openml_benchmark_bundle(
            Path(str(args.out_path)),
            config,
            bundle=build_result.bundle,
        )
    else:
        out_path = write_openml_benchmark_bundle(Path(str(args.out_path)), config)
    print(f"wrote benchmark bundle: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
