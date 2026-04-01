"""OpenML benchmark bundle helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import openml

from tab_foundry.bench.artifacts import write_json
from tab_foundry.bench.openml_benchmark import (
    normalize_benchmark_bundle,
    prepare_openml_benchmark_task,
    read_required_openml_quality,
)
from tab_foundry.bench.openml_bundle.config import (
    OpenMLBenchmarkBundleBuildResult,
    OpenMLBenchmarkBundleConfig,
    OpenMLBenchmarkCandidateReportEntry,
    OpenMLBenchmarkTaskCandidate,
    parse_max_classes_arg,
)
from tab_foundry.bench.openml_bundle.discovery import (
    collect_discovered_task_candidates,
    task_listing_rows_for_config,
)
from tab_foundry.bench.openml_bundle.reporting import (
    build_openml_benchmark_bundle as _build_openml_benchmark_bundle,
    build_openml_benchmark_bundle_result as _build_openml_benchmark_bundle_result,
    render_openml_benchmark_candidate_report,
    write_openml_benchmark_bundle as _write_openml_benchmark_bundle,
)
from tab_foundry.bench.openml_bundle.selection import resolve_selected_tasks
from tab_foundry.bench.openml_task_source_registry import (
    DEFAULT_OPENML_TASK_SOURCE,
    task_ids_for_source,
    task_source_names,
)


def _resolve_selected_tasks(
    config: OpenMLBenchmarkBundleConfig,
) -> tuple[list[Any], int, tuple[OpenMLBenchmarkCandidateReportEntry, ...]]:
    return resolve_selected_tasks(
        config,
        prepare_openml_benchmark_task_fn=prepare_openml_benchmark_task,
        get_task_fn=openml.tasks.get_task,
        task_ids_for_source_fn=task_ids_for_source,
        read_required_openml_quality_fn=read_required_openml_quality,
        collect_discovered_task_candidates_fn=lambda discovery_config: collect_discovered_task_candidates(
            discovery_config,
            task_listing_rows_fn=lambda listing_config: task_listing_rows_for_config(
                listing_config,
                list_tasks_fn=openml.tasks.list_tasks,
            ),
        ),
    )


def build_openml_benchmark_bundle_result(
    config: OpenMLBenchmarkBundleConfig,
) -> OpenMLBenchmarkBundleBuildResult:
    return _build_openml_benchmark_bundle_result(
        config,
        resolve_selected_tasks_fn=_resolve_selected_tasks,
        normalize_benchmark_bundle_fn=normalize_benchmark_bundle,
    )


def build_openml_benchmark_bundle(config: OpenMLBenchmarkBundleConfig) -> dict[str, Any]:
    return _build_openml_benchmark_bundle(
        config,
        build_openml_benchmark_bundle_result_fn=build_openml_benchmark_bundle_result,
    )


def write_openml_benchmark_bundle(
    path: Path,
    config: OpenMLBenchmarkBundleConfig,
    *,
    bundle: Mapping[str, Any] | None = None,
) -> Path:
    return _write_openml_benchmark_bundle(
        path,
        config,
        bundle=bundle,
        build_openml_benchmark_bundle_fn=build_openml_benchmark_bundle,
        normalize_benchmark_bundle_fn=normalize_benchmark_bundle,
        write_json_fn=write_json,
    )


__all__ = [
    "DEFAULT_OPENML_TASK_SOURCE",
    "OpenMLBenchmarkBundleBuildResult",
    "OpenMLBenchmarkBundleConfig",
    "OpenMLBenchmarkCandidateReportEntry",
    "OpenMLBenchmarkTaskCandidate",
    "build_openml_benchmark_bundle",
    "build_openml_benchmark_bundle_result",
    "parse_max_classes_arg",
    "prepare_openml_benchmark_task",
    "read_required_openml_quality",
    "render_openml_benchmark_candidate_report",
    "task_ids_for_source",
    "task_source_names",
    "write_openml_benchmark_bundle",
]
