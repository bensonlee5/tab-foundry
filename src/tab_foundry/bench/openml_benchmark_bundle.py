"""Helpers for building pinned OpenML benchmark bundles."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import openml

from tab_foundry.bench.artifacts import write_json
from tab_foundry.bench.nanotabpfn import normalize_benchmark_bundle, prepare_openml_benchmark_task, read_required_openml_quality
from tab_foundry.bench.openml_bundle.config import (
    OpenMLBenchmarkBundleBuildResult,
    OpenMLBenchmarkBundleConfig,
    OpenMLBenchmarkCandidateReportEntry,
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
from tab_foundry.bench.openml_bundle.selection import resolve_selected_tasks as _resolve_selected_tasks_impl
from tab_foundry.bench.openml_task_source_registry import task_ids_for_source


def build_openml_benchmark_bundle_result(
    config: OpenMLBenchmarkBundleConfig,
) -> OpenMLBenchmarkBundleBuildResult:
    return _build_openml_benchmark_bundle_result_impl(
        config,
        resolve_selected_tasks_fn=lambda cfg: _resolve_selected_tasks_impl(
            cfg,
            prepare_openml_benchmark_task_fn=prepare_openml_benchmark_task,
            get_task_fn=openml.tasks.get_task,
            task_ids_for_source_fn=task_ids_for_source,
            read_required_openml_quality_fn=read_required_openml_quality,
            collect_discovered_task_candidates_fn=lambda discovery_cfg: _collect_discovered_task_candidates_impl(
                discovery_cfg,
                task_listing_rows_fn=lambda listing_cfg: _task_listing_rows_for_config_impl(
                    listing_cfg,
                    list_tasks_fn=openml.tasks.list_tasks,
                ),
            ),
        ),
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
