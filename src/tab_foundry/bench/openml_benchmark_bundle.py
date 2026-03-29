"""Compatibility wrapper for pinned OpenML benchmark bundle helpers."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import openml
from tab_realdata_hub import openml as hub_openml
from tab_realdata_hub.openml import (
    OpenMLBundleBuildResult as OpenMLBenchmarkBundleBuildResult,
    OpenMLBundleConfig as OpenMLBenchmarkBundleConfig,
    OpenMLCandidateReportEntry as OpenMLBenchmarkCandidateReportEntry,
)

DEFAULT_OPENML_TASK_SOURCE = hub_openml.DEFAULT_OPENML_TASK_SOURCE
prepare_openml_benchmark_task = hub_openml.prepare_task
read_required_openml_quality = hub_openml.read_required_quality
parse_max_classes_arg = hub_openml.parse_max_classes_arg


def task_source_names() -> tuple[str, ...]:
    return hub_openml.task_source_names()


def task_ids_for_source(task_source: str) -> tuple[int, ...]:
    return hub_openml.task_ids_for_source(task_source)


def _resolved_wrapper_config(
    config: OpenMLBenchmarkBundleConfig,
) -> OpenMLBenchmarkBundleConfig:
    if config.discover_from_openml or config.task_ids is not None:
        return config
    return replace(
        config,
        task_ids=task_ids_for_source(config.task_source),
    )


def build_openml_benchmark_bundle_result(
    config: OpenMLBenchmarkBundleConfig,
) -> OpenMLBenchmarkBundleBuildResult:
    return hub_openml.build_bundle_result(
        _resolved_wrapper_config(config),
        prepare_task_fn=prepare_openml_benchmark_task,
        get_task_fn=openml.tasks.get_task,
        list_tasks_fn=openml.tasks.list_tasks,
    )


def render_openml_benchmark_candidate_report(
    entries: Sequence[OpenMLBenchmarkCandidateReportEntry],
) -> str:
    return hub_openml.render_candidate_report(entries)


def build_openml_benchmark_bundle(config: OpenMLBenchmarkBundleConfig) -> dict[str, Any]:
    """Build one normalized benchmark bundle from the notebook task set."""

    return build_openml_benchmark_bundle_result(config).bundle


def write_openml_benchmark_bundle(
    path: Path,
    config: OpenMLBenchmarkBundleConfig,
    *,
    bundle: Mapping[str, Any] | None = None,
) -> Path:
    """Write one normalized benchmark bundle to disk."""

    return hub_openml.write_bundle(
        path,
        _resolved_wrapper_config(config),
        bundle=bundle,
    )
