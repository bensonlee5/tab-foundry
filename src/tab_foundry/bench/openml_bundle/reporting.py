"""Reporting and file-writing helpers for OpenML benchmark bundles."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from tab_foundry.bench.openml_bundle.config import (
    OpenMLBenchmarkBundleBuildResult,
    OpenMLBenchmarkBundleConfig,
    OpenMLBenchmarkCandidateReportEntry,
)


def bundle_selection_payload(config: OpenMLBenchmarkBundleConfig, *, max_classes: int) -> dict[str, Any]:
    payload = {
        "new_instances": int(config.new_instances),
        "task_type": str(config.task_type),
        "max_features": int(config.max_features),
        "max_missing_pct": float(config.max_missing_pct),
    }
    if config.task_type == "supervised_classification":
        payload["max_classes"] = int(max_classes)
        payload["min_minority_class_pct"] = float(config.min_minority_class_pct)
    return payload


def build_openml_benchmark_bundle_result(
    config: OpenMLBenchmarkBundleConfig,
    *,
    resolve_selected_tasks_fn: Any,
    normalize_benchmark_bundle_fn: Any,
) -> OpenMLBenchmarkBundleBuildResult:
    selected_tasks, effective_max_classes, report_entries = resolve_selected_tasks_fn(config)
    payload = {
        "name": str(config.bundle_name),
        "version": int(config.version),
        "selection": bundle_selection_payload(config, max_classes=effective_max_classes),
        "task_ids": [int(prepared.task_id) for prepared in selected_tasks],
        "tasks": [dict(prepared.observed_task) for prepared in selected_tasks],
    }
    return OpenMLBenchmarkBundleBuildResult(
        bundle=normalize_benchmark_bundle_fn(payload),
        report_entries=report_entries,
    )


def render_openml_benchmark_candidate_report(
    entries: Sequence[OpenMLBenchmarkCandidateReportEntry],
) -> str:
    if not entries:
        return ""
    accepted = [entry for entry in entries if entry.status == "accepted"]
    rejected = [entry for entry in entries if entry.status == "rejected"]
    lines = [
        "OpenML discovery candidate report:",
        f"- accepted={len(accepted)}",
        f"- rejected={len(rejected)}",
    ]
    if accepted:
        lines.append("Accepted:")
        for entry in accepted:
            lines.append(
                "- "
                f"task_id={entry.task_id} dataset_id={entry.dataset_id} "
                f"dataset_name={entry.dataset_name!r} reason={entry.reason}"
            )
    if rejected:
        lines.append("Rejected:")
        for entry in rejected:
            lines.append(
                "- "
                f"task_id={entry.task_id} dataset_id={entry.dataset_id} "
                f"dataset_name={entry.dataset_name!r} reason={entry.reason}"
            )
    return "\n".join(lines)


def build_openml_benchmark_bundle(
    config: OpenMLBenchmarkBundleConfig,
    *,
    build_openml_benchmark_bundle_result_fn: Any,
) -> dict[str, Any]:
    """Build one normalized benchmark bundle from the notebook task set."""

    return build_openml_benchmark_bundle_result_fn(config).bundle


def write_openml_benchmark_bundle(
    path: Path,
    config: OpenMLBenchmarkBundleConfig,
    *,
    bundle: Mapping[str, Any] | None,
    build_openml_benchmark_bundle_fn: Any,
    normalize_benchmark_bundle_fn: Any,
    write_json_fn: Any,
) -> Path:
    """Write one normalized benchmark bundle to disk."""

    payload = (
        build_openml_benchmark_bundle_fn(config)
        if bundle is None
        else normalize_benchmark_bundle_fn(dict(bundle))
    )
    return write_json_fn(path.expanduser().resolve(), payload)
