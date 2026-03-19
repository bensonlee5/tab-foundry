"""Selection helpers for OpenML benchmark bundle building."""

from __future__ import annotations

from collections import Counter
from typing import Any

from openml.tasks import TaskType

from tab_foundry.bench.nanotabpfn import PreparedOpenMLBenchmarkTask
from tab_foundry.bench.openml_bundle.config import (
    OpenMLBenchmarkBundleConfig,
    OpenMLBenchmarkCandidateReportEntry,
    OpenMLBenchmarkTaskCandidate,
)
from tab_foundry.bench.openml_bundle.discovery import task_type_value


def validate_prepared_task(
    prepared: PreparedOpenMLBenchmarkTask,
    *,
    config: OpenMLBenchmarkBundleConfig,
) -> None:
    if int(prepared.observed_task["n_rows"]) != int(config.new_instances):
        raise RuntimeError(
            "observed row count mismatch after subsampling: "
            f"expected={config.new_instances}, actual={prepared.observed_task['n_rows']}"
        )
    number_of_features = float(prepared.qualities["NumberOfFeatures"])
    if number_of_features > float(config.max_features):
        raise RuntimeError(
            f"number_of_features={number_of_features:g} exceeds max_features={config.max_features}"
        )
    missing_pct = float(prepared.qualities["PercentageOfInstancesWithMissingValues"])
    if missing_pct > float(config.max_missing_pct):
        raise RuntimeError(
            f"missing_pct={missing_pct:g} exceeds max_missing_pct={config.max_missing_pct:g}"
        )
    if config.task_type == "supervised_classification":
        number_of_classes = float(prepared.qualities["NumberOfClasses"])
        if config.max_classes is not None and number_of_classes > float(config.max_classes):
            raise RuntimeError(
                f"number_of_classes={number_of_classes:g} exceeds max_classes={config.max_classes}"
            )
        minority_class_pct = float(prepared.qualities["MinorityClassPercentage"])
        if minority_class_pct < float(config.min_minority_class_pct):
            raise RuntimeError(
                "minority_class_pct="
                f"{minority_class_pct:g} below min_minority_class_pct={config.min_minority_class_pct:g}"
            )


def collect_task_candidates(
    config: OpenMLBenchmarkBundleConfig,
    *,
    get_task_fn: Any,
    task_ids_for_source_fn: Any,
    read_required_openml_quality_fn: Any,
) -> list[OpenMLBenchmarkTaskCandidate]:
    resolved_task_ids = (
        tuple(int(task_id) for task_id in config.task_ids)
        if config.task_ids is not None
        else tuple(int(task_id) for task_id in task_ids_for_source_fn(config.task_source))
    )
    resolved_candidates: list[OpenMLBenchmarkTaskCandidate] = []
    for task_id in resolved_task_ids:
        task = get_task_fn(int(task_id), download_splits=False)
        expected_task_type = (
            TaskType.SUPERVISED_CLASSIFICATION
            if config.task_type == "supervised_classification"
            else TaskType.SUPERVISED_REGRESSION
        )
        if task_type_value(task.task_type_id) != task_type_value(expected_task_type):
            continue
        task_any: Any = task
        dataset = task_any.get_dataset(download_data=False)
        dataset_any: Any = dataset
        raw_qualities = dataset_any.qualities
        candidate = OpenMLBenchmarkTaskCandidate(
            task_id=int(task_id),
            number_of_features=read_required_openml_quality_fn(
                raw_qualities,
                task_id=int(task_id),
                quality_name="NumberOfFeatures",
            ),
            number_of_classes=(
                None
                if config.task_type != "supervised_classification"
                else read_required_openml_quality_fn(
                    raw_qualities,
                    task_id=int(task_id),
                    quality_name="NumberOfClasses",
                )
            ),
            missing_pct=read_required_openml_quality_fn(
                raw_qualities,
                task_id=int(task_id),
                quality_name="PercentageOfInstancesWithMissingValues",
            ),
            minority_class_pct=(
                None
                if config.task_type != "supervised_classification"
                else read_required_openml_quality_fn(
                    raw_qualities,
                    task_id=int(task_id),
                    quality_name="MinorityClassPercentage",
                )
            ),
        )
        keep_candidate = (
            candidate.number_of_features <= float(config.max_features)
            and candidate.missing_pct <= float(config.max_missing_pct)
        )
        if config.task_type == "supervised_classification":
            keep_candidate = (
                keep_candidate
                and candidate.minority_class_pct is not None
                and candidate.minority_class_pct >= float(config.min_minority_class_pct)
            )
            if config.max_classes is not None:
                keep_candidate = (
                    keep_candidate
                    and candidate.number_of_classes is not None
                    and candidate.number_of_classes <= float(config.max_classes)
                )
        if keep_candidate:
            resolved_candidates.append(candidate)
    return resolved_candidates


def resolve_selected_tasks(
    config: OpenMLBenchmarkBundleConfig,
    *,
    prepare_openml_benchmark_task_fn: Any,
    get_task_fn: Any,
    task_ids_for_source_fn: Any,
    read_required_openml_quality_fn: Any,
    collect_discovered_task_candidates_fn: Any,
) -> tuple[list[PreparedOpenMLBenchmarkTask], int, tuple[OpenMLBenchmarkCandidateReportEntry, ...]]:
    report_entries: list[OpenMLBenchmarkCandidateReportEntry] = []
    if config.discover_from_openml:
        eligible_candidates, discovery_report_entries = collect_discovered_task_candidates_fn(config)
        report_entries.extend(discovery_report_entries)
    else:
        eligible_candidates = collect_task_candidates(
            config,
            get_task_fn=get_task_fn,
            task_ids_for_source_fn=task_ids_for_source_fn,
            read_required_openml_quality_fn=read_required_openml_quality_fn,
        )
    if not eligible_candidates:
        raise RuntimeError("OpenML benchmark bundle produced no eligible tasks")

    effective_max_classes = (
        0
        if config.task_type != "supervised_classification"
        else (
            max(
                int(candidate.number_of_classes)
                for candidate in eligible_candidates
                if candidate.number_of_classes is not None
            )
            if config.max_classes is None
            else int(config.max_classes)
        )
    )
    selected_candidates = (
        eligible_candidates
        if config.task_type != "supervised_classification"
        else [
            candidate
            for candidate in eligible_candidates
            if candidate.number_of_classes is not None
            and int(candidate.number_of_classes) <= effective_max_classes
        ]
    )
    if not selected_candidates:
        raise RuntimeError("OpenML benchmark bundle produced no tasks after task-type filtering")
    selected_tasks: list[PreparedOpenMLBenchmarkTask] = []
    for candidate in selected_candidates:
        try:
            prepared = prepare_openml_benchmark_task_fn(
                int(candidate.task_id),
                new_instances=int(config.new_instances),
                task_type=str(config.task_type),
            )
            validate_prepared_task(prepared, config=config)
        except RuntimeError as exc:
            if config.discover_from_openml:
                report_entries.append(
                    OpenMLBenchmarkCandidateReportEntry(
                        task_id=int(candidate.task_id),
                        dataset_id=candidate.dataset_id,
                        dataset_name=candidate.dataset_name,
                        estimation_procedure=candidate.estimation_procedure,
                        status="rejected",
                        reason=str(exc),
                    )
                )
                continue
            raise
        if config.discover_from_openml:
            report_entries.append(
                OpenMLBenchmarkCandidateReportEntry(
                    task_id=int(prepared.task_id),
                    dataset_id=candidate.dataset_id,
                    dataset_name=str(prepared.dataset_name),
                    estimation_procedure=candidate.estimation_procedure,
                    status="accepted",
                    reason="validated via prepare_openml_benchmark_task",
                )
            )
        selected_tasks.append(prepared)
    if config.discover_from_openml and len(selected_tasks) < int(config.min_task_count):
        raise RuntimeError(
            "OpenML benchmark bundle validated task count is below min_task_count: "
            f"validated={len(selected_tasks)}, min_task_count={config.min_task_count}"
        )
    dataset_name_counts = Counter(str(prepared.dataset_name) for prepared in selected_tasks)
    duplicate_dataset_names = sorted(name for name, count in dataset_name_counts.items() if count > 1)
    if duplicate_dataset_names:
        raise RuntimeError(
            "OpenML benchmark bundle produced duplicate dataset names after validation: "
            f"{duplicate_dataset_names}"
        )
    return (
        sorted(selected_tasks, key=lambda prepared: int(prepared.task_id)),
        int(effective_max_classes),
        tuple(sorted(report_entries, key=lambda entry: (entry.status, int(entry.task_id)))),
    )
