"""Discovery helpers for OpenML benchmark bundle building."""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

from openml.tasks import TaskType

from tab_foundry.bench.openml_bundle.config import (
    OpenMLBenchmarkBundleConfig,
    OpenMLBenchmarkCandidateReportEntry,
    OpenMLBenchmarkTaskCandidate,
)


def task_type_value(task_type: TaskType | int) -> int:
    return int(task_type.value) if isinstance(task_type, TaskType) else int(task_type)


def coerce_finite_float(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{context} must be numeric")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{context} must be numeric") from exc
    if not math.isfinite(numeric):
        raise RuntimeError(f"{context} must be finite")
    return numeric


def lookup_task_listing_value(row: Mapping[str, Any], *names: str) -> Any:
    normalized_keys = {str(key).casefold(): key for key in row}
    for name in names:
        key = normalized_keys.get(str(name).casefold())
        if key is not None:
            return row[key]
    raise KeyError(names[0])


def task_listing_records(task_listing: Any) -> list[Mapping[str, Any]]:
    if hasattr(task_listing, "to_dict"):
        records = task_listing.to_dict(orient="records")
    elif isinstance(task_listing, dict):
        records = list(task_listing.values())
    else:
        raise RuntimeError("OpenML task listing must be a dataframe or dict of rows")
    if not isinstance(records, list):
        raise RuntimeError("OpenML task listing must resolve into a list of rows")
    normalized: list[Mapping[str, Any]] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise RuntimeError(f"OpenML task listing row {index} must be a mapping")
        normalized.append(record)
    return normalized


def task_listing_rows_for_config(
    config: OpenMLBenchmarkBundleConfig,
    *,
    list_tasks_fn: Any,
) -> list[Mapping[str, Any]]:
    expected_task_type = (
        TaskType.SUPERVISED_CLASSIFICATION
        if config.task_type == "supervised_classification"
        else TaskType.SUPERVISED_REGRESSION
    )
    listing_filters: dict[str, Any] = {}
    if int(config.min_instances) > 1:
        listing_filters["number_instances"] = f"{int(config.min_instances)}.."
    listing_filters["number_features"] = f"..{int(config.max_features)}"
    if config.task_type == "supervised_classification" and config.max_classes is not None:
        listing_filters["number_classes"] = int(config.max_classes)
    if float(config.max_missing_pct) <= 0.0:
        listing_filters["number_missing_values"] = 0
    try:
        task_listing = list_tasks_fn(
            task_type=expected_task_type,
            output_format="dataframe",
            **listing_filters,
        )
    except Exception:
        task_listing = list_tasks_fn(
            task_type=expected_task_type,
            output_format="dataframe",
        )
    return task_listing_records(task_listing)


def candidate_matches_listing_filters(
    candidate: OpenMLBenchmarkTaskCandidate,
    config: OpenMLBenchmarkBundleConfig,
) -> tuple[bool, str]:
    if candidate.number_of_instances is not None and candidate.number_of_instances < float(config.min_instances):
        return (
            False,
            f"number_of_instances={candidate.number_of_instances:g} below min_instances={config.min_instances}",
        )
    if candidate.number_of_features > float(config.max_features):
        return (
            False,
            f"number_of_features={candidate.number_of_features:g} exceeds max_features={config.max_features}",
        )
    if candidate.missing_pct > float(config.max_missing_pct):
        return (
            False,
            f"missing_pct={candidate.missing_pct:g} exceeds max_missing_pct={config.max_missing_pct:g}",
        )
    if config.task_type == "supervised_classification":
        if candidate.number_of_classes is None:
            return False, "number_of_classes missing from task listing"
        if config.max_classes is not None and candidate.number_of_classes > float(config.max_classes):
            return (
                False,
                f"number_of_classes={candidate.number_of_classes:g} exceeds max_classes={config.max_classes}",
            )
        if candidate.minority_class_pct is None:
            return False, "minority_class_pct missing from task listing"
        if candidate.minority_class_pct < float(config.min_minority_class_pct):
            return (
                False,
                "minority_class_pct="
                f"{candidate.minority_class_pct:g} below min_minority_class_pct={config.min_minority_class_pct:g}",
            )
    return True, "listing filters matched"


def candidate_from_task_listing_row(
    row: Mapping[str, Any],
    *,
    config: OpenMLBenchmarkBundleConfig,
) -> OpenMLBenchmarkTaskCandidate:
    task_id = int(
        coerce_finite_float(
            lookup_task_listing_value(row, "tid", "task_id"),
            context="task listing tid",
        )
    )
    dataset_id = int(
        coerce_finite_float(
            lookup_task_listing_value(row, "did", "data_id"),
            context="task listing did",
        )
    )
    dataset_name = str(lookup_task_listing_value(row, "name")).strip()
    if not dataset_name:
        raise RuntimeError("task listing dataset name must be non-empty")
    number_of_instances = coerce_finite_float(
        lookup_task_listing_value(row, "NumberOfInstances"),
        context=f"task listing NumberOfInstances for task {task_id}",
    )
    number_of_features = coerce_finite_float(
        lookup_task_listing_value(row, "NumberOfFeatures"),
        context=f"task listing NumberOfFeatures for task {task_id}",
    )
    missing_instances = coerce_finite_float(
        lookup_task_listing_value(row, "NumberOfInstancesWithMissingValues"),
        context=f"task listing NumberOfInstancesWithMissingValues for task {task_id}",
    )
    missing_pct = 0.0 if number_of_instances <= 0.0 else (100.0 * missing_instances / number_of_instances)
    number_of_classes = None
    minority_class_pct = None
    if config.task_type == "supervised_classification":
        number_of_classes = coerce_finite_float(
            lookup_task_listing_value(row, "NumberOfClasses"),
            context=f"task listing NumberOfClasses for task {task_id}",
        )
        minority_class_size = coerce_finite_float(
            lookup_task_listing_value(row, "MinorityClassSize"),
            context=f"task listing MinorityClassSize for task {task_id}",
        )
        minority_class_pct = (
            0.0 if number_of_instances <= 0.0 else (100.0 * minority_class_size / number_of_instances)
        )
    estimation_procedure_raw = row.get("estimation_procedure")
    estimation_procedure = None if estimation_procedure_raw is None else str(estimation_procedure_raw).strip()
    return OpenMLBenchmarkTaskCandidate(
        task_id=task_id,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        estimation_procedure=estimation_procedure,
        number_of_instances=number_of_instances,
        number_of_features=number_of_features,
        number_of_classes=number_of_classes,
        missing_pct=missing_pct,
        minority_class_pct=minority_class_pct,
    )


def is_preferred_ten_fold_cv(candidate: OpenMLBenchmarkTaskCandidate) -> bool:
    estimation_procedure = candidate.estimation_procedure
    if estimation_procedure is None:
        return False
    normalized = estimation_procedure.strip().casefold()
    return "10-fold" in normalized and "crossvalidation" in normalized.replace(" ", "")


def dedupe_discovered_candidates(
    candidates: list[OpenMLBenchmarkTaskCandidate],
) -> tuple[list[OpenMLBenchmarkTaskCandidate], list[OpenMLBenchmarkCandidateReportEntry]]:
    grouped: dict[int, list[OpenMLBenchmarkTaskCandidate]] = {}
    for candidate in candidates:
        if candidate.dataset_id is None:
            raise RuntimeError(f"discovered task {candidate.task_id} is missing a dataset_id")
        grouped.setdefault(int(candidate.dataset_id), []).append(candidate)
    selected: list[OpenMLBenchmarkTaskCandidate] = []
    report_entries: list[OpenMLBenchmarkCandidateReportEntry] = []
    for dataset_id, grouped_candidates in grouped.items():
        preferred = min(
            grouped_candidates,
            key=lambda candidate: (
                0 if is_preferred_ten_fold_cv(candidate) else 1,
                int(candidate.task_id),
            ),
        )
        selected.append(preferred)
        for candidate in grouped_candidates:
            if candidate.task_id == preferred.task_id:
                continue
            report_entries.append(
                OpenMLBenchmarkCandidateReportEntry(
                    task_id=int(candidate.task_id),
                    dataset_id=candidate.dataset_id,
                    dataset_name=candidate.dataset_name,
                    estimation_procedure=candidate.estimation_procedure,
                    status="rejected",
                    reason=(
                        f"duplicate dataset_id={dataset_id}; preferred task_id={preferred.task_id} "
                        f"via estimation_procedure={preferred.estimation_procedure or '<missing>'}"
                    ),
                )
            )
    return selected, sorted(report_entries, key=lambda entry: int(entry.task_id))


def collect_discovered_task_candidates(
    config: OpenMLBenchmarkBundleConfig,
    *,
    task_listing_rows_fn: Any,
) -> tuple[list[OpenMLBenchmarkTaskCandidate], list[OpenMLBenchmarkCandidateReportEntry]]:
    eligible_candidates: list[OpenMLBenchmarkTaskCandidate] = []
    report_entries: list[OpenMLBenchmarkCandidateReportEntry] = []
    for row in task_listing_rows_fn(config):
        try:
            candidate = candidate_from_task_listing_row(row, config=config)
            keep_candidate, reason = candidate_matches_listing_filters(candidate, config)
        except (KeyError, RuntimeError) as exc:
            raw_task_id = row.get("tid", row.get("task_id", -1))
            report_entries.append(
                OpenMLBenchmarkCandidateReportEntry(
                    task_id=int(raw_task_id),
                    dataset_id=None,
                    dataset_name=None if row.get("name") is None else str(row.get("name")),
                    estimation_procedure=(
                        None
                        if row.get("estimation_procedure") is None
                        else str(row.get("estimation_procedure"))
                    ),
                    status="rejected",
                    reason=str(exc),
                )
            )
            continue
        if keep_candidate:
            eligible_candidates.append(candidate)
            continue
        report_entries.append(
            OpenMLBenchmarkCandidateReportEntry(
                task_id=int(candidate.task_id),
                dataset_id=candidate.dataset_id,
                dataset_name=candidate.dataset_name,
                estimation_procedure=candidate.estimation_procedure,
                status="rejected",
                reason=reason,
            )
        )
    deduped_candidates, duplicate_report_entries = dedupe_discovered_candidates(eligible_candidates)
    report_entries.extend(duplicate_report_entries)
    return sorted(deduped_candidates, key=lambda candidate: int(candidate.task_id)), sorted(
        report_entries,
        key=lambda entry: int(entry.task_id),
    )
