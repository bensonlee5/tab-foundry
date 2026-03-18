"""Helpers for building pinned OpenML benchmark bundles."""

from __future__ import annotations

import argparse
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import openml
from openml.tasks import TaskType

from tab_foundry.bench.artifacts import write_json
from tab_foundry.bench.nanotabpfn import (
    PreparedOpenMLBenchmarkTask,
    normalize_benchmark_bundle,
    prepare_openml_benchmark_task,
    read_required_openml_quality,
)
from tab_foundry.bench.openml_task_source_registry import (
    DEFAULT_OPENML_TASK_SOURCE,
    task_ids_for_source,
    task_source_names,
)


@dataclass(slots=True, frozen=True)
class OpenMLBenchmarkBundleConfig:
    """Configuration for generating a pinned OpenML bundle."""

    bundle_name: str
    version: int
    task_source: str = DEFAULT_OPENML_TASK_SOURCE
    task_type: str = "supervised_classification"
    new_instances: int = 200
    max_features: int = 10
    max_classes: int | None = 2
    max_missing_pct: float = 0.0
    min_minority_class_pct: float = 2.5
    task_ids: tuple[int, ...] | None = None
    discover_from_openml: bool = False
    min_instances: int = 1
    min_task_count: int = 1

    def resolved_task_ids(self) -> tuple[int, ...]:
        """Resolve custom task ids or fall back to the named pinned source pool."""

        if self.discover_from_openml:
            raise RuntimeError("OpenML discovery mode does not use pinned task ids")
        if self.task_ids is not None:
            return tuple(int(task_id) for task_id in self.task_ids)
        return task_ids_for_source(self.task_source)


@dataclass(slots=True, frozen=True)
class OpenMLBenchmarkTaskCandidate:
    """Notebook-source task metadata used for bundle filtering."""

    task_id: int
    number_of_features: float
    number_of_classes: float | None
    missing_pct: float
    minority_class_pct: float | None
    dataset_id: int | None = None
    dataset_name: str | None = None
    estimation_procedure: str | None = None
    number_of_instances: float | None = None


@dataclass(slots=True, frozen=True)
class OpenMLBenchmarkCandidateReportEntry:
    """One task-candidate decision recorded during discovery."""

    task_id: int
    status: str
    reason: str
    dataset_id: int | None = None
    dataset_name: str | None = None
    estimation_procedure: str | None = None


@dataclass(slots=True, frozen=True)
class OpenMLBenchmarkBundleBuildResult:
    """Final bundle payload and optional discovery report entries."""

    bundle: dict[str, Any]
    report_entries: tuple[OpenMLBenchmarkCandidateReportEntry, ...] = ()


def _task_type_value(task_type: TaskType | int) -> int:
    return int(task_type.value) if isinstance(task_type, TaskType) else int(task_type)


def _bundle_selection_payload(config: OpenMLBenchmarkBundleConfig, *, max_classes: int) -> dict[str, Any]:
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


def _coerce_finite_float(value: Any, *, context: str) -> float:
    if isinstance(value, bool):
        raise RuntimeError(f"{context} must be numeric")
    try:
        numeric = float(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{context} must be numeric") from exc
    if not math.isfinite(numeric):
        raise RuntimeError(f"{context} must be finite")
    return numeric


def _lookup_task_listing_value(row: Mapping[str, Any], *names: str) -> Any:
    normalized_keys = {str(key).casefold(): key for key in row}
    for name in names:
        key = normalized_keys.get(str(name).casefold())
        if key is not None:
            return row[key]
    raise KeyError(names[0])


def _task_listing_records(task_listing: Any) -> list[Mapping[str, Any]]:
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


def _task_listing_rows_for_config(config: OpenMLBenchmarkBundleConfig) -> list[Mapping[str, Any]]:
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
        task_listing = openml.tasks.list_tasks(
            task_type=expected_task_type,
            output_format="dataframe",
            **listing_filters,
        )
    except Exception:
        task_listing = openml.tasks.list_tasks(
            task_type=expected_task_type,
            output_format="dataframe",
        )
    return _task_listing_records(task_listing)


def _candidate_matches_listing_filters(
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


def _candidate_from_task_listing_row(
    row: Mapping[str, Any],
    *,
    config: OpenMLBenchmarkBundleConfig,
) -> OpenMLBenchmarkTaskCandidate:
    task_id = int(
        _coerce_finite_float(
            _lookup_task_listing_value(row, "tid", "task_id"),
            context="task listing tid",
        )
    )
    dataset_id = int(
        _coerce_finite_float(
            _lookup_task_listing_value(row, "did", "data_id"),
            context="task listing did",
        )
    )
    dataset_name = str(_lookup_task_listing_value(row, "name")).strip()
    if not dataset_name:
        raise RuntimeError("task listing dataset name must be non-empty")
    number_of_instances = _coerce_finite_float(
        _lookup_task_listing_value(row, "NumberOfInstances"),
        context=f"task listing NumberOfInstances for task {task_id}",
    )
    number_of_features = _coerce_finite_float(
        _lookup_task_listing_value(row, "NumberOfFeatures"),
        context=f"task listing NumberOfFeatures for task {task_id}",
    )
    missing_instances = _coerce_finite_float(
        _lookup_task_listing_value(row, "NumberOfInstancesWithMissingValues"),
        context=f"task listing NumberOfInstancesWithMissingValues for task {task_id}",
    )
    missing_pct = 0.0 if number_of_instances <= 0.0 else (100.0 * missing_instances / number_of_instances)
    number_of_classes = None
    minority_class_pct = None
    if config.task_type == "supervised_classification":
        number_of_classes = _coerce_finite_float(
            _lookup_task_listing_value(row, "NumberOfClasses"),
            context=f"task listing NumberOfClasses for task {task_id}",
        )
        minority_class_size = _coerce_finite_float(
            _lookup_task_listing_value(row, "MinorityClassSize"),
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


def _is_preferred_ten_fold_cv(candidate: OpenMLBenchmarkTaskCandidate) -> bool:
    estimation_procedure = candidate.estimation_procedure
    if estimation_procedure is None:
        return False
    normalized = estimation_procedure.strip().casefold()
    return "10-fold" in normalized and "crossvalidation" in normalized.replace(" ", "")


def _dedupe_discovered_candidates(
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
                0 if _is_preferred_ten_fold_cv(candidate) else 1,
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


def _validate_prepared_task(
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


def _collect_discovered_task_candidates(
    config: OpenMLBenchmarkBundleConfig,
) -> tuple[list[OpenMLBenchmarkTaskCandidate], list[OpenMLBenchmarkCandidateReportEntry]]:
    eligible_candidates: list[OpenMLBenchmarkTaskCandidate] = []
    report_entries: list[OpenMLBenchmarkCandidateReportEntry] = []
    for row in _task_listing_rows_for_config(config):
        try:
            candidate = _candidate_from_task_listing_row(row, config=config)
            keep_candidate, reason = _candidate_matches_listing_filters(candidate, config)
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
    deduped_candidates, duplicate_report_entries = _dedupe_discovered_candidates(eligible_candidates)
    report_entries.extend(duplicate_report_entries)
    return sorted(deduped_candidates, key=lambda candidate: int(candidate.task_id)), sorted(
        report_entries,
        key=lambda entry: int(entry.task_id),
    )


def _collect_task_candidates(config: OpenMLBenchmarkBundleConfig) -> list[OpenMLBenchmarkTaskCandidate]:
    if config.discover_from_openml:
        discovered_candidates, _report_entries = _collect_discovered_task_candidates(config)
        return discovered_candidates
    resolved_candidates: list[OpenMLBenchmarkTaskCandidate] = []
    for task_id in config.resolved_task_ids():
        task = openml.tasks.get_task(int(task_id), download_splits=False)
        expected_task_type = (
            TaskType.SUPERVISED_CLASSIFICATION
            if config.task_type == "supervised_classification"
            else TaskType.SUPERVISED_REGRESSION
        )
        if _task_type_value(task.task_type_id) != _task_type_value(expected_task_type):
            continue
        task_any: Any = task
        dataset = task_any.get_dataset(download_data=False)
        dataset_any: Any = dataset
        raw_qualities = dataset_any.qualities
        candidate = OpenMLBenchmarkTaskCandidate(
            task_id=int(task_id),
            number_of_features=read_required_openml_quality(
                raw_qualities,
                task_id=int(task_id),
                quality_name="NumberOfFeatures",
            ),
            number_of_classes=(
                None
                if config.task_type != "supervised_classification"
                else read_required_openml_quality(
                    raw_qualities,
                    task_id=int(task_id),
                    quality_name="NumberOfClasses",
                )
            ),
            missing_pct=read_required_openml_quality(
                raw_qualities,
                task_id=int(task_id),
                quality_name="PercentageOfInstancesWithMissingValues",
            ),
            minority_class_pct=(
                None
                if config.task_type != "supervised_classification"
                else read_required_openml_quality(
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
        if keep_candidate:
            resolved_candidates.append(candidate)
    return resolved_candidates


def _resolve_selected_tasks(
    config: OpenMLBenchmarkBundleConfig,
) -> tuple[list[PreparedOpenMLBenchmarkTask], int, tuple[OpenMLBenchmarkCandidateReportEntry, ...]]:
    report_entries: list[OpenMLBenchmarkCandidateReportEntry] = []
    if config.discover_from_openml:
        eligible_candidates, discovery_report_entries = _collect_discovered_task_candidates(config)
        report_entries.extend(discovery_report_entries)
    else:
        eligible_candidates = _collect_task_candidates(config)
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
            prepared = prepare_openml_benchmark_task(
                int(candidate.task_id),
                new_instances=int(config.new_instances),
                task_type=str(config.task_type),
            )
            _validate_prepared_task(prepared, config=config)
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


def build_openml_benchmark_bundle_result(
    config: OpenMLBenchmarkBundleConfig,
) -> OpenMLBenchmarkBundleBuildResult:
    selected_tasks, effective_max_classes, report_entries = _resolve_selected_tasks(config)
    payload = {
        "name": str(config.bundle_name),
        "version": int(config.version),
        "selection": _bundle_selection_payload(config, max_classes=effective_max_classes),
        "task_ids": [int(prepared.task_id) for prepared in selected_tasks],
        "tasks": [dict(prepared.observed_task) for prepared in selected_tasks],
    }
    return OpenMLBenchmarkBundleBuildResult(
        bundle=normalize_benchmark_bundle(payload),
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

    payload = build_openml_benchmark_bundle(config) if bundle is None else normalize_benchmark_bundle(dict(bundle))
    return write_json(path.expanduser().resolve(), payload)


def _parse_max_classes_arg(raw_value: str) -> int | None:
    normalized = str(raw_value).strip().lower()
    if normalized == "auto":
        return None
    value = int(normalized)
    if value <= 0:
        raise ValueError("max_classes must be a positive int or 'auto'")
    return value


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
