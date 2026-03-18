"""Helpers for building pinned OpenML benchmark bundles."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

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

    def resolved_task_ids(self) -> tuple[int, ...]:
        """Resolve custom task ids or fall back to the named pinned source pool."""

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


def _collect_task_candidates(config: OpenMLBenchmarkBundleConfig) -> list[OpenMLBenchmarkTaskCandidate]:
    candidates: list[OpenMLBenchmarkTaskCandidate] = []
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
            candidates.append(candidate)
    return candidates


def _resolve_selected_tasks(
    config: OpenMLBenchmarkBundleConfig,
) -> tuple[list[PreparedOpenMLBenchmarkTask], int]:
    eligible_candidates = _collect_task_candidates(config)
    if not eligible_candidates:
        raise RuntimeError("OpenML benchmark bundle produced no eligible tasks")

    effective_max_classes = (
        0
        if config.task_type != "supervised_classification"
        else (
            max(int(candidate.number_of_classes) for candidate in eligible_candidates if candidate.number_of_classes is not None)
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
    selected_tasks = [
        prepare_openml_benchmark_task(
            int(candidate.task_id),
            new_instances=int(config.new_instances),
            task_type=str(config.task_type),
        )
        for candidate in selected_candidates
    ]
    return sorted(selected_tasks, key=lambda prepared: int(prepared.task_id)), int(effective_max_classes)


def build_openml_benchmark_bundle(config: OpenMLBenchmarkBundleConfig) -> dict[str, Any]:
    """Build one normalized benchmark bundle from the notebook task set."""

    selected_tasks, effective_max_classes = _resolve_selected_tasks(config)
    payload = {
        "name": str(config.bundle_name),
        "version": int(config.version),
        "selection": _bundle_selection_payload(config, max_classes=effective_max_classes),
        "task_ids": [int(prepared.task_id) for prepared in selected_tasks],
        "tasks": [dict(prepared.observed_task) for prepared in selected_tasks],
    }
    return normalize_benchmark_bundle(payload)


def write_openml_benchmark_bundle(path: Path, config: OpenMLBenchmarkBundleConfig) -> Path:
    """Write one normalized benchmark bundle to disk."""

    return write_json(path.expanduser().resolve(), build_openml_benchmark_bundle(config))


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
    parser.add_argument("--new-instances", type=int, default=200, help="Subsampled row count used by the benchmark")
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
    )
    out_path = write_openml_benchmark_bundle(Path(str(args.out_path)), config)
    print(f"wrote benchmark bundle: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
