"""OpenML benchmark dataset preparation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
from tab_realdata_hub.openml import (
    PreparedOpenMLTask as PreparedOpenMLBenchmarkTask,
    get_feature_preprocessor,
    prepare_task as _prepare_openml_task,
    read_required_quality as read_required_openml_quality,
)

from .bundle import (
    _CLASSIFICATION_TASK_TYPE,
    benchmark_bundle_task_type,
    load_benchmark_bundle,
)
from .dataset_common import _assert_finite_benchmark_datasets

__all__ = [
    "PreparedOpenMLBenchmarkTask",
    "get_feature_preprocessor",
    "load_openml_benchmark_datasets",
    "prepare_openml_benchmark_task",
    "read_required_openml_quality",
]


def prepare_openml_benchmark_task(
    task_id: int,
    *,
    new_instances: int,
    task_type: str,
) -> PreparedOpenMLBenchmarkTask:
    """Load and preprocess one OpenML task using the shared OpenML helper."""

    return _prepare_openml_task(
        task_id,
        new_instances=new_instances,
        task_type=task_type,
    )


def load_openml_benchmark_datasets(
    *,
    new_instances: int = 200,
    benchmark_bundle_path: Path | None = None,
    allow_missing_values: bool = False,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    """Load the nanoTabPFN OpenML benchmark suite."""

    bundle = load_benchmark_bundle(
        benchmark_bundle_path,
        allow_missing_values=allow_missing_values,
    )
    selection = cast(dict[str, Any], bundle["selection"])
    selection_task_type = benchmark_bundle_task_type(bundle)
    expected_new_instances = int(selection["new_instances"])
    if new_instances != expected_new_instances:
        raise RuntimeError(
            "benchmark bundle selection mismatch: "
            f"expected new_instances={expected_new_instances}, got {new_instances}"
        )
    expected_tasks = cast(list[dict[str, Any]], bundle["tasks"])
    expected_by_task_id = {int(task["task_id"]): task for task in expected_tasks}
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    benchmark_tasks: list[dict[str, Any]] = []
    for task_id in cast(list[int], bundle["task_ids"]):
        prepared = prepare_openml_benchmark_task(
            int(task_id),
            new_instances=new_instances,
            task_type=selection_task_type,
        )
        number_of_features = prepared.qualities["NumberOfFeatures"]
        missing_pct = prepared.qualities["PercentageOfInstancesWithMissingValues"]
        if number_of_features > int(selection["max_features"]):
            raise RuntimeError(
                "benchmark bundle drift: "
                f"task {task_id} exceeds max_features expected<={selection['max_features']}, actual={number_of_features}"
            )
        if missing_pct > float(selection["max_missing_pct"]):
            raise RuntimeError(
                "benchmark bundle drift: "
                f"task {task_id} exceeds max_missing_pct expected<={selection['max_missing_pct']}, actual={missing_pct}"
            )
        if selection_task_type == _CLASSIFICATION_TASK_TYPE:
            number_of_classes = prepared.qualities["NumberOfClasses"]
            minority_class_pct = prepared.qualities["MinorityClassPercentage"]
            if number_of_classes > int(selection["max_classes"]):
                raise RuntimeError(
                    "benchmark bundle drift: "
                    f"task {task_id} exceeds max_classes expected<={selection['max_classes']}, actual={number_of_classes}"
                )
            if minority_class_pct < float(selection["min_minority_class_pct"]):
                raise RuntimeError(
                    "benchmark bundle drift: "
                    "task "
                    f"{task_id} violates min_minority_class_pct expected>={selection['min_minority_class_pct']}, "
                    f"actual={minority_class_pct}"
                )
        expected_task = expected_by_task_id[int(task_id)]
        if prepared.observed_task != expected_task:
            raise RuntimeError(
                "benchmark bundle drift: "
                f"task {task_id} metadata mismatch expected={expected_task}, actual={prepared.observed_task}"
            )

        datasets[prepared.dataset_name] = (prepared.x, prepared.y)
        benchmark_tasks.append(dict(prepared.observed_task))
    if not datasets:
        raise RuntimeError("OpenML benchmark produced no datasets after filtering")
    if len(benchmark_tasks) != len(expected_tasks):
        raise RuntimeError(
            "benchmark bundle drift: "
            f"task count mismatch expected={len(expected_tasks)}, actual={len(benchmark_tasks)}"
        )
    if not allow_missing_values:
        _assert_finite_benchmark_datasets(
            datasets,
            context=f"benchmark bundle {bundle['name']!r}",
        )
    return datasets, benchmark_tasks
