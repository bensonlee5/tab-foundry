"""Benchmark bundle loading and validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, cast


BENCHMARK_BUNDLE_FILENAME = "nanotabpfn_openml_binary_medium_v1.json"
_CLASSIFICATION_TASK_TYPE = "supervised_classification"
_REGRESSION_TASK_TYPE = "supervised_regression"
_ALLOWED_BUNDLE_SELECTION_TASK_TYPES = {
    _CLASSIFICATION_TASK_TYPE,
    _REGRESSION_TASK_TYPE,
}


def default_benchmark_bundle_path() -> Path:
    """Return the repo-tracked canonical benchmark bundle path."""

    return Path(__file__).resolve().parents[1] / BENCHMARK_BUNDLE_FILENAME


def _normalize_selection(payload: Any) -> dict[str, Any]:
    """Validate and normalize benchmark bundle selection metadata."""

    if not isinstance(payload, dict):
        raise RuntimeError("benchmark bundle selection must be an object")
    task_type = payload.get("task_type", _CLASSIFICATION_TASK_TYPE)
    if task_type not in _ALLOWED_BUNDLE_SELECTION_TASK_TYPES:
        raise RuntimeError(
            "benchmark bundle selection.task_type must be one of "
            f"{sorted(_ALLOWED_BUNDLE_SELECTION_TASK_TYPES)!r}"
        )
    expected_keys = (
        {
            "new_instances",
            "max_features",
            "max_classes",
            "max_missing_pct",
            "min_minority_class_pct",
        }
        | ({"task_type"} if "task_type" in payload else set())
        if task_type == _CLASSIFICATION_TASK_TYPE
        else {
            "new_instances",
            "task_type",
            "max_features",
            "max_missing_pct",
        }
    )
    actual_keys = set(payload.keys())
    if actual_keys != expected_keys:
        raise RuntimeError(
            "benchmark bundle selection keys mismatch: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )

    new_instances = payload["new_instances"]
    max_features = payload["max_features"]
    max_missing_pct = payload["max_missing_pct"]

    if not isinstance(new_instances, int) or isinstance(new_instances, bool) or new_instances <= 0:
        raise RuntimeError("benchmark bundle selection.new_instances must be a positive int")
    if not isinstance(max_features, int) or isinstance(max_features, bool) or max_features <= 0:
        raise RuntimeError("benchmark bundle selection.max_features must be a positive int")
    if not isinstance(max_missing_pct, (int, float)) or not 0 <= float(max_missing_pct) <= 100:
        raise RuntimeError("benchmark bundle selection.max_missing_pct must be a percentage between 0 and 100")
    normalized = {
        "new_instances": int(new_instances),
        "task_type": str(task_type),
        "max_features": int(max_features),
        "max_missing_pct": float(max_missing_pct),
    }
    if task_type == _CLASSIFICATION_TASK_TYPE:
        max_classes = payload["max_classes"]
        min_minority_class_pct = payload["min_minority_class_pct"]
        if not isinstance(max_classes, int) or isinstance(max_classes, bool) or max_classes <= 0:
            raise RuntimeError("benchmark bundle selection.max_classes must be a positive int")
        if not isinstance(min_minority_class_pct, (int, float)) or not 0 <= float(min_minority_class_pct) <= 100:
            raise RuntimeError(
                "benchmark bundle selection.min_minority_class_pct must be a percentage between 0 and 100"
            )
        normalized["max_classes"] = int(max_classes)
        normalized["min_minority_class_pct"] = float(min_minority_class_pct)
    return normalized


def normalize_benchmark_bundle(payload: Any) -> dict[str, Any]:
    """Validate and normalize benchmark bundle metadata."""

    if not isinstance(payload, dict):
        raise RuntimeError("benchmark bundle must be a JSON object")
    expected_keys = {"name", "version", "selection", "task_ids", "tasks"}
    actual_keys = set(payload.keys())
    if actual_keys != expected_keys:
        raise RuntimeError(
            "benchmark bundle keys mismatch: "
            f"missing={sorted(expected_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )

    name = payload["name"]
    version = payload["version"]
    selection = payload["selection"]
    task_ids = payload["task_ids"]
    tasks = payload["tasks"]
    if not isinstance(name, str) or not name.strip():
        raise RuntimeError("benchmark bundle name must be a non-empty string")
    if not isinstance(version, int) or version <= 0:
        raise RuntimeError("benchmark bundle version must be a positive int")
    if not isinstance(task_ids, list) or not task_ids:
        raise RuntimeError("benchmark bundle task_ids must be a non-empty list")
    if not isinstance(tasks, list) or not tasks:
        raise RuntimeError("benchmark bundle tasks must be a non-empty list")

    normalized_selection = _normalize_selection(selection)
    selection_task_type = str(normalized_selection["task_type"])
    normalized_task_ids = [int(task_id) for task_id in task_ids]
    normalized_tasks: list[dict[str, Any]] = []
    for index, task_payload in enumerate(tasks):
        if not isinstance(task_payload, dict):
            raise RuntimeError(f"benchmark bundle task {index} must be an object")
        task_keys = (
            {"task_id", "dataset_name", "n_rows", "n_features", "n_classes"}
            if selection_task_type == _CLASSIFICATION_TASK_TYPE
            else {"task_id", "dataset_name", "n_rows", "n_features"}
        )
        actual_task_keys = set(task_payload.keys())
        if actual_task_keys != task_keys:
            raise RuntimeError(
                f"benchmark bundle task keys mismatch at index {index}: "
                f"expected={sorted(task_keys)}, actual={sorted(actual_task_keys)}"
            )
        dataset_name = task_payload["dataset_name"]
        if not isinstance(dataset_name, str) or not dataset_name.strip():
            raise RuntimeError(f"benchmark bundle task dataset_name must be non-empty at index {index}")
        normalized_task = {
            "task_id": int(task_payload["task_id"]),
            "dataset_name": str(dataset_name),
            "n_rows": int(task_payload["n_rows"]),
            "n_features": int(task_payload["n_features"]),
        }
        if selection_task_type == _CLASSIFICATION_TASK_TYPE:
            normalized_task["n_classes"] = int(task_payload["n_classes"])
        normalized_tasks.append(normalized_task)

    if normalized_task_ids != [int(task["task_id"]) for task in normalized_tasks]:
        raise RuntimeError("benchmark bundle task_ids must match tasks[].task_id order exactly")

    return {
        "name": str(name),
        "version": int(version),
        "selection": normalized_selection,
        "task_ids": normalized_task_ids,
        "tasks": normalized_tasks,
    }


def _validate_bundle_missing_value_policy(
    bundle: Mapping[str, Any],
    *,
    allow_missing_values: bool,
    source_path: Path,
) -> None:
    if allow_missing_values:
        return
    selection = cast(dict[str, Any], bundle["selection"])
    max_missing_pct = float(selection["max_missing_pct"])
    if max_missing_pct > 0.0:
        raise RuntimeError(
            "benchmark bundle permits missing-valued inputs while allow_missing_values=False: "
            f"path={source_path}, max_missing_pct={max_missing_pct}"
        )


def benchmark_bundle_allows_missing_values(bundle: Mapping[str, Any]) -> bool:
    """Return whether the bundle contract permits missing-valued inputs."""

    selection = cast(dict[str, Any], bundle["selection"])
    raw_max_missing_pct = selection.get("max_missing_pct")
    if not isinstance(raw_max_missing_pct, (int, float)):
        return False
    return bool(float(raw_max_missing_pct) > 0.0)


def benchmark_bundle_task_type(bundle: Mapping[str, Any]) -> str:
    """Return the bundle task type."""

    selection = cast(dict[str, Any], bundle["selection"])
    task_type = selection.get("task_type", _CLASSIFICATION_TASK_TYPE)
    if task_type not in _ALLOWED_BUNDLE_SELECTION_TASK_TYPES:
        raise RuntimeError(
            "benchmark bundle selection.task_type must be one of "
            f"{sorted(_ALLOWED_BUNDLE_SELECTION_TASK_TYPES)!r}"
        )
    return str(task_type)


def load_benchmark_bundle(
    path: Path | None = None,
    *,
    allow_missing_values: bool = False,
) -> dict[str, Any]:
    """Load and validate the canonical benchmark bundle metadata."""

    bundle_path = (path or default_benchmark_bundle_path()).expanduser().resolve()
    with bundle_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    try:
        bundle = normalize_benchmark_bundle(payload)
    except RuntimeError as exc:
        raise RuntimeError(f"{exc}: {bundle_path}") from exc
    _validate_bundle_missing_value_policy(
        bundle,
        allow_missing_values=bool(allow_missing_values),
        source_path=bundle_path,
    )
    return bundle


def load_benchmark_bundle_for_execution(path: Path | None = None) -> tuple[dict[str, Any], bool]:
    """Load a bundle and resolve whether execution should allow missing values."""

    bundle = load_benchmark_bundle(path, allow_missing_values=True)
    return bundle, benchmark_bundle_allows_missing_values(bundle)


def benchmark_bundle_summary(
    bundle: Mapping[str, Any],
    *,
    source_path: Path,
) -> dict[str, Any]:
    """Build compact bundle metadata for run summaries."""

    task_ids = [int(task_id) for task_id in cast(list[Any], bundle["task_ids"])]
    selection_raw = bundle.get("selection")
    selection = (
        cast(dict[str, Any], json.loads(json.dumps(selection_raw, sort_keys=True)))
        if isinstance(selection_raw, Mapping)
        else None
    )
    allow_missing_values = (
        None
        if not isinstance(selection_raw, Mapping)
        else benchmark_bundle_allows_missing_values(bundle)
    )
    return {
        "name": str(bundle["name"]),
        "version": int(bundle["version"]),
        "source_path": str(source_path.expanduser().resolve()),
        "task_count": int(len(task_ids)),
        "task_ids": task_ids,
        "selection": selection,
        "allow_missing_values": allow_missing_values,
        "all_tasks_no_missing": None if allow_missing_values is None else (not allow_missing_values),
    }
