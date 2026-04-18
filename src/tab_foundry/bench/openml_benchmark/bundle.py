"""Benchmark bundle loading and validation helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, cast

from tab_foundry.repo_paths import normalize_repo_relative_path, repo_root


BENCHMARK_BUNDLE_FILENAME = "openml_classification_medium_v1.json"
DEFAULT_ANCHOR_CONTROL_BASELINE_ID = "cls_benchmark_linear_multiclass_medium_v1"
_CLASSIFICATION_TASK_TYPE = "supervised_classification"
_PERCENT_SCALE_MAX = 100.0
_REGRESSION_TASK_TYPE = "supervised_regression"
_OPTIONAL_BUNDLE_KEYS = {"tasks"}
_ALLOWED_BUNDLE_SELECTION_TASK_TYPES = {
    _CLASSIFICATION_TASK_TYPE,
    _REGRESSION_TASK_TYPE,
}
_REPO_TRACKED_BUNDLE_SENTINEL = ("src", "tab_foundry", "bench")


def default_benchmark_bundle_path() -> Path:
    """Return the repo-tracked canonical benchmark bundle path."""

    return repo_root() / "src" / "tab_foundry" / "bench" / BENCHMARK_BUNDLE_FILENAME


def default_benchmark_manifest_path() -> Path:
    """Return the repo-local canonical materialized benchmark manifest path."""

    bundle_stem = Path(BENCHMARK_BUNDLE_FILENAME).stem
    return repo_root() / "data" / "manifests" / "bench" / bundle_stem / "manifest.parquet"


def default_anchor_control_baseline_id() -> str:
    """Return the canonical control-baseline id paired with the default anchor benchmark."""

    return DEFAULT_ANCHOR_CONTROL_BASELINE_ID


def _default_repo_root() -> Path:
    return repo_root().expanduser().resolve()


def _portable_repo_relative_bundle_path(
    path: Path,
    *,
    repo_root: Path,
) -> Path | None:
    resolved = path.expanduser().resolve()
    try:
        return resolved.relative_to(repo_root)
    except ValueError:
        pass

    parts = resolved.parts
    sentinel = _REPO_TRACKED_BUNDLE_SENTINEL
    sentinel_length = len(sentinel)
    for index in range(len(parts) - sentinel_length + 1):
        if parts[index : index + sentinel_length] != sentinel:
            continue
        suffix = Path(*parts[index:])
        if (repo_root / suffix).exists():
            return suffix
    return None


def canonical_benchmark_bundle_source_path(
    value: str | Path,
    *,
    repo_root: Path | None = None,
) -> str:
    """Return one portable bundle-path identity for persistence and reuse matching."""

    resolved_repo_root = (repo_root or _default_repo_root()).expanduser().resolve()
    path = value if isinstance(value, Path) else Path(str(value).strip()).expanduser()
    resolved_path = (
        path.resolve()
        if path.is_absolute()
        else (resolved_repo_root / path).resolve()
    )
    portable = _portable_repo_relative_bundle_path(
        resolved_path,
        repo_root=resolved_repo_root,
    )
    if portable is not None:
        return str(portable)
    return normalize_repo_relative_path(
        resolved_path,
        root=resolved_repo_root,
    )


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
        | ({"min_classes"} if "min_classes" in payload else set())
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
    if not isinstance(max_missing_pct, (int, float)) or not 0 <= float(max_missing_pct) <= _PERCENT_SCALE_MAX:
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
        min_classes = payload.get("min_classes")
        if not isinstance(max_classes, int) or isinstance(max_classes, bool) or max_classes <= 0:
            raise RuntimeError("benchmark bundle selection.max_classes must be a positive int")
        if min_classes is not None and (
            not isinstance(min_classes, int)
            or isinstance(min_classes, bool)
            or min_classes <= 0
            or min_classes > max_classes
        ):
            raise RuntimeError(
                "benchmark bundle selection.min_classes must be a positive int "
                "no larger than max_classes"
            )
        if not isinstance(min_minority_class_pct, (int, float)) or not 0 <= float(min_minority_class_pct) <= _PERCENT_SCALE_MAX:
            raise RuntimeError(
                "benchmark bundle selection.min_minority_class_pct must be a percentage between 0 and 100"
            )
        normalized["max_classes"] = int(max_classes)
        if min_classes is not None:
            normalized["min_classes"] = int(min_classes)
        normalized["min_minority_class_pct"] = float(min_minority_class_pct)
    return normalized


def normalize_benchmark_bundle(payload: Any) -> dict[str, Any]:
    """Validate and normalize benchmark bundle metadata."""

    if not isinstance(payload, dict):
        raise RuntimeError("benchmark bundle must be a JSON object")
    required_keys = {"name", "version", "selection", "task_ids"}
    expected_keys = required_keys | _OPTIONAL_BUNDLE_KEYS
    actual_keys = set(payload.keys())
    if not required_keys.issubset(actual_keys) or not actual_keys.issubset(expected_keys):
        raise RuntimeError(
            "benchmark bundle keys mismatch: "
            f"missing={sorted(required_keys - actual_keys)}, "
            f"extra={sorted(actual_keys - expected_keys)}"
        )

    name = payload["name"]
    version = payload["version"]
    selection = payload["selection"]
    task_ids = payload["task_ids"]
    tasks = payload.get("tasks", [])
    if not isinstance(name, str) or not name.strip():
        raise RuntimeError("benchmark bundle name must be a non-empty string")
    if not isinstance(version, int) or version <= 0:
        raise RuntimeError("benchmark bundle version must be a positive int")
    if not isinstance(task_ids, list) or not task_ids:
        raise RuntimeError("benchmark bundle task_ids must be a non-empty list")
    if not isinstance(tasks, list):
        raise RuntimeError("benchmark bundle tasks must be a list when present")

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

    if normalized_tasks and normalized_task_ids != [int(task["task_id"]) for task in normalized_tasks]:
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
        "source_path": canonical_benchmark_bundle_source_path(source_path),
        "task_count": int(len(task_ids)),
        "task_ids": task_ids,
        "selection": selection,
        "allow_missing_values": allow_missing_values,
        "all_tasks_no_missing": None if allow_missing_values is None else (not allow_missing_values),
    }


def _normalized_bundle_summary_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    task_ids = [int(task_id) for task_id in cast(list[Any], payload.get("task_ids", []))]
    selection_raw = payload.get("selection")
    selection = (
        cast(dict[str, Any], json.loads(json.dumps(selection_raw, sort_keys=True)))
        if isinstance(selection_raw, Mapping)
        else None
    )
    source_path_value = payload.get("source_path")
    source_path = (
        canonical_benchmark_bundle_source_path(str(source_path_value))
        if isinstance(source_path_value, str) and source_path_value.strip()
        else None
    )
    return {
        "name": str(payload["name"]),
        "version": int(payload["version"]),
        "source_path": source_path,
        "task_count": int(payload["task_count"]),
        "task_ids": task_ids,
        "selection": selection,
        "allow_missing_values": (
            None
            if payload.get("allow_missing_values") is None
            else bool(payload["allow_missing_values"])
        ),
        "all_tasks_no_missing": (
            None
            if payload.get("all_tasks_no_missing") is None
            else bool(payload["all_tasks_no_missing"])
        ),
    }


def default_anchor_benchmark_summary() -> dict[str, Any]:
    """Return the canonical medium-multiclass anchor benchmark summary."""

    bundle_path = default_benchmark_bundle_path()
    bundle = load_benchmark_bundle(bundle_path, allow_missing_values=True)
    return benchmark_bundle_summary(bundle, source_path=bundle_path)


def validate_default_anchor_benchmark_summary(
    bundle_summary: Mapping[str, Any] | None,
) -> list[str]:
    """Return contract issues when the default anchor benchmark resolves to a stale surface."""

    if bundle_summary is None:
        return ["default anchor benchmark summary is missing from the resolved manifest surface"]
    expected = _normalized_bundle_summary_payload(default_anchor_benchmark_summary())
    actual = _normalized_bundle_summary_payload(bundle_summary)
    issues: list[str] = []
    for key in (
        "name",
        "version",
        "source_path",
        "task_count",
        "task_ids",
        "selection",
        "allow_missing_values",
        "all_tasks_no_missing",
    ):
        if actual[key] != expected[key]:
            issues.append(
                "default anchor benchmark mismatch for "
                f"{key}: expected={expected[key]!r} actual={actual[key]!r}"
            )
    return issues
