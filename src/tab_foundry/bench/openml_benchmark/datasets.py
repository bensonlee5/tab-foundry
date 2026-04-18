"""Benchmark-surface dataset helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np
from tab_realdata_hub.manifest import LoadedManifestDatasets, load_manifest_datasets
from tab_realdata_hub.openml import (
    PreparedOpenMLTask as PreparedOpenMLBenchmarkTask,
    get_feature_preprocessor,
    prepare_task as _prepare_openml_task,
    read_required_quality as read_required_openml_quality,
)
from tab_foundry.data.dataset import load_manifest_record_metadata

from .bundle import default_benchmark_manifest_path, validate_default_anchor_benchmark_summary
from .dataset_common import BenchmarkDataset, _assert_finite_benchmark_datasets

__all__ = [
    "PreparedOpenMLBenchmarkTask",
    "get_feature_preprocessor",
    "benchmark_manifest_allows_missing_values",
    "benchmark_manifest_bundle_summary",
    "benchmark_manifest_task_type",
    "load_benchmark_manifest_datasets",
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


def _task_record_metadata(record: Mapping[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata")
    if not isinstance(metadata, Mapping):
        raise RuntimeError(f"benchmark task record omitted metadata: {record!r}")
    return dict(metadata)


def benchmark_manifest_task_type(task_records: list[dict[str, Any]]) -> str:
    """Return the shared task type encoded by a manifest-backed benchmark surface."""

    task_values = {str(record["task"]) for record in task_records}
    if not task_values:
        raise RuntimeError("manifest-backed benchmark surface has no task records")
    if len(task_values) != 1:
        raise RuntimeError(
            f"manifest-backed benchmark surface mixes task types: {sorted(task_values)!r}"
        )
    task = next(iter(task_values))
    if task == "classification":
        return "supervised_classification"
    if task == "regression":
        return "supervised_regression"
    raise RuntimeError(f"unsupported manifest-backed benchmark task type: {task!r}")


def benchmark_manifest_allows_missing_values(task_records: list[dict[str, Any]]) -> bool:
    """Return whether the manifest-backed benchmark surface explicitly allows missing values."""

    allow_flags: set[bool] = set()
    for record in task_records:
        benchmark_bundle = _task_record_metadata(record).get("benchmark_bundle")
        if isinstance(benchmark_bundle, Mapping) and "allow_missing_values" in benchmark_bundle:
            allow_flags.add(bool(benchmark_bundle["allow_missing_values"]))
    if not allow_flags:
        return False
    if len(allow_flags) != 1:
        raise RuntimeError(
            "manifest-backed benchmark surface mixes allow_missing_values provenance"
        )
    return next(iter(allow_flags))


def benchmark_manifest_bundle_summary(task_records: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Reconstruct compact source-bundle provenance when present in manifest metadata."""

    benchmark_bundles: list[dict[str, Any]] = []
    task_ids: list[int] = []
    for record in task_records:
        benchmark_bundle = _task_record_metadata(record).get("benchmark_bundle")
        if not isinstance(benchmark_bundle, Mapping):
            return None
        payload = dict(benchmark_bundle)
        benchmark_bundles.append(payload)
        task_id = payload.get("task_id")
        if not isinstance(task_id, int):
            raise RuntimeError(f"benchmark bundle metadata omitted task_id: {payload!r}")
        task_ids.append(int(task_id))
    first = benchmark_bundles[0]
    for payload in benchmark_bundles[1:]:
        comparable = dict(payload)
        comparable.pop("task_id", None)
        baseline = dict(first)
        baseline.pop("task_id", None)
        if comparable != baseline:
            raise RuntimeError(
                "manifest-backed benchmark surface mixes source bundle provenance"
            )
    selection = first.get("selection")
    return {
        "name": str(first["name"]),
        "version": int(first["version"]),
        "source_path": str(first["source_path"]),
        "task_count": int(len(task_ids)),
        "task_ids": sorted(task_ids),
        "selection": None if not isinstance(selection, Mapping) else dict(selection),
        "allow_missing_values": bool(first.get("allow_missing_values", False)),
        "all_tasks_no_missing": not bool(first.get("allow_missing_values", False)),
    }


def load_benchmark_manifest_datasets(
    *,
    benchmark_manifest_path: Path,
    allow_missing_values: bool | None = None,
) -> tuple[dict[str, BenchmarkDataset], list[dict[str, Any]], dict[str, Any]]:
    """Load a manifest-backed benchmark surface."""

    resolved_manifest_path = benchmark_manifest_path.expanduser().resolve()
    if resolved_manifest_path.suffix.lower() == ".json":
        raise RuntimeError(
            "benchmark execution now requires a materialized manifest parquet; "
            "materialize the benchmark bundle first and pass `--benchmark-manifest-path` "
            f"instead of the bundle JSON: {resolved_manifest_path}"
        )
    # Load first with missing values admitted, then enforce the benchmark-surface policy
    # from persisted provenance below.
    loaded: LoadedManifestDatasets = load_manifest_datasets(
        resolved_manifest_path,
        allow_missing_values=True,
    )
    task_records = [dict(record) for record in loaded.task_records]
    if not task_records:
        raise RuntimeError(
            f"manifest-backed benchmark surface produced no task records: {benchmark_manifest_path}"
        )
    surface_allow_missing_values = benchmark_manifest_allows_missing_values(task_records)
    if allow_missing_values is not None and bool(allow_missing_values) != surface_allow_missing_values:
        raise RuntimeError(
            "benchmark manifest allow_missing_values mismatch: "
            f"expected={bool(allow_missing_values)}, actual={surface_allow_missing_values}, "
            f"path={benchmark_manifest_path}"
        )
    datasets: dict[str, BenchmarkDataset] = {}
    for task_record in task_records:
        dataset_name = str(task_record["dataset_name"])
        if dataset_name not in loaded.datasets:
            raise RuntimeError(
                "manifest-backed benchmark surface task record is missing a loaded dataset: "
                f"dataset={dataset_name!r}"
            )
        x, y = loaded.datasets[dataset_name]
        manifest_record = task_record.get("manifest_record")
        if not isinstance(manifest_record, Mapping):
            raise RuntimeError(
                "manifest-backed benchmark task record omitted manifest_record payload: "
                f"dataset={dataset_name!r}"
            )
        _metadata, feature_types = load_manifest_record_metadata(
            resolved_manifest_path,
            record=manifest_record,
            expected_feature_count=int(np.asarray(x).shape[1]),
            require_feature_types=False,
        )
        dataset_x = np.asarray(x, dtype=np.float32)
        dataset_y = np.asarray(y)
        if feature_types is None:
            datasets[dataset_name] = (dataset_x, dataset_y)
        else:
            datasets[dataset_name] = (
                dataset_x,
                dataset_y,
                feature_types,
            )
    if not surface_allow_missing_values:
        _assert_finite_benchmark_datasets(
            datasets,
            context=f"benchmark manifest {benchmark_manifest_path!s}",
        )
    bundle_summary = benchmark_manifest_bundle_summary(task_records)
    if resolved_manifest_path == default_benchmark_manifest_path().expanduser().resolve():
        anchor_contract_issues = validate_default_anchor_benchmark_summary(bundle_summary)
        if anchor_contract_issues:
            raise RuntimeError(
                "default anchor benchmark manifest drift detected:\n- "
                + "\n- ".join(anchor_contract_issues)
            )
    return datasets, task_records, {
        "manifest_path": str(loaded.manifest_path),
        "contract_version": int(loaded.contract_version),
        "manifest_sha256": str(loaded.manifest_sha256),
        "task_type": benchmark_manifest_task_type(task_records),
        "allow_missing_values": surface_allow_missing_values,
        "benchmark_bundle": bundle_summary,
        "persisted_summary": loaded.persisted_summary,
    }
