"""Manifest inspection and comparison helpers."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, cast

import pyarrow.parquet as pq

from .manifest import MANIFEST_SUMMARY_METADATA_KEY


def _read_persisted_manifest_summary(manifest_path: Path) -> dict[str, Any] | None:
    metadata = pq.ParquetFile(manifest_path).schema_arrow.metadata or {}
    raw_summary = metadata.get(MANIFEST_SUMMARY_METADATA_KEY)
    if raw_summary is None:
        return None
    payload = json.loads(raw_summary.decode("utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"persisted manifest summary must be an object: {manifest_path}")
    return cast(dict[str, Any], payload)


def _distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": int(min(values)),
        "max": int(max(values)),
        "mean": float(sum(values) / float(len(values))),
    }


def inspect_manifest(manifest_path: Path) -> dict[str, Any]:
    """Inspect one manifest parquet file and summarize its contents."""

    resolved_manifest = manifest_path.expanduser().resolve()
    if not resolved_manifest.exists():
        raise RuntimeError(f"manifest does not exist: {resolved_manifest}")
    if not resolved_manifest.is_file():
        raise RuntimeError(f"manifest path is not a file: {resolved_manifest}")

    table = pq.read_table(resolved_manifest)
    records = cast(list[dict[str, Any]], table.to_pylist())
    if not records:
        raise RuntimeError(f"manifest has zero rows: {resolved_manifest}")

    split_counts: Counter[str] = Counter()
    task_counts: Counter[str] = Counter()
    task_split_counts: dict[str, Counter[str]] = {}
    task_train_test_record_counts: dict[str, Counter[str]] = {}
    filter_status_counts: Counter[str] = Counter()
    missing_value_status_counts: Counter[str] = Counter()
    task_missing_value_status_counts: dict[str, Counter[str]] = {}
    n_features_values: list[int] = []
    classification_n_classes: list[int] = []
    source_roots: set[str] = set()
    dataset_ids: set[str] = set()

    for record in records:
        split = str(record.get("split", "unknown"))
        task = str(record.get("task", "unknown"))
        split_counts[split] += 1
        task_counts[task] += 1
        task_split_counts.setdefault(task, Counter())[split] += 1
        if record.get("n_train") is not None and int(record["n_train"]) > 0:
            task_train_test_record_counts.setdefault(task, Counter())["train"] += 1
        if record.get("n_test") is not None and int(record["n_test"]) > 0:
            task_train_test_record_counts.setdefault(task, Counter())["test"] += 1

        raw_filter_status = record.get("filter_status")
        filter_status = "missing" if raw_filter_status is None else str(raw_filter_status)
        filter_status_counts[filter_status] += 1

        raw_missing_value_status = record.get("missing_value_status")
        missing_value_status = (
            "missing" if raw_missing_value_status is None else str(raw_missing_value_status)
        )
        missing_value_status_counts[missing_value_status] += 1
        task_missing_value_status_counts.setdefault(task, Counter())[missing_value_status] += 1

        raw_n_features = record.get("n_features")
        if raw_n_features is not None:
            n_features_values.append(int(raw_n_features))

        if task == "classification" and record.get("n_classes") is not None:
            classification_n_classes.append(int(record["n_classes"]))

        source_root = record.get("source_root_id")
        if isinstance(source_root, str) and source_root.strip():
            source_roots.add(source_root)
        dataset_id = record.get("dataset_id")
        if isinstance(dataset_id, str) and dataset_id.strip():
            dataset_ids.add(dataset_id)

    n_class_histogram = Counter(classification_n_classes)
    return {
        "manifest_path": str(resolved_manifest),
        "total_records": len(records),
        "split_counts": dict(sorted(split_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "task_split_counts": {
            str(task): dict(sorted(counts.items()))
            for task, counts in sorted(task_split_counts.items())
        },
        "task_train_test_record_counts": {
            str(task): dict(sorted(counts.items()))
            for task, counts in sorted(task_train_test_record_counts.items())
        },
        "filter_status_counts": dict(sorted(filter_status_counts.items())),
        "missing_value_status_counts": dict(sorted(missing_value_status_counts.items())),
        "task_missing_value_status_counts": {
            str(task): dict(sorted(counts.items()))
            for task, counts in sorted(task_missing_value_status_counts.items())
        },
        "n_features": (
            None
            if not n_features_values
            else {
                "min": int(min(n_features_values)),
                "max": int(max(n_features_values)),
            }
        ),
        "classification_n_classes": (
            None
            if not classification_n_classes
            else {
                "min": int(min(classification_n_classes)),
                "max": int(max(classification_n_classes)),
                "histogram": {
                    str(class_count): int(count)
                    for class_count, count in sorted(n_class_histogram.items())
                },
            }
        ),
        "unique_source_root_count": len(source_roots),
        "unique_dataset_id_count": len(dataset_ids),
        "persisted_summary": _read_persisted_manifest_summary(resolved_manifest),
    }


def manifest_characteristics(manifest_path: Path) -> dict[str, Any]:
    """Return a richer manifest summary for training and corpus surfaces."""

    resolved_manifest = manifest_path.expanduser().resolve()
    parquet_file = pq.ParquetFile(resolved_manifest)
    table = parquet_file.read()
    rows = cast(list[dict[str, Any]], table.to_pylist())
    missing_value_statuses = [
        str(status).strip()
        if isinstance((status := row.get("missing_value_status")), str) and str(status).strip()
        else None
        for row in rows
    ]
    split_counts = Counter(str(row.get("split", "missing")) for row in rows)
    task_counts = Counter(str(row.get("task", "missing")) for row in rows)
    filter_status_counts = Counter(str(row.get("filter_status", "missing")) for row in rows)
    missing_value_status_counts = Counter(str(row.get("missing_value_status", "missing")) for row in rows)
    has_complete_missing_value_metadata = bool(rows) and all(
        status is not None for status in missing_value_statuses
    )
    missing_value_policies = sorted(
        {
            str(row["missing_value_policy"])
            for row in rows
            if isinstance(row.get("missing_value_policy"), str) and row["missing_value_policy"].strip()
        }
    )
    source_root_ids = sorted(
        {
            str(row["source_root_id"])
            for row in rows
            if isinstance(row.get("source_root_id"), str) and row["source_root_id"].strip()
        }
    )
    shard_counts = Counter(
        str(row["source_shard_relpath"])
        for row in rows
        if isinstance(row.get("source_shard_relpath"), str) and row["source_shard_relpath"].strip()
    )
    total_rows = [
        int(row["n_train"]) + int(row["n_test"])
        for row in rows
        if row.get("n_train") is not None and row.get("n_test") is not None
    ]
    n_features = [
        int(row["n_features"])
        for row in rows
        if row.get("n_features") is not None and int(row["n_features"]) >= 0
    ]
    n_classes = [
        int(row["n_classes"])
        for row in rows
        if row.get("n_classes") is not None
    ]
    raw_metadata = parquet_file.schema_arrow.metadata or {}
    persisted_summary = None
    raw_summary = raw_metadata.get(MANIFEST_SUMMARY_METADATA_KEY)
    if raw_summary is not None:
        persisted_summary = json.loads(raw_summary.decode("utf-8"))
    return {
        "record_count": int(len(rows)),
        "split_counts": dict(sorted(split_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "row_count_distribution": _distribution(total_rows),
        "feature_count_distribution": _distribution(n_features),
        "class_count_distribution": _distribution(n_classes),
        "filter_status_counts": dict(sorted(filter_status_counts.items())),
        "missing_value_status_counts": dict(sorted(missing_value_status_counts.items())),
        "missing_value_policy": None if len(missing_value_policies) != 1 else missing_value_policies[0],
        "all_records_no_missing": (
            None
            if not has_complete_missing_value_metadata
            else missing_value_status_counts.get("contains_nan_or_inf", 0) == 0
        ),
        "persisted_summary": persisted_summary,
        "source_root_ids": source_root_ids,
        "source_shard_relpath_summary": {
            "unique_count": int(len(shard_counts)),
            "top_counts": [
                {"relpath": relpath, "count": int(count)}
                for relpath, count in shard_counts.most_common(10)
            ],
        },
    }


def compare_jsonlike_payloads(
    left: Any,
    right: Any,
    *,
    prefix: str = "",
) -> dict[str, dict[str, Any]]:
    """Return recursive differences between two JSON-like payloads."""

    if isinstance(left, dict) and isinstance(right, dict):
        differences: dict[str, dict[str, Any]] = {}
        for key in sorted(set(left.keys()) | set(right.keys())):
            next_prefix = f"{prefix}.{key}" if prefix else str(key)
            differences.update(compare_jsonlike_payloads(left.get(key), right.get(key), prefix=next_prefix))
        return differences
    if left == right:
        return {}
    return {prefix: {"left": left, "right": right}}
