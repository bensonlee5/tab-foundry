"""Manifest inspection helpers."""

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
