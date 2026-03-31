"""Lightweight manifest summaries and streaming characteristics helpers."""

from __future__ import annotations

from collections import Counter
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, cast

import pyarrow.parquet as pq


_SUMMARY_METADATA_KEY = b"tab_foundry_manifest_summary"


def _read_json_mapping(raw: bytes, *, context: str) -> dict[str, Any]:
    payload = json.loads(raw.decode("utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{context} must decode to a JSON object")
    return {str(key): value for key, value in payload.items()}


def _parquet_file(path: Path) -> pq.ParquetFile:
    resolved = path.expanduser().resolve()
    return pq.ParquetFile(resolved)


def load_manifest_persisted_summary(manifest_path: Path) -> dict[str, Any] | None:
    parquet_file = _parquet_file(manifest_path)
    metadata = parquet_file.schema_arrow.metadata or {}
    raw_summary = metadata.get(_SUMMARY_METADATA_KEY)
    if raw_summary is None:
        return None
    return _read_json_mapping(raw_summary, context="manifest persisted summary")


def _string_counter(values: Iterable[Any]) -> Counter[str]:
    counter: Counter[str] = Counter()
    for value in values:
        if value is None:
            counter["missing"] += 1
            continue
        normalized = str(value).strip()
        counter[normalized or "missing"] += 1
    return counter


def _append_positive_ints(target: list[int], values: Iterable[Any]) -> None:
    for value in values:
        if value is None or isinstance(value, bool):
            continue
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            continue
        if parsed > 0:
            target.append(parsed)


def _distribution_payload(values: list[int]) -> dict[str, Any] | None:
    if not values:
        return None
    return {
        "min": int(min(values)),
        "max": int(max(values)),
        "count": int(len(values)),
    }


def _all_records_no_missing(status_counts: Mapping[str, int]) -> bool | None:
    normalized = {
        str(key): int(value)
        for key, value in status_counts.items()
        if int(value) > 0
    }
    if not normalized:
        return None
    non_missing_keys = {key for key in normalized if key != "missing"}
    if not non_missing_keys:
        return None
    if non_missing_keys == {"clean"}:
        return True
    if any(key not in {"clean", "missing"} for key in normalized):
        return False
    return None


def _iter_manifest_batches(
    parquet_file: pq.ParquetFile,
    *,
    columns: list[str],
    batch_size: int = 4096,
) -> Iterable[dict[str, list[Any]]]:
    selected_columns = [column for column in columns if column in parquet_file.schema.names]
    if not selected_columns:
        return
    for batch in parquet_file.iter_batches(columns=selected_columns, batch_size=batch_size):
        batch_payload: dict[str, list[Any]] = {}
        for index, name in enumerate(batch.schema.names):
            batch_payload[str(name)] = batch.column(index).to_pylist()
        yield batch_payload


def inspect_manifest_summary(manifest_path: Path) -> dict[str, Any]:
    parquet_file = _parquet_file(manifest_path)
    metadata = parquet_file.metadata
    total_records = 0 if metadata is None else int(metadata.num_rows)
    split_counts: Counter[str] = Counter()
    for batch in _iter_manifest_batches(parquet_file, columns=["split"]):
        split_counts.update(_string_counter(batch.get("split", [])))
    return {
        "manifest_path": str(manifest_path.expanduser().resolve()),
        "total_records": int(total_records),
        "split_counts": dict(sorted(split_counts.items())),
        "persisted_summary": load_manifest_persisted_summary(manifest_path),
    }


def compute_manifest_characteristics(manifest_path: Path) -> dict[str, Any]:
    parquet_file = _parquet_file(manifest_path)
    metadata = parquet_file.metadata
    record_count = 0 if metadata is None else int(metadata.num_rows)
    split_counts: Counter[str] = Counter()
    missing_value_policy_counts: Counter[str] = Counter()
    missing_value_status_counts: Counter[str] = Counter()
    row_counts: list[int] = []
    feature_counts: list[int] = []
    class_counts: list[int] = []

    for batch in _iter_manifest_batches(
        parquet_file,
        columns=[
            "split",
            "missing_value_policy",
            "missing_value_status",
            "n_train",
            "n_test",
            "n_features",
            "n_classes",
        ],
    ):
        split_counts.update(_string_counter(batch.get("split", [])))
        missing_value_policy_counts.update(_string_counter(batch.get("missing_value_policy", [])))
        missing_value_status_counts.update(_string_counter(batch.get("missing_value_status", [])))
        _append_positive_ints(feature_counts, batch.get("n_features", []))
        _append_positive_ints(class_counts, batch.get("n_classes", []))
        n_train_values = batch.get("n_train", [])
        n_test_values = batch.get("n_test", [])
        if n_train_values or n_test_values:
            for n_train, n_test in zip(n_train_values, n_test_values, strict=False):
                if n_train is None or n_test is None:
                    continue
                try:
                    total_rows = int(n_train) + int(n_test)
                except (TypeError, ValueError):
                    continue
                if total_rows > 0:
                    row_counts.append(total_rows)

    persisted_summary = load_manifest_persisted_summary(manifest_path)
    if not missing_value_policy_counts and persisted_summary is not None:
        persisted_policy = persisted_summary.get("missing_value_policy")
        if isinstance(persisted_policy, str) and persisted_policy.strip():
            missing_value_policy_counts[str(persisted_policy).strip()] = int(record_count)
    if not missing_value_status_counts:
        missing_value_status_counts["missing"] = int(record_count)

    missing_value_policy = None
    if len(missing_value_policy_counts) == 1:
        missing_value_policy = next(iter(missing_value_policy_counts))
    elif persisted_summary is not None:
        persisted_policy = persisted_summary.get("missing_value_policy")
        if isinstance(persisted_policy, str) and persisted_policy.strip():
            missing_value_policy = str(persisted_policy).strip()

    payload: dict[str, Any] = {
        "record_count": int(record_count),
        "split_counts": dict(sorted(split_counts.items())),
        "missing_value_policy": missing_value_policy,
        "missing_value_status_counts": dict(sorted(missing_value_status_counts.items())),
        "all_records_no_missing": _all_records_no_missing(missing_value_status_counts),
        "persisted_summary": persisted_summary,
    }
    row_distribution = _distribution_payload(row_counts)
    if row_distribution is not None:
        payload["row_count_distribution"] = row_distribution
    feature_distribution = _distribution_payload(feature_counts)
    if feature_distribution is not None:
        payload["feature_count_distribution"] = feature_distribution
    class_distribution = _distribution_payload(class_counts)
    if class_distribution is not None:
        payload["class_count_distribution"] = class_distribution
    return payload


def load_manifest_characteristics_sidecar(sidecar_path: Path) -> dict[str, Any] | None:
    resolved = sidecar_path.expanduser().resolve()
    if not resolved.exists():
        return None
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"manifest characteristics sidecar must decode to a JSON object: {resolved}")
    return cast(dict[str, Any], {str(key): value for key, value in payload.items()})


def write_manifest_characteristics_sidecar(
    *,
    manifest_path: Path,
    sidecar_path: Path,
) -> dict[str, Any]:
    payload = compute_manifest_characteristics(manifest_path)
    resolved_sidecar = sidecar_path.expanduser().resolve()
    resolved_sidecar.parent.mkdir(parents=True, exist_ok=True)
    resolved_sidecar.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
