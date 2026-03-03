"""Dataset loader for parquet task bundles."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from tab_foundry.types import TaskBatch


def _packed_x_to_matrix(x_column: Any) -> np.ndarray:
    rows = x_column.to_numpy(zero_copy_only=False)
    if rows.size == 0:
        raise RuntimeError("packed split has zero rows")
    try:
        x = np.vstack(rows).astype(np.float32, copy=False)
    except ValueError as exc:
        raise RuntimeError("packed x column has ragged row lengths") from exc
    if x.ndim != 2:
        raise RuntimeError(f"packed x column did not decode to rank-2 matrix, got shape={x.shape}")
    return x


def _read_packed_split(split_path: Path, *, dataset_index: int) -> tuple[np.ndarray, np.ndarray]:
    try:
        table = pq.read_table(
            split_path,
            filters=[("dataset_index", "=", int(dataset_index))],
            columns=["row_index", "x", "y"],
        )
    except Exception as exc:  # pragma: no cover - pyarrow error typing is backend-specific
        raise RuntimeError(
            f"failed to read packed split parquet path={split_path}, dataset_index={dataset_index}"
        ) from exc

    if table.num_rows <= 0:
        raise RuntimeError(
            f"packed split has zero rows for dataset_index={dataset_index}: path={split_path}"
        )

    row_index = table["row_index"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    x = _packed_x_to_matrix(table["x"])
    y = table["y"].to_numpy(zero_copy_only=False)
    if row_index.shape[0] != x.shape[0] or row_index.shape[0] != y.shape[0]:
        raise RuntimeError(
            "packed split row count mismatch: "
            f"path={split_path}, dataset_index={dataset_index}, "
            f"row_index={row_index.shape[0]}, x={x.shape[0]}, y={y.shape[0]}"
        )

    order = np.argsort(row_index, kind="stable")
    if not np.array_equal(order, np.arange(order.shape[0])):
        row_index = row_index[order]
        x = x[order]
        y = y[order]

    unique = np.unique(row_index)
    if unique.shape[0] != row_index.shape[0]:
        raise RuntimeError(
            f"packed split row_index values must be unique: path={split_path}, dataset_index={dataset_index}"
        )
    return x, y


def _impute_mean(x_train: np.ndarray, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = x_train.copy()
    test = x_test.copy()
    means = np.nanmean(train, axis=0)
    means = np.where(np.isnan(means), 0.0, means)

    train_nan = np.isnan(train)
    if np.any(train_nan):
        train[train_nan] = np.take(means, np.where(train_nan)[1])

    test_nan = np.isnan(test)
    if np.any(test_nan):
        test[test_nan] = np.take(means, np.where(test_nan)[1])
    return train, test


def _remap_labels(
    y_train: np.ndarray, y_test: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int, np.ndarray]:
    """Remap labels to contiguous [0, num_classes) range.

    Returns (remapped_train, remapped_test, num_classes, valid_mask) where
    valid_mask is a boolean array over y_test indicating which test samples
    had labels present in the train split.
    """
    train_i64 = y_train.astype(np.int64, copy=False)
    test_i64 = y_test.astype(np.int64, copy=False)
    values = np.unique(train_i64)
    if values.size == 0:
        raise RuntimeError("train split has no class labels")

    remapped_train = np.searchsorted(values, train_i64).astype(np.int64, copy=False)
    test_pos = np.searchsorted(values, test_i64)
    clamped = np.clip(test_pos, 0, values.shape[0] - 1)
    in_bounds = test_pos < values.shape[0]
    valid_mask = in_bounds & (values[clamped] == test_i64)
    remapped_test = test_pos[valid_mask].astype(np.int64, copy=False)
    return remapped_train, remapped_test, int(values.shape[0]), valid_mask


def _resolve_record_path(manifest_path: Path, raw_path: str) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (manifest_path.parent / path).resolve()


def _read_ndjson_record_by_offset(
    metadata_path: Path,
    *,
    offset_bytes: int,
    size_bytes: int,
) -> dict[str, Any]:
    if offset_bytes < 0 or size_bytes <= 0:
        raise RuntimeError(
            "metadata byte offset/size must be non-negative and non-zero: "
            f"path={metadata_path}, offset={offset_bytes}, size={size_bytes}"
        )

    with metadata_path.open("rb") as handle:
        handle.seek(offset_bytes)
        raw = handle.read(size_bytes)
    if len(raw) != size_bytes:
        raise RuntimeError(
            "failed to read full metadata slice from NDJSON: "
            f"path={metadata_path}, offset={offset_bytes}, size={size_bytes}, got={len(raw)}"
        )

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as exc:  # pragma: no cover - defensive parse context
        raise RuntimeError(
            "failed to parse metadata NDJSON record: "
            f"path={metadata_path}, offset={offset_bytes}, size={size_bytes}"
        ) from exc
    if not isinstance(payload, dict):
        raise RuntimeError(
            f"metadata NDJSON payload must be an object: path={metadata_path}, offset={offset_bytes}"
        )
    return payload


def _subsample_rows(
    x: np.ndarray,
    y: np.ndarray,
    *,
    cap: int | None,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if cap is None or cap <= 0 or x.shape[0] <= cap:
        return x, y
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(x.shape[0], size=cap, replace=False))
    return x[idx], y[idx]


class CauchyParquetTaskDataset(Dataset[TaskBatch]):
    """Lazily load one dataset-task at a time from manifest records."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        split: str,
        task: str,
        train_row_cap: int | None = None,
        test_row_cap: int | None = None,
        impute_missing: bool = True,
        seed: int = 0,
    ) -> None:
        self.manifest_path = manifest_path.expanduser().resolve()
        self.split = split
        self.task = task
        self.train_row_cap = train_row_cap
        self.test_row_cap = test_row_cap
        self.impute_missing = impute_missing
        self.seed = int(seed)

        table = pq.read_table(self.manifest_path)
        records: list[dict[str, Any]] = table.to_pylist()
        self.records = [
            record for record in records if record.get("split") == split and record.get("task") == task
        ]
        if not self.records:
            raise RuntimeError(
                f"no records found for split={split!r}, task={task!r} in {self.manifest_path}"
            )

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> TaskBatch:
        record = self.records[index]
        required_keys = {
            "dataset_index",
            "train_path",
            "test_path",
            "metadata_path",
            "metadata_offset_bytes",
            "metadata_size_bytes",
        }
        missing = sorted(required_keys - set(record))
        if missing:
            raise RuntimeError(
                "manifest record is missing required packed-contract fields: "
                f"missing={missing}, split={self.split}, task={self.task}"
            )
        dataset_index = int(record["dataset_index"])
        train_path = _resolve_record_path(self.manifest_path, str(record["train_path"]))
        test_path = _resolve_record_path(self.manifest_path, str(record["test_path"]))
        metadata_path = _resolve_record_path(self.manifest_path, str(record["metadata_path"]))
        metadata_offset_bytes = int(record["metadata_offset_bytes"])
        metadata_size_bytes = int(record["metadata_size_bytes"])

        x_train, y_train = _read_packed_split(train_path, dataset_index=dataset_index)
        x_test, y_test = _read_packed_split(test_path, dataset_index=dataset_index)
        metadata_record = _read_ndjson_record_by_offset(
            metadata_path,
            offset_bytes=metadata_offset_bytes,
            size_bytes=metadata_size_bytes,
        )
        metadata_dataset_index = int(metadata_record.get("dataset_index", -1))
        if metadata_dataset_index != dataset_index:
            raise RuntimeError(
                "metadata dataset_index mismatch for manifest record: "
                f"manifest={dataset_index}, metadata={metadata_dataset_index}, path={metadata_path}"
            )
        metadata = metadata_record.get("metadata")
        if not isinstance(metadata, dict):
            raise RuntimeError(
                f"metadata record missing object payload at key 'metadata': path={metadata_path}"
            )

        expected_n_train = int(record.get("n_train", -1))
        expected_n_test = int(record.get("n_test", -1))
        if expected_n_train >= 0 and int(x_train.shape[0]) != expected_n_train:
            raise RuntimeError(
                "train row count mismatch for packed split: "
                f"dataset_index={dataset_index}, expected={expected_n_train}, got={x_train.shape[0]}"
            )
        if expected_n_test >= 0 and int(x_test.shape[0]) != expected_n_test:
            raise RuntimeError(
                "test row count mismatch for packed split: "
                f"dataset_index={dataset_index}, expected={expected_n_test}, got={x_test.shape[0]}"
            )
        expected_n_features = int(record.get("n_features", -1))
        if expected_n_features >= 0:
            if int(x_train.shape[1]) != expected_n_features or int(x_test.shape[1]) != expected_n_features:
                raise RuntimeError(
                    "feature count mismatch for packed split: "
                    f"dataset_index={dataset_index}, expected={expected_n_features}, "
                    f"got_train={x_train.shape[1]}, got_test={x_test.shape[1]}"
                )

        x_train, y_train = _subsample_rows(
            x_train,
            y_train,
            cap=self.train_row_cap,
            seed=self.seed + index * 2 + 1,
        )
        x_test, y_test = _subsample_rows(
            x_test,
            y_test,
            cap=self.test_row_cap,
            seed=self.seed + index * 2 + 2,
        )

        if self.impute_missing:
            x_train, x_test = _impute_mean(x_train, x_test)

        num_classes: int | None = None
        if self.task == "classification":
            n_test_before = int(y_test.shape[0])
            y_train, y_test, num_classes, valid_mask = _remap_labels(y_train, y_test)
            if not bool(np.all(valid_mask)):
                x_test = x_test[valid_mask]
            n_test_after = int(y_test.shape[0])
            if n_test_after <= 0:
                dataset_id = str(record.get("dataset_id", "<unknown>"))
                raise RuntimeError(
                    "classification test split has zero rows after filtering unseen labels; "
                    f"dataset_id={dataset_id}, split={self.split}, n_test_before={n_test_before}, "
                    f"n_test_after={n_test_after}"
                )
            y_train_t = torch.from_numpy(y_train.astype(np.int64, copy=False))
            y_test_t = torch.from_numpy(y_test.astype(np.int64, copy=False))
        else:
            y_train_t = torch.from_numpy(y_train.astype(np.float32, copy=False))
            y_test_t = torch.from_numpy(y_test.astype(np.float32, copy=False))

        return TaskBatch(
            x_train=torch.from_numpy(x_train.astype(np.float32, copy=False)),
            y_train=y_train_t,
            x_test=torch.from_numpy(x_test.astype(np.float32, copy=False)),
            y_test=y_test_t,
            metadata=metadata,
            num_classes=num_classes,
        )
