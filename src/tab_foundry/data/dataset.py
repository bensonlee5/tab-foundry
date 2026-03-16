"""Dataset loader for parquet task bundles."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from tab_foundry.data.validation import assert_no_non_finite_values
from tab_foundry.preprocessing import preprocess_runtime_task_arrays
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
    expected_sha256: str,
) -> dict[str, Any]:
    if offset_bytes < 0 or size_bytes <= 0:
        raise RuntimeError(
            "metadata byte offset/size must be non-negative and non-zero: "
            f"path={metadata_path}, offset={offset_bytes}, size={size_bytes}"
        )
    if len(expected_sha256) != 64:
        raise RuntimeError(
            "metadata SHA-256 must be a 64-char hex digest: "
            f"path={metadata_path}, offset={offset_bytes}, digest={expected_sha256!r}"
        )

    with metadata_path.open("rb") as handle:
        handle.seek(offset_bytes)
        raw = handle.read(size_bytes)
    if len(raw) != size_bytes:
        raise RuntimeError(
            "failed to read full metadata slice from NDJSON: "
            f"path={metadata_path}, offset={offset_bytes}, size={size_bytes}, got={len(raw)}"
        )
    actual_sha256 = sha256(raw).hexdigest()
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "metadata NDJSON checksum mismatch: "
            f"path={metadata_path}, offset={offset_bytes}, size={size_bytes}, "
            f"expected={expected_sha256}, actual={actual_sha256}"
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


@dataclass(slots=True)
class _LoadedManifestTaskRecord:
    record: dict[str, Any]
    metadata: dict[str, Any]
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def _load_manifest_task_record(
    manifest_path: Path,
    *,
    split: str,
    task: str,
    record: dict[str, Any],
) -> _LoadedManifestTaskRecord:
    required_keys = {
        "dataset_index",
        "train_path",
        "test_path",
        "metadata_path",
        "metadata_offset_bytes",
        "metadata_size_bytes",
        "metadata_sha256",
    }
    missing = sorted(required_keys - set(record))
    if missing:
        raise RuntimeError(
            "manifest record is missing required packed-contract fields: "
            f"missing={missing}, split={split}, task={task}"
        )
    dataset_index = int(record["dataset_index"])
    train_path = _resolve_record_path(manifest_path, str(record["train_path"]))
    test_path = _resolve_record_path(manifest_path, str(record["test_path"]))
    metadata_path = _resolve_record_path(manifest_path, str(record["metadata_path"]))
    metadata_offset_bytes = int(record["metadata_offset_bytes"])
    metadata_size_bytes = int(record["metadata_size_bytes"])
    metadata_sha256 = str(record["metadata_sha256"])

    x_train, y_train = _read_packed_split(train_path, dataset_index=dataset_index)
    x_test, y_test = _read_packed_split(test_path, dataset_index=dataset_index)
    metadata_record = _read_ndjson_record_by_offset(
        metadata_path,
        offset_bytes=metadata_offset_bytes,
        size_bytes=metadata_size_bytes,
        expected_sha256=metadata_sha256,
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

    return _LoadedManifestTaskRecord(
        record=record,
        metadata=metadata,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )


class PackedParquetTaskDataset(Dataset[TaskBatch]):
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
        all_nan_fill: float = 0.0,
        label_mapping: str = "train_only_remap",
        unseen_test_label_policy: str = "filter",
        allow_missing_values: bool = False,
        seed: int = 0,
    ) -> None:
        self.manifest_path = manifest_path.expanduser().resolve()
        self.split = split
        self.task = task
        self.train_row_cap = train_row_cap
        self.test_row_cap = test_row_cap
        self.impute_missing = impute_missing
        self.all_nan_fill = float(all_nan_fill)
        self.label_mapping = str(label_mapping)
        self.unseen_test_label_policy = str(unseen_test_label_policy)
        self.allow_missing_values = bool(allow_missing_values)
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
        if (
            not self.allow_missing_values
            and str(record.get("missing_value_status", "")).strip() == "contains_nan_or_inf"
        ):
            raise RuntimeError(
                "manifest record contains NaN or Inf while allow_missing_values=False: "
                f"dataset_id={record.get('dataset_id')!r}, manifest_path={self.manifest_path}"
            )
        loaded = _load_manifest_task_record(
            self.manifest_path,
            split=self.split,
            task=self.task,
            record=record,
        )
        x_train = loaded.x_train
        y_train = loaded.y_train
        x_test = loaded.x_test
        y_test = loaded.y_test
        metadata = loaded.metadata
        if not self.allow_missing_values:
            assert_no_non_finite_values(
                {
                    "x_train": x_train,
                    "y_train": y_train,
                    "x_test": x_test,
                    "y_test": y_test,
                },
                context=(
                    "manifest-backed task dataset "
                    f"dataset_id={record.get('dataset_id')!r}"
                ),
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

        processed = preprocess_runtime_task_arrays(
            task=self.task,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            impute_missing=self.impute_missing,
            all_nan_fill=self.all_nan_fill,
            label_mapping=self.label_mapping,
            unseen_test_label_policy=self.unseen_test_label_policy,
        )
        x_train = processed.x_train
        y_train = processed.y_train
        x_test = processed.x_test
        y_test = processed.y_test if processed.y_test is not None else y_test
        num_classes = processed.num_classes

        metadata_out = dict(metadata)

        if self.task == "classification":
            if y_test is None:
                raise RuntimeError("classification preprocessing must produce y_test")
            n_test_after = int(y_test.shape[0])
            if n_test_after <= 0:
                dataset_id = str(record.get("dataset_id", "<unknown>"))
                raise RuntimeError(
                    "classification test split has zero rows after filtering unseen labels; "
                    f"dataset_id={dataset_id}, split={self.split}, n_test_after={n_test_after}"
                )
            y_train_t = torch.from_numpy(np.asarray(y_train, dtype=np.int64))
            y_test_t = torch.from_numpy(np.asarray(y_test, dtype=np.int64))
        else:
            y_train_t = torch.from_numpy(np.asarray(y_train, dtype=np.float32))
            if y_test is None:
                raise RuntimeError("regression preprocessing must produce y_test")
            y_test_t = torch.from_numpy(np.asarray(y_test, dtype=np.float32))

        return TaskBatch(
            x_train=torch.from_numpy(np.asarray(x_train, dtype=np.float32)),
            y_train=y_train_t,
            x_test=torch.from_numpy(np.asarray(x_test, dtype=np.float32)),
            y_test=y_test_t,
            metadata=metadata_out,
            num_classes=num_classes,
        )
