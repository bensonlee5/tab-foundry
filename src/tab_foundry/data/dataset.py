"""Dataset loader for parquet task bundles."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset

from tab_realdata_hub import manifest as manifest_module
from tab_realdata_hub.validation import assert_no_non_finite_values
from tab_foundry.feature_types import normalize_feature_types
from tab_foundry.preprocessing import preprocess_runtime_task_arrays
from tab_foundry.types import TaskBatch

TaskSignature = tuple[int, int, int, int | None]

_LOAD_MANIFEST_RECORD_CATALOG = getattr(manifest_module, "load_manifest_record_catalog", None)
_LOAD_MANIFEST_RECORD_TEACHER_CONDITIONALS = getattr(
    manifest_module,
    "load_manifest_record_teacher_conditionals",
    None,
)


def _read_ndjson_record_by_offset(
    ndjson_path: Path,
    *,
    offset_bytes: int,
    size_bytes: int,
    expected_sha256: str,
) -> dict[str, Any]:
    with ndjson_path.open("rb") as handle:
        handle.seek(offset_bytes)
        raw_payload = handle.read(size_bytes)
    observed_sha256 = sha256(raw_payload).hexdigest()
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            "NDJSON record digest mismatch: "
            f"path={ndjson_path}, offset={offset_bytes}, expected={expected_sha256}, observed={observed_sha256}"
        )
    try:
        payload = json.loads(raw_payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            "failed to parse NDJSON record: "
            f"path={ndjson_path}, offset={offset_bytes}, size={size_bytes}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(
            f"NDJSON payload must be an object: path={ndjson_path}, offset={offset_bytes}"
        )
    return {str(key): value for key, value in cast(Mapping[str, Any], payload).items()}


def _manifest_row_catalog_locator(record: Mapping[str, Any]) -> tuple[str, int, int, str]:
    return (
        str(record["catalog_path"]),
        int(record["catalog_offset_bytes"]),
        int(record["catalog_size_bytes"]),
        str(record["catalog_sha256"]),
    )


def _fallback_load_manifest_record_catalog(
    manifest_path: Path,
    *,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    raw_path, offset_bytes, size_bytes, expected_sha256 = _manifest_row_catalog_locator(record)
    catalog_path = _resolve_record_path(manifest_path, raw_path)
    return _read_ndjson_record_by_offset(
        catalog_path,
        offset_bytes=offset_bytes,
        size_bytes=size_bytes,
        expected_sha256=expected_sha256,
    )


def _teacher_probs_to_matrix(values: list[list[float]]) -> np.ndarray:
    if not values:
        return np.empty((0, 0), dtype=np.float32)
    return np.asarray(values, dtype=np.float32)


def _fallback_load_manifest_record_teacher_conditionals(
    manifest_path: Path,
    *,
    record: Mapping[str, Any],
) -> np.ndarray | None:
    raw_path = record.get("teacher_conditionals_path")
    if raw_path is None:
        return None
    teacher_path = _resolve_record_path(manifest_path, str(raw_path))
    dataset_index = int(record["dataset_index"])
    try:
        table = pq.read_table(
            teacher_path,
            filters=[("dataset_index", "=", dataset_index)],
            columns=["row_index", "class_probs"],
        )
    except Exception as exc:  # pragma: no cover - pyarrow error typing is backend-specific
        raise RuntimeError(
            "failed to read teacher_conditionals parquet: "
            f"path={teacher_path}, dataset_index={dataset_index}"
        ) from exc
    if table.num_rows <= 0:
        return np.empty((0, 0), dtype=np.float32)
    row_index = table["row_index"].to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
    order = np.argsort(row_index, kind="stable")
    probs = _teacher_probs_to_matrix(table["class_probs"].to_pylist())
    if not np.array_equal(order, np.arange(order.shape[0])):
        probs = probs[order]
    return probs


def _load_manifest_record_catalog(
    manifest_path: Path,
    *,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    if callable(_LOAD_MANIFEST_RECORD_CATALOG):
        payload = _LOAD_MANIFEST_RECORD_CATALOG(manifest_path, record=record)
        return {str(key): value for key, value in cast(Mapping[str, Any], payload).items()}
    return _fallback_load_manifest_record_catalog(manifest_path, record=record)


def _load_manifest_record_teacher_conditionals(
    manifest_path: Path,
    *,
    record: Mapping[str, Any],
) -> np.ndarray | None:
    if callable(_LOAD_MANIFEST_RECORD_TEACHER_CONDITIONALS):
        return _LOAD_MANIFEST_RECORD_TEACHER_CONDITIONALS(manifest_path, record=record)
    return _fallback_load_manifest_record_teacher_conditionals(manifest_path, record=record)


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


def _record_identity_text(record: dict[str, Any]) -> str:
    dataset_id = record.get("dataset_id", "<unknown>")
    identity_key = record.get("dataset_identity_key")
    parts = [f"dataset_id={dataset_id!r}"]
    if isinstance(identity_key, str) and identity_key and identity_key != dataset_id:
        parts.append(f"dataset_identity_key={identity_key!r}")
    return ", ".join(parts)


def _copy_jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _copy_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_copy_jsonable(item) for item in value]
    if isinstance(value, tuple):
        return [_copy_jsonable(item) for item in value]
    return value


def _compatibility_metadata_from_catalog(
    *,
    catalog_record: Mapping[str, Any],
    record: Mapping[str, Any],
    teacher_probabilities: np.ndarray | None,
) -> dict[str, Any]:
    raw_metadata = catalog_record.get("metadata")
    if isinstance(raw_metadata, Mapping):
        materialized_metadata = cast(dict[str, Any], _copy_jsonable(raw_metadata))
        if teacher_probabilities is not None:
            materialized_teacher_conditionals = materialized_metadata.get("teacher_conditionals")
            if isinstance(materialized_teacher_conditionals, Mapping):
                teacher_payload = cast(
                    dict[str, Any], _copy_jsonable(materialized_teacher_conditionals)
                )
                teacher_payload["test_probs"] = teacher_probabilities.tolist()
                materialized_metadata["teacher_conditionals"] = teacher_payload
        return materialized_metadata

    metadata: dict[str, Any] = {}
    dataset_id = catalog_record.get("dataset_id")
    if isinstance(dataset_id, str) and dataset_id.strip():
        metadata["dataset_id"] = dataset_id

    group_ids = catalog_record.get("group_ids")
    if isinstance(group_ids, Mapping):
        metadata["split_groups"] = {
            str(key): _copy_jsonable(value)
            for key, value in group_ids.items()
        }

    feature_types = catalog_record.get("feature_types")
    if feature_types is not None:
        metadata["feature_types"] = _copy_jsonable(feature_types)

    n_classes = catalog_record.get("n_classes")
    if n_classes is not None:
        metadata["n_classes"] = int(n_classes)

    teacher_summary = (
        cast(Mapping[str, Any], catalog_record["teacher_conditionals"])
        if isinstance(catalog_record.get("teacher_conditionals"), Mapping)
        else None
    )
    teacher_enabled = bool(
        teacher_summary is not None or record.get("teacher_conditionals_path") is not None
    )
    teacher_available = bool(
        teacher_summary is not None and teacher_summary.get("available") is True
    )
    metadata["posterior_predictive"] = {
        "teacher_conditional_export_enabled": teacher_enabled,
        "teacher_conditionals_available": teacher_available,
    }
    if teacher_summary is not None:
        teacher_conditionals_payload: dict[str, Any] = {
            "class_labels": _copy_jsonable(teacher_summary.get("class_labels")),
            "optimal_log_loss_per_test_cell": teacher_summary.get(
                "optimal_log_loss_per_test_cell"
            ),
        }
        if teacher_probabilities is not None:
            teacher_conditionals_payload["test_probs"] = teacher_probabilities.tolist()
        metadata["teacher_conditionals"] = teacher_conditionals_payload
    return metadata


def load_manifest_record_metadata(
    manifest_path: Path,
    *,
    record: Mapping[str, Any],
    expected_feature_count: int | None = None,
    require_feature_types: bool = True,
) -> tuple[dict[str, Any], list[str] | None]:
    """Load one packed-manifest metadata payload plus validated feature types."""

    required_keys = {"dataset_index"}
    locator_keys = {
        "catalog_path",
        "catalog_offset_bytes",
        "catalog_size_bytes",
        "catalog_sha256",
    }
    missing = sorted(required_keys - set(record))
    if missing or not locator_keys.issubset(record):
        raise RuntimeError(
            "manifest record is missing required packed-contract fields: "
            f"missing={missing}"
        )

    dataset_index = int(record["dataset_index"])
    catalog_record = _load_manifest_record_catalog(
        manifest_path,
        record=record,
    )
    metadata_dataset_index = int(catalog_record.get("dataset_index", -1))
    if metadata_dataset_index != dataset_index:
        raise RuntimeError(
            "metadata dataset_index mismatch for manifest record: "
            f"manifest={dataset_index}, metadata={metadata_dataset_index}, path={manifest_path}"
        )
    teacher_probabilities = _load_manifest_record_teacher_conditionals(
        manifest_path,
        record=record,
    )
    metadata = _compatibility_metadata_from_catalog(
        catalog_record=catalog_record,
        record=record,
        teacher_probabilities=teacher_probabilities,
    )
    raw_feature_types = catalog_record.get("feature_types", metadata.get("feature_types"))
    raw_metadata = catalog_record.get("metadata")
    if raw_feature_types is None and isinstance(raw_metadata, Mapping):
        raw_feature_types = raw_metadata.get("feature_types")
    if raw_feature_types is None:
        if require_feature_types:
            raise RuntimeError(
                "manifest-backed task dataset requires explicit feature_types metadata: "
                f"dataset_index={dataset_index}, path={manifest_path}"
            )
        return metadata, None
    try:
        feature_types = normalize_feature_types(
            raw_feature_types,
            expected_count=expected_feature_count,
            context="metadata_record.feature_types",
        )
    except ValueError as exc:
        raise RuntimeError(
            "invalid manifest-backed feature_types metadata: "
            f"dataset_index={dataset_index}, path={manifest_path}: {exc}"
        ) from exc
    return metadata, feature_types


def load_manifest_record_catalog(
    manifest_path: Path,
    *,
    record: Mapping[str, Any],
) -> dict[str, Any]:
    """Load the public packed-catalog record for one manifest row."""

    required_keys = {"dataset_index"}
    locator_keys = {
        "catalog_path",
        "catalog_offset_bytes",
        "catalog_size_bytes",
        "catalog_sha256",
    }
    missing = sorted(required_keys - set(record))
    if missing or not locator_keys.issubset(record):
        raise RuntimeError(
            "manifest record is missing required packed-contract fields: "
            f"missing={missing}"
        )

    dataset_index = int(record["dataset_index"])
    catalog_record = _load_manifest_record_catalog(
        manifest_path,
        record=record,
    )
    metadata_dataset_index = int(catalog_record.get("dataset_index", -1))
    if metadata_dataset_index != dataset_index:
        raise RuntimeError(
            "metadata dataset_index mismatch for manifest record: "
            f"manifest={dataset_index}, metadata={metadata_dataset_index}, path={manifest_path}"
        )
    return catalog_record


def _read_packed_split_targets(
    split_path: Path,
    *,
    dataset_index: int,
) -> np.ndarray:
    try:
        table = pq.read_table(
            split_path,
            filters=[("dataset_index", "=", int(dataset_index))],
            columns=["row_index", "y"],
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
    y = table["y"].to_numpy(zero_copy_only=False)
    if row_index.shape[0] != y.shape[0]:
        raise RuntimeError(
            "packed split row count mismatch: "
            f"path={split_path}, dataset_index={dataset_index}, "
            f"row_index={row_index.shape[0]}, y={y.shape[0]}"
        )

    order = np.argsort(row_index, kind="stable")
    if not np.array_equal(order, np.arange(order.shape[0])):
        row_index = row_index[order]
        y = y[order]

    unique = np.unique(row_index)
    if unique.shape[0] != row_index.shape[0]:
        raise RuntimeError(
            f"packed split row_index values must be unique: path={split_path}, dataset_index={dataset_index}"
        )
    return np.asarray(y)


@dataclass(slots=True)
class _LoadedManifestTaskRecord:
    record: dict[str, Any]
    metadata: dict[str, Any]
    feature_types: list[str]
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
    required_keys = {"dataset_index", "train_path", "test_path"}
    missing = sorted(required_keys - set(record))
    if missing:
        raise RuntimeError(
            "manifest record is missing required packed-contract fields: "
            f"missing={missing}, split={split}, task={task}"
        )
    dataset_index = int(record["dataset_index"])
    train_path = _resolve_record_path(manifest_path, str(record["train_path"]))
    test_path = _resolve_record_path(manifest_path, str(record["test_path"]))
    x_train, y_train = _read_packed_split(train_path, dataset_index=dataset_index)
    x_test, y_test = _read_packed_split(test_path, dataset_index=dataset_index)
    metadata, feature_types = load_manifest_record_metadata(
        manifest_path,
        record=record,
        expected_feature_count=int(x_train.shape[1]),
    )
    if feature_types is None:
        raise RuntimeError(
            "manifest-backed task dataset requires explicit feature_types metadata: "
            f"{_record_identity_text(record)}"
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
        feature_types=feature_types,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
    )


def _normalize_teacher_conditionals_for_runtime(
    metadata: dict[str, Any],
    *,
    raw_train_labels: np.ndarray,
    valid_test_mask: np.ndarray | None,
) -> None:
    teacher_conditionals = metadata.get("teacher_conditionals")
    if not isinstance(teacher_conditionals, Mapping):
        return
    class_labels_raw = teacher_conditionals.get("class_labels")
    test_probs_raw = teacher_conditionals.get("test_probs")
    if not isinstance(class_labels_raw, list) or test_probs_raw is None:
        return

    try:
        class_labels = [int(value) for value in class_labels_raw]
        probabilities = np.asarray(test_probs_raw, dtype=np.float64)
    except Exception:
        return
    if probabilities.ndim != 2:
        return

    if valid_test_mask is not None:
        mask = np.asarray(valid_test_mask, dtype=bool)
        if probabilities.shape[0] == mask.shape[0]:
            probabilities = probabilities[mask]

    observed_train_labels = np.unique(np.asarray(raw_train_labels, dtype=np.int64))
    if observed_train_labels.size <= 0:
        return
    observed_label_list = [int(label) for label in observed_train_labels.tolist()]
    if not all(label in class_labels for label in observed_label_list):
        return

    column_order = [class_labels.index(label) for label in observed_label_list]
    probabilities = probabilities[:, column_order]
    row_mass = probabilities.sum(axis=1, keepdims=True)
    if np.any(row_mass <= 0.0):
        return
    probabilities = probabilities / row_mass

    teacher_payload = {
        str(key): _copy_jsonable(value)
        for key, value in teacher_conditionals.items()
        if key not in {"class_labels", "test_probs"}
    }
    teacher_payload["class_labels"] = list(range(len(observed_label_list)))
    teacher_payload["test_probs"] = probabilities.tolist()
    metadata["teacher_conditionals"] = teacher_payload


class PackedParquetTaskDataset(Dataset[TaskBatch]):
    """Lazily load one dataset-task at a time from manifest records."""

    def __init__(
        self,
        manifest_path: Path,
        *,
        split: str,
        task: str,
        impute_missing: bool = True,
        all_nan_fill: float = 0.0,
        label_mapping: str = "train_only_remap",
        unseen_test_label_policy: str = "filter",
        allow_missing_values: bool = False,
    ) -> None:
        self.manifest_path = manifest_path.expanduser().resolve()
        self.split = split
        self.task = task
        self.impute_missing = impute_missing
        self.all_nan_fill = float(all_nan_fill)
        self.label_mapping = str(label_mapping)
        self.unseen_test_label_policy = str(unseen_test_label_policy)
        self.allow_missing_values = bool(allow_missing_values)

        table = pq.read_table(self.manifest_path)
        records: list[dict[str, Any]] = table.to_pylist()
        self.records = [
            record for record in records if record.get("split") == split and record.get("task") == task
        ]
        if not self.records:
            raise RuntimeError(
                f"no records found for split={split!r}, task={task!r} in {self.manifest_path}"
            )
        self._task_signature_cache: dict[int, TaskSignature] = {}

    def __len__(self) -> int:
        return len(self.records)

    @staticmethod
    def _task_signature(batch: TaskBatch) -> TaskSignature:
        return (
            int(batch.x_train.shape[0]),
            int(batch.x_test.shape[0]),
            int(batch.x_train.shape[1]),
            None if batch.num_classes is None else int(batch.num_classes),
        )

    def _materialize_task_batch(self, index: int) -> TaskBatch:
        record = self.records[index]
        if (
            not self.allow_missing_values
            and str(record.get("missing_value_status", "")).strip() == "contains_nan_or_inf"
        ):
            raise RuntimeError(
                "manifest record contains NaN or Inf while allow_missing_values=False: "
                f"{_record_identity_text(record)}, manifest_path={self.manifest_path}"
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
        feature_types = loaded.feature_types
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
                    f"{_record_identity_text(record)}"
                ),
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
        metadata_out["feature_types"] = list(feature_types)
        if self.task == "classification":
            metadata_out["n_classes"] = None if num_classes is None else int(num_classes)
            _normalize_teacher_conditionals_for_runtime(
                metadata_out,
                raw_train_labels=np.asarray(loaded.y_train, dtype=np.int64),
                valid_test_mask=processed.valid_test_mask,
            )

        if self.task == "classification":
            if y_test is None:
                raise RuntimeError("classification preprocessing must produce y_test")
            n_test_after = int(y_test.shape[0])
            if n_test_after <= 0:
                raise RuntimeError(
                    "classification test split has zero rows after filtering unseen labels; "
                    f"{_record_identity_text(record)}, split={self.split}, "
                    f"n_test_after={n_test_after}"
                )
            y_train_t = torch.from_numpy(np.asarray(y_train, dtype=np.int64))
            y_test_t = torch.from_numpy(np.asarray(y_test, dtype=np.int64))
        else:
            y_train_t = torch.from_numpy(np.asarray(y_train, dtype=np.float32))
            if y_test is None:
                raise RuntimeError("regression preprocessing must produce y_test")
            y_test_t = torch.from_numpy(np.asarray(y_test, dtype=np.float32))

        batch = TaskBatch(
            x_train=torch.from_numpy(np.asarray(x_train, dtype=np.float32)),
            y_train=y_train_t,
            x_test=torch.from_numpy(np.asarray(x_test, dtype=np.float32)),
            y_test=y_test_t,
            metadata=metadata_out,
            num_classes=num_classes,
        )
        self._task_signature_cache[int(index)] = self._task_signature(batch)
        return batch

    def __getitem__(self, index: int) -> TaskBatch:
        return self._materialize_task_batch(int(index))

    def _record_n_features(self, record: dict[str, Any]) -> int | None:
        raw_n_features = record.get("n_features")
        if raw_n_features is None:
            return None
        try:
            n_features = int(raw_n_features)
        except (TypeError, ValueError):
            return None
        if n_features <= 0:
            return None
        return n_features

    def _fast_task_signature(self, index: int) -> TaskSignature | None:
        record = self.records[index]
        if (
            not self.allow_missing_values
            and str(record.get("missing_value_status", "")).strip() == "contains_nan_or_inf"
        ):
            raise RuntimeError(
                "manifest record contains NaN or Inf while allow_missing_values=False: "
                f"{_record_identity_text(record)}, manifest_path={self.manifest_path}"
            )
        n_features = self._record_n_features(record)
        if n_features is None:
            return None
        required_keys = {"dataset_index", "train_path", "test_path"}
        missing = sorted(required_keys - set(record))
        if missing:
            raise RuntimeError(
                "manifest record is missing required packed-contract fields: "
                f"missing={missing}, split={self.split}, task={self.task}"
            )
        dataset_index = int(record["dataset_index"])
        train_path = _resolve_record_path(self.manifest_path, str(record["train_path"]))
        test_path = _resolve_record_path(self.manifest_path, str(record["test_path"]))
        train_labels_raw = _read_packed_split_targets(train_path, dataset_index=dataset_index)
        test_labels_raw = _read_packed_split_targets(test_path, dataset_index=dataset_index)

        expected_n_train = int(record.get("n_train", -1))
        expected_n_test = int(record.get("n_test", -1))
        if expected_n_train >= 0 and int(train_labels_raw.shape[0]) != expected_n_train:
            raise RuntimeError(
                "train row count mismatch for packed split: "
                f"dataset_index={dataset_index}, expected={expected_n_train}, got={train_labels_raw.shape[0]}"
            )
        if expected_n_test >= 0 and int(test_labels_raw.shape[0]) != expected_n_test:
            raise RuntimeError(
                "test row count mismatch for packed split: "
                f"dataset_index={dataset_index}, expected={expected_n_test}, got={test_labels_raw.shape[0]}"
            )

        train_labels = np.asarray(train_labels_raw)
        test_labels = np.asarray(test_labels_raw)
        if self.task != "classification":
            return (
                int(train_labels.shape[0]),
                int(test_labels.shape[0]),
                int(n_features),
                None,
            )
        if self.label_mapping != "train_only_remap" or self.unseen_test_label_policy != "filter":
            return None

        train_targets = np.asarray(train_labels, dtype=np.int64)
        label_values = np.unique(train_targets)
        if label_values.size <= 0:
            raise RuntimeError("classification train split has no labels")
        test_targets = np.asarray(test_labels, dtype=np.int64)
        test_pos = np.searchsorted(label_values, test_targets)
        test_in_bounds = test_pos < label_values.shape[0]
        test_clamped = np.clip(test_pos, 0, label_values.shape[0] - 1)
        valid_test = test_in_bounds & (label_values[test_clamped] == test_targets)
        n_test_after = int(valid_test.sum())
        if n_test_after <= 0:
            raise RuntimeError(
                "classification test split has zero rows after filtering unseen labels; "
                f"{_record_identity_text(record)}, split={self.split}, "
                f"n_test_after={n_test_after}"
            )
        return (
            int(train_targets.shape[0]),
            n_test_after,
            int(n_features),
            int(label_values.shape[0]),
        )

    def task_signature(self, index: int) -> TaskSignature:
        resolved_index = int(index)
        cached = self._task_signature_cache.get(resolved_index)
        if cached is not None:
            return cached
        signature = self._fast_task_signature(resolved_index)
        if signature is None:
            batch = self._materialize_task_batch(resolved_index)
            return self._task_signature(batch)
        self._task_signature_cache[resolved_index] = signature
        return signature
