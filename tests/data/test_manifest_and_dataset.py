from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from tab_foundry.data.dataset import CauchyParquetTaskDataset, _remap_labels
from tab_foundry.data.manifest import build_manifest


def _classification_metadata(*, n_features: int, n_classes: int = 3, seed: int = 7) -> dict[str, Any]:
    return {
        "n_features": n_features,
        "n_classes": n_classes,
        "seed": seed,
        "curriculum": {"stage": 1},
        "config": {"dataset": {"task": "classification"}},
    }


def _classification_arrays(
    *,
    n_train: int = 16,
    n_test: int = 8,
    n_features: int = 4,
    n_classes: int = 3,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_train = rng.standard_normal((n_train, n_features)).astype(np.float32)
    x_test = rng.standard_normal((n_test, n_features)).astype(np.float32)
    y_train = np.tile(np.arange(n_classes, dtype=np.int64), int(np.ceil(n_train / n_classes)))[:n_train]
    y_test = np.tile(np.arange(n_classes, dtype=np.int64), int(np.ceil(n_test / n_classes)))[:n_test]
    rng.shuffle(y_train)
    rng.shuffle(y_test)
    return x_train, y_train, x_test, y_test


def _build_split_table(
    rows: list[tuple[int, np.ndarray, np.ndarray]],
) -> pa.Table:
    dataset_indices: list[int] = []
    row_indices: list[int] = []
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    for dataset_index, x, y in rows:
        for row_index in range(int(x.shape[0])):
            dataset_indices.append(int(dataset_index))
            row_indices.append(int(row_index))
            x_rows.append(x[row_index].astype(np.float32, copy=False).tolist())
            y_rows.append(int(y[row_index]))

    return pa.table(
        {
            "dataset_index": pa.array(dataset_indices, type=pa.int64()),
            "row_index": pa.array(row_indices, type=pa.int64()),
            "x": pa.array(x_rows, type=pa.list_(pa.float32())),
            "y": pa.array(y_rows, type=pa.int64()),
        }
    )


def _write_packed_shard(
    shard_dir: Path,
    *,
    datasets: list[dict[str, Any]],
) -> dict[int, tuple[int, int]]:
    shard_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [
        (
            int(dataset["dataset_index"]),
            dataset["x_train"],
            dataset["y_train"],
        )
        for dataset in datasets
    ]
    test_rows = [
        (
            int(dataset["dataset_index"]),
            dataset["x_test"],
            dataset["y_test"],
        )
        for dataset in datasets
    ]
    pq.write_table(_build_split_table(train_rows), shard_dir / "train.parquet")
    pq.write_table(_build_split_table(test_rows), shard_dir / "test.parquet")

    offsets: dict[int, tuple[int, int]] = {}
    with (shard_dir / "metadata.ndjson").open("wb") as handle:
        for dataset in datasets:
            payload = {
                "dataset_index": int(dataset["dataset_index"]),
                "n_train": int(dataset["x_train"].shape[0]),
                "n_test": int(dataset["x_test"].shape[0]),
                "n_features": int(dataset["x_train"].shape[1]),
                "feature_types": list(dataset["feature_types"]),
                "metadata": dict(dataset["metadata"]),
            }
            raw = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
            offset = int(handle.tell())
            handle.write(raw)
            offsets[int(dataset["dataset_index"])] = (offset, len(raw))

    return offsets


def _write_dataset(
    shard_dir: Path,
    *,
    dataset_index: int = 0,
    n_train: int = 16,
    n_test: int = 8,
    n_features: int = 4,
    seed: int = 7,
) -> dict[int, tuple[int, int]]:
    x_train, y_train, x_test, y_test = _classification_arrays(
        n_train=n_train,
        n_test=n_test,
        n_features=n_features,
        seed=seed,
    )
    metadata = _classification_metadata(n_features=n_features, seed=seed)
    dataset = {
        "dataset_index": dataset_index,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "feature_types": ["num"] * n_features,
        "metadata": metadata,
    }
    return _write_packed_shard(shard_dir, datasets=[dataset])


def test_manifest_and_dataset_loading(tmp_path: Path) -> None:
    shard_dir = tmp_path / "run" / "shard_00000"
    _ = _write_dataset(shard_dir)

    manifest_path = tmp_path / "manifest.parquet"
    summary = build_manifest([tmp_path / "run"], manifest_path)
    assert summary.total_records == 1

    table = pq.read_table(manifest_path)
    split = table["split"][0].as_py()
    ds = CauchyParquetTaskDataset(manifest_path, split=split, task="classification")
    sample = ds[0]
    assert sample.x_train.ndim == 2
    assert sample.y_test.ndim == 1
    assert sample.metadata["config"]["dataset"]["task"] == "classification"


def test_remap_labels_uses_train_only() -> None:
    y_train = np.array([10, 20, 10], dtype=np.int64)
    y_test = np.array([20, 10], dtype=np.int64)
    remapped_train, remapped_test, num_classes, _ = _remap_labels(y_train, y_test)
    assert num_classes == 2
    assert np.array_equal(remapped_train, np.array([0, 1, 0], dtype=np.int64))
    assert np.array_equal(remapped_test, np.array([1, 0], dtype=np.int64))


def test_remap_labels_filters_unseen_test_classes() -> None:
    y_train = np.array([0, 0, 1], dtype=np.int64)
    y_test = np.array([0, 2], dtype=np.int64)
    x_test = np.array([[1], [2]])
    _, remapped_test, _, valid_mask = _remap_labels(y_train, y_test)
    assert np.array_equal(remapped_test, np.array([0], dtype=np.int64))
    assert np.array_equal(valid_mask, np.array([True, False]))
    assert np.array_equal(x_test[valid_mask], np.array([[1]]))


def test_dataset_raises_when_unseen_filter_removes_all_test_rows(tmp_path: Path) -> None:
    shard_dir = tmp_path / "run" / "shard_00000"
    dataset = {
        "dataset_index": 0,
        "x_train": np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=np.float32),
        "y_train": np.array([0, 0, 0], dtype=np.int64),
        "x_test": np.array([[4.0, 6.0], [5.0, 7.0]], dtype=np.float32),
        "y_test": np.array([1, 1], dtype=np.int64),
        "feature_types": ["num", "num"],
        "metadata": _classification_metadata(n_features=2, n_classes=2, seed=7),
    }
    _ = _write_packed_shard(shard_dir, datasets=[dataset])

    manifest_path = tmp_path / "manifest.parquet"
    _ = build_manifest([tmp_path / "run"], manifest_path)
    row = pq.read_table(manifest_path).to_pylist()[0]
    ds = CauchyParquetTaskDataset(
        manifest_path=manifest_path,
        split=str(row["split"]),
        task="classification",
    )
    with pytest.raises(RuntimeError, match="zero rows after filtering unseen labels"):
        _ = ds[0]


def test_manifest_dataset_id_and_split_are_stable_across_root_paths(tmp_path: Path) -> None:
    root_a = tmp_path / "root_a" / "run" / "shard_00000"
    root_b = tmp_path / "root_b" / "run" / "shard_00000"
    _ = _write_dataset(root_a)
    _ = _write_dataset(root_b)

    manifest_a = tmp_path / "manifest_a.parquet"
    manifest_b = tmp_path / "manifest_b.parquet"
    _ = build_manifest([tmp_path / "root_a" / "run"], manifest_a)
    _ = build_manifest([tmp_path / "root_b" / "run"], manifest_b)

    row_a = pq.read_table(manifest_a).to_pylist()[0]
    row_b = pq.read_table(manifest_b).to_pylist()[0]
    assert row_a["dataset_id"] != row_b["dataset_id"]
    assert row_a["source_root_id"] != row_b["source_root_id"]
    assert row_a["dataset_id"].startswith("root_")
    assert row_b["dataset_id"].startswith("root_")


def test_dataset_resolves_relative_paths_from_manifest_location(tmp_path: Path) -> None:
    shard_dir = tmp_path / "run" / "shard_00000"
    offsets = _write_dataset(shard_dir)
    metadata_offset, metadata_size = offsets[0]

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "relative_manifest.parquet"
    record = {
        "dataset_id": "root_deadbeef0000/shard_00000/dataset_000000",
        "source_root_id": "deadbeef0000",
        "split": "train",
        "task": "classification",
        "shard_id": 0,
        "dataset_index": 0,
        "train_path": os.path.relpath(shard_dir / "train.parquet", manifest_dir),
        "test_path": os.path.relpath(shard_dir / "test.parquet", manifest_dir),
        "metadata_path": os.path.relpath(shard_dir / "metadata.ndjson", manifest_dir),
        "metadata_offset_bytes": metadata_offset,
        "metadata_size_bytes": metadata_size,
        "n_train": 16,
        "n_test": 8,
        "n_features": 4,
        "n_classes": 3,
        "seed": 7,
        "curriculum_stage": 1,
    }
    pq.write_table(pa.Table.from_pylist([record]), manifest_path)

    run_dir = tmp_path / "elsewhere"
    run_dir.mkdir(parents=True, exist_ok=True)
    current = Path.cwd()
    try:
        os.chdir(run_dir)
        ds = CauchyParquetTaskDataset(manifest_path, split="train", task="classification")
        sample = ds[0]
    finally:
        os.chdir(current)

    assert sample.x_train.shape[1] == 4
    assert sample.y_train.dtype == torch.int64


def test_manifest_paths_are_relative_to_manifest_dir(tmp_path: Path) -> None:
    shard_dir = tmp_path / "run" / "shard_00000"
    _ = _write_dataset(shard_dir)

    manifest_path = tmp_path / "manifests" / "default.parquet"
    _ = build_manifest([tmp_path / "run"], manifest_path)
    row = pq.read_table(manifest_path).to_pylist()[0]

    assert not Path(str(row["train_path"])).is_absolute()
    assert not Path(str(row["test_path"])).is_absolute()
    assert not Path(str(row["metadata_path"])).is_absolute()
    assert int(row["metadata_offset_bytes"]) >= 0
    assert int(row["metadata_size_bytes"]) > 0


def test_manifest_multi_root_order_is_deterministic(tmp_path: Path) -> None:
    root_a = tmp_path / "root_a" / "run" / "shard_00000"
    root_b = tmp_path / "root_b" / "run" / "shard_00000"
    _ = _write_dataset(root_a)
    _ = _write_dataset(root_b)

    manifest_ab = tmp_path / "manifest_ab.parquet"
    manifest_ba = tmp_path / "manifest_ba.parquet"
    _ = build_manifest([tmp_path / "root_a" / "run", tmp_path / "root_b" / "run"], manifest_ab)
    _ = build_manifest([tmp_path / "root_b" / "run", tmp_path / "root_a" / "run"], manifest_ba)

    rows_ab = pq.read_table(manifest_ab).to_pylist()
    rows_ba = pq.read_table(manifest_ba).to_pylist()
    assert rows_ab == rows_ba
