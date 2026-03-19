"""Synthetic Iris task generation for the smoke harness."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


def build_split_table(rows: list[tuple[int, np.ndarray, np.ndarray]]) -> pa.Table:
    dataset_indices: list[int] = []
    row_indices: list[int] = []
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    for dataset_index, x, y in rows:
        for row_index in range(int(x.shape[0])):
            dataset_indices.append(int(dataset_index))
            row_indices.append(int(row_index))
            x_rows.append(np.asarray(x[row_index], dtype=np.float32).tolist())
            y_rows.append(int(y[row_index]))

    return pa.table(
        {
            "dataset_index": pa.array(dataset_indices, type=pa.int64()),
            "row_index": pa.array(row_indices, type=pa.int64()),
            "x": pa.array(x_rows, type=pa.list_(pa.float32())),
            "y": pa.array(y_rows, type=pa.int64()),
        }
    )


def binary_iris_arrays() -> tuple[np.ndarray, np.ndarray]:
    iris = load_iris()
    x = np.asarray(iris.data[iris.target != 0], dtype=np.float32)
    y = np.asarray(iris.target[iris.target != 0] - 1, dtype=np.int64)
    return x, y


def write_iris_tasks(
    generated_dir: Path,
    *,
    num_tasks: int,
    seed: int,
    test_size: float,
) -> Path:
    if num_tasks <= 0:
        raise ValueError(f"num_tasks must be > 0, got {num_tasks}")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

    if generated_dir.exists():
        shutil.rmtree(generated_dir)
    shard_dir = generated_dir / "shard_00000"
    shard_dir.mkdir(parents=True, exist_ok=True)

    x, y = binary_iris_arrays()
    train_rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    test_rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    metadata_records: list[dict[str, Any]] = []
    for dataset_index in range(num_tasks):
        split_seed = int(seed + dataset_index)
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=split_seed,
            stratify=y,
        )
        train_rows.append((dataset_index, x_train, y_train))
        test_rows.append((dataset_index, x_test, y_test))
        metadata_records.append(
            {
                "dataset_index": int(dataset_index),
                "n_train": int(x_train.shape[0]),
                "n_test": int(x_test.shape[0]),
                "n_features": int(x_train.shape[1]),
                "feature_types": ["num"] * int(x_train.shape[1]),
                "metadata": {
                    "n_features": int(x_train.shape[1]),
                    "n_classes": 2,
                    "seed": split_seed,
                    "filter": {"mode": "deferred", "status": "accepted", "accepted": True},
                    "config": {"dataset": {"task": "classification"}},
                    "source": {"name": "iris_binary_smoke"},
                },
            }
        )

    pq.write_table(build_split_table(train_rows), shard_dir / "train.parquet")
    pq.write_table(build_split_table(test_rows), shard_dir / "test.parquet")
    with (shard_dir / "metadata.ndjson").open("wb") as handle:
        for record in metadata_records:
            serialized = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
            handle.write(serialized)
    return generated_dir
