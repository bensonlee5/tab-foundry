from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from tests.data.manifest_and_dataset_cases import (
    _classification_arrays,
    _classification_metadata,
    _write_packed_shard,
)


def write_task_batch_manifest_from_specs(
    tmp_path: Path,
    *,
    task_specs: list[Mapping[str, Any]],
) -> Path:
    shard_dir = tmp_path / "manifest_data" / "shard_00000"
    datasets: list[dict[str, object]] = []
    split_by_dataset_index: dict[int, str] = {}

    for raw_spec in task_specs:
        dataset_index = int(raw_spec["dataset_index"])
        n_train = int(raw_spec.get("n_train", 6))
        n_test = int(raw_spec.get("n_test", 3))
        n_features = int(raw_spec.get("n_features", 4))
        n_classes = int(raw_spec.get("n_classes", 3))
        seed = int(raw_spec.get("seed", dataset_index))
        split_by_dataset_index[dataset_index] = str(raw_spec.get("split", "train"))
        x_train, y_train, x_test, y_test = _classification_arrays(
            n_train=n_train,
            n_test=n_test,
            n_features=n_features,
            n_classes=n_classes,
            seed=seed,
        )
        datasets.append(
            {
                "dataset_index": dataset_index,
                "x_train": x_train,
                "y_train": y_train,
                "x_test": x_test,
                "y_test": y_test,
                "feature_types": ["num"] * n_features,
                "metadata": _classification_metadata(
                    n_features=n_features,
                    n_classes=n_classes,
                    seed=seed,
                ),
            }
        )

    offsets = _write_packed_shard(shard_dir, datasets=datasets)
    manifest_rows: list[dict[str, object]] = []
    for dataset in datasets:
        dataset_index = int(dataset["dataset_index"])
        offset, size, digest = offsets[dataset_index]
        manifest_rows.append(
            {
                "dataset_id": f"root_a/shard_00000/dataset_{dataset_index:06d}",
                "source_root_id": "root_a",
                "source_shard_relpath": "shard_00000",
                "split": split_by_dataset_index[dataset_index],
                "task": "classification",
                "dataset_index": dataset_index,
                "train_path": "manifest_data/shard_00000/train.parquet",
                "test_path": "manifest_data/shard_00000/test.parquet",
                "metadata_path": "manifest_data/shard_00000/metadata.ndjson",
                "metadata_offset_bytes": offset,
                "metadata_size_bytes": size,
                "metadata_sha256": digest,
                "n_train": int(dataset["x_train"].shape[0]),
                "n_test": int(dataset["x_test"].shape[0]),
                "n_features": int(dataset["x_train"].shape[1]),
                "n_classes": int(dict(dataset["metadata"]).get("n_classes", 3)),
                "seed": int(dict(dataset["metadata"]).get("seed", 0)),
                "filter_mode": "deferred",
                "filter_status": "not_run",
                "filter_accepted": None,
                "missing_value_policy": "allow_any",
                "missing_value_status": "clean",
            }
        )
    manifest_path = tmp_path / "manifest.parquet"
    pq.write_table(pa.Table.from_pylist(manifest_rows), manifest_path)
    return manifest_path
