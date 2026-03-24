from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from tab_foundry.data.dataset import PackedParquetTaskDataset
from tab_foundry.data.factory import _ManifestTaskBatchSampler, build_task_loader
from tab_foundry.training.batching import (
    collate_task_batch,
    move_batch,
    task_batch_diagnostics,
)
from tab_foundry.types import TaskBatch

from tests.data.manifest_and_dataset_cases import _classification_metadata, _write_packed_shard


def _sample_batch() -> TaskBatch:
    return TaskBatch(
        x_train=torch.randn(4, 3),
        y_train=torch.randint(0, 3, (4,)),
        x_test=torch.randn(2, 3),
        y_test=torch.randint(0, 3, (2,)),
        metadata={"dataset_id": "d0"},
        num_classes=3,
    )


class _StubManifestTaskDataset:
    def __len__(self) -> int:
        return 15

    def task_signature(self, index: int) -> tuple[int, int, int, int | None]:
        del index
        return (6, 3, 4, 3)


def _write_manifest_dataset(tmp_path: Path) -> Path:
    shard_dir = tmp_path / "manifest_data" / "shard_00000"
    dataset = {
        "dataset_index": 1,
        "x_train": np.asarray(
            [
                [0.0, 0.1, 0.2],
                [1.0, 1.1, 1.2],
                [2.0, 2.1, 2.2],
                [3.0, 3.1, 3.2],
            ],
            dtype=np.float32,
        ),
        "y_train": np.asarray([0, 1, 2, 2], dtype=np.int64),
        "x_test": np.asarray(
            [
                [4.0, 4.1, 4.2],
                [5.0, 5.1, 5.2],
                [6.0, 6.1, 6.2],
                [7.0, 7.1, 7.2],
            ],
            dtype=np.float32,
        ),
        "y_test": np.asarray([0, 1, 2, 2], dtype=np.int64),
        "feature_types": ["num"] * 3,
        "metadata": _classification_metadata(n_features=3, n_classes=3, seed=11),
    }
    offsets = _write_packed_shard(shard_dir, datasets=[dataset])
    offset, size, digest = offsets[1]
    manifest_path = tmp_path / "manifest.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "dataset_id": "root_a/shard_00000/dataset_000001",
                    "source_root_id": "root_a",
                    "source_shard_relpath": "shard_00000",
                    "split": "train",
                    "task": "classification",
                    "dataset_index": 1,
                    "train_path": "manifest_data/shard_00000/train.parquet",
                    "test_path": "manifest_data/shard_00000/test.parquet",
                    "metadata_path": "manifest_data/shard_00000/metadata.ndjson",
                    "metadata_offset_bytes": offset,
                    "metadata_size_bytes": size,
                    "metadata_sha256": digest,
                    "n_train": 4,
                    "n_test": 4,
                    "n_features": 3,
                    "n_classes": 3,
                    "seed": 11,
                    "filter_mode": "deferred",
                    "filter_status": "not_run",
                    "filter_accepted": None,
                    "missing_value_policy": "allow_any",
                    "missing_value_status": "clean",
                }
            ]
        ),
        manifest_path,
    )
    return manifest_path


def test_collate_task_batch_single_item() -> None:
    batch = _sample_batch()
    out = collate_task_batch([batch])
    assert out is batch


def test_collate_task_batch_rejects_non_singleton() -> None:
    batch = _sample_batch()
    with pytest.raises(RuntimeError, match="Only batch_size=1 is supported for task-level batching"):
        _ = collate_task_batch([batch, batch])


def test_collate_task_batch_stacks_exact_shape_tasks() -> None:
    batch_a = _sample_batch()
    batch_b = _sample_batch()

    out = collate_task_batch([batch_a, batch_b], requested_task_batch_size=4)

    assert tuple(out.x_train.shape) == (2, 4, 3)
    assert tuple(out.y_train.shape) == (2, 4)
    assert tuple(out.x_test.shape) == (2, 2, 3)
    assert tuple(out.y_test.shape) == (2, 2)
    assert out.metadata == {
        "task_members": [batch_a.metadata, batch_b.metadata],
        "task_batch_size_requested": 4,
        "task_batch_size_actual": 2,
        "task_batch_signature": "4x2x3x3",
        "task_batch_mode": "batched",
    }
    assert task_batch_diagnostics(out) == {
        "task_batch_size_requested": 4,
        "task_batch_size_actual": 2,
        "task_batch_mode": "batched",
        "task_batch_signature": "4x2x3x3",
        "task_members": [batch_a.metadata, batch_b.metadata],
    }


def test_collate_task_batch_emits_singleton_fallback_metadata() -> None:
    batch = _sample_batch()

    out = collate_task_batch([batch], requested_task_batch_size=8)

    assert out.x_train is batch.x_train
    assert out.metadata == {
        "task_members": [batch.metadata],
        "task_batch_size_requested": 8,
        "task_batch_size_actual": 1,
        "task_batch_signature": "4x2x3x3",
        "task_batch_mode": "singleton_fallback",
    }
    assert task_batch_diagnostics(out)["task_batch_mode"] == "singleton_fallback"


def test_collate_task_batch_rejects_shape_mismatch() -> None:
    batch_a = _sample_batch()
    batch_b = TaskBatch(
        x_train=torch.randn(5, 3),
        y_train=torch.randint(0, 3, (5,)),
        x_test=torch.randn(2, 3),
        y_test=torch.randint(0, 3, (2,)),
        metadata={"dataset_id": "d1"},
        num_classes=3,
    )

    with pytest.raises(RuntimeError, match="Only shape-compatible tasks can be tensor-batched"):
        _ = collate_task_batch([batch_a, batch_b], requested_task_batch_size=2)


def test_manifest_task_batch_sampler_batches_multi_item_remainder_together() -> None:
    sampler = _ManifestTaskBatchSampler(
        _StubManifestTaskDataset(),
        task_batch_size=8,
        shuffle=False,
        seed=0,
    )

    assert list(sampler) == [list(range(8)), list(range(8, 15))]
    assert len(sampler) == 2


def test_build_task_loader_does_not_materialize_manifest_tasks_for_signatures(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = PackedParquetTaskDataset(
        _write_manifest_dataset(tmp_path),
        split="train",
        task="classification",
    )

    def _explode(_index: int) -> TaskBatch:
        raise AssertionError("loader construction should not materialize task batches")

    monkeypatch.setattr(dataset, "_materialize_task_batch", _explode)

    loader = build_task_loader(
        dataset,
        num_workers=0,
        shuffle=False,
        seed=0,
        task_batch_size=2,
    )

    assert loader.batch_sampler is not None


def test_task_signature_matches_loaded_batch_after_caps_and_label_filtering(
    tmp_path: Path,
) -> None:
    dataset = PackedParquetTaskDataset(
        _write_manifest_dataset(tmp_path),
        split="train",
        task="classification",
        train_row_cap=2,
        test_row_cap=3,
        seed=5,
    )

    signature = dataset.task_signature(0)
    loaded_batch = dataset[0]

    assert signature == dataset._task_signature(loaded_batch)
    assert signature != (4, 4, 3, 3)


def test_move_batch_moves_tensors_and_preserves_metadata() -> None:
    batch = _sample_batch()
    out = move_batch(batch, torch.device("cpu"))
    assert out.x_train.device.type == "cpu"
    assert out.y_train.device.type == "cpu"
    assert out.x_test.device.type == "cpu"
    assert out.y_test.device.type == "cpu"
    assert out.metadata == batch.metadata
    assert out.num_classes == batch.num_classes
