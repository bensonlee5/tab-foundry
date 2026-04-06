from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

from accelerate.data_loader import BatchSamplerShard
import numpy as np
from omegaconf import OmegaConf
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch import nn

import tab_foundry.data.dataset as dataset_module
import tab_foundry.data.sources.manifest as manifest_source_module
import tab_foundry.training.trainer_metrics as trainer_metrics_module
from tab_foundry.data.dataset import PackedParquetTaskDataset
from tab_foundry.data.factory import _ManifestTaskBatchSampler, build_task_loader
from tab_foundry.model.outputs import ClassificationOutput
from tab_foundry.task_batching import (
    collate_task_batch,
    move_batch,
    task_batch_diagnostics,
)
from tab_foundry.types import TaskBatch

from tests.support.manifest_and_dataset_cases import _classification_metadata, _write_packed_shard


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
    def __init__(
        self,
        *,
        size: int = 15,
        signatures: list[tuple[int, int, int, int | None]] | None = None,
    ) -> None:
        self.signatures = None if signatures is None else list(signatures)
        self.size = len(self.signatures) if self.signatures is not None else int(size)

    def __len__(self) -> int:
        return self.size

    def task_signature(self, index: int) -> tuple[int, int, int, int | None]:
        if self.signatures is not None:
            return self.signatures[int(index)]
        del index
        return (6, 3, 4, 3)


def _manifest_task_payload(
    *,
    dataset_index: int,
    order_tag: str,
    n_train: int = 4,
    n_test: int = 4,
    n_features: int = 3,
    n_classes: int = 3,
    seed: int = 11,
) -> dict[str, object]:
    x_train = np.arange(n_train * n_features, dtype=np.float32).reshape(n_train, n_features)
    x_test = np.arange(n_test * n_features, dtype=np.float32).reshape(n_test, n_features)
    metadata = _classification_metadata(n_features=n_features, n_classes=n_classes, seed=seed)
    metadata["order_tag"] = order_tag
    return {
        "dataset_index": int(dataset_index),
        "x_train": x_train + float(dataset_index),
        "y_train": np.arange(n_train, dtype=np.int64) % n_classes,
        "x_test": x_test + float(dataset_index) + 0.5,
        "y_test": np.arange(n_test, dtype=np.int64) % n_classes,
        "feature_types": ["floating"] * n_features,
        "metadata": metadata,
    }


def _write_manifest_dataset(
    tmp_path: Path,
    *,
    datasets: list[dict[str, object]] | None = None,
) -> Path:
    shard_dir = tmp_path / "manifest_data" / "shard_00000"
    resolved_datasets = list(datasets) if datasets is not None else [
        {
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
            "feature_types": ["floating"] * 3,
            "metadata": _classification_metadata(n_features=3, n_classes=3, seed=11),
        }
    ]
    offsets = _write_packed_shard(shard_dir, datasets=resolved_datasets)
    manifest_path = tmp_path / "manifest.parquet"
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "dataset_id": f"root_a/shard_00000/dataset_{int(dataset['dataset_index']):06d}",
                    "source_root_id": "root_a",
                    "source_shard_relpath": "shard_00000",
                    "split": "train",
                    "task": "classification",
                    "dataset_index": int(dataset["dataset_index"]),
                    "train_path": "manifest_data/shard_00000/train.parquet",
                    "test_path": "manifest_data/shard_00000/test.parquet",
                    "catalog_path": "manifest_data/shard_00000/metadata.ndjson",
                    "catalog_offset_bytes": offset,
                    "catalog_size_bytes": size,
                    "catalog_sha256": digest,
                    "n_train": int(np.asarray(dataset["x_train"]).shape[0]),
                    "n_test": int(np.asarray(dataset["x_test"]).shape[0]),
                    "n_features": int(np.asarray(dataset["x_train"]).shape[1]),
                    "n_classes": int(dict(dataset["metadata"]).get("n_classes", 3)),
                    "seed": int(dict(dataset["metadata"]).get("seed", 0)),
                    "filter_mode": "deferred",
                    "filter_status": "not_run",
                    "filter_accepted": None,
                    "missing_value_policy": "allow_any",
                    "missing_value_status": "clean",
                }
                for dataset in resolved_datasets
                for (offset, size, digest) in [offsets[int(dataset["dataset_index"])]]
            ]
        ),
        manifest_path,
    )
    return manifest_path


class _FakeEvalAccelerator:
    def __init__(self) -> None:
        self.device = torch.device("cpu")

    def autocast(self):
        return nullcontext()

    def reduce(self, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        if reduction != "sum":
            raise ValueError("only sum reduction is supported in fake accelerator")
        return tensor


class _EvalOrderClassifier(nn.Module):
    def forward(self, batch: TaskBatch) -> ClassificationOutput:
        n_rows = int(batch.y_test.reshape(-1).shape[0])
        return ClassificationOutput(
            logits=torch.zeros((n_rows, 3), dtype=torch.float32),
            num_classes=3,
        )


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


def test_collate_task_batch_resolves_feature_type_ids_when_metadata_is_present() -> None:
    batch_a = TaskBatch(
        x_train=torch.randn(4, 3),
        y_train=torch.randint(0, 3, (4,)),
        x_test=torch.randn(2, 3),
        y_test=torch.randint(0, 3, (2,)),
        metadata={"feature_types": ["floating", "integer", "bool"]},
        num_classes=3,
    )
    batch_b = TaskBatch(
        x_train=torch.randn(4, 3),
        y_train=torch.randint(0, 3, (4,)),
        x_test=torch.randn(2, 3),
        y_test=torch.randint(0, 3, (2,)),
        metadata={"feature_types": ["integer", "floating", "unknown"]},
        num_classes=3,
    )

    out = collate_task_batch([batch_a, batch_b], requested_task_batch_size=2)

    assert out.feature_type_ids is not None
    assert out.feature_type_ids.dtype == torch.int64
    assert tuple(out.feature_type_ids.shape) == (2, 3)
    assert out.feature_type_ids.tolist() == [[2, 1, 0], [1, 2, 4]]


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

    assert sampler.batch_size == 8
    assert sampler.drop_last is False
    assert list(sampler) == [list(range(8)), list(range(8, 15))]
    assert len(sampler) == 2


def test_manifest_task_batch_sampler_preserves_manifest_order_without_shuffle() -> None:
    signatures = [
        (2, 2, 3, 2),
        (3, 2, 3, 2),
        (2, 2, 3, 2),
        (3, 2, 3, 2),
    ]
    sampler = _ManifestTaskBatchSampler(
        _StubManifestTaskDataset(signatures=signatures),
        task_batch_size=2,
        shuffle=False,
        seed=0,
    )

    assert list(sampler) == [[0], [1], [2], [3]]
    assert len(sampler) == 4


def test_manifest_task_batch_sampler_batches_contiguous_runs_without_shuffle() -> None:
    signatures = [
        (2, 2, 3, 2),
        (2, 2, 3, 2),
        (3, 2, 3, 2),
        (3, 2, 3, 2),
        (3, 2, 3, 2),
    ]
    sampler = _ManifestTaskBatchSampler(
        _StubManifestTaskDataset(signatures=signatures),
        task_batch_size=2,
        shuffle=False,
        seed=0,
    )

    assert list(sampler) == [[0, 1], [2, 3], [4]]
    assert len(sampler) == 3


def test_manifest_task_batch_sampler_keeps_regrouped_batches_when_shuffled() -> None:
    signatures = [
        (2, 2, 3, 2),
        (3, 2, 3, 2),
        (2, 2, 3, 2),
        (3, 2, 3, 2),
    ]
    sampler = _ManifestTaskBatchSampler(
        _StubManifestTaskDataset(signatures=signatures),
        task_batch_size=2,
        shuffle=True,
        seed=0,
    )

    normalized_batches = sorted(sorted(batch) for batch in list(sampler))
    assert normalized_batches == [[0, 2], [1, 3]]
    assert len(sampler) == 2


def test_manifest_task_batch_sampler_supports_accelerate_sharding_without_padding() -> None:
    sampler = _ManifestTaskBatchSampler(
        _StubManifestTaskDataset(size=5),
        task_batch_size=4,
        shuffle=False,
        seed=0,
    )

    shard_zero = BatchSamplerShard(
        sampler,
        num_processes=2,
        process_index=0,
        split_batches=False,
        even_batches=False,
    )
    shard_one = BatchSamplerShard(
        sampler,
        num_processes=2,
        process_index=1,
        split_batches=False,
        even_batches=False,
    )

    assert list(shard_zero) == [list(range(4))]
    assert list(shard_one) == [[4]]


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


def test_build_task_loader_construction_defers_task_signature_resolution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = PackedParquetTaskDataset(
        _write_manifest_dataset(tmp_path),
        split="train",
        task="classification",
    )

    def _explode_signature(_index: int) -> tuple[int, int, int, int | None]:
        raise AssertionError("loader construction should not resolve task signatures")

    monkeypatch.setattr(dataset, "task_signature", _explode_signature)

    loader = build_task_loader(
        dataset,
        num_workers=0,
        shuffle=False,
        seed=0,
        task_batch_size=2,
    )

    assert loader.batch_sampler is not None
    with pytest.raises(AssertionError, match="loader construction should not resolve task signatures"):
        _ = next(iter(loader))


def test_build_task_loader_omits_worker_only_overlap_kwargs_when_num_workers_is_zero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _capture_loader(_dataset: object, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr("tab_foundry.data.factory.DataLoader", _capture_loader)

    loader = build_task_loader(
        [_sample_batch()],
        num_workers=0,
        shuffle=False,
        seed=0,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=4,
    )

    assert loader.pin_memory is True
    assert loader.num_workers == 0
    assert "persistent_workers" not in captured
    assert "prefetch_factor" not in captured


def test_build_task_loader_forwards_overlap_kwargs_for_manifest_batching(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _capture_loader(_dataset: object, **kwargs: object) -> SimpleNamespace:
        captured.update(kwargs)
        return SimpleNamespace(**kwargs)

    monkeypatch.setattr("tab_foundry.data.factory.DataLoader", _capture_loader)
    dataset = PackedParquetTaskDataset(
        _write_manifest_dataset(tmp_path),
        split="train",
        task="classification",
    )

    loader = build_task_loader(
        dataset,
        num_workers=2,
        shuffle=False,
        seed=0,
        task_batch_size=2,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    assert loader.pin_memory is True
    assert loader.num_workers == 2
    assert loader.persistent_workers is True
    assert loader.prefetch_factor == 2
    assert loader.batch_sampler is not None


def test_packed_parquet_task_dataset_can_seed_shuffle_manifest_records(tmp_path: Path) -> None:
    manifest_path = _write_manifest_dataset(
        tmp_path,
        datasets=[
            _manifest_task_payload(dataset_index=1, order_tag="task_0"),
            _manifest_task_payload(dataset_index=2, order_tag="task_1"),
            _manifest_task_payload(dataset_index=3, order_tag="task_2"),
            _manifest_task_payload(dataset_index=4, order_tag="task_3"),
            _manifest_task_payload(dataset_index=5, order_tag="task_4"),
        ],
    )

    dataset = PackedParquetTaskDataset(
        manifest_path,
        split="train",
        task="classification",
        shuffle_records=True,
        shuffle_seed=7,
    )

    observed = [int(record["dataset_index"]) for record in dataset.records]
    expected = [
        [1, 2, 3, 4, 5][int(index)]
        for index in np.random.default_rng(7).permutation(5).tolist()
    ]

    assert observed == expected


def test_build_task_loader_preserves_manifest_order_when_shuffle_is_false(tmp_path: Path) -> None:
    manifest_path = _write_manifest_dataset(
        tmp_path,
        datasets=[
            _manifest_task_payload(dataset_index=1, order_tag="task_0", n_train=2, n_test=2, n_classes=2),
            _manifest_task_payload(dataset_index=2, order_tag="task_1", n_train=3, n_test=2, n_classes=2),
            _manifest_task_payload(dataset_index=3, order_tag="task_2", n_train=2, n_test=2, n_classes=2),
            _manifest_task_payload(dataset_index=4, order_tag="task_3", n_train=3, n_test=2, n_classes=2),
        ],
    )
    dataset = PackedParquetTaskDataset(
        manifest_path,
        split="train",
        task="classification",
    )

    loader = build_task_loader(
        dataset,
        num_workers=0,
        shuffle=False,
        seed=0,
        task_batch_size=2,
    )

    batches = list(loader)

    assert [int(batch.metadata["task_batch_size_actual"]) for batch in batches] == [1, 1, 1, 1]
    assert [batch.metadata["task_members"][0]["order_tag"] for batch in batches] == [
        "task_0",
        "task_1",
        "task_2",
        "task_3",
    ]


def test_build_manifest_task_dataset_shuffles_train_split_only(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_manifest_dataset(tmp_path)
    captured_calls: list[dict[str, object]] = []

    def _capture_dataset(*args: object, **kwargs: object) -> str:
        captured_calls.append({str(key): value for key, value in kwargs.items()})
        return "captured"

    monkeypatch.setattr(manifest_source_module, "PackedParquetTaskDataset", _capture_dataset)

    train_dataset = manifest_source_module.build_manifest_task_dataset(
        OmegaConf.create({"source": "manifest", "manifest_path": str(manifest_path)}),
        split="train",
        task="classification",
        seed=11,
    )
    val_dataset = manifest_source_module.build_manifest_task_dataset(
        OmegaConf.create({"source": "manifest", "manifest_path": str(manifest_path)}),
        split="val",
        task="classification",
        seed=11,
    )

    assert train_dataset == "captured"
    assert val_dataset == "captured"
    assert captured_calls[0]["shuffle_records"] is True
    assert captured_calls[0]["shuffle_seed"] == 11
    assert captured_calls[1]["shuffle_records"] is False
    assert captured_calls[1]["shuffle_seed"] == 11


def test_evaluate_loader_respects_manifest_order_for_capped_shuffle_false_loader(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    manifest_path = _write_manifest_dataset(
        tmp_path,
        datasets=[
            _manifest_task_payload(dataset_index=1, order_tag="task_0", n_train=2, n_test=2, n_classes=2),
            _manifest_task_payload(dataset_index=2, order_tag="task_1", n_train=3, n_test=2, n_classes=2),
            _manifest_task_payload(dataset_index=3, order_tag="task_2", n_train=2, n_test=2, n_classes=2),
            _manifest_task_payload(dataset_index=4, order_tag="task_3", n_train=3, n_test=2, n_classes=2),
        ],
    )
    dataset = PackedParquetTaskDataset(
        manifest_path,
        split="train",
        task="classification",
    )
    loader = build_task_loader(
        dataset,
        num_workers=0,
        shuffle=False,
        seed=0,
        task_batch_size=2,
    )
    seen_order_tags: list[str] = []

    def _fake_compute(
        _output: object,
        batch: TaskBatch,
        *,
        task: str,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        assert task == "classification"
        seen_order_tags.extend(member["order_tag"] for member in batch.metadata["task_members"])
        return torch.tensor(1.0), {"acc": 0.5}

    monkeypatch.setattr(trainer_metrics_module, "_compute_loss_and_metrics", _fake_compute)

    metrics = trainer_metrics_module._evaluate_loader(
        _EvalOrderClassifier(),
        loader,
        accelerator=_FakeEvalAccelerator(),
        task="classification",
        max_batches=2,
    )

    assert seen_order_tags == ["task_0", "task_1"]
    assert metrics["val_loss"] == pytest.approx(1.0)
    assert metrics["acc"] == pytest.approx(0.5)


def test_task_signature_matches_loaded_batch_after_label_filtering(
    tmp_path: Path,
) -> None:
    dataset = PackedParquetTaskDataset(
        _write_manifest_dataset(tmp_path),
        split="train",
        task="classification",
    )

    signature = dataset.task_signature(0)
    loaded_batch = dataset[0]

    assert signature == dataset._task_signature(loaded_batch)


def test_task_signature_fast_path_reads_filtered_split_targets(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = PackedParquetTaskDataset(
        _write_manifest_dataset(tmp_path),
        split="train",
        task="classification",
    )
    original_read_table = dataset_module.pq.read_table
    captured_calls: list[dict[str, object]] = []

    def _capture_read_table(path: Path, *args: object, **kwargs: object):
        if Path(path).name in {"train.parquet", "test.parquet"}:
            captured_calls.append(
                {
                    "name": Path(path).name,
                    "filters": kwargs.get("filters"),
                    "columns": kwargs.get("columns"),
                }
            )
        return original_read_table(path, *args, **kwargs)

    monkeypatch.setattr(dataset_module.pq, "read_table", _capture_read_table)

    _ = dataset.task_signature(0)

    assert captured_calls == [
        {
            "name": "train.parquet",
            "filters": [("dataset_index", "=", 1)],
            "columns": ["row_index", "y"],
        },
        {
            "name": "test.parquet",
            "filters": [("dataset_index", "=", 1)],
            "columns": ["row_index", "y"],
        },
    ]


def test_move_batch_moves_tensors_and_preserves_metadata() -> None:
    batch = _sample_batch()
    out = move_batch(batch, torch.device("cpu"))
    assert out.x_train.device.type == "cpu"
    assert out.y_train.device.type == "cpu"
    assert out.x_test.device.type == "cpu"
    assert out.y_test.device.type == "cpu"
    assert out.metadata == batch.metadata
    assert out.num_classes == batch.num_classes


def test_move_batch_moves_feature_type_ids_when_present() -> None:
    batch = TaskBatch(
        x_train=torch.randn(4, 3),
        y_train=torch.randint(0, 3, (4,)),
        x_test=torch.randn(2, 3),
        y_test=torch.randint(0, 3, (2,)),
        metadata={"feature_types": ["floating", "integer", "bool"]},
        num_classes=3,
        feature_type_ids=torch.tensor([[2, 1, 0]], dtype=torch.int64),
    )

    out = move_batch(batch, torch.device("cpu"))

    assert out.feature_type_ids is not None
    assert out.feature_type_ids.device.type == "cpu"
    assert out.feature_type_ids.tolist() == [[2, 1, 0]]


def test_move_batch_forwards_non_blocking_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    sample = _sample_batch()
    batch = TaskBatch(
        x_train=sample.x_train,
        y_train=sample.y_train,
        x_test=sample.x_test,
        y_test=sample.y_test,
        metadata=sample.metadata,
        num_classes=sample.num_classes,
        feature_type_ids=torch.tensor([[2, 1, 0]], dtype=torch.int64),
    )
    tracked_ids = {
        id(batch.x_train),
        id(batch.y_train),
        id(batch.x_test),
        id(batch.y_test),
        id(batch.feature_type_ids),
    }
    seen_flags: list[bool] = []
    original_to = torch.Tensor.to

    def _capture_to(self: torch.Tensor, *args: object, **kwargs: object) -> torch.Tensor:
        if id(self) in tracked_ids:
            seen_flags.append(bool(kwargs.get("non_blocking", False)))
        return original_to(self, *args, **kwargs)

    monkeypatch.setattr(torch.Tensor, "to", _capture_to)

    _ = move_batch(batch, torch.device("cpu"), non_blocking=True)

    assert seen_flags == [True, True, True, True, True]
