"""Dataset and DataLoader construction helpers."""

from __future__ import annotations

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from omegaconf import DictConfig

from tab_foundry.types import TaskBatch

from .dataset import PackedParquetTaskDataset
from .nanoprior import NanoPriorTaskDataset


def _collate_task_batch(items: list[TaskBatch]) -> TaskBatch:
    if len(items) != 1:
        raise RuntimeError("Only batch_size=1 is supported for task-level batching")
    return items[0]


def build_task_dataset(
    data_cfg: DictConfig,
    *,
    split: str,
    task: str,
    seed: int,
) -> Dataset[TaskBatch]:
    """Build one task dataset from the configured backing source."""

    source = str(getattr(data_cfg, "source", "manifest")).strip().lower()
    if source == "manifest":
        return PackedParquetTaskDataset(
            manifest_path=Path(str(data_cfg.manifest_path)),
            split=split,
            task=task,
            train_row_cap=(int(data_cfg.train_row_cap) if data_cfg.train_row_cap is not None else None),
            test_row_cap=(int(data_cfg.test_row_cap) if data_cfg.test_row_cap is not None else None),
            seed=seed,
        )

    if source == "nanoprior":
        if task != "classification":
            raise RuntimeError("nano prior dataset only supports classification tasks")
        prior_dump_path = getattr(data_cfg, "prior_dump_path", None)
        if prior_dump_path is None:
            raise RuntimeError("data.prior_dump_path must be set when data.source=nanoprior")
        offset_key = f"prior_{split}_offset"
        size_key = f"prior_{split}_size"
        offset_value = getattr(data_cfg, offset_key, None)
        size_value = getattr(data_cfg, size_key, None)
        if offset_value is None or size_value is None:
            raise RuntimeError(
                f"data.{offset_key} and data.{size_key} must be set when data.source=nanoprior"
            )
        return NanoPriorTaskDataset(
            Path(str(prior_dump_path)),
            offset=int(offset_value),
            size=int(size_value),
        )

    raise ValueError(f"Unsupported data.source: {source!r}")


def build_task_loader(
    dataset: Dataset[TaskBatch],
    *,
    num_workers: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[TaskBatch]:
    """Build a task loader with deterministic seeded shuffling."""

    generator: torch.Generator | None = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=int(num_workers),
        collate_fn=_collate_task_batch,
        generator=generator,
    )
