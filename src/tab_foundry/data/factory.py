"""Dataset and DataLoader construction helpers."""

from __future__ import annotations

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from tab_foundry.types import TaskBatch

from .sources import build_source_dataset


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
    preprocessing_cfg: DictConfig | None = None,
) -> Dataset[TaskBatch]:
    """Build one task dataset from the configured backing source."""

    return build_source_dataset(
        data_cfg,
        split=split,
        task=task,
        seed=seed,
        preprocessing_cfg=preprocessing_cfg,
    )


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
