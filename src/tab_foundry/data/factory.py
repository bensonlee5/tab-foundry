"""Dataset and DataLoader construction helpers."""

from __future__ import annotations

from collections import OrderedDict
from functools import partial
from typing import TypeVar

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Sampler

from tab_foundry.data.dataset import PackedParquetTaskDataset
from tab_foundry.task_batching import collate_task_batch
from tab_foundry.types import TaskBatch

from .sources import build_source_dataset

_T = TypeVar("_T")


class _ManifestTaskBatchSampler(Sampler[list[int]]):
    """Deterministic exact-shape batch sampler for manifest-backed task batches."""

    def __init__(
        self,
        dataset: PackedParquetTaskDataset,
        *,
        task_batch_size: int,
        shuffle: bool,
        seed: int,
    ) -> None:
        self.dataset = dataset
        self.task_batch_size = int(task_batch_size)
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self._epoch = 0
        self._counts_by_signature: OrderedDict[tuple[int, int, int, int | None], int] = OrderedDict()
        for index in range(len(dataset)):
            signature = dataset.task_signature(index)
            self._counts_by_signature[signature] = self._counts_by_signature.get(signature, 0) + 1

    @staticmethod
    def _shuffle_items(items: list[_T], *, generator: torch.Generator) -> list[_T]:
        if len(items) <= 1:
            return list(items)
        order = torch.randperm(len(items), generator=generator).tolist()
        return [items[idx] for idx in order]

    def _build_batches(self, *, epoch_seed: int) -> list[list[int]]:
        generator = torch.Generator()
        generator.manual_seed(int(epoch_seed))
        buckets: OrderedDict[tuple[int, int, int, int | None], list[int]] = OrderedDict()
        for index in range(len(self.dataset)):
            signature = self.dataset.task_signature(index)
            buckets.setdefault(signature, []).append(index)

        signature_order = list(buckets)
        if self.shuffle:
            signature_order = [
                signature
                for signature in self._shuffle_items(signature_order, generator=generator)
            ]

        batches: list[list[int]] = []
        for signature in signature_order:
            indices = list(buckets[signature])
            if self.shuffle:
                indices = [int(index) for index in self._shuffle_items(indices, generator=generator)]
            if self.task_batch_size <= 1:
                batches.extend([[int(index)] for index in indices])
                continue
            full_batch_limit = (len(indices) // self.task_batch_size) * self.task_batch_size
            for start in range(0, full_batch_limit, self.task_batch_size):
                batches.append(indices[start : start + self.task_batch_size])
            remainder = indices[full_batch_limit:]
            if len(remainder) == 1:
                batches.append([int(remainder[0])])
            elif remainder:
                batches.append([int(index) for index in remainder])
        if self.shuffle:
            batches = [list(batch) for batch in self._shuffle_items(batches, generator=generator)]
        return batches

    def __iter__(self):
        batches = self._build_batches(epoch_seed=self.seed + self._epoch)
        self._epoch += 1
        yield from batches

    def __len__(self) -> int:
        return sum(
            (count // self.task_batch_size) + (1 if count % self.task_batch_size > 0 else 0)
            for count in self._counts_by_signature.values()
        )


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
    task_batch_size: int = 1,
) -> DataLoader[TaskBatch]:
    """Build a task loader with deterministic seeded shuffling."""

    resolved_task_batch_size = int(task_batch_size)
    if resolved_task_batch_size <= 0:
        raise ValueError(f"task_batch_size must be >= 1, got {resolved_task_batch_size}")

    collate = partial(
        collate_task_batch,
        requested_task_batch_size=resolved_task_batch_size,
    )
    generator: torch.Generator | None = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
    if resolved_task_batch_size > 1:
        if not isinstance(dataset, PackedParquetTaskDataset):
            raise RuntimeError(
                "training.task_batch_size > 1 requires a manifest-backed PackedParquetTaskDataset"
            )
        return DataLoader(
            dataset,
            batch_sampler=_ManifestTaskBatchSampler(
                dataset,
                task_batch_size=resolved_task_batch_size,
                shuffle=shuffle,
                seed=seed,
            ),
            num_workers=int(num_workers),
            collate_fn=collate,
        )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        num_workers=int(num_workers),
        collate_fn=collate,
        generator=generator,
    )
