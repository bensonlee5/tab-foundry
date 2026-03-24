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
        if self.task_batch_size <= 0:
            raise ValueError(f"task_batch_size must be >= 1, got {self.task_batch_size}")
        self.shuffle = bool(shuffle)
        self.seed = int(seed)
        self.batch_size = int(task_batch_size)
        self.drop_last = False
        self._epoch = 0
        self._length_cache: int | None = None

    @staticmethod
    def _shuffle_items(items: list[_T], *, generator: torch.Generator) -> list[_T]:
        if len(items) <= 1:
            return list(items)
        order = torch.randperm(len(items), generator=generator).tolist()
        return [items[idx] for idx in order]

    def _ordered_indices(self, *, generator: torch.Generator | None) -> list[int]:
        indices = [int(index) for index in range(len(self.dataset))]
        if not self.shuffle or generator is None:
            return indices
        return [int(index) for index in self._shuffle_items(indices, generator=generator)]

    def _signature_counts(self) -> OrderedDict[tuple[int, int, int, int | None], int]:
        counts: OrderedDict[tuple[int, int, int, int | None], int] = OrderedDict()
        for index in range(len(self.dataset)):
            signature = self.dataset.task_signature(index)
            counts[signature] = counts.get(signature, 0) + 1
        return counts

    def _iter_grouped_batches(self, *, ordered_indices: list[int]):
        buckets: OrderedDict[tuple[int, int, int, int | None], list[int]] = OrderedDict()
        for index in ordered_indices:
            signature = self.dataset.task_signature(index)
            bucket = buckets.setdefault(signature, [])
            bucket.append(int(index))
            if len(bucket) == self.task_batch_size:
                yield list(bucket)
                bucket.clear()
        for remainder in buckets.values():
            if remainder:
                yield list(remainder)

    def _iter_contiguous_batches(self, *, ordered_indices: list[int]):
        current_signature: tuple[int, int, int, int | None] | None = None
        current_batch: list[int] = []
        for index in ordered_indices:
            signature = self.dataset.task_signature(index)
            if current_signature is not None and signature != current_signature:
                if current_batch:
                    yield list(current_batch)
                    current_batch.clear()
                current_signature = signature
            elif current_signature is None:
                current_signature = signature
            current_batch.append(int(index))
            if len(current_batch) == self.task_batch_size:
                yield list(current_batch)
                current_batch.clear()
        if current_batch:
            yield list(current_batch)

    def _iter_batches(self, *, epoch_seed: int):
        generator = torch.Generator()
        generator.manual_seed(int(epoch_seed))
        ordered_indices = self._ordered_indices(generator=generator)
        if self.shuffle:
            yield from self._iter_grouped_batches(ordered_indices=ordered_indices)
            return
        yield from self._iter_contiguous_batches(ordered_indices=ordered_indices)

    def _count_signature_batches(self) -> int:
        return sum(
            (count // self.task_batch_size) + (1 if count % self.task_batch_size > 0 else 0)
            for count in self._signature_counts().values()
        )

    def _count_contiguous_batches(self) -> int:
        batch_count = 0
        current_signature: tuple[int, int, int, int | None] | None = None
        current_run_length = 0
        for index in range(len(self.dataset)):
            signature = self.dataset.task_signature(index)
            if current_signature is None or signature == current_signature:
                current_signature = signature
                current_run_length += 1
                continue
            batch_count += (current_run_length // self.task_batch_size) + (
                1 if current_run_length % self.task_batch_size > 0 else 0
            )
            current_signature = signature
            current_run_length = 1
        if current_run_length > 0:
            batch_count += (current_run_length // self.task_batch_size) + (
                1 if current_run_length % self.task_batch_size > 0 else 0
            )
        return batch_count

    def __iter__(self):
        batches = self._iter_batches(epoch_seed=self.seed + self._epoch)
        self._epoch += 1
        yield from batches

    def __len__(self) -> int:
        if self._length_cache is None:
            if self.shuffle:
                self._length_cache = self._count_signature_batches()
            else:
                self._length_cache = self._count_contiguous_batches()
        return int(self._length_cache)


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
