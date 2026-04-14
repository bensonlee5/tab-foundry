"""Dataset and DataLoader construction helpers."""

from __future__ import annotations

from collections import OrderedDict
from collections.abc import Sized
from functools import partial
from typing import Any, TypeVar, cast

import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset, Sampler

from tab_foundry.data.dataset import PackedParquetTaskDataset
from tab_foundry.task_batching import collate_task_batch
from tab_foundry.types import TaskBatch

from .sources import build_source_dataset

_T = TypeVar("_T")
_VALID_TASK_BATCH_CACHE_MODES = frozenset({"off", "eager_full", "bounded_streaming"})
_Signature = tuple[int, int, int, int | None]
_SignatureFamily = tuple[int, int, int]


class _ManifestTaskBatchSampler(Sampler[list[int]]):
    """Deterministic exact-shape batch sampler for manifest-backed task batches."""

    def __init__(
        self,
        dataset: PackedParquetTaskDataset,
        *,
        task_batch_size: int,
        shuffle: bool,
        seed: int,
        max_batches: int | None = None,
        signature_family_run_length: int = 1,
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
        self.max_batches = None if max_batches is None else int(max_batches)
        if self.max_batches is not None and self.max_batches <= 0:
            raise ValueError(f"max_batches must be >= 1 when provided, got {self.max_batches}")
        self.signature_family_run_length = int(signature_family_run_length)
        if self.signature_family_run_length <= 0:
            raise ValueError(
                "signature_family_run_length must be >= 1, "
                f"got {self.signature_family_run_length}"
            )
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

    @staticmethod
    def _signature_family(signature: _Signature) -> _SignatureFamily:
        return int(signature[0]), int(signature[1]), int(signature[2])

    def _signature_counts(self) -> OrderedDict[_Signature, int]:
        counts: OrderedDict[_Signature, int] = OrderedDict()
        for index in range(len(self.dataset)):
            signature = self.dataset.task_signature(index)
            counts[signature] = counts.get(signature, 0) + 1
        return counts

    def _iter_grouped_batches(self, *, ordered_indices: list[int]):
        buckets: OrderedDict[_Signature, list[int]] = OrderedDict()
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

    def _iter_family_grouped_batches(self, *, ordered_indices: list[int]):
        grouped_batches = list(self._iter_grouped_batches(ordered_indices=ordered_indices))
        if self.max_batches is not None:
            prefix_batches = grouped_batches[: int(self.max_batches)]
            suffix_batches = grouped_batches[int(self.max_batches) :]
            yield from self._emit_family_runs_from_batches(prefix_batches)
            yield from self._emit_family_runs_from_batches(suffix_batches)
            return
        full_batches = [batch for batch in grouped_batches if len(batch) == self.task_batch_size]
        remainder_batches = [batch for batch in grouped_batches if len(batch) < self.task_batch_size]
        yield from self._emit_family_runs_from_batches(full_batches)
        yield from self._emit_family_runs_from_batches(remainder_batches)

    def _emit_family_runs_from_batches(self, batches: list[list[int]]):
        family_runs: OrderedDict[_SignatureFamily, list[list[int]]] = OrderedDict()
        for batch in batches:
            if not batch:
                continue
            family = self._signature_family(self.dataset.task_signature(batch[0]))
            family_runs.setdefault(family, []).append(list(batch))
        while True:
            emitted = False
            for family_batches in family_runs.values():
                run_length = 0
                while family_batches and run_length < self.signature_family_run_length:
                    yield family_batches.pop(0)
                    run_length += 1
                    emitted = True
            if not emitted:
                break

    def _iter_contiguous_batches(self, *, ordered_indices: list[int]):
        current_signature: _Signature | None = None
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
            if self.signature_family_run_length > 1:
                yield from self._iter_family_grouped_batches(ordered_indices=ordered_indices)
                return
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
        current_signature: _Signature | None = None
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
        if self.max_batches is None:
            yield from batches
            return
        for batch_index, batch in enumerate(batches):
            if batch_index >= self.max_batches:
                break
            yield batch

    def __len__(self) -> int:
        if self._length_cache is None:
            if self.shuffle:
                self._length_cache = self._count_signature_batches()
            else:
                self._length_cache = self._count_contiguous_batches()
        if self.max_batches is None:
            return int(self._length_cache)
        return min(int(self._length_cache), int(self.max_batches))


class _CachedTaskBatchDataset(Dataset[TaskBatch]):
    """Eager in-memory cache of preprocessed exact-shape task batches."""

    def __init__(self, batches: list[TaskBatch]) -> None:
        if not batches:
            raise RuntimeError("task-batch cache requires at least one batch")
        self._batches = list(batches)

    def __len__(self) -> int:
        return len(self._batches)

    def __getitem__(self, index: int) -> TaskBatch:
        return self._batches[int(index)]


def _identity_task_batch(batch: Any) -> TaskBatch:
    return cast(TaskBatch, batch)


def _resolve_task_batch_cache_mode(
    *,
    cache_task_batches: bool,
    cache_mode: str | None,
) -> str:
    if cache_mode is None:
        return "eager_full" if bool(cache_task_batches) else "off"
    normalized = str(cache_mode).strip().lower()
    if normalized not in _VALID_TASK_BATCH_CACHE_MODES:
        raise ValueError(
            "cache_mode must be one of "
            f"{sorted(_VALID_TASK_BATCH_CACHE_MODES)}, got {cache_mode!r}"
        )
    return normalized


def _materialize_task_batch_cache(
    dataset: Dataset[TaskBatch],
    *,
    task_batch_size: int,
    shuffle: bool,
    seed: int,
) -> _CachedTaskBatchDataset:
    requested_task_batch_size = int(task_batch_size)
    if requested_task_batch_size <= 0:
        raise ValueError(f"task_batch_size must be >= 1, got {requested_task_batch_size}")
    if requested_task_batch_size > 1:
        if not isinstance(dataset, PackedParquetTaskDataset):
            raise RuntimeError(
                "runtime.loader_task_batch_cache=true with training.task_batch_size > 1 "
                "requires a manifest-backed PackedParquetTaskDataset"
            )
        index_batches = _ManifestTaskBatchSampler(
            dataset,
            task_batch_size=requested_task_batch_size,
            shuffle=shuffle,
            seed=seed,
            signature_family_run_length=1,
        )
        return _CachedTaskBatchDataset(
            [
                collate_task_batch(
                    [dataset[int(index)] for index in index_batch],
                    requested_task_batch_size=requested_task_batch_size,
                )
                for index_batch in index_batches
            ]
        )

    if not isinstance(dataset, Sized):
        raise RuntimeError("runtime.loader_task_batch_cache=true requires a sized dataset")
    ordered_indices = [int(index) for index in range(len(dataset))]
    if shuffle and len(ordered_indices) > 1:
        generator = torch.Generator()
        generator.manual_seed(int(seed))
        order = torch.randperm(len(ordered_indices), generator=generator).tolist()
        ordered_indices = [ordered_indices[int(index)] for index in order]
    return _CachedTaskBatchDataset(
        [
            collate_task_batch(
                [dataset[int(index)]],
                requested_task_batch_size=1,
            )
            for index in ordered_indices
        ]
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
    pin_memory: bool = False,
    persistent_workers: bool = False,
    prefetch_factor: int | None = None,
    cache_task_batches: bool = False,
    cache_mode: str | None = None,
    max_batches: int | None = None,
    signature_family_run_length: int = 1,
) -> DataLoader[TaskBatch]:
    """Build a task loader with deterministic seeded shuffling."""

    resolved_task_batch_size = int(task_batch_size)
    if resolved_task_batch_size <= 0:
        raise ValueError(f"task_batch_size must be >= 1, got {resolved_task_batch_size}")
    resolved_num_workers = int(num_workers)
    resolved_cache_mode = _resolve_task_batch_cache_mode(
        cache_task_batches=cache_task_batches,
        cache_mode=cache_mode,
    )
    if resolved_cache_mode == "eager_full":
        if resolved_num_workers != 0:
            raise ValueError("runtime.loader_task_batch_cache=true requires runtime.num_workers=0")
        cached_dataset = _materialize_task_batch_cache(
            dataset,
            task_batch_size=resolved_task_batch_size,
            shuffle=shuffle,
            seed=seed,
        )
        cache_generator: torch.Generator | None = None
        if shuffle:
            cache_generator = torch.Generator()
            cache_generator.manual_seed(int(seed))
        return DataLoader(
            cached_dataset,
            batch_size=None,
            shuffle=shuffle,
            collate_fn=cast(Any, _identity_task_batch),
            generator=cache_generator,
            num_workers=0,
            pin_memory=bool(pin_memory),
        )
    if resolved_cache_mode == "bounded_streaming":
        if max_batches is None:
            raise ValueError(
                "cache_mode='bounded_streaming' requires max_batches to bound the loader horizon"
            )
        if not isinstance(dataset, PackedParquetTaskDataset):
            raise RuntimeError(
                "cache_mode='bounded_streaming' requires a manifest-backed PackedParquetTaskDataset"
            )
        batch_sampler = _ManifestTaskBatchSampler(
            dataset,
            task_batch_size=resolved_task_batch_size,
            shuffle=shuffle,
            seed=seed,
            max_batches=max_batches,
            signature_family_run_length=signature_family_run_length,
        )
        if resolved_num_workers > 0:
            if prefetch_factor is None:
                return DataLoader(
                    dataset,
                    batch_sampler=batch_sampler,
                    collate_fn=partial(
                        collate_task_batch,
                        requested_task_batch_size=resolved_task_batch_size,
                    ),
                    num_workers=resolved_num_workers,
                    pin_memory=bool(pin_memory),
                    persistent_workers=bool(persistent_workers),
                )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=partial(
                    collate_task_batch,
                    requested_task_batch_size=resolved_task_batch_size,
                ),
                num_workers=resolved_num_workers,
                pin_memory=bool(pin_memory),
                persistent_workers=bool(persistent_workers),
                prefetch_factor=int(prefetch_factor),
            )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=partial(
                collate_task_batch,
                requested_task_batch_size=resolved_task_batch_size,
            ),
            num_workers=0,
            pin_memory=bool(pin_memory),
        )
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
        batch_sampler = _ManifestTaskBatchSampler(
            dataset,
            task_batch_size=resolved_task_batch_size,
            shuffle=shuffle,
            seed=seed,
            signature_family_run_length=signature_family_run_length,
        )
        if resolved_num_workers > 0:
            if prefetch_factor is None:
                return DataLoader(
                    dataset,
                    batch_sampler=batch_sampler,
                    collate_fn=collate,
                    num_workers=resolved_num_workers,
                    pin_memory=bool(pin_memory),
                    persistent_workers=bool(persistent_workers),
                )
            return DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=collate,
                num_workers=resolved_num_workers,
                pin_memory=bool(pin_memory),
                persistent_workers=bool(persistent_workers),
                prefetch_factor=int(prefetch_factor),
            )
        return DataLoader(
            dataset,
            batch_sampler=batch_sampler,
            collate_fn=collate,
            num_workers=resolved_num_workers,
            pin_memory=bool(pin_memory),
        )
    if resolved_num_workers > 0:
        if prefetch_factor is None:
            return DataLoader(
                dataset,
                batch_size=1,
                shuffle=shuffle,
                collate_fn=collate,
                generator=generator,
                num_workers=resolved_num_workers,
                pin_memory=bool(pin_memory),
                persistent_workers=bool(persistent_workers),
            )
        return DataLoader(
            dataset,
            batch_size=1,
            shuffle=shuffle,
            collate_fn=collate,
            generator=generator,
            num_workers=resolved_num_workers,
            pin_memory=bool(pin_memory),
            persistent_workers=bool(persistent_workers),
            prefetch_factor=int(prefetch_factor),
        )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        collate_fn=collate,
        generator=generator,
        num_workers=resolved_num_workers,
        pin_memory=bool(pin_memory),
    )
