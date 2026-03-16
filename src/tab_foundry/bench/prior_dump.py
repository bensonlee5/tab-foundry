"""HDF5 prior-dump readers for benchmark/debug training workflows."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from tab_foundry.data.validation import assert_no_non_finite_values
from tab_foundry.types import TaskBatch


@dataclass(slots=True, frozen=True)
class PriorDumpStep:
    """One optimizer step worth of tasks from a nanoTabPFN prior dump."""

    step_index: int
    dataset_indices: tuple[int, ...]
    train_test_split_index: int
    tasks: tuple[TaskBatch, ...]
    x_batch: torch.Tensor | None = None
    y_batch: torch.Tensor | None = None


class PriorDumpTaskBatchReader:
    """Iterate a nanoTabPFN prior dump with the notebook's batching semantics."""

    def __init__(
        self,
        path: Path,
        *,
        num_steps: int,
        batch_size: int,
        allow_missing_values: bool = False,
    ) -> None:
        self.path = path.expanduser().resolve()
        self.num_steps = int(num_steps)
        self.batch_size = int(batch_size)
        self.allow_missing_values = bool(allow_missing_values)
        if self.num_steps <= 0:
            raise ValueError(f"num_steps must be >= 1, got {self.num_steps}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")

    def __iter__(self) -> Iterator[PriorDumpStep]:
        import h5py

        if not self.path.exists():
            raise RuntimeError(f"prior dump does not exist: {self.path}")
        pointer = 0
        with h5py.File(self.path, "r") as handle:
            raw_max_classes = np.asarray(handle["max_num_classes"]).reshape(-1)
            if raw_max_classes.size != 1:
                raise RuntimeError("prior dump max_num_classes must contain exactly one value")
            max_num_classes = int(raw_max_classes[0])
            if max_num_classes != 2:
                raise RuntimeError(
                    "phase 1 prior-dump training only supports binary classification; "
                    f"got max_num_classes={max_num_classes}"
                )

            x_ds = handle["X"]
            y_ds = handle["y"]
            num_features_ds = handle["num_features"]
            num_datapoints_ds = handle["num_datapoints"]
            split_ds = handle["single_eval_pos"]
            dataset_count = int(x_ds.shape[0])
            if dataset_count <= 0:
                raise RuntimeError("prior dump contains no datasets")

            for step_index in range(1, self.num_steps + 1):
                end = min(pointer + self.batch_size, dataset_count)
                if end <= pointer:
                    raise RuntimeError("prior dump batch selection produced no datasets")

                batch_dataset_indices = tuple(range(pointer, end))
                num_features = np.asarray(num_features_ds[pointer:end], dtype=np.int64)
                num_datapoints = np.asarray(num_datapoints_ds[pointer:end], dtype=np.int64)
                split_values = np.asarray(split_ds[pointer:end], dtype=np.int64)
                if split_values.size == 0:
                    raise RuntimeError("prior dump batch is empty")
                first_split = int(split_values[0])
                max_num_features = int(num_features.max())
                max_num_datapoints = int(num_datapoints.max())
                x_batch = torch.from_numpy(
                    np.asarray(
                        x_ds[pointer:end, :max_num_datapoints, :max_num_features],
                        dtype=np.float32,
                    )
                )
                y_batch = torch.from_numpy(
                    np.asarray(
                        y_ds[pointer:end, :max_num_datapoints],
                        dtype=np.float32,
                    )
                )
                if not self.allow_missing_values:
                    assert_no_non_finite_values(
                        {
                            "x_batch": x_batch.numpy(),
                            "y_batch": y_batch.numpy(),
                        },
                        context=(
                            "prior dump batch "
                            f"path={self.path}, dataset_indices={batch_dataset_indices}"
                        ),
                    )
                tasks: list[TaskBatch] = []
                for local_index, dataset_index in enumerate(batch_dataset_indices):
                    n_features = int(num_features[local_index])
                    n_datapoints = int(num_datapoints[local_index])
                    if n_features <= 0:
                        raise RuntimeError(
                            f"prior dump dataset {dataset_index} has invalid num_features={n_features}"
                        )
                    if n_datapoints <= 0:
                        raise RuntimeError(
                            f"prior dump dataset {dataset_index} has invalid num_datapoints={n_datapoints}"
                        )
                    if first_split <= 0 or first_split >= n_datapoints:
                        raise RuntimeError(
                            "phase 1 prior-dump training requires "
                            f"0 < train_test_split_index < num_datapoints, got "
                            f"split={first_split}, num_datapoints={n_datapoints}, dataset_index={dataset_index}"
                        )

                    x = np.asarray(
                        x_ds[dataset_index, :n_datapoints, :n_features],
                        dtype=np.float32,
                    )
                    y = np.asarray(
                        y_ds[dataset_index, :n_datapoints],
                        dtype=np.int64,
                    )
                    if y.shape[0] != n_datapoints:
                        raise RuntimeError(
                            f"prior dump dataset {dataset_index} label shape mismatch: "
                            f"expected {n_datapoints}, got {y.shape[0]}"
                        )
                    if np.any((y < 0) | (y > 1)):
                        raise RuntimeError(
                            f"phase 1 prior-dump training requires binary labels in {{0,1}}, "
                            f"got dataset_index={dataset_index}, labels={sorted(set(y.tolist()))}"
                        )

                    tasks.append(
                        TaskBatch(
                            x_train=torch.from_numpy(x[:first_split].copy()),
                            y_train=torch.from_numpy(y[:first_split].copy()),
                            x_test=torch.from_numpy(x[first_split:].copy()),
                            y_test=torch.from_numpy(y[first_split:].copy()),
                            metadata={
                                "source": "nanotabpfn_prior_dump",
                                "prior_dump_path": str(self.path),
                                "dataset_index": int(dataset_index),
                                "num_features": n_features,
                                "num_datapoints": n_datapoints,
                                "train_test_split_index": first_split,
                                "raw_single_eval_pos": int(split_values[local_index]),
                            },
                            num_classes=2,
                        )
                    )

                yield PriorDumpStep(
                    step_index=step_index,
                    dataset_indices=batch_dataset_indices,
                    train_test_split_index=first_split,
                    tasks=tuple(tasks),
                    x_batch=x_batch,
                    y_batch=y_batch,
                )

                pointer += self.batch_size
                if pointer >= dataset_count:
                    pointer = 0

    def __len__(self) -> int:
        return self.num_steps
