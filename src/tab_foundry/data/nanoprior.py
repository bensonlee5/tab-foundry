"""HDF5-backed nanoTabPFN prior task loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from tab_foundry.types import TaskBatch


@dataclass(slots=True, frozen=True)
class NanoPriorDumpSummary:
    """Compact metadata for one nanoTabPFN prior dump."""

    path: Path
    num_tasks: int
    max_rows: int
    max_features: int
    max_num_classes: int


def inspect_nano_prior_dump(prior_dump_path: Path) -> NanoPriorDumpSummary:
    """Load metadata needed to validate one nanoTabPFN prior dump."""

    path = prior_dump_path.expanduser().resolve()
    with h5py.File(path, "r") as handle:
        x = handle["X"]
        max_num_classes_raw = np.asarray(handle["max_num_classes"][()], dtype=np.int64).reshape(-1)
        if max_num_classes_raw.size <= 0:
            raise RuntimeError(f"nano prior dump is missing max_num_classes payload: {path}")
        max_num_classes = int(max_num_classes_raw[0])
        return NanoPriorDumpSummary(
            path=path,
            num_tasks=int(x.shape[0]),
            max_rows=int(x.shape[1]),
            max_features=int(x.shape[2]),
            max_num_classes=max_num_classes,
        )


class NanoPriorTaskDataset(Dataset[TaskBatch]):
    """Expose one HDF5 prior task per dataset item."""

    def __init__(
        self,
        prior_dump_path: Path,
        *,
        offset: int,
        size: int,
    ) -> None:
        self.summary = inspect_nano_prior_dump(prior_dump_path)
        if self.summary.max_num_classes != 2:
            raise RuntimeError(
                "nanoTabPFN prior dump must be binary classification for the nano-aligned path: "
                f"max_num_classes={self.summary.max_num_classes}"
            )
        self.offset = int(offset)
        self.size = int(size)
        if self.offset < 0:
            raise ValueError(f"nano prior offset must be >= 0, got {self.offset}")
        if self.size <= 0:
            raise ValueError(f"nano prior size must be > 0, got {self.size}")
        if self.offset + self.size > self.summary.num_tasks:
            raise ValueError(
                "nano prior slice exceeds dataset count: "
                f"offset={self.offset}, size={self.size}, num_tasks={self.summary.num_tasks}"
            )
        self._handle: h5py.File | None = None

    def __len__(self) -> int:
        return self.size

    def _file(self) -> h5py.File:
        if self._handle is None:
            self._handle = h5py.File(self.summary.path, "r")
        return self._handle

    def __getitem__(self, index: int) -> TaskBatch:
        local_index = int(index)
        if local_index < 0 or local_index >= self.size:
            raise IndexError(local_index)
        task_index = self.offset + local_index
        handle = self._file()

        num_features = int(handle["num_features"][task_index])
        num_datapoints = int(handle["num_datapoints"][task_index])
        split_index = int(handle["single_eval_pos"][task_index])
        if num_features <= 0 or num_features > self.summary.max_features:
            raise RuntimeError(f"invalid num_features for task_index={task_index}: {num_features}")
        if num_datapoints <= 1 or num_datapoints > self.summary.max_rows:
            raise RuntimeError(f"invalid num_datapoints for task_index={task_index}: {num_datapoints}")
        if split_index <= 0 or split_index >= num_datapoints:
            raise RuntimeError(
                "invalid train/test split index for task_index="
                f"{task_index}: split_index={split_index}, num_datapoints={num_datapoints}"
            )

        x = np.asarray(
            handle["X"][task_index, :num_datapoints, :num_features],
            dtype=np.float32,
        )
        y = np.asarray(handle["y"][task_index, :num_datapoints], dtype=np.int64)
        num_classes = int(np.unique(y).size)
        if num_classes > 2:
            raise RuntimeError(f"task_index={task_index} has unsupported class count: {num_classes}")

        metadata: dict[str, Any] = {
            "task_index": int(task_index),
            "num_features": int(num_features),
            "num_datapoints": int(num_datapoints),
            "single_eval_pos": int(split_index),
            "prior_dump_path": str(self.summary.path),
        }
        return TaskBatch(
            x_train=torch.from_numpy(x[:split_index]),
            y_train=torch.from_numpy(y[:split_index]),
            x_test=torch.from_numpy(x[split_index:]),
            y_test=torch.from_numpy(y[split_index:]),
            metadata=metadata,
            num_classes=max(2, num_classes),
        )

    def __del__(self) -> None:  # pragma: no cover - best-effort resource cleanup
        handle = getattr(self, "_handle", None)
        if handle is not None:
            handle.close()
