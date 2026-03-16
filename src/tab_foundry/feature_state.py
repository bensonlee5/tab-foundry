"""Shared task-time feature-state containers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass(slots=True)
class TaskFeatureState:
    """Model-ready feature metadata and categorical ids."""

    categorical_mask: torch.Tensor
    categorical_cardinalities: torch.Tensor
    x_train_categorical_ids: torch.Tensor
    x_test_categorical_ids: torch.Tensor

    def to(self, device: torch.device) -> "TaskFeatureState":
        return TaskFeatureState(
            categorical_mask=self.categorical_mask.to(device),
            categorical_cardinalities=self.categorical_cardinalities.to(device),
            x_train_categorical_ids=self.x_train_categorical_ids.to(device),
            x_test_categorical_ids=self.x_test_categorical_ids.to(device),
        )


@dataclass(slots=True)
class PreprocessedFeatureState:
    """Numpy-backed feature metadata emitted by preprocessing."""

    categorical_mask: np.ndarray
    categorical_cardinalities: np.ndarray
    x_train_categorical_ids: np.ndarray
    x_test_categorical_ids: np.ndarray

    def filter_test_rows(self, mask: np.ndarray) -> "PreprocessedFeatureState":
        return PreprocessedFeatureState(
            categorical_mask=np.asarray(self.categorical_mask, dtype=bool),
            categorical_cardinalities=np.asarray(self.categorical_cardinalities, dtype=np.int64),
            x_train_categorical_ids=np.asarray(self.x_train_categorical_ids, dtype=np.int64),
            x_test_categorical_ids=np.asarray(self.x_test_categorical_ids, dtype=np.int64)[mask],
        )

    def to_task_feature_state(self) -> TaskFeatureState:
        return TaskFeatureState(
            categorical_mask=torch.from_numpy(np.asarray(self.categorical_mask, dtype=bool)),
            categorical_cardinalities=torch.from_numpy(
                np.asarray(self.categorical_cardinalities, dtype=np.int64)
            ),
            x_train_categorical_ids=torch.from_numpy(
                np.asarray(self.x_train_categorical_ids, dtype=np.int64)
            ),
            x_test_categorical_ids=torch.from_numpy(
                np.asarray(self.x_test_categorical_ids, dtype=np.int64)
            ),
        )
