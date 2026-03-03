"""Common project types."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass(slots=True)
class TaskBatch:
    """Container for one tabular task (train+test split)."""

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    metadata: dict[str, Any]
    num_classes: int | None = None


@dataclass(slots=True)
class TrainResult:
    """Summary of a training run."""

    output_dir: Path
    best_checkpoint: Path | None
    latest_checkpoint: Path | None
    global_step: int
    metrics: dict[str, float]


@dataclass(slots=True)
class EvalResult:
    """Summary of an evaluation run."""

    checkpoint: Path
    metrics: dict[str, float]
