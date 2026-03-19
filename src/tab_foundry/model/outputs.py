"""Shared model outputs and common constructor defaults."""

from __future__ import annotations

from dataclasses import dataclass

import torch


DEFAULT_HEAD_HIDDEN_DIM = 1024


@dataclass(slots=True)
class ClassificationOutput:
    """Classifier forward output."""

    logits: torch.Tensor | None
    num_classes: int
    class_probs: torch.Tensor | None = None
    path_logits: list[torch.Tensor] | None = None
    path_targets: list[torch.Tensor] | None = None
    path_sample_counts: list[int] | None = None
    aux_metrics: dict[str, float] | None = None
