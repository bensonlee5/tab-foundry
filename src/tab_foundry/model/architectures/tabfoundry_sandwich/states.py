"""Typed intermediate states for the sandwich architecture family."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class SandwichRawInputState:
    """Canonical sandwich input state before feature encoding."""

    x_all: torch.Tensor
    y_train: torch.Tensor
    y_test: torch.Tensor | None
    train_test_split_index: int
    num_classes: int
    feature_type_ids: torch.Tensor


@dataclass(slots=True)
class SandwichFeatureState:
    """Feature-encoded sandwich state with shape [B, R, C, E]."""

    raw_state: SandwichRawInputState
    feature_cells: torch.Tensor


@dataclass(slots=True)
class SandwichClassificationState:
    """Classification-ready sandwich state before perceiver execution."""

    feature_state: SandwichFeatureState
    full_cell_stream: torch.Tensor
    summary_input: torch.Tensor
    row_tokens: torch.Tensor
