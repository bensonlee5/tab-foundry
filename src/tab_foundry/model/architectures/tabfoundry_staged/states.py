"""Typed intermediate states for the staged tabfoundry family."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from tab_foundry.feature_state import TaskFeatureState


@dataclass(slots=True)
class RawInputState:
    """Canonical raw-input state before feature encoding."""

    x_all: torch.Tensor
    y_train: torch.Tensor
    y_test: torch.Tensor | None
    train_test_split_index: int
    num_classes: int
    feature_state: TaskFeatureState | None


@dataclass(slots=True)
class CellTableState:
    """Cell-table state with shape [B, R, C, E]."""

    cells: torch.Tensor
    train_test_split_index: int
    num_classes: int


@dataclass(slots=True)
class RowState:
    """Row state with shape [B, R, E]."""

    rows: torch.Tensor
    train_test_split_index: int
    num_classes: int


@dataclass(slots=True)
class HeadOutputState:
    """Head-ready output state over the full row sequence."""

    rows: torch.Tensor
    train_test_split_index: int
    num_classes: int
