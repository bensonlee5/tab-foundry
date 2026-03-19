"""Direct-head staged forward helpers."""

from __future__ import annotations

from typing import Any

import torch

from .forward_common import (
    build_raw_input_state,
    condition_rows,
    context_train_embeddings,
    encode_to_cell_state,
    pool_rows,
)
from .states import HeadOutputState, RawInputState


def build_direct_head_state(model: Any, raw_state: RawInputState) -> HeadOutputState:
    cell_state = encode_to_cell_state(model, raw_state)
    row_state = pool_rows(model, cell_state)
    if model.context_encoder is None:
        return HeadOutputState(
            rows=row_state.rows,
            train_test_split_index=row_state.train_test_split_index,
            num_classes=row_state.num_classes,
        )
    return condition_rows(
        model,
        row_state,
        train_target_embeddings=context_train_embeddings(model, raw_state.y_train),
    )


def forward_batched(
    model: Any,
    *,
    x_all: torch.Tensor,
    y_train: torch.Tensor,
    train_test_split_index: int,
) -> torch.Tensor:
    if model.surface.head == "many_class":
        raise RuntimeError("forward_batched() is only supported for direct-head staged recipes")
    raw_state = build_raw_input_state(
        model,
        x_all=x_all,
        y_train=y_train.to(torch.int64),
        y_test=None,
        train_test_split_index=train_test_split_index,
        num_classes=max(2, int(y_train.max().item()) + 1),
    )
    head_state = build_direct_head_state(model, raw_state)
    test_rows = head_state.rows[:, train_test_split_index:, :]
    return model.direct_head(test_rows)
