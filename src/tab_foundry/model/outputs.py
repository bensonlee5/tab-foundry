"""Shared model outputs and common constructor defaults."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import torch


DEFAULT_HEAD_HIDDEN_DIM = 1024
_CLASSIFICATION_TENSOR_DIMENSIONS: Final[int] = 2


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


def _resolve_expected_class_count(
    output: ClassificationOutput,
    *,
    expected_num_classes: int | None,
    context: str,
) -> int:
    resolved_output_num_classes = int(output.num_classes)
    if resolved_output_num_classes <= 0:
        raise RuntimeError(
            f"{context} requires output.num_classes > 0, got {resolved_output_num_classes}"
        )
    if expected_num_classes is None:
        return resolved_output_num_classes
    resolved_expected_num_classes = int(expected_num_classes)
    if resolved_expected_num_classes <= 0:
        raise RuntimeError(
            f"{context} requires expected_num_classes > 0, got {resolved_expected_num_classes}"
        )
    if resolved_output_num_classes != resolved_expected_num_classes:
        raise RuntimeError(
            f"{context} produced output.num_classes={resolved_output_num_classes}, "
            f"expected {resolved_expected_num_classes}"
        )
    return resolved_expected_num_classes


def _validate_classification_tensor(
    tensor: torch.Tensor,
    *,
    expected_rows: int | None,
    expected_width: int,
    width_mode: str,
    name: str,
    context: str,
) -> None:
    if tensor.ndim != _CLASSIFICATION_TENSOR_DIMENSIONS:
        raise RuntimeError(
            f"{context} requires {name} to be 2D, got shape={tuple(int(dim) for dim in tensor.shape)}"
        )
    rows = int(tensor.shape[0])
    width = int(tensor.shape[1])
    if expected_rows is not None and rows != int(expected_rows):
        raise RuntimeError(f"{context} produced {name} rows={rows}, expected {int(expected_rows)}")
    if width_mode == "at_least":
        if width < expected_width:
            raise RuntimeError(f"{context} produced {name} width={width}, expected at least {expected_width}")
        return
    if width_mode != "exact":
        raise ValueError(f"unsupported width_mode={width_mode!r}")
    if width != expected_width:
        raise RuntimeError(f"{context} produced {name} width={width}, expected {expected_width}")


def validate_classification_output_contract(
    output: ClassificationOutput,
    *,
    expected_rows: int | None = None,
    expected_num_classes: int | None = None,
    context: str,
) -> int:
    """Validate a classification output against one consumer contract."""

    resolved_num_classes = _resolve_expected_class_count(
        output,
        expected_num_classes=expected_num_classes,
        context=context,
    )
    if output.logits is not None:
        _validate_classification_tensor(
            output.logits,
            expected_rows=expected_rows,
            expected_width=resolved_num_classes,
            width_mode="at_least",
            name="logits",
            context=context,
        )
    if output.class_probs is not None:
        _validate_classification_tensor(
            output.class_probs,
            expected_rows=expected_rows,
            expected_width=resolved_num_classes,
            width_mode="exact",
            name="class_probs",
            context=context,
        )
    return resolved_num_classes
