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


def validate_classification_path_terms_contract(
    output: ClassificationOutput,
    *,
    expected_rows: int | None = None,
    context: str,
) -> list[int]:
    """Validate many-class path-term payloads against one consumer contract."""

    if output.path_logits is None or output.path_targets is None:
        raise RuntimeError(f"{context} missing path_logits or path_targets")
    if len(output.path_logits) != len(output.path_targets):
        raise RuntimeError(f"{context} produced path_logits and path_targets length mismatch")

    counts = output.path_sample_counts
    if counts is not None and len(counts) != len(output.path_logits):
        raise RuntimeError(f"{context} produced path_sample_counts length mismatch")

    total_count = 0
    resolved_counts: list[int] = []
    for index, (logits, targets) in enumerate(zip(output.path_logits, output.path_targets, strict=True)):
        if logits.ndim != _CLASSIFICATION_TENSOR_DIMENSIONS:
            raise RuntimeError(
                f"{context} requires path_logits[{index}] to be 2D, "
                f"got shape={tuple(int(dim) for dim in logits.shape)}"
            )
        if targets.ndim != 1:
            raise RuntimeError(
                f"{context} requires path_targets[{index}] to be 1D, "
                f"got shape={tuple(int(dim) for dim in targets.shape)}"
            )

        logits_rows = int(logits.shape[0])
        targets_rows = int(targets.shape[0])
        count_i = logits_rows if counts is None else int(counts[index])
        if count_i < 0:
            raise RuntimeError(f"{context} produced path_sample_counts[{index}]={count_i}, expected >= 0")
        if count_i != logits_rows:
            raise RuntimeError(
                f"{context} produced path_sample_counts[{index}]={count_i}, "
                f"but path_logits[{index}] rows={logits_rows}"
            )
        if count_i != targets_rows:
            raise RuntimeError(
                f"{context} produced path_sample_counts[{index}]={count_i}, "
                f"but path_targets[{index}] rows={targets_rows}"
            )

        resolved_counts.append(count_i)
        total_count += count_i

    if expected_rows is not None and total_count < int(expected_rows):
        raise RuntimeError(
            f"{context} produced path_sample_counts total={total_count}, expected at least {int(expected_rows)}"
        )
    return resolved_counts
