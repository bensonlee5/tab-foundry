"""Shared model inspection helpers for developer tooling."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any

import torch

from tab_foundry.types import TaskBatch

from .architectures.tabfoundry_staged.resolved import resolve_staged_surface
from .factory import build_model_from_spec
from .spec import ModelBuildSpec


@dataclass(slots=True, frozen=True)
class SyntheticForwardBatch:
    """Deterministic synthetic inputs for model construction checks."""

    task_batch: TaskBatch
    x_all: torch.Tensor
    y_train_batched: torch.Tensor
    train_test_split_index: int
    expected_output_kind: str
    expected_num_classes: int
    expected_test_rows: int


@dataclass(slots=True, frozen=True)
class SyntheticReferenceArrays:
    """Deterministic runtime arrays for export/reference smoke checks."""

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    expected_num_classes: int


def parameter_counts_from_model_spec(spec: ModelBuildSpec) -> dict[str, int]:
    """Return total and trainable parameter counts for one resolved model spec."""

    model = build_model_from_spec(spec)
    total_params = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable_params = sum(
        int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
    }


def model_surface_payload(spec: ModelBuildSpec) -> dict[str, Any]:
    """Render the resolved model surface for CLI and artifact summaries."""

    payload: dict[str, Any] = {
        "arch": str(spec.arch),
        "stage": None if spec.stage is None else str(spec.stage),
        "stage_label": None if spec.stage_label is None else str(spec.stage_label),
        "input_normalization": str(spec.input_normalization),
        "feature_group_size": int(spec.feature_group_size),
        "many_class_base": int(spec.many_class_base),
        "build_spec": spec.to_dict(),
    }
    if spec.arch != "tabfoundry_staged":
        return payload

    surface = resolve_staged_surface(spec)
    payload.update(
        {
            "benchmark_profile": str(surface.benchmark_profile),
            "module_selection": surface.module_selection(),
            "module_hyperparameters": surface.component_hyperparameters(),
            "task_contract": dict(asdict(surface.task_contract)),
        }
    )
    return payload


def synthetic_forward_batch(spec: ModelBuildSpec) -> SyntheticForwardBatch:
    """Build one deterministic synthetic task batch for forward-only checks."""

    train_rows = 3
    test_rows = 2
    feature_count = 4
    total_rows = train_rows + test_rows
    x_all = torch.arange(total_rows * feature_count, dtype=torch.float32).reshape(total_rows, feature_count)
    x_all = (x_all / float(feature_count)) - 1.0
    num_classes, output_kind = _resolved_forward_shape(spec)
    y_all = torch.arange(total_rows, dtype=torch.int64).remainder(num_classes)
    x_train = x_all[:train_rows].clone()
    x_test = x_all[train_rows:].clone()
    y_train = y_all[:train_rows].clone()
    y_test = y_all[train_rows:].clone()
    task_batch = TaskBatch(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        metadata={"source": "synthetic_forward_check"},
        num_classes=int(num_classes),
    )
    return SyntheticForwardBatch(
        task_batch=task_batch,
        x_all=x_all.unsqueeze(0),
        y_train_batched=y_train.unsqueeze(0),
        train_test_split_index=int(train_rows),
        expected_output_kind=output_kind,
        expected_num_classes=int(num_classes),
        expected_test_rows=int(test_rows),
    )


def synthetic_reference_arrays(
    spec: ModelBuildSpec,
    *,
    include_missing_inputs: bool,
) -> SyntheticReferenceArrays:
    """Build deterministic runtime arrays for reference-consumer smoke checks."""

    num_classes, _ = _resolved_forward_shape(spec)
    train_rows = max(int(num_classes) + 1, 4)
    test_rows = 2
    feature_count = 3
    total_rows = train_rows + test_rows

    x_all = torch.arange(total_rows * feature_count, dtype=torch.float32).reshape(total_rows, feature_count)
    x_all = (x_all / float(feature_count)) + 1.0
    x_train = x_all[:train_rows].clone()
    x_test = x_all[train_rows:].clone()
    if include_missing_inputs:
        x_train[0, 1] = float("nan")
        x_test[0, 0] = float("nan")

    y_train = torch.arange(train_rows, dtype=torch.int64).remainder(int(num_classes))
    y_train = y_train + 100
    return SyntheticReferenceArrays(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        expected_num_classes=int(num_classes),
    )


def _resolved_forward_shape(spec: ModelBuildSpec) -> tuple[int, str]:
    if spec.arch == "tabfoundry_simple":
        return 2, "logits"

    surface = resolve_staged_surface(spec)
    if surface.head == "many_class":
        return max(int(spec.many_class_base) + 1, int(surface.task_contract.min_classes)), "class_probs"
    if surface.head == "small_class":
        return int(spec.many_class_base), "logits"

    target_classes = 2 if surface.task_contract.max_classes == 2 else 3
    if surface.task_contract.max_classes is not None:
        target_classes = min(target_classes, int(surface.task_contract.max_classes))
    target_classes = max(target_classes, int(surface.task_contract.min_classes))
    return int(target_classes), "logits"
