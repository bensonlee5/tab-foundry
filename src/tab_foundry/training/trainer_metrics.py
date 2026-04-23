"""Metric and evaluation helpers shared by training and evaluation."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any, Callable

from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader

from tab_foundry.model.outputs import (
    CellLikelihoodOutput,
    ClassificationOutput,
    validate_cell_likelihood_output_contract,
    validate_classification_output_contract,
    validate_classification_path_terms_contract,
)
from tab_foundry.task_batching import move_batch, task_batch_diagnostics
from tab_foundry.types import TaskBatch

from .distributed import _global_mean_from_local
from .losses import classification_loss, classification_z_loss, hierarchical_nll_loss

_TASK_BATCH_TENSOR_DIMENSIONS = 3


def cycle_loader(loader: DataLoader[TaskBatch]) -> Iterator[TaskBatch]:
    while True:
        yield from loader


def _compute_loss_and_metrics(
    output: ClassificationOutput | CellLikelihoodOutput,
    batch: TaskBatch,
    *,
    task: str,
    classification_z_loss_coeff: float = 0.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    if task != "classification":
        raise RuntimeError(
            "Only classification training/evaluation is supported in this branch; "
            f"got task={task!r}."
        )
    if isinstance(output, CellLikelihoodOutput):
        expected_batch_size = (
            1 if batch.x_train.ndim != _TASK_BATCH_TENSOR_DIMENSIONS else int(batch.x_train.shape[0])
        )
        expected_shape = (
            expected_batch_size,
            int(batch.x_train.shape[-2]) + int(batch.x_test.shape[-2]),
            int(batch.x_train.shape[-1]),
        )
        validate_cell_likelihood_output_contract(
            output,
            expected_shape=expected_shape,
            context="cell_bpc training/evaluation",
        )
        if output.bpc is None or output.bpf is None:
            raise RuntimeError("cell_bpc training/evaluation requires output.bpc and output.bpf")
        metrics = {
            "bpc": float(output.bpc.detach().item()),
            "bpf": float(output.bpf.detach().item()),
        }
        if output.aux_metrics is not None:
            metrics.update(output.aux_metrics)
        return output.bpc, metrics
    if not isinstance(output, ClassificationOutput):
        raise TypeError("classification run expected ClassificationOutput")
    target = batch.y_test.to(torch.int64).reshape(-1)
    n_test = int(target.shape[0])
    if n_test <= 0:
        raise RuntimeError("classification batch has zero test labels")
    expected_num_classes = None if batch.num_classes is None else int(batch.num_classes)
    resolved_num_classes = validate_classification_output_contract(
        output,
        expected_rows=n_test,
        expected_num_classes=expected_num_classes,
        context="classification training/evaluation",
    )
    z_loss_coeff = max(0.0, float(classification_z_loss_coeff))

    if output.logits is not None:
        logits = output.logits[:, :resolved_num_classes]
        ce_loss = classification_loss(logits, target)
        z_loss = classification_z_loss(logits) if z_loss_coeff > 0.0 else None
        loss = ce_loss if z_loss is None else ce_loss + (z_loss_coeff * z_loss)
        acc = (logits.argmax(dim=-1) == target).float().mean().item()
        cls_metrics = {"acc": float(acc)}
        if z_loss is not None:
            cls_metrics.update(
                {
                    "classification_ce_loss": float(ce_loss.detach().item()),
                    "classification_z_loss": float(z_loss.detach().item()),
                    "classification_z_loss_coeff": float(z_loss_coeff),
                }
            )
        if output.aux_metrics is not None:
            cls_metrics.update(output.aux_metrics)
        return loss, cls_metrics

    if output.class_probs is not None:
        probs = output.class_probs
        loss = hierarchical_nll_loss(probs, target)
        acc = (probs.argmax(dim=-1) == target).float().mean().item()
        cls_metrics = {"acc": float(acc)}
        if output.aux_metrics is not None:
            cls_metrics.update(output.aux_metrics)
        return loss, cls_metrics

    counts = validate_classification_path_terms_contract(
        output,
        expected_rows=n_test,
        context="classification training/evaluation",
    )
    path_logits = output.path_logits
    path_targets = output.path_targets
    if path_logits is None or path_targets is None:
        raise RuntimeError("classification training/evaluation missing path logits or targets")
    weighted_total: torch.Tensor | None = None
    weighted_ce_total: torch.Tensor | None = None
    weighted_z_total: torch.Tensor | None = None
    total_edges = 0
    for logits, targets, sample_count in zip(path_logits, path_targets, counts, strict=True):
        count_i = int(sample_count)
        if count_i <= 0:
            continue
        ce_term = classification_loss(logits, targets.to(torch.int64))
        z_term = classification_z_loss(logits) if z_loss_coeff > 0.0 else None
        term = ce_term if z_term is None else ce_term + (z_loss_coeff * z_term)
        contrib = term * float(count_i)
        weighted_total = contrib if weighted_total is None else weighted_total + contrib
        ce_contrib = ce_term * float(count_i)
        weighted_ce_total = ce_contrib if weighted_ce_total is None else weighted_ce_total + ce_contrib
        if z_term is not None:
            z_contrib = z_term * float(count_i)
            weighted_z_total = z_contrib if weighted_z_total is None else weighted_z_total + z_contrib
        total_edges += count_i
    if weighted_total is None or total_edges <= 0 or n_test <= 0:
        raise RuntimeError("path-based many-class output has no valid terms")
    loss = weighted_total / float(n_test)
    path_metrics: dict[str, float] = {}
    if weighted_z_total is not None and weighted_ce_total is not None:
        path_metrics.update(
            {
                "classification_ce_loss": float(
                    (weighted_ce_total / float(n_test)).detach().item()
                ),
                "classification_z_loss": float(
                    (weighted_z_total / float(n_test)).detach().item()
                ),
                "classification_z_loss_coeff": float(z_loss_coeff),
            }
        )
    if output.aux_metrics is not None:
        path_metrics.update(output.aux_metrics)
    return loss, path_metrics


def _evaluate_loader(
    model: torch.nn.Module,
    loader: DataLoader[TaskBatch],
    *,
    accelerator: Accelerator,
    task: str,
    max_batches: int,
    non_blocking_device_transfer: bool = False,
    model_forward: Callable[[TaskBatch], Any] | None = None,
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    count = 0
    tasks_seen = 0
    if task != "classification":
        raise RuntimeError(
            "Only classification evaluation is supported in this branch; "
            f"got task={task!r}."
        )
    metric_sums: dict[str, float] = {}
    metric_counts: dict[str, int] = {}

    with torch.no_grad():
        for batch in loader:
            if max_batches <= 0:
                break
            if tasks_seen >= max_batches:
                break
            actual_task_count = int(task_batch_diagnostics(batch)["task_batch_size_actual"])
            if tasks_seen > 0 and tasks_seen + actual_task_count > max_batches:
                break
            batch = move_batch(
                batch,
                accelerator.device,
                non_blocking=non_blocking_device_transfer,
            )
            with accelerator.autocast():
                output = model(batch) if model_forward is None else model_forward(batch)
                loss, metrics = _compute_loss_and_metrics(output, batch, task=task)
            loss_sum += float(loss.detach().item()) * float(actual_task_count)
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + (float(value) * float(actual_task_count))
                metric_counts[key] = metric_counts.get(key, 0) + actual_task_count
            count += actual_task_count
            tasks_seen += actual_task_count

    model.train()
    dev = accelerator.device
    val_loss = _global_mean_from_local(
        accelerator,
        local_sum=loss_sum,
        local_count=count,
        device=dev,
        default=float("inf"),
    )
    reduced_metrics = {"val_loss": val_loss}
    for key, local_sum in metric_sums.items():
        reduced_metrics[key] = _global_mean_from_local(
            accelerator,
            local_sum=local_sum,
            local_count=metric_counts.get(key, 0),
            device=dev,
            default=0.0,
        )
    return reduced_metrics


def _expected_metric_keys(task: str) -> set[str]:
    if task != "classification":
        raise RuntimeError(
            "Only classification metrics are supported in this branch; "
            f"got task={task!r}."
        )
    return {
        "acc",
        "bpc",
        "bpf",
        "grad_norm",
        "many_class_nodes_visited",
        "many_class_avg_path_depth",
        "many_class_empty_nodes",
        "classification_ce_loss",
        "classification_z_loss",
        "classification_z_loss_coeff",
    }
