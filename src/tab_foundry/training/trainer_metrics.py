"""Metric and evaluation helpers shared by training and evaluation."""

from __future__ import annotations

from collections.abc import Iterator

from accelerate import Accelerator
import torch
from torch.utils.data import DataLoader

from tab_foundry.model.outputs import ClassificationOutput, validate_classification_output_contract
from tab_foundry.types import TaskBatch

from .batching import move_batch
from .distributed import _global_mean_from_local
from .losses import classification_loss, hierarchical_nll_loss


def cycle_loader(loader: DataLoader[TaskBatch]) -> Iterator[TaskBatch]:
    while True:
        yield from loader


def _compute_loss_and_metrics(
    output: ClassificationOutput,
    batch: TaskBatch,
    *,
    task: str,
) -> tuple[torch.Tensor, dict[str, float]]:
    if task != "classification":
        raise RuntimeError(
            "Only classification training/evaluation is supported in this branch; "
            f"got task={task!r}."
        )
    if not isinstance(output, ClassificationOutput):
        raise TypeError("classification run expected ClassificationOutput")
    n_test = int(batch.y_test.shape[0])
    if n_test <= 0:
        raise RuntimeError("classification batch has zero test labels")
    expected_num_classes = None if batch.num_classes is None else int(batch.num_classes)
    resolved_num_classes = validate_classification_output_contract(
        output,
        expected_rows=n_test,
        expected_num_classes=expected_num_classes,
        context="classification training/evaluation",
    )

    if output.logits is not None:
        logits = output.logits[:, :resolved_num_classes]
        target = batch.y_test.to(torch.int64)
        loss = classification_loss(logits, target)
        acc = (logits.argmax(dim=-1) == target).float().mean().item()
        cls_metrics = {"acc": float(acc)}
        if output.aux_metrics is not None:
            cls_metrics.update(output.aux_metrics)
        return loss, cls_metrics

    if output.class_probs is not None:
        probs = output.class_probs
        target = batch.y_test.to(torch.int64)
        loss = hierarchical_nll_loss(probs, target)
        acc = (probs.argmax(dim=-1) == target).float().mean().item()
        cls_metrics = {"acc": float(acc)}
        if output.aux_metrics is not None:
            cls_metrics.update(output.aux_metrics)
        return loss, cls_metrics

    if output.path_logits is None or output.path_targets is None:
        raise RuntimeError("many-class output missing class_probs and path terms")
    if len(output.path_logits) != len(output.path_targets):
        raise RuntimeError("path_logits and path_targets length mismatch")

    counts = (
        output.path_sample_counts
        if output.path_sample_counts is not None
        else [int(logits.shape[0]) for logits in output.path_logits]
    )
    if len(counts) != len(output.path_logits):
        raise RuntimeError("path_sample_counts length mismatch")
    weighted_total: torch.Tensor | None = None
    total_edges = 0
    for logits, targets, sample_count in zip(
        output.path_logits, output.path_targets, counts, strict=True
    ):
        count_i = int(sample_count)
        if count_i <= 0:
            continue
        term = classification_loss(logits, targets.to(torch.int64))
        contrib = term * float(count_i)
        weighted_total = contrib if weighted_total is None else weighted_total + contrib
        total_edges += count_i
    if weighted_total is None or total_edges <= 0 or n_test <= 0:
        raise RuntimeError("path-based many-class output has no valid terms")
    loss = weighted_total / float(n_test)
    path_metrics: dict[str, float] = {}
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
) -> dict[str, float]:
    model.eval()
    loss_sum = 0.0
    score_sum = 0.0
    count = 0
    if task != "classification":
        raise RuntimeError(
            "Only classification evaluation is supported in this branch; "
            f"got task={task!r}."
        )
    metric_name = "acc"

    with torch.no_grad():
        for step, batch in enumerate(loader):
            if step >= max_batches:
                break
            batch = move_batch(batch, accelerator.device)
            with accelerator.autocast():
                output = model(batch)
                loss, metrics = _compute_loss_and_metrics(output, batch, task=task)
            loss_sum += float(loss.detach().item())
            score_sum += float(metrics[metric_name])
            count += 1

    model.train()
    dev = accelerator.device
    val_loss = _global_mean_from_local(
        accelerator,
        local_sum=loss_sum,
        local_count=count,
        device=dev,
        default=float("inf"),
    )
    val_score = _global_mean_from_local(
        accelerator,
        local_sum=score_sum,
        local_count=count,
        device=dev,
        default=0.0,
    )
    return {"val_loss": val_loss, metric_name: val_score}


def _expected_metric_keys(task: str) -> set[str]:
    if task != "classification":
        raise RuntimeError(
            "Only classification metrics are supported in this branch; "
            f"got task={task!r}."
        )
    return {
        "acc",
        "grad_norm",
        "many_class_nodes_visited",
        "many_class_avg_path_depth",
        "many_class_empty_nodes",
    }
