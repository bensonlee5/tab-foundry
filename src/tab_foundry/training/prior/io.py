"""I/O helpers for exact prior-dump training."""

from __future__ import annotations

import torch

from tab_foundry.feature_types import DEFAULT_FEATURE_TYPE


def _stack_feature_types(
    tasks,
    *,
    feature_count: int,
    allow_padding: bool = False,
) -> list[list[str]] | None:
    resolved: list[list[str]] = []
    missing_indices: list[int] = []
    for index, task in enumerate(tasks):
        raw_feature_types = task.metadata.get("feature_types")
        if raw_feature_types is None:
            missing_indices.append(index)
            continue
        if not isinstance(raw_feature_types, list):
            raise RuntimeError(
                "prior-dump task feature_types metadata must be a list of strings, "
                f"got task[{index}]={type(raw_feature_types).__name__}"
            )
        feature_types = [str(value).strip() for value in raw_feature_types]
        if len(feature_types) != int(feature_count):
            if allow_padding and len(feature_types) < int(feature_count):
                feature_types = feature_types + [DEFAULT_FEATURE_TYPE] * (
                    int(feature_count) - len(feature_types)
                )
            else:
                raise RuntimeError(
                    "prior-dump task feature_types length must match the task feature count, "
                    f"got task[{index}]={len(feature_types)} expected={int(feature_count)}"
                )
        resolved.append(feature_types)
    if missing_indices:
        if resolved:
            raise RuntimeError(
                "prior-dump task batch mixes tasks with and without feature_types metadata: "
                f"missing task indices={missing_indices}"
            )
        return None
    return resolved or None


def stack_prior_step(
    prior_step,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[list[str]] | None]:
    tasks = prior_step.tasks
    if len(tasks) <= 0:
        raise RuntimeError(f"prior dump step {prior_step.step_index} produced no tasks")
    if prior_step.x_batch is not None and prior_step.y_batch is not None:
        feature_count = int(prior_step.x_batch.shape[2])
        feature_types_batch = _stack_feature_types(
            tasks,
            feature_count=feature_count,
            allow_padding=True,
        )
        x_batch = prior_step.x_batch.to(device=device, dtype=torch.float32)
        y_batch = prior_step.y_batch.to(device=device, dtype=torch.float32)
        return (
            x_batch,
            y_batch[:, : prior_step.train_test_split_index],
            y_batch,
            feature_types_batch,
        )

    first_x_all = torch.cat([tasks[0].x_train, tasks[0].x_test], dim=0)
    feature_count = int(first_x_all.shape[1])
    feature_types_batch = _stack_feature_types(tasks, feature_count=feature_count)
    row_count = int(first_x_all.shape[0])
    for task in tasks[1:]:
        x_all = torch.cat([task.x_train, task.x_test], dim=0)
        if tuple(x_all.shape) != (row_count, feature_count):
            raise RuntimeError(
                "exact nanoTabPFN parity requires rectangular prior batches, but got "
                f"mixed x_all shapes {(row_count, feature_count)} and {tuple(x_all.shape)}"
            )
        y_all = torch.cat([task.y_train, task.y_test], dim=0)
        if tuple(y_all.shape) != (row_count,):
            raise RuntimeError(
                "exact nanoTabPFN parity requires matching label lengths across the batch, but got "
                f"{row_count} and {tuple(y_all.shape)}"
            )
    x_batch = torch.stack(
        [torch.cat([task.x_train, task.x_test], dim=0) for task in tasks],
        dim=0,
    ).to(device=device, dtype=torch.float32)
    y_train_batch = torch.stack([task.y_train for task in tasks], dim=0).to(
        device=device,
        dtype=torch.float32,
    )
    y_all_batch = torch.stack(
        [torch.cat([task.y_train, task.y_test], dim=0) for task in tasks],
        dim=0,
    ).to(device=device, dtype=torch.float32)
    return x_batch, y_train_batch, y_all_batch, feature_types_batch


def save_eval_mode_checkpoint(
    prepared_opts: list[tuple[str, torch.optim.Optimizer]],
    *,
    path,
    model: torch.nn.Module,
    global_step: int,
    cfg,
    restore_training: bool,
    set_optimizer_training_mode_fn,
    save_checkpoint_fn,
) -> None:
    set_optimizer_training_mode_fn(prepared_opts, training=False)
    save_checkpoint_fn(
        path,
        model_state=model.state_dict(),
        global_step=global_step,
        cfg=cfg,
    )
    if restore_training:
        set_optimizer_training_mode_fn(prepared_opts, training=True)
