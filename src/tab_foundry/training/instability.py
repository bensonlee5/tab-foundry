"""Instability telemetry helpers for training-style runs."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn

from tab_foundry.timestamps import utc_now as _shared_utc_now
from tab_foundry.types import TaskBatch


LOSS_EMA_ALPHA = 0.1
TRAINING_TELEMETRY_SCHEMA = "tab-foundry-training-telemetry-v4"
CLASSIFICATION_OBJECTIVE_METRIC = "final_log_loss_at_matched_regime_budget"
CELL_BPC_OBJECTIVE_METRIC = "final_bpc_at_matched_regime_budget"
_TASK_BATCH_NDIM = 3
_WINDOW_EARLY = "early_1_25"
_WINDOW_POST_WARMUP = "post_warmup_100"
_WINDOW_FINAL = "final_10pct"
_EARLY_WINDOW_MAX_STEP = 25
_MIN_LINEAR_SLOPE_POINTS = 2
_POST_WARMUP_WINDOW_MAX_RECORDS = 100
_FINAL_WINDOW_FRACTION = 0.1
_TRACKED_ACTIVATIONS = (
    "post_feature_encoder",
    "pre_transformer",
    "post_column_encoder",
    "post_row_pool",
    "post_context_encoder",
)
_UPPER_BLOCK_START = 8
_UPPER_BLOCK_END = 11
_STAGE_LOCAL_GRADIENT_MODULES = (
    "column_encoder",
    "row_pool",
    "context_encoder",
)

_TOP_LEVEL_GRADIENT_MODULES = (
    "tokenizer",
    "feature_encoder",
    "post_encoder_norm",
    "target_encoder",
    "target_conditioner",
    "column_encoder",
    "row_pool",
    "context_encoder",
    "context_label_embed",
    "digit_position_embed",
    "direct_head",
    "decoder",
)
_GLOBAL_GRAD_NORM_KINDS = ("finite", "nan", "pos_inf", "neg_inf")


def _utc_now() -> str:
    return _shared_utc_now()


def gradient_history_path(output_dir: Path) -> Path:
    """Return the canonical module-gradient history path for one run."""

    return output_dir.expanduser().resolve() / "gradient_history.jsonl"


def telemetry_path(output_dir: Path) -> Path:
    """Return the canonical telemetry path for one run."""

    return output_dir.expanduser().resolve() / "telemetry.json"


def reset_peak_device_memory_stats(device: torch.device | None) -> None:
    """Reset peak CUDA memory stats for one device when available."""

    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return
    try:
        torch.cuda.reset_peak_memory_stats(device)
    except RuntimeError:
        return


def peak_device_memory_summary(device: torch.device | None) -> dict[str, int | None]:
    """Return peak CUDA allocated and reserved memory in bytes."""

    if device is None or device.type != "cuda" or not torch.cuda.is_available():
        return {
            "peak_vram_allocated": None,
            "peak_vram_reserved": None,
        }
    try:
        return {
            "peak_vram_allocated": int(torch.cuda.max_memory_allocated(device)),
            "peak_vram_reserved": int(torch.cuda.max_memory_reserved(device)),
        }
    except RuntimeError:
        return {
            "peak_vram_allocated": None,
            "peak_vram_reserved": None,
        }


def task_batch_examples_seen(batch: TaskBatch) -> int:
    """Return the number of logical tasks contained in one task batch."""

    if batch.x_train.ndim == _TASK_BATCH_NDIM:
        return int(batch.x_train.shape[0])
    return 1


def task_batch_token_count(batch: TaskBatch) -> int:
    """Return the logical cell-token count for one task batch."""

    if batch.x_train.ndim == _TASK_BATCH_NDIM:
        task_count = int(batch.x_train.shape[0])
        n_train = int(batch.x_train.shape[1])
        n_test = int(batch.x_test.shape[1])
        n_features = int(batch.x_train.shape[2])
    else:
        task_count = 1
        n_train = int(batch.x_train.shape[0])
        n_test = int(batch.x_test.shape[0])
        n_features = int(batch.x_train.shape[1])
    return int(task_count * (n_train + n_test) * n_features)


def tensor_batch_examples_seen(x_batch: torch.Tensor) -> int:
    """Return the number of logical tasks in one prior-dump tensor batch."""

    if x_batch.ndim != _TASK_BATCH_NDIM:
        raise RuntimeError(
            "prior-dump runtime summaries require x_batch with shape "
            f"(tasks, rows, features), got {tuple(int(dim) for dim in x_batch.shape)}"
        )
    return int(x_batch.shape[0])


def tensor_batch_token_count(x_batch: torch.Tensor) -> int:
    """Return the logical cell-token count for one prior-dump tensor batch."""

    if x_batch.ndim != _TASK_BATCH_NDIM:
        raise RuntimeError(
            "prior-dump runtime summaries require x_batch with shape "
            f"(tasks, rows, features), got {tuple(int(dim) for dim in x_batch.shape)}"
        )
    return int(x_batch.shape[0] * x_batch.shape[1] * x_batch.shape[2])


def objective_metric_for_task(
    task: str | None,
    *,
    loss_surface: str | None = None,
) -> str | None:
    """Return the default objective metric for one supported task."""

    normalized = "" if task is None else str(task).strip().lower()
    normalized_loss_surface = "" if loss_surface is None else str(loss_surface).strip().lower()
    if normalized == "classification":
        if normalized_loss_surface == "cell_bpc":
            return CELL_BPC_OBJECTIVE_METRIC
        return CLASSIFICATION_OBJECTIVE_METRIC
    return None


def total_grad_norm(parameters) -> float:
    """Compute the L2 norm across all parameter gradients."""

    total_sq = 0.0
    found_grad = False
    for parameter in parameters:
        if parameter.grad is None:
            continue
        grad = parameter.grad.detach()
        if grad.is_sparse:
            grad = grad.coalesce().values()
        norm = float(torch.linalg.vector_norm(grad).item())
        total_sq += norm * norm
        found_grad = True
    if not found_grad:
        return 0.0
    return math.sqrt(total_sq)


def normalize_grad_norm_value(value: object, *, fallback: float) -> float:
    """Normalize a grad-norm return value to one finite float."""

    if value is None:
        return float(fallback)
    if isinstance(value, torch.Tensor):
        value_f = float(value.detach().item())
        return value_f if math.isfinite(value_f) else float(fallback)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        value_f = float(value)
        return value_f if math.isfinite(value_f) else float(fallback)
    return float(fallback)


def gradient_module_map(model: nn.Module) -> dict[str, nn.Module]:
    """Resolve the stable module names used in per-step gradient telemetry."""

    modules: dict[str, nn.Module] = {}
    for name in _TOP_LEVEL_GRADIENT_MODULES:
        raw = getattr(model, name, None)
        if isinstance(raw, nn.Module):
            modules[name] = raw
    raw_blocks = getattr(model, "transformer_blocks", None)
    if isinstance(raw_blocks, nn.ModuleList):
        for index, block in enumerate(raw_blocks):
            modules[f"transformer_blocks.{index}"] = block
    return modules


def module_grad_norms(model: nn.Module) -> dict[str, float]:
    """Compute module-level gradient norms for the active model surface."""

    return {
        name: float(total_grad_norm(module.parameters()))
        for name, module in gradient_module_map(model).items()
    }


def update_loss_ema(
    train_loss: float,
    *,
    previous_ema: float | None,
    alpha: float = LOSS_EMA_ALPHA,
) -> float:
    """Update the exponentially weighted moving average for train loss."""

    loss_value = float(train_loss)
    if previous_ema is None:
        return loss_value
    resolved_alpha = float(alpha)
    if not 0.0 < resolved_alpha <= 1.0:
        raise ValueError(f"loss ema alpha must be in (0, 1], got {resolved_alpha}")
    return resolved_alpha * loss_value + (1.0 - resolved_alpha) * float(previous_ema)


def train_loss_delta(train_loss: float, *, previous_train_loss: float | None) -> float | None:
    """Compute the additive train-loss delta from the previous step."""

    if previous_train_loss is None:
        return None
    return float(train_loss) - float(previous_train_loss)


def history_loss_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, float | int | None]:
    """Summarize train-loss volatility from history-style records."""

    weighted_losses: list[tuple[float, float]] = []
    losses: list[float] = []
    for record in records:
        raw_loss = record.get("train_loss")
        if raw_loss is None:
            continue
        loss_value = float(raw_loss)
        if not math.isfinite(loss_value):
            continue
        raw_weight = record.get("task_batch_size_actual")
        weight = 1.0
        if raw_weight is not None:
            try:
                resolved_weight = int(raw_weight)
            except (TypeError, ValueError):
                resolved_weight = 1
            if resolved_weight > 0:
                weight = float(resolved_weight)
        losses.append(loss_value)
        weighted_losses.append((loss_value, weight))
    deltas = [
        float(record["train_loss_delta"])
        for record in records
        if record.get("train_loss_delta") is not None
        and math.isfinite(float(record["train_loss_delta"]))
    ]
    if not losses:
        return {
            "record_count": int(len(records)),
            "initial_train_loss": None,
            "final_train_loss": None,
            "min_train_loss": None,
            "max_train_loss": None,
            "mean_train_loss": None,
            "train_loss_variance": None,
            "max_abs_train_loss_delta": None,
        }
    total_weight = sum(weight for _loss, weight in weighted_losses)
    mean_loss = (
        sum(loss_value * weight for loss_value, weight in weighted_losses) / float(total_weight)
        if total_weight > 0.0
        else sum(losses) / float(len(losses))
    )
    variance = (
        sum(weight * ((loss_value - mean_loss) ** 2) for loss_value, weight in weighted_losses)
        / float(total_weight)
        if total_weight > 0.0 and len(losses) > 1
        else 0.0
    )
    return {
        "record_count": int(len(records)),
        "initial_train_loss": float(losses[0]),
        "final_train_loss": float(losses[-1]),
        "min_train_loss": float(min(losses)),
        "max_train_loss": float(max(losses)),
        "mean_train_loss": float(mean_loss),
        "train_loss_variance": float(variance),
        "max_abs_train_loss_delta": None if not deltas else float(max(abs(delta) for delta in deltas)),
    }


def _mapping_value_history(
    records: Sequence[Mapping[str, Any]],
    *,
    key: str,
) -> dict[str, list[float]]:
    history: dict[str, list[float]] = {}
    for record in records:
        raw_values = record.get(key)
        if not isinstance(raw_values, Mapping):
            continue
        for name, value in raw_values.items():
            value_f = float(value)
            if not math.isfinite(value_f):
                continue
            history.setdefault(str(name), []).append(value_f)
    return history


def _record_global_grad_norm_kind(record: Mapping[str, Any]) -> str | None:
    raw_kind = record.get("global_grad_norm_kind")
    if isinstance(raw_kind, str):
        normalized_kind = raw_kind.strip()
        if normalized_kind in _GLOBAL_GRAD_NORM_KINDS:
            return normalized_kind

    raw_value = record.get("global_grad_norm")
    if raw_value is None:
        return None
    value = float(raw_value)
    if math.isnan(value):
        return "nan"
    if math.isinf(value):
        return "pos_inf" if value > 0.0 else "neg_inf"
    if math.isfinite(value):
        return "finite"
    return None


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / float(len(values)))


def grad_norm_summary_from_values(values: Sequence[float]) -> dict[str, float | None]:
    """Summarize one finite grad-norm history with nullable outputs."""

    return {
        "mean_grad_norm": _mean_or_none(values),
        "max_grad_norm": None if not values else float(max(values)),
        "final_grad_norm": None if not values else float(values[-1]),
    }


def grad_norm_summary_from_running_totals(
    *,
    grad_norm_sum: float,
    grad_norm_count: int,
    max_grad_norm: float,
    final_grad_norm: float,
) -> dict[str, float | None]:
    """Summarize grad norms from running totals while preserving missing state."""

    if grad_norm_count <= 0:
        return grad_norm_summary_from_values(())
    return {
        "mean_grad_norm": float(grad_norm_sum / float(grad_norm_count)),
        "max_grad_norm": float(max_grad_norm),
        "final_grad_norm": float(final_grad_norm),
    }


def _linear_slope_or_none(points: Sequence[tuple[float, float]]) -> float | None:
    if len(points) < _MIN_LINEAR_SLOPE_POINTS:
        return None
    xs = [float(x_value) for x_value, _ in points]
    ys = [float(y_value) for _, y_value in points]
    mean_x = sum(xs) / float(len(xs))
    mean_y = sum(ys) / float(len(ys))
    denom = sum((x_value - mean_x) ** 2 for x_value in xs)
    if denom == 0.0:
        return None
    numer = sum((x_value - mean_x) * (y_value - mean_y) for x_value, y_value in zip(xs, ys, strict=True))
    return float(numer / denom)


def _ratio_or_none(numerator: float | None, denominator: float | None) -> float | None:
    if numerator is None or denominator is None or denominator == 0.0:
        return None
    return float(numerator / denominator)


def _sorted_records(records: Sequence[Mapping[str, Any]]) -> list[Mapping[str, Any]]:
    return sorted(records, key=lambda record: int(record.get("step", 0)))


def _warmup_end_step(training_surface_record: Mapping[str, Any] | None) -> int:
    if not isinstance(training_surface_record, Mapping):
        return 0
    raw_training = training_surface_record.get("training")
    if not isinstance(raw_training, Mapping):
        return 0
    raw_stages = raw_training.get("schedule_stages")
    if not isinstance(raw_stages, list) or not raw_stages:
        return 0
    first_stage = raw_stages[0]
    if not isinstance(first_stage, Mapping):
        return 0
    steps_raw = first_stage.get("steps")
    warmup_ratio_raw = first_stage.get("warmup_ratio", 0.0)
    if not isinstance(steps_raw, int):
        return 0
    if not isinstance(warmup_ratio_raw, (int, float)):
        return 0
    steps = int(steps_raw)
    warmup_ratio = float(warmup_ratio_raw)
    if steps <= 1 or warmup_ratio <= 0.0:
        return 0
    return min(steps - 1, max(1, int(math.ceil(float(steps) * warmup_ratio))))


def _windowed_gradient_records(
    records: Sequence[Mapping[str, Any]],
    *,
    warmup_end_step: int,
) -> dict[str, list[Mapping[str, Any]]]:
    ordered = _sorted_records(records)
    early = [
        record
        for record in ordered
        if 1 <= int(record.get("step", 0)) <= _EARLY_WINDOW_MAX_STEP
    ]
    post_warmup = [
        record for record in ordered if int(record.get("step", 0)) > warmup_end_step
    ][:_POST_WARMUP_WINDOW_MAX_RECORDS]
    final_count = (
        0
        if not ordered
        else max(1, int(math.ceil(float(len(ordered)) * _FINAL_WINDOW_FRACTION)))
    )
    final_window = [] if final_count <= 0 else ordered[-final_count:]
    return {
        _WINDOW_EARLY: early,
        _WINDOW_POST_WARMUP: post_warmup,
        _WINDOW_FINAL: final_window,
    }


def _module_balance_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    warmup_end_step: int,
) -> dict[str, Any]:
    windows = _windowed_gradient_records(records, warmup_end_step=warmup_end_step)
    window_summaries: dict[str, Any] = {}
    for window_name, window_records in windows.items():
        feature_values: list[float] = []
        head_values: list[float] = []
        for record in window_records:
            raw_modules = record.get("module_grad_norms")
            if not isinstance(raw_modules, Mapping):
                continue
            feature_raw = raw_modules.get("feature_encoder")
            head_raw = raw_modules.get("direct_head")
            if feature_raw is None or head_raw is None:
                continue
            feature_value = float(feature_raw)
            head_value = float(head_raw)
            if not math.isfinite(feature_value) or not math.isfinite(head_value):
                continue
            feature_values.append(feature_value)
            head_values.append(head_value)
        feature_mean = _mean_or_none(feature_values)
        head_mean = _mean_or_none(head_values)
        window_summaries[window_name] = {
            "record_count": int(len(window_records)),
            "paired_record_count": int(len(feature_values)),
            "feature_encoder_mean_grad_norm": feature_mean,
            "direct_head_mean_grad_norm": head_mean,
            "feature_encoder_to_direct_head_mean_ratio": _ratio_or_none(feature_mean, head_mean),
            "direct_head_to_feature_encoder_mean_ratio": _ratio_or_none(head_mean, feature_mean),
        }
    return {
        "warmup_end_step": int(warmup_end_step),
        "windows": window_summaries,
    }


def _module_window_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    warmup_end_step: int,
    module_names: Sequence[str],
) -> dict[str, Any]:
    windows = _windowed_gradient_records(records, warmup_end_step=warmup_end_step)
    modules: dict[str, Any] = {}
    for module_name in module_names:
        window_summaries: dict[str, Any] = {}
        for window_name, window_records in windows.items():
            values: list[float] = []
            for record in window_records:
                raw_modules = record.get("module_grad_norms")
                if not isinstance(raw_modules, Mapping):
                    continue
                raw_value = raw_modules.get(module_name)
                if raw_value is None:
                    continue
                value = float(raw_value)
                if math.isfinite(value):
                    values.append(value)
            window_summaries[window_name] = {
                "record_count": int(len(values)),
                **grad_norm_summary_from_values(values),
            }
        modules[module_name] = {
            "windows": window_summaries,
        }
    return {
        "warmup_end_step": int(warmup_end_step),
        "modules": modules,
    }


def _stage_local_gradient_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    warmup_end_step: int,
) -> dict[str, Any]:
    return _module_window_summary(
        records,
        warmup_end_step=warmup_end_step,
        module_names=_STAGE_LOCAL_GRADIENT_MODULES,
    )


def _activation_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    warmup_end_step: int,
) -> dict[str, Any]:
    windows = _windowed_gradient_records(records, warmup_end_step=warmup_end_step)
    tracked: dict[str, Any] = {}
    for activation_name in _TRACKED_ACTIVATIONS:
        window_summaries: dict[str, Any] = {}
        for window_name, window_records in windows.items():
            values: list[float] = []
            for record in window_records:
                raw_activations = record.get("activation_norms")
                if not isinstance(raw_activations, Mapping):
                    continue
                raw_value = raw_activations.get(activation_name)
                if raw_value is None:
                    continue
                value = float(raw_value)
                if math.isfinite(value):
                    values.append(value)
            mean_value = _mean_or_none(values)
            final_value = None if not values else float(values[-1])
            window_summaries[window_name] = {
                "record_count": int(len(values)),
                "mean": mean_value,
                "max": None if not values else float(max(values)),
                "final": final_value,
            }
        early_mean = window_summaries[_WINDOW_EARLY]["mean"]
        final_mean = window_summaries[_WINDOW_FINAL]["mean"]
        tracked[activation_name] = {
            "windows": window_summaries,
            "early_to_final_mean_delta": None
            if early_mean is None or final_mean is None
            else float(final_mean - early_mean),
            "early_to_final_mean_ratio": _ratio_or_none(final_mean, early_mean),
        }
    return {
        "warmup_end_step": int(warmup_end_step),
        "tracked_activations": tracked,
        "upper_transformer_blocks": _upper_transformer_block_summary(
            records,
            warmup_end_step=warmup_end_step,
        ),
    }


def _upper_transformer_block_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    warmup_end_step: int,
) -> dict[str, Any]:
    ordered = _sorted_records(records)
    windows = _windowed_gradient_records(ordered, warmup_end_step=warmup_end_step)
    activation_history = _mapping_value_history(ordered, key="activation_norms")
    transformer_block_names = [
        activation_name
        for activation_name in activation_history
        if activation_name.startswith("post_transformer_block_")
    ]
    selected_names = [
        activation_name
        for activation_name in sorted(
            transformer_block_names,
            key=lambda name: int(name.removeprefix("post_transformer_block_")),
        )
        if _UPPER_BLOCK_START
        <= int(activation_name.removeprefix("post_transformer_block_"))
        <= _UPPER_BLOCK_END
    ]
    if not selected_names:
        return {
            "block_names": [],
            "aggregate": {
                "final_window_mean": None,
                "post_warmup_mean_slope": None,
            },
            "blocks": {},
        }

    blocks: dict[str, Any] = {}
    final_window_means: list[float] = []
    post_warmup_slopes: list[float] = []
    post_warmup_records = windows[_WINDOW_POST_WARMUP]
    final_records = windows[_WINDOW_FINAL]
    for block_name in selected_names:
        final_points: list[float] = []
        slope_points: list[tuple[float, float]] = []
        for record in final_records:
            raw_activations = record.get("activation_norms")
            if not isinstance(raw_activations, Mapping):
                continue
            raw_value = raw_activations.get(block_name)
            if raw_value is None:
                continue
            value = float(raw_value)
            if math.isfinite(value):
                final_points.append(value)
        for record in post_warmup_records:
            raw_activations = record.get("activation_norms")
            if not isinstance(raw_activations, Mapping):
                continue
            raw_value = raw_activations.get(block_name)
            if raw_value is None:
                continue
            value = float(raw_value)
            if not math.isfinite(value):
                continue
            slope_points.append((float(record.get("step", 0)), value))
        final_window_mean = _mean_or_none(final_points)
        post_warmup_slope = _linear_slope_or_none(slope_points)
        if final_window_mean is not None:
            final_window_means.append(final_window_mean)
        if post_warmup_slope is not None:
            post_warmup_slopes.append(post_warmup_slope)
        blocks[block_name] = {
            "final_window_mean": final_window_mean,
            "post_warmup_slope": post_warmup_slope,
            "final_window_record_count": int(len(final_points)),
            "post_warmup_record_count": int(len(slope_points)),
        }

    return {
        "block_names": selected_names,
        "aggregate": {
            "final_window_mean": _mean_or_none(final_window_means),
            "post_warmup_mean_slope": _mean_or_none(post_warmup_slopes),
        },
        "blocks": blocks,
    }


def diagnostics_summary(
    records: Sequence[Mapping[str, Any]],
    *,
    training_surface_record: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Summarize clipped-step, module-balance, and activation-window diagnostics."""

    ordered = _sorted_records(records)
    clipped_step_count = sum(1 for record in ordered if bool(record.get("grad_clip_triggered", False)))
    warmup_end_step = _warmup_end_step(training_surface_record)
    window_records = _windowed_gradient_records(ordered, warmup_end_step=warmup_end_step)
    return {
        "windowing": {
            "warmup_end_step": int(warmup_end_step),
            "window_record_counts": {
                window_name: int(len(window)) for window_name, window in window_records.items()
            },
        },
        "task_batching": _task_batching_summary(ordered),
        "grad_clip": {
            "record_count": int(len(ordered)),
            "clipped_step_count": int(clipped_step_count),
            "clipped_step_fraction": 0.0
            if not ordered
            else float(clipped_step_count / float(len(ordered))),
        },
        "stage_local_gradients": _stage_local_gradient_summary(
            ordered,
            warmup_end_step=warmup_end_step,
        ),
        "module_balance": {
            "feature_encoder_vs_direct_head": _module_balance_summary(
                ordered,
                warmup_end_step=warmup_end_step,
            )
        },
        "activation_windows": _activation_summary(
            ordered,
            warmup_end_step=warmup_end_step,
        ),
    }


def _task_batching_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    requested_sizes: set[int] = set()
    actual_batch_sizes: dict[str, int] = {}
    signature_counts: dict[str, int] = {}
    batched_step_count = 0
    singleton_fallback_count = 0
    task_batch_microstep_count = 0
    for record in records:
        raw_requested = record.get("task_batch_size_requested")
        if raw_requested is not None:
            requested_sizes.add(int(raw_requested))
        raw_actual = record.get("task_batch_size_actual")
        if raw_actual is not None:
            actual_batch_sizes[str(int(raw_actual))] = actual_batch_sizes.get(str(int(raw_actual)), 0) + 1
        raw_batched_count = record.get("task_batch_batched_count")
        batched_count = (
            int(raw_batched_count)
            if raw_batched_count is not None
            else (1 if raw_actual is not None and int(raw_actual) > 1 else 0)
        )
        if batched_count > 0:
            batched_step_count += 1
        task_batch_microstep_count += max(batched_count, 0)
        raw_fallback_count = record.get("task_batch_singleton_fallback_count")
        if raw_fallback_count is not None:
            fallback_count = int(raw_fallback_count)
            singleton_fallback_count += fallback_count
            task_batch_microstep_count += max(fallback_count, 0)
        raw_signature_counts = record.get("task_batch_signature_counts")
        if isinstance(raw_signature_counts, Mapping):
            for signature, count in raw_signature_counts.items():
                signature_counts[str(signature)] = signature_counts.get(str(signature), 0) + int(count)
    record_count = int(len(records))
    fraction_denominator = task_batch_microstep_count if task_batch_microstep_count > 0 else record_count
    return {
        "record_count": record_count,
        "requested_task_batch_sizes": sorted(requested_sizes),
        "actual_task_batch_size_counts": actual_batch_sizes,
        "batched_step_count": int(batched_step_count),
        "singleton_fallback_count": int(singleton_fallback_count),
        "singleton_fallback_fraction": 0.0
        if fraction_denominator <= 0
        else float(singleton_fallback_count / float(fraction_denominator)),
        "signature_counts": signature_counts,
    }


def gradient_trace_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize global and module-level gradients from gradient-history records."""

    global_grad_norms = [
        float(record["global_grad_norm"])
        for record in records
        if record.get("global_grad_norm") is not None
        and math.isfinite(float(record["global_grad_norm"]))
    ]
    non_finite_global_grad_norm_counts = {kind: 0 for kind in _GLOBAL_GRAD_NORM_KINDS if kind != "finite"}
    for record in records:
        kind = _record_global_grad_norm_kind(record)
        if kind in non_finite_global_grad_norm_counts:
            non_finite_global_grad_norm_counts[kind] += 1

    module_history = _mapping_value_history(records, key="module_grad_norms")
    activation_history = _mapping_value_history(records, key="activation_norms")

    return {
        "record_count": int(len(records)),
        "global": grad_norm_summary_from_values(global_grad_norms),
        "non_finite_global_grad_norm_counts": non_finite_global_grad_norm_counts,
        "final_global_grad_norm_kind": None
        if not records
        else _record_global_grad_norm_kind(records[-1]),
        "modules": {
            name: grad_norm_summary_from_values(values)
            for name, values in sorted(module_history.items())
        },
        "activations": {
            name: {
                "mean": float(sum(values) / float(len(values))),
                "max": float(max(values)),
                "final": float(values[-1]),
            }
            for name, values in sorted(activation_history.items())
        },
    }


def _normalize_payload_values(payload: Mapping[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for key, value in payload.items():
        if isinstance(value, Path):
            normalized[str(key)] = str(value.expanduser().resolve())
        elif isinstance(value, dict):
            normalized[str(key)] = _normalize_payload_values(value)
        elif isinstance(value, list):
            normalized[str(key)] = [
                str(item.expanduser().resolve()) if isinstance(item, Path) else item for item in value
            ]
        else:
            normalized[str(key)] = value
    return normalized


def _training_surface_data_payload(
    training_surface_record: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    if not isinstance(training_surface_record, Mapping):
        return None
    raw_data = training_surface_record.get("data")
    if not isinstance(raw_data, Mapping):
        return None
    return raw_data


def _manifest_characteristics_payload(
    training_surface_record: Mapping[str, Any] | None,
) -> Mapping[str, Any] | None:
    data_payload = _training_surface_data_payload(training_surface_record)
    if not isinstance(data_payload, Mapping):
        return None
    raw_manifest = data_payload.get("manifest")
    if not isinstance(raw_manifest, Mapping):
        return None
    raw_characteristics = raw_manifest.get("characteristics")
    if not isinstance(raw_characteristics, Mapping):
        return None
    return raw_characteristics


def _curriculum_summary(
    training_surface_record: Mapping[str, Any] | None,
) -> tuple[str | None, dict[str, Any] | None]:
    data_payload = _training_surface_data_payload(training_surface_record)
    if data_payload is None:
        return None, None
    raw_provenance = data_payload.get("dagzoo_provenance")
    if not isinstance(raw_provenance, Mapping):
        corpus_variant = data_payload.get("surface_label")
        return (
            None if corpus_variant is None else str(corpus_variant),
            None,
        )

    invocations_payload = raw_provenance.get("invocations")
    invocations = invocations_payload if isinstance(invocations_payload, list) else []
    generate_run_ids: list[str] = []
    invocation_mix: list[dict[str, Any]] = []
    source_families: set[str] = set()
    for raw_invocation in invocations:
        if not isinstance(raw_invocation, Mapping):
            continue
        raw_handoff = raw_invocation.get("handoff")
        handoff = raw_handoff if isinstance(raw_handoff, Mapping) else {}
        generate_run_id = handoff.get("generate_run_id")
        if isinstance(generate_run_id, str) and generate_run_id.strip():
            generate_run_ids.append(str(generate_run_id))
        source_family = handoff.get("source_family")
        if isinstance(source_family, str) and source_family.strip():
            source_families.add(str(source_family))
        invocation_mix.append(
            {
                "invocation_id": raw_invocation.get("invocation_id"),
                "requested_config_ref": raw_invocation.get("requested_config_ref"),
                "num_datasets": raw_invocation.get("num_datasets"),
                "rows": raw_invocation.get("rows"),
                "source_family": source_family,
                "generate_run_id": generate_run_id,
                "generated_corpus_id": handoff.get("generated_corpus_id"),
            }
        )
    corpus_variant = raw_provenance.get("corpus_variant")
    curriculum_id: str | None = None
    unique_generate_run_ids = sorted({run_id for run_id in generate_run_ids if run_id})
    if len(unique_generate_run_ids) == 1:
        curriculum_id = unique_generate_run_ids[0]
    elif isinstance(corpus_variant, str) and corpus_variant.strip():
        curriculum_id = str(corpus_variant)
    else:
        raw_recipe_id = data_payload.get("recipe_id")
        if isinstance(raw_recipe_id, str) and raw_recipe_id.strip():
            curriculum_id = str(raw_recipe_id)
    summary = {
        "corpus_variant": raw_provenance.get("corpus_variant"),
        "config_refs": raw_provenance.get("config_refs"),
        "source_families": sorted(source_families),
        "invocation_mix": invocation_mix,
    }
    return curriculum_id, summary


def _scm_complexity_summary(
    training_surface_record: Mapping[str, Any] | None,
) -> dict[str, Any] | None:
    characteristics = _manifest_characteristics_payload(training_surface_record)
    data_payload = _training_surface_data_payload(training_surface_record)
    raw_provenance = (
        data_payload.get("dagzoo_provenance")
        if isinstance(data_payload, Mapping)
        else None
    )
    summary: dict[str, Any] = {}
    if isinstance(characteristics, Mapping):
        for key in (
            "row_count_distribution",
            "feature_count_distribution",
            "class_count_distribution",
        ):
            if key in characteristics:
                summary[key] = characteristics[key]
    if isinstance(raw_provenance, Mapping):
        invocations_payload = raw_provenance.get("invocations")
        invocations = invocations_payload if isinstance(invocations_payload, list) else []
        source_families = sorted(
            {
                str(handoff.get("source_family"))
                for raw_invocation in invocations
                if isinstance(raw_invocation, Mapping)
                and isinstance(
                    (handoff := raw_invocation.get("handoff")),
                    Mapping,
                )
                and isinstance(handoff.get("source_family"), str)
                and str(handoff.get("source_family")).strip()
            }
        )
        if source_families:
            summary["source_families"] = source_families
        rows = [
            int(raw_invocation["rows"])
            for raw_invocation in invocations
            if isinstance(raw_invocation, Mapping)
            and raw_invocation.get("rows") is not None
        ]
        if rows:
            summary["curriculum_rows"] = {
                "min": int(min(rows)),
                "max": int(max(rows)),
                "count": int(len(rows)),
            }
    return summary or None


def build_runtime_summary(
    *,
    train_elapsed_seconds: float,
    wall_elapsed_seconds: float,
    examples_seen: int,
    tokens_seen: int,
    peak_memory_summary: Mapping[str, int | None] | None,
) -> dict[str, Any]:
    """Build runtime telemetry derived from loop counters."""

    train_elapsed = float(train_elapsed_seconds)
    wall_elapsed = float(wall_elapsed_seconds)
    throughput_examples = None
    throughput_tokens = None
    if train_elapsed > 0.0:
        throughput_examples = float(examples_seen) / train_elapsed
        throughput_tokens = float(tokens_seen) / train_elapsed
    peak_allocated = None
    peak_reserved = None
    if isinstance(peak_memory_summary, Mapping):
        raw_allocated = peak_memory_summary.get("peak_vram_allocated")
        raw_reserved = peak_memory_summary.get("peak_vram_reserved")
        peak_allocated = None if raw_allocated is None else int(raw_allocated)
        peak_reserved = None if raw_reserved is None else int(raw_reserved)
    return {
        "peak_vram_allocated": peak_allocated,
        "peak_vram_reserved": peak_reserved,
        "throughput_examples_per_second": throughput_examples,
        "throughput_tokens_per_second": throughput_tokens,
        "non_train_overhead_seconds": max(0.0, wall_elapsed - train_elapsed),
    }


def build_regime_budget_summary(
    *,
    task: str | None,
    loss_surface: str | None,
    training_surface_record: Mapping[str, Any] | None,
    global_step: int,
    tokens_seen: int,
) -> dict[str, Any]:
    """Build data-regime budget metadata from persisted surface context."""

    characteristics = _manifest_characteristics_payload(training_surface_record)
    split_counts = (
        characteristics.get("split_counts")
        if isinstance(characteristics, Mapping)
        else None
    )
    unique_task_budget = None
    if isinstance(split_counts, Mapping) and split_counts.get("train") is not None:
        unique_task_budget = int(split_counts["train"])
    elif isinstance(characteristics, Mapping) and characteristics.get("record_count") is not None:
        unique_task_budget = int(characteristics["record_count"])
    curriculum_id, curriculum_mix = _curriculum_summary(training_surface_record)
    objective_metric = objective_metric_for_task(task, loss_surface=loss_surface)
    tokens_per_step = None
    if int(global_step) > 0:
        tokens_per_step = float(tokens_seen) / float(global_step)
    return {
        "tokens_per_step": tokens_per_step,
        "tokens_seen": int(tokens_seen),
        "token_budget": int(tokens_seen),
        "unique_task_budget": unique_task_budget,
        "objective_metric": objective_metric,
        "curriculum_id": curriculum_id,
        "curriculum_mix": curriculum_mix,
        "scm_complexity_summary": _scm_complexity_summary(training_surface_record),
    }


def build_training_telemetry(
    *,
    run_dir: Path,
    success: bool,
    artifacts: Mapping[str, Any],
    checkpoint_snapshots: Sequence[Mapping[str, Any]],
    history_records: Sequence[Mapping[str, Any]],
    gradient_records: Sequence[Mapping[str, Any]],
    task: str | None = None,
    global_step: int | None = None,
    runtime_summary: Mapping[str, Any] | None = None,
    regime_budget: Mapping[str, Any] | None = None,
    missingness: Mapping[str, Any] | None = None,
    training_surface_record: Mapping[str, Any] | None = None,
    wandb: Mapping[str, Any] | None = None,
    error: BaseException | None = None,
) -> dict[str, Any]:
    """Build the canonical training telemetry payload for one run."""

    training_surface_context = None
    if training_surface_record is not None:
        labels = training_surface_record.get("labels")
        manifest = None
        raw_data = training_surface_record.get("data")
        if isinstance(raw_data, Mapping):
            raw_manifest = raw_data.get("manifest")
            if isinstance(raw_manifest, Mapping):
                raw_characteristics = raw_manifest.get("characteristics")
                if isinstance(raw_characteristics, Mapping):
                    manifest = {
                        "missing_value_policy": raw_characteristics.get("missing_value_policy"),
                        "missing_value_status_counts": raw_characteristics.get(
                            "missing_value_status_counts"
                        ),
                        "all_records_no_missing": raw_characteristics.get("all_records_no_missing"),
                    }
        training_surface_context = {
            "labels": dict(labels) if isinstance(labels, Mapping) else None,
            "manifest_missingness": manifest,
        }

    payload: dict[str, Any] = {
        "schema": TRAINING_TELEMETRY_SCHEMA,
        "generated_at_utc": _utc_now(),
        "success": bool(success),
        "run_dir": str(run_dir.expanduser().resolve()),
        "task": None if task is None else str(task),
        "global_step": None if global_step is None else int(global_step),
        "artifacts": _normalize_payload_values(artifacts),
        "checkpoint_snapshots": [
            _normalize_payload_values(snapshot) for snapshot in checkpoint_snapshots
        ],
        "loss_summary": history_loss_summary(history_records),
        "gradient_summary": gradient_trace_summary(gradient_records),
        "diagnostics": diagnostics_summary(
            gradient_records,
            training_surface_record=training_surface_record,
        ),
        "runtime_summary": (
            None if runtime_summary is None else _normalize_payload_values(runtime_summary)
        ),
        "regime_budget": (
            None if regime_budget is None else _normalize_payload_values(regime_budget)
        ),
        "missingness": None if missingness is None else _normalize_payload_values(missingness),
        "training_surface_context": training_surface_context,
        "wandb": None if wandb is None else _normalize_payload_values(wandb),
    }
    if error is not None:
        payload["error"] = {"type": type(error).__name__, "message": str(error)}
    return payload


def write_training_telemetry(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write one training telemetry payload with stable formatting."""

    resolved_path = path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return resolved_path
