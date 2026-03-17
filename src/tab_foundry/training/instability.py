"""Instability telemetry helpers for training-style runs."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from torch import nn


LOSS_EMA_ALPHA = 0.1
TRAINING_TELEMETRY_SCHEMA = "tab-foundry-training-telemetry-v1"
_WINDOW_EARLY = "early_1_25"
_WINDOW_POST_WARMUP = "post_warmup_100"
_WINDOW_FINAL = "final_10pct"
_TRACKED_ACTIVATIONS = ("post_feature_encoder", "pre_transformer")

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


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def gradient_history_path(output_dir: Path) -> Path:
    """Return the canonical module-gradient history path for one run."""

    return output_dir.expanduser().resolve() / "gradient_history.jsonl"


def telemetry_path(output_dir: Path) -> Path:
    """Return the canonical telemetry path for one run."""

    return output_dir.expanduser().resolve() / "telemetry.json"


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

    losses = [
        float(record["train_loss"])
        for record in records
        if record.get("train_loss") is not None and math.isfinite(float(record["train_loss"]))
    ]
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
    mean_loss = sum(losses) / float(len(losses))
    variance = (
        sum((loss_value - mean_loss) ** 2 for loss_value in losses) / float(len(losses))
        if len(losses) > 1
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


def _mean_or_none(values: Sequence[float]) -> float | None:
    if not values:
        return None
    return float(sum(values) / float(len(values)))


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
    early = [record for record in ordered if 1 <= int(record.get("step", 0)) <= 25]
    post_warmup = [record for record in ordered if int(record.get("step", 0)) > warmup_end_step][:100]
    final_count = 0 if not ordered else max(1, int(math.ceil(float(len(ordered)) * 0.1)))
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
        "grad_clip": {
            "record_count": int(len(ordered)),
            "clipped_step_count": int(clipped_step_count),
            "clipped_step_fraction": 0.0
            if not ordered
            else float(clipped_step_count / float(len(ordered))),
        },
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


def gradient_trace_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize global and module-level gradients from gradient-history records."""

    global_grad_norms = [
        float(record["global_grad_norm"])
        for record in records
        if record.get("global_grad_norm") is not None
        and math.isfinite(float(record["global_grad_norm"]))
    ]
    module_history = _mapping_value_history(records, key="module_grad_norms")
    activation_history = _mapping_value_history(records, key="activation_norms")

    return {
        "record_count": int(len(records)),
        "global": {
            "mean_grad_norm": None
            if not global_grad_norms
            else float(sum(global_grad_norms) / float(len(global_grad_norms))),
            "max_grad_norm": None if not global_grad_norms else float(max(global_grad_norms)),
            "final_grad_norm": None if not global_grad_norms else float(global_grad_norms[-1]),
        },
        "modules": {
            name: {
                "mean_grad_norm": float(sum(values) / float(len(values))),
                "max_grad_norm": float(max(values)),
                "final_grad_norm": float(values[-1]),
            }
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


def build_training_telemetry(
    *,
    run_dir: Path,
    success: bool,
    artifacts: Mapping[str, Any],
    checkpoint_snapshots: Sequence[Mapping[str, Any]],
    history_records: Sequence[Mapping[str, Any]],
    gradient_records: Sequence[Mapping[str, Any]],
    missingness: Mapping[str, Any] | None = None,
    training_surface_record: Mapping[str, Any] | None = None,
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
        "missingness": None if missingness is None else _normalize_payload_values(missingness),
        "training_surface_context": training_surface_context,
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
