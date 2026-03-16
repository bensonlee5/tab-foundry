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

_TOP_LEVEL_GRADIENT_MODULES = (
    "tokenizer",
    "feature_encoder",
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


def gradient_trace_summary(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Summarize global and module-level gradients from gradient-history records."""

    global_grad_norms = [
        float(record["global_grad_norm"])
        for record in records
        if record.get("global_grad_norm") is not None
        and math.isfinite(float(record["global_grad_norm"]))
    ]
    module_history: dict[str, list[float]] = {}
    for record in records:
        raw_module_grad_norms = record.get("module_grad_norms")
        if not isinstance(raw_module_grad_norms, Mapping):
            continue
        for name, value in raw_module_grad_norms.items():
            value_f = float(value)
            if not math.isfinite(value_f):
                continue
            module_history.setdefault(str(name), []).append(value_f)

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
