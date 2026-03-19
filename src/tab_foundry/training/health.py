"""Run-health summaries for training-style artifacts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, cast

from .instability import (
    _UPPER_BLOCK_END,
    _UPPER_BLOCK_START,
    _mean_or_none,
    _warmup_end_step,
    _windowed_gradient_records,
    build_training_telemetry,
    gradient_history_path,
    telemetry_path,
)


_WARN_CLIPPED_STEP_FRACTION = 0.05
_FAIL_CLIPPED_STEP_FRACTION = 0.20
_WARN_UPPER_BLOCK_SLOPE = 0.02
_FAIL_UPPER_BLOCK_SLOPE = 0.10
_FAIL_UPPER_BLOCK_FINAL_TO_EARLY_RATIO = 2.0
_WARN_FINAL_TRAIN_LOSS_MULTIPLIER = 1.05


def _read_json_mapping(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected a JSON object at {path}")
    return cast(dict[str, Any], payload)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(cast(dict[str, Any], payload))
    return records


def _optional_finite_float(value: Any, *, context: str) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RuntimeError(f"{context} must be finite when present")
    return numeric


def _upper_block_final_to_early_ratio(
    gradient_records: list[dict[str, Any]],
    *,
    training_surface_record: Mapping[str, Any] | None,
) -> float | None:
    warmup_end_step = _warmup_end_step(training_surface_record)
    windows = _windowed_gradient_records(gradient_records, warmup_end_step=warmup_end_step)
    early_records = windows["early_1_25"]
    final_records = windows["final_10pct"]
    block_names = [
        f"post_transformer_block_{block_index}"
        for block_index in range(_UPPER_BLOCK_START, _UPPER_BLOCK_END + 1)
    ]
    early_values: list[float] = []
    final_values: list[float] = []
    for record in early_records:
        raw_norms = record.get("activation_norms")
        if not isinstance(raw_norms, Mapping):
            continue
        for block_name in block_names:
            value = raw_norms.get(block_name)
            if value is None:
                continue
            numeric = float(value)
            if math.isfinite(numeric):
                early_values.append(numeric)
    for record in final_records:
        raw_norms = record.get("activation_norms")
        if not isinstance(raw_norms, Mapping):
            continue
        for block_name in block_names:
            value = raw_norms.get(block_name)
            if value is None:
                continue
            numeric = float(value)
            if math.isfinite(numeric):
                final_values.append(numeric)
    early_mean = _mean_or_none(early_values)
    final_mean = _mean_or_none(final_values)
    if early_mean is None or final_mean is None or early_mean == 0.0:
        return None
    return float(final_mean / early_mean)


def _rebuild_telemetry(
    *,
    run_dir: Path,
    history_records: list[dict[str, Any]],
    gradient_records: list[dict[str, Any]],
    training_surface_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    return build_training_telemetry(
        run_dir=run_dir,
        success=True,
        artifacts={
            "train_history_jsonl": str((run_dir / "train_history.jsonl").resolve()),
            "gradient_history_jsonl": str(gradient_history_path(run_dir)),
            "telemetry_json": str(telemetry_path(run_dir)),
            "training_surface_record_json": (
                None
                if training_surface_record is None
                else str((run_dir / "training_surface_record.json").resolve())
            ),
        },
        checkpoint_snapshots=[],
        history_records=history_records,
        gradient_records=gradient_records,
        training_surface_record=training_surface_record,
    )


def _metrics_from_telemetry(
    telemetry_payload: Mapping[str, Any],
    *,
    gradient_records: list[dict[str, Any]],
    training_surface_record: Mapping[str, Any] | None,
) -> dict[str, Any]:
    loss_summary = telemetry_payload.get("loss_summary")
    diagnostics = telemetry_payload.get("diagnostics")
    if not isinstance(loss_summary, Mapping):
        raise RuntimeError("telemetry payload is missing loss_summary")
    if not isinstance(diagnostics, Mapping):
        raise RuntimeError("telemetry payload is missing diagnostics")
    grad_clip = diagnostics.get("grad_clip")
    activation_windows = diagnostics.get("activation_windows")
    if not isinstance(grad_clip, Mapping):
        raise RuntimeError("telemetry payload is missing diagnostics.grad_clip")
    if not isinstance(activation_windows, Mapping):
        raise RuntimeError("telemetry payload is missing diagnostics.activation_windows")
    upper_blocks = activation_windows.get("upper_transformer_blocks")
    if not isinstance(upper_blocks, Mapping):
        raise RuntimeError("telemetry payload is missing diagnostics.activation_windows.upper_transformer_blocks")
    aggregate = upper_blocks.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise RuntimeError(
            "telemetry payload is missing diagnostics.activation_windows.upper_transformer_blocks.aggregate"
        )
    return {
        "initial_train_loss": _optional_finite_float(
            loss_summary.get("initial_train_loss"),
            context="loss_summary.initial_train_loss",
        ),
        "final_train_loss": _optional_finite_float(
            loss_summary.get("final_train_loss"),
            context="loss_summary.final_train_loss",
        ),
        "clipped_step_fraction": _optional_finite_float(
            grad_clip.get("clipped_step_fraction"),
            context="diagnostics.grad_clip.clipped_step_fraction",
        ),
        "upper_block_post_warmup_mean_slope": _optional_finite_float(
            aggregate.get("post_warmup_mean_slope"),
            context="diagnostics.activation_windows.upper_transformer_blocks.aggregate.post_warmup_mean_slope",
        ),
        "upper_block_final_window_mean": _optional_finite_float(
            aggregate.get("final_window_mean"),
            context="diagnostics.activation_windows.upper_transformer_blocks.aggregate.final_window_mean",
        ),
        "upper_block_final_to_early_ratio": _upper_block_final_to_early_ratio(
            gradient_records,
            training_surface_record=training_surface_record,
        ),
    }


def _summary_for_verdict(
    verdict: str,
    *,
    metrics: Mapping[str, Any],
    has_error: bool,
) -> str:
    if verdict == "fail":
        if has_error:
            return "recorded error present; inspect telemetry before trusting this run"
        return (
            "instability signals exceed fail thresholds "
            f"(clip={metrics['clipped_step_fraction']}, "
            f"slope={metrics['upper_block_post_warmup_mean_slope']}, "
            f"ratio={metrics['upper_block_final_to_early_ratio']})"
        )
    if verdict == "warn":
        return (
            "no fatal error recorded, but at least one instability heuristic is elevated "
            f"(clip={metrics['clipped_step_fraction']}, "
            f"slope={metrics['upper_block_post_warmup_mean_slope']}, "
            f"loss={metrics['initial_train_loss']}->{metrics['final_train_loss']})"
        )
    return (
        "gradient clipping, upper-block drift, and train-loss trajectory all remain within "
        "the current triage thresholds"
    )


def health_check(run_dir: Path) -> dict[str, Any]:
    """Summarize run telemetry into one triage verdict."""

    resolved_run_dir = run_dir.expanduser().resolve()
    telemetry_json_path = telemetry_path(resolved_run_dir)
    history_jsonl_path = resolved_run_dir / "train_history.jsonl"
    gradient_jsonl_path = gradient_history_path(resolved_run_dir)
    training_surface_record_path = resolved_run_dir / "training_surface_record.json"

    telemetry_payload = _read_json_mapping(telemetry_json_path)
    history_records = _read_jsonl(history_jsonl_path)
    gradient_records = _read_jsonl(gradient_jsonl_path)
    training_surface_record = _read_json_mapping(training_surface_record_path)

    if telemetry_payload is None:
        if not history_records or not gradient_records:
            raise RuntimeError(
                "health-check requires telemetry.json or both train_history.jsonl and "
                "gradient_history.jsonl under the selected run directory"
            )
        telemetry_payload = _rebuild_telemetry(
            run_dir=resolved_run_dir,
            history_records=history_records,
            gradient_records=gradient_records,
            training_surface_record=training_surface_record,
        )
        source = "reconstructed"
    else:
        source = "telemetry"

    metrics = _metrics_from_telemetry(
        telemetry_payload,
        gradient_records=gradient_records,
        training_surface_record=training_surface_record,
    )
    raw_error = telemetry_payload.get("error")
    has_error = bool(raw_error) or not bool(telemetry_payload.get("success", True))

    fail = has_error
    clipped_step_fraction = cast(float | None, metrics["clipped_step_fraction"])
    upper_block_slope = cast(float | None, metrics["upper_block_post_warmup_mean_slope"])
    upper_block_ratio = cast(float | None, metrics["upper_block_final_to_early_ratio"])
    initial_train_loss = cast(float | None, metrics["initial_train_loss"])
    final_train_loss = cast(float | None, metrics["final_train_loss"])

    if clipped_step_fraction is not None and clipped_step_fraction > _FAIL_CLIPPED_STEP_FRACTION:
        fail = True
    if upper_block_slope is not None and upper_block_slope > _FAIL_UPPER_BLOCK_SLOPE:
        fail = True
    if (
        upper_block_ratio is not None
        and upper_block_ratio > _FAIL_UPPER_BLOCK_FINAL_TO_EARLY_RATIO
    ):
        fail = True

    warn = False
    if not fail:
        if clipped_step_fraction is not None and clipped_step_fraction > _WARN_CLIPPED_STEP_FRACTION:
            warn = True
        if upper_block_slope is not None and upper_block_slope > _WARN_UPPER_BLOCK_SLOPE:
            warn = True
        if (
            initial_train_loss is not None
            and final_train_loss is not None
            and final_train_loss > initial_train_loss * _WARN_FINAL_TRAIN_LOSS_MULTIPLIER
        ):
            warn = True

    verdict = "fail" if fail else "warn" if warn else "ok"
    return {
        "run_dir": str(resolved_run_dir),
        "source": source,
        "verdict": verdict,
        "summary": _summary_for_verdict(verdict, metrics=metrics, has_error=has_error),
        "metrics": metrics,
        "telemetry_error": raw_error,
    }
