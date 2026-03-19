"""Screening helpers for train-only system-delta rows."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, cast

from .queue_updates import read_jsonl


PRIMARY_TOLERANCE = 0.05
SECONDARY_TOLERANCE = 0.10


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected mapping payload at {path}")
    return cast(dict[str, Any], payload)


def _relative_gap(left: float, right: float) -> float:
    scale = max(abs(left), abs(right), 1.0e-12)
    return float(abs(left - right) / scale)


def _optional_float(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RuntimeError(f"{key} must be finite when present")
    return numeric


def screen_metrics(*, run_dir: Path) -> dict[str, Any]:
    telemetry_payload = _read_json(run_dir / "telemetry.json")
    diagnostics = telemetry_payload.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        raise RuntimeError("telemetry.json omitted diagnostics payload")
    activation_windows = diagnostics.get("activation_windows")
    if not isinstance(activation_windows, Mapping):
        raise RuntimeError("telemetry.json omitted activation_windows diagnostics")
    upper_blocks = activation_windows.get("upper_transformer_blocks")
    if not isinstance(upper_blocks, Mapping):
        raise RuntimeError("telemetry.json omitted upper_transformer_blocks diagnostics")
    aggregate = upper_blocks.get("aggregate")
    if not isinstance(aggregate, Mapping):
        raise RuntimeError("telemetry.json omitted upper_transformer_blocks.aggregate")
    grad_clip = diagnostics.get("grad_clip")
    if not isinstance(grad_clip, Mapping):
        raise RuntimeError("telemetry.json omitted grad_clip diagnostics")

    history_records = read_jsonl(run_dir / "train_history.jsonl")
    final_train_loss_ema = None
    if history_records:
        final_record = history_records[-1]
        raw_loss_ema = final_record.get("train_loss_ema")
        if raw_loss_ema is not None:
            final_train_loss_ema = float(raw_loss_ema)
            if not math.isfinite(final_train_loss_ema):
                raise RuntimeError("train_history final train_loss_ema must be finite")

    block_names = upper_blocks.get("block_names", [])
    if not isinstance(block_names, list):
        raise RuntimeError("upper_transformer_blocks.block_names must be a list")
    block_payloads = upper_blocks.get("blocks", {})
    if not isinstance(block_payloads, Mapping):
        raise RuntimeError("upper_transformer_blocks.blocks must be a mapping")

    return {
        "upper_block_names": [str(name) for name in block_names],
        "upper_block_final_window_mean": _optional_float(aggregate, "final_window_mean"),
        "upper_block_post_warmup_mean_slope": _optional_float(aggregate, "post_warmup_mean_slope"),
        "clipped_step_fraction": _optional_float(cast(Mapping[str, Any], grad_clip), "clipped_step_fraction"),
        "final_train_loss_ema": final_train_loss_ema,
        "upper_blocks": {
            str(name): {
                "final_window_mean": _optional_float(cast(Mapping[str, Any], payload), "final_window_mean"),
                "post_warmup_slope": _optional_float(cast(Mapping[str, Any], payload), "post_warmup_slope"),
            }
            for name, payload in block_payloads.items()
            if isinstance(payload, Mapping)
        },
    }


def pick_screen_winner(
    *,
    candidates: list[dict[str, Any]],
    tie_break_preference: str,
) -> dict[str, Any]:
    if len(candidates) < 2:
        raise RuntimeError("screen winner selection requires at least two candidates")
    normalized_candidates = sorted(candidates, key=lambda candidate: int(candidate["order"]))

    def _metric(candidate: Mapping[str, Any], key: str) -> float | None:
        metrics = candidate.get("screen_metrics")
        if not isinstance(metrics, Mapping):
            raise RuntimeError(f"candidate row {candidate.get('order')} is missing screen_metrics")
        return _optional_float(cast(Mapping[str, Any], metrics), key)

    best_primary_candidate = _best_by_from_pool(
        normalized_candidates,
        key="upper_block_final_window_mean",
    )[0]
    best_primary = _metric(best_primary_candidate, "upper_block_final_window_mean")
    assert best_primary is not None
    tied_primary = [
        candidate
        for candidate in normalized_candidates
        if (candidate_primary := _metric(candidate, "upper_block_final_window_mean")) is not None
        and _relative_gap(candidate_primary, best_primary) <= PRIMARY_TOLERANCE
    ]
    if len(tied_primary) == 1:
        winner = tied_primary[0]
        reason = "lower upper-block final-window mean"
    else:
        secondary_pool = _best_by_from_pool(
            tied_primary,
            key="upper_block_post_warmup_mean_slope",
        )
        best_secondary_candidate = secondary_pool[0]
        best_secondary = _metric(best_secondary_candidate, "upper_block_post_warmup_mean_slope")
        assert best_secondary is not None
        tied_secondary = [
            candidate
            for candidate in tied_primary
            if (candidate_secondary := _metric(candidate, "upper_block_post_warmup_mean_slope")) is not None
            and _relative_gap(candidate_secondary, best_secondary) <= SECONDARY_TOLERANCE
        ]
        if len(tied_secondary) == 1:
            winner = tied_secondary[0]
            reason = "lower upper-block post-warmup slope after a primary tie"
        else:
            tertiary_pool = _best_by_from_pool(
                tied_secondary,
                key="clipped_step_fraction",
            )
            if len(tertiary_pool) == 1:
                winner = tertiary_pool[0]
                reason = "lower clipped-step fraction after upper-block ties"
            else:
                quaternary_pool = _best_by_from_pool(
                    tertiary_pool,
                    key="final_train_loss_ema",
                )
                preferred = next(
                    (
                        candidate
                        for candidate in quaternary_pool
                        if str(candidate.get("value")) == tie_break_preference
                    ),
                    quaternary_pool[0],
                )
                winner = preferred
                reason = (
                    "tie-break preference after upper-block, slope, clip-rate, and loss-ema ties"
                )

    return {
        "winning_order": int(winner["order"]),
        "winning_value": str(winner["value"]),
        "reason": reason,
    }


def _best_by_from_pool(pool: list[dict[str, Any]], *, key: str) -> list[dict[str, Any]]:
    if not pool:
        raise RuntimeError("screen winner pool cannot be empty")
    best_value: float | None = None
    best: list[dict[str, Any]] = []
    for candidate in pool:
        metrics = candidate.get("screen_metrics")
        if not isinstance(metrics, Mapping):
            raise RuntimeError(f"candidate row {candidate.get('order')} is missing screen_metrics")
        value = _optional_float(cast(Mapping[str, Any], metrics), key)
        if value is None:
            raise RuntimeError(f"candidate row {candidate.get('order')} omitted required metric {key!r}")
        if best_value is None or value < best_value:
            best_value = value
            best = [candidate]
            continue
        if value == best_value:
            best.append(candidate)
    return best
