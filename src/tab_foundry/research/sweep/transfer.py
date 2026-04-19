"""Faithful training-dynamics transfer helpers for sweep execution and reporting."""

from __future__ import annotations

import math
from typing import Any, Mapping, cast


TRANSFER_SCREEN_SURFACE_ROLE = "classification_training_dynamics_transfer_screen"
TRANSFER_SURFACE_ROLE = "classification_training_dynamics_transfer"
TRANSFER_REGIME_B = "B"
TRANSFER_REGIME_D = "D"
SUPPORTED_TRANSFER_REGIMES = frozenset({TRANSFER_REGIME_B, TRANSFER_REGIME_D})
DEFAULT_TRANSFER_MIN_LR_RATIO = 1.0e-3
DEFAULT_MAX_TRANSFER_DRIFT = 0.02
DEFAULT_FIXED_BATCH_REGIME_B = 64


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_string(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return str(value).strip()


def is_transfer_surface_role(surface_role: str | None) -> bool:
    return str(surface_role or "").strip() in {
        TRANSFER_SCREEN_SURFACE_ROLE,
        TRANSFER_SURFACE_ROLE,
    }


def round_half_up(value: float) -> int:
    if not math.isfinite(value):
        raise RuntimeError(f"round_half_up requires a finite value, got {value!r}")
    if value < 0:
        raise RuntimeError(f"round_half_up requires a non-negative value, got {value!r}")
    return int(math.floor(value + 0.5))


def nearest_realizable_effective_batch(
    *,
    target_effective_batch: float,
    task_batch_size: int,
) -> int:
    if task_batch_size <= 0:
        raise RuntimeError(f"task_batch_size must be > 0, got {task_batch_size}")
    if not math.isfinite(target_effective_batch) or target_effective_batch <= 0.0:
        raise RuntimeError(
            "target_effective_batch must be a finite positive float, "
            f"got {target_effective_batch!r}"
        )
    grad_accum_steps = max(1, round_half_up(float(target_effective_batch) / float(task_batch_size)))
    return int(task_batch_size * grad_accum_steps)


def resolve_transfer_schedule(
    *,
    regime_label: str,
    base_lr_max: float,
    base_momentum: float,
    base_effective_batch: int,
    base_effective_budget: int,
    target_effective_budget: int,
    task_batch_size: int,
    fixed_effective_batch: int | None = None,
    min_lr_ratio: float = DEFAULT_TRANSFER_MIN_LR_RATIO,
    max_budget_drift: float = DEFAULT_MAX_TRANSFER_DRIFT,
) -> dict[str, Any]:
    normalized_regime = str(regime_label).strip().upper()
    if normalized_regime not in SUPPORTED_TRANSFER_REGIMES:
        raise RuntimeError(
            f"unsupported transfer regime {regime_label!r}; expected one of {sorted(SUPPORTED_TRANSFER_REGIMES)}"
        )
    if base_effective_budget <= 0 or target_effective_budget <= 0:
        raise RuntimeError(
            "effective budgets must be positive ints; "
            f"got base={base_effective_budget}, target={target_effective_budget}"
        )
    if base_effective_batch <= 0:
        raise RuntimeError(f"base_effective_batch must be > 0, got {base_effective_batch}")
    if not (0.0 < float(base_momentum) < 1.0):
        raise RuntimeError(f"base_momentum must lie in (0, 1), got {base_momentum}")
    if float(base_lr_max) <= 0.0:
        raise RuntimeError(f"base_lr_max must be > 0, got {base_lr_max}")
    if float(min_lr_ratio) <= 0.0:
        raise RuntimeError(f"min_lr_ratio must be > 0, got {min_lr_ratio}")

    ratio = float(target_effective_budget) / float(base_effective_budget)
    alpha0 = 1.0 - float(base_momentum)
    if normalized_regime == TRANSFER_REGIME_B:
        target_alpha = alpha0 * (ratio ** -0.5)
        target_lr_max = float(base_lr_max) * (ratio ** -0.75)
        target_effective_batch = float(
            DEFAULT_FIXED_BATCH_REGIME_B if fixed_effective_batch is None else fixed_effective_batch
        )
        formula_label = "Theorem 2 fixed-batch transfer"
    else:
        target_alpha = alpha0 * (ratio ** (-1.0 / 3.0))
        target_lr_max = float(base_lr_max) * (ratio ** (-7.0 / 12.0))
        target_effective_batch = float(base_effective_batch) * (ratio ** (1.0 / 6.0))
        formula_label = "Theorem 3 joint-transfer proxy"

    if not (0.0 < target_alpha < 1.0):
        raise RuntimeError(
            f"derived alpha must lie in (0, 1), got alpha={target_alpha} for regime {normalized_regime}"
        )
    target_momentum = 1.0 - target_alpha
    if not (0.0 < target_momentum < 1.0):
        raise RuntimeError(
            "derived momentum must lie in (0, 1), "
            f"got momentum={target_momentum} for regime {normalized_regime}"
        )
    realized_effective_batch = nearest_realizable_effective_batch(
        target_effective_batch=target_effective_batch,
        task_batch_size=int(task_batch_size),
    )
    grad_accum_steps = int(realized_effective_batch // int(task_batch_size))
    max_steps = round_half_up(float(target_effective_budget) / float(realized_effective_batch))
    realized_effective_budget = int(max_steps * realized_effective_batch)
    budget_drift = float(realized_effective_budget) / float(target_effective_budget) - 1.0
    batch_drift = float(realized_effective_batch) / float(target_effective_batch) - 1.0
    if abs(budget_drift) > float(max_budget_drift):
        raise RuntimeError(
            "derived transfer row exceeded the allowed effective-budget drift: "
            f"target={target_effective_budget}, realized={realized_effective_budget}, "
            f"drift={budget_drift:+.6f}, limit={max_budget_drift:.6f}"
        )

    return {
        "regime_label": normalized_regime,
        "formula_label": formula_label,
        "base_effective_budget": int(base_effective_budget),
        "target_effective_budget": int(target_effective_budget),
        "realized_effective_budget": int(realized_effective_budget),
        "base_effective_batch": int(base_effective_batch),
        "target_effective_batch": float(target_effective_batch),
        "realized_effective_batch": int(realized_effective_batch),
        "base_lr_max": float(base_lr_max),
        "target_lr_max": float(target_lr_max),
        "base_momentum": float(base_momentum),
        "target_momentum": float(target_momentum),
        "base_alpha": float(alpha0),
        "target_alpha": float(target_alpha),
        "grad_accum_steps": int(grad_accum_steps),
        "max_steps": int(max_steps),
        "min_lr": float(target_lr_max) * float(min_lr_ratio),
        "budget_drift": float(budget_drift),
        "batch_drift": float(batch_drift),
    }


def row_transfer_context(row: Mapping[str, Any]) -> dict[str, Any] | None:
    payload: dict[str, Any] = {}
    for key in ("transfer_context", "transfer_resolution"):
        raw_payload = row.get(key)
        if isinstance(raw_payload, Mapping):
            payload.update(dict(cast(Mapping[str, Any], raw_payload)))
    return payload or None


def imported_baseline_provenance(row: Mapping[str, Any]) -> dict[str, Any] | None:
    raw_payload = row.get("imported_baseline_provenance")
    if not isinstance(raw_payload, Mapping):
        return None
    return dict(cast(Mapping[str, Any], raw_payload))


def annotate_rows_with_transfer_context(
    *,
    rows: list[dict[str, Any]],
    surface_role: str | None = None,
) -> dict[str, Any] | None:
    if not is_transfer_surface_role(surface_role):
        return None

    for row in rows:
        context = row_transfer_context(row)
        provenance = imported_baseline_provenance(row)
        row["transfer_regime_label"] = None if context is None else _optional_string(context.get("regime_label"))
        row["transfer_phase"] = None if context is None else _optional_string(context.get("phase"))
        row["transfer_formula_label"] = None if context is None else _optional_string(context.get("formula_label"))
        row["transfer_base_budget_label"] = None if context is None else _optional_string(
            context.get("base_budget_label")
        )
        row["transfer_target_budget_label"] = None if context is None else _optional_string(
            context.get("target_budget_label")
        )
        row["transfer_candidate_label"] = None if context is None else _optional_string(
            context.get("candidate_label")
        )
        row["target_effective_batch"] = None if context is None else _optional_float(
            context.get("target_effective_batch")
        )
        row["realized_effective_batch"] = None if context is None else _optional_int(
            context.get("realized_effective_batch")
        )
        row["target_effective_budget"] = None if context is None else _optional_int(
            context.get("target_effective_budget")
        )
        row["realized_effective_budget"] = None if context is None else _optional_int(
            context.get("realized_effective_budget")
        )
        row["budget_drift"] = None if context is None else _optional_float(context.get("budget_drift"))
        row["batch_drift"] = None if context is None else _optional_float(context.get("batch_drift"))
        row["imported_baseline_provenance"] = provenance

    eligible_rows = [
        row
        for row in rows
        if row.get("transfer_regime_label") is not None and _optional_float(row.get("final_log_loss")) is not None
    ]
    if not eligible_rows:
        return {
            "eligible_row_count": 0,
            "best_row": None,
            "fastest_row": None,
            "regime_leaderboard": [],
            "imported_baseline_orders": sorted(
                int(row["order"])
                for row in rows
                if imported_baseline_provenance(row) is not None
            ),
        }

    best_row = min(
        eligible_rows,
        key=lambda row: (
            float(row["final_log_loss"]),
            float(row["end_to_end_wall_seconds"])
            if row.get("end_to_end_wall_seconds") is not None
            else math.inf,
            int(row["order"]),
        ),
    )
    fastest_candidates = [
        row for row in eligible_rows if _optional_float(row.get("end_to_end_wall_seconds")) is not None
    ]
    fastest_row = (
        None
        if not fastest_candidates
        else min(
            fastest_candidates,
            key=lambda row: (
                float(row["end_to_end_wall_seconds"]),
                float(row["final_log_loss"]),
                int(row["order"]),
            ),
        )
    )
    regimes: dict[str, list[Mapping[str, Any]]] = {}
    for row in eligible_rows:
        regime_label = str(row["transfer_regime_label"])
        regimes.setdefault(regime_label, []).append(row)
    regime_leaderboard: list[dict[str, Any]] = []
    for regime_label, regime_rows in regimes.items():
        benchmark_rows = [
            row
            for row in regime_rows
            if str(row.get("status", "")).strip().lower() == "completed"
        ]
        if not benchmark_rows:
            continue
        best_regime_row = min(
            benchmark_rows,
            key=lambda row: (
                float(row["final_log_loss"]),
                float(row["end_to_end_wall_seconds"])
                if row.get("end_to_end_wall_seconds") is not None
                else math.inf,
                int(row["order"]),
            ),
        )
        mean_log_loss = sum(float(row["final_log_loss"]) for row in benchmark_rows) / float(
            len(benchmark_rows)
        )
        mean_wall = (
            sum(
                float(row["end_to_end_wall_seconds"])
                for row in benchmark_rows
                if row.get("end_to_end_wall_seconds") is not None
            )
            / float(
                sum(1 for row in benchmark_rows if row.get("end_to_end_wall_seconds") is not None)
            )
            if any(row.get("end_to_end_wall_seconds") is not None for row in benchmark_rows)
            else None
        )
        regime_leaderboard.append(
            {
                "regime_label": regime_label,
                "row_count": len(benchmark_rows),
                "orders": sorted(int(row["order"]) for row in benchmark_rows),
                "best_row_order": int(best_regime_row["order"]),
                "best_log_loss": float(best_regime_row["final_log_loss"]),
                "mean_benchmark_log_loss": float(mean_log_loss),
                "mean_end_to_end_wall_seconds": None if mean_wall is None else float(mean_wall),
                "imported_baseline": all(imported_baseline_provenance(row) is not None for row in benchmark_rows),
            }
        )
    regime_leaderboard.sort(
        key=lambda payload: (
            float(payload["mean_benchmark_log_loss"]),
            math.inf
            if payload["mean_end_to_end_wall_seconds"] is None
            else float(payload["mean_end_to_end_wall_seconds"]),
            str(payload["regime_label"]),
        )
    )
    return {
        "eligible_row_count": len(eligible_rows),
        "best_row": {
            "order": int(best_row["order"]),
            "regime_label": str(best_row["transfer_regime_label"]),
            "final_log_loss": float(best_row["final_log_loss"]),
            "end_to_end_wall_seconds": _optional_float(best_row.get("end_to_end_wall_seconds")),
            "target_budget_label": _optional_string(best_row.get("transfer_target_budget_label")),
        },
        "fastest_row": (
            None
            if fastest_row is None
            else {
                "order": int(fastest_row["order"]),
                "regime_label": str(fastest_row["transfer_regime_label"]),
                "final_log_loss": float(fastest_row["final_log_loss"]),
                "end_to_end_wall_seconds": float(fastest_row["end_to_end_wall_seconds"]),
                "target_budget_label": _optional_string(fastest_row.get("transfer_target_budget_label")),
            }
        ),
        "regime_leaderboard": regime_leaderboard,
        "imported_baseline_orders": sorted(
            int(row["order"])
            for row in rows
            if imported_baseline_provenance(row) is not None
        ),
    }
