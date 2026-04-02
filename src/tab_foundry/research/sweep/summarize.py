"""Sweep-result summarization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

from .materialize import load_system_delta_queue_for_inspection, ordered_rows
from .objective_metrics import (
    is_classification_objective_metric,
    objective_metric_from_queue_metrics,
)


_WARN_CLIPPED_STEP_FRACTION = 0.05
_FAIL_CLIPPED_STEP_FRACTION = 0.20
_WARN_UPPER_BLOCK_SLOPE = 0.02
_FAIL_UPPER_BLOCK_SLOPE = 0.10


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _row_metric_payload(row: Mapping[str, Any]) -> Mapping[str, Any]:
    benchmark_metrics = row.get("benchmark_metrics")
    if isinstance(benchmark_metrics, Mapping):
        return cast(Mapping[str, Any], benchmark_metrics)
    screen_metrics = row.get("screen_metrics")
    if isinstance(screen_metrics, Mapping):
        return cast(Mapping[str, Any], screen_metrics)
    return {}


def _stability_verdict(row: Mapping[str, Any], metrics: Mapping[str, Any]) -> str:
    status = str(row.get("status", "")).strip().lower()
    if status == "blocked":
        return "blocked"
    clipped_step_fraction = _optional_float(metrics.get("clipped_step_fraction"))
    upper_block_slope = _optional_float(metrics.get("upper_block_post_warmup_mean_slope"))
    has_stability_signal = clipped_step_fraction is not None or upper_block_slope is not None
    if clipped_step_fraction is not None and clipped_step_fraction > _FAIL_CLIPPED_STEP_FRACTION:
        return "fail"
    if upper_block_slope is not None and upper_block_slope > _FAIL_UPPER_BLOCK_SLOPE:
        return "fail"
    if clipped_step_fraction is not None and clipped_step_fraction > _WARN_CLIPPED_STEP_FRACTION:
        return "warn"
    if upper_block_slope is not None and upper_block_slope > _WARN_UPPER_BLOCK_SLOPE:
        return "warn"
    if status in {"completed", "screened"} and has_stability_signal:
        return "ok"
    return "n/a"


def summarize_sweep(
    *,
    sweep_id: str | None = None,
    include_screened: bool = False,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    queue = load_system_delta_queue_for_inspection(
        sweep_id=sweep_id,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
    )
    rows_payload: list[dict[str, Any]] = []
    for row in ordered_rows(queue):
        status = str(row.get("status", "")).strip().lower()
        if status == "screened" and not include_screened:
            continue
        metrics = _row_metric_payload(row)
        objective_metric = objective_metric_from_queue_metrics(metrics)
        rows_payload.append(
            {
                "order": int(row["order"]),
                "delta_id": str(row["delta_id"]),
                "status": status,
                "decision": None if row.get("decision") is None else str(row["decision"]),
                "run_id": None if row.get("run_id") is None else str(row["run_id"]),
                "stability": _stability_verdict(row, metrics),
                "objective_metric": objective_metric,
                "final_roc_auc": _optional_float(metrics.get("final_roc_auc")),
                "delta_final_roc_auc": _optional_float(metrics.get("delta_final_roc_auc")),
                "final_bpc": _optional_float(metrics.get("final_bpc")),
                "delta_final_bpc": _optional_float(metrics.get("delta_final_bpc")),
                "final_log_loss": _optional_float(metrics.get("final_log_loss")),
                "delta_final_log_loss": _optional_float(metrics.get("delta_final_log_loss")),
                "clipped_step_fraction": _optional_float(metrics.get("clipped_step_fraction")),
                "upper_block_post_warmup_mean_slope": _optional_float(
                    metrics.get("upper_block_post_warmup_mean_slope")
                ),
                "upper_block_final_window_mean": _optional_float(
                    metrics.get("upper_block_final_window_mean")
                ),
                "final_train_loss_ema": _optional_float(metrics.get("final_train_loss_ema")),
            }
        )
    return {
        "sweep_id": str(queue["sweep_id"]),
        "row_count": len(rows_payload),
        "include_screened": bool(include_screened),
        "rows": rows_payload,
    }


def _format_float(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def render_sweep_summary_table(payload: Mapping[str, Any]) -> str:
    rows = cast(list[dict[str, Any]], payload["rows"])
    headers = [
        "ord",
        "delta_id",
        "status",
        "decision",
        "stability",
        "d_primary",
        "d_roc_auc",
        "clip_frac",
        "upper_slope",
        "run_id",
    ]
    rendered_rows: list[list[str]] = []
    for row in rows:
        if is_classification_objective_metric(cast(str | None, row.get("objective_metric"))):
            primary_delta = cast(float | None, row["delta_final_log_loss"])
        else:
            primary_delta = (
                cast(float | None, row["delta_final_bpc"])
                if row.get("delta_final_bpc") is not None
                else cast(float | None, row["delta_final_log_loss"])
            )
        rendered_rows.append(
            [
                f"{int(row['order']):02d}",
                str(row["delta_id"]),
                str(row["status"]),
                str(row["decision"] or "n/a"),
                str(row["stability"]),
                _format_float(primary_delta, signed=True),
                _format_float(cast(float | None, row["delta_final_roc_auc"]), signed=True),
                _format_float(cast(float | None, row["clipped_step_fraction"])),
                _format_float(cast(float | None, row["upper_block_post_warmup_mean_slope"])),
                str(row["run_id"] or "n/a"),
            ]
        )
    widths = [
        max([len(header), *(len(row[index]) for row in rendered_rows)])
        for index, header in enumerate(headers)
    ]
    lines = [
        f"Sweep summary: sweep_id={payload['sweep_id']} rows={payload['row_count']}",
        "  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)),
        "  ".join("-" * widths[index] for index in range(len(headers))),
    ]
    for rendered_row in rendered_rows:
        lines.append(
            "  ".join(value.ljust(widths[index]) for index, value in enumerate(rendered_row))
        )
    return "\n".join(lines)
