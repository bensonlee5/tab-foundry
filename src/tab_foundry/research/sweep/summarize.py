"""Sweep-result summarization helpers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from .materialize import load_system_delta_queue_for_inspection, ordered_rows
from .paths_io import default_catalog_path, default_sweep_index_path, default_sweeps_root


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
        rows_payload.append(
            {
                "order": int(row["order"]),
                "delta_id": str(row["delta_id"]),
                "status": status,
                "decision": None if row.get("decision") is None else str(row["decision"]),
                "run_id": None if row.get("run_id") is None else str(row["run_id"]),
                "stability": _stability_verdict(row, metrics),
                "final_roc_auc": _optional_float(metrics.get("final_roc_auc")),
                "delta_final_roc_auc": _optional_float(metrics.get("delta_final_roc_auc")),
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
        "d_roc_auc",
        "d_log_loss",
        "clip_frac",
        "upper_slope",
        "run_id",
    ]
    rendered_rows: list[list[str]] = []
    for row in rows:
        rendered_rows.append(
            [
                f"{int(row['order']):02d}",
                str(row["delta_id"]),
                str(row["status"]),
                str(row["decision"] or "n/a"),
                str(row["stability"]),
                _format_float(cast(float | None, row["delta_final_roc_auc"]), signed=True),
                _format_float(cast(float | None, row["delta_final_log_loss"]), signed=True),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Summarize local system-delta sweep results")
    parser.add_argument("--sweep-id", default=None, help="Sweep id to inspect; defaults to the active sweep")
    parser.add_argument("--json", action="store_true", help="Emit machine-readable JSON")
    parser.add_argument(
        "--include-screened",
        action="store_true",
        help="Include screened rows alongside completed or blocked rows",
    )
    parser.add_argument(
        "--catalog-path",
        default=str(default_catalog_path()),
        help="Path to reference/system_delta_catalog.yaml",
    )
    parser.add_argument(
        "--index-path",
        default=str(default_sweep_index_path()),
        help="Path to reference/system_delta_sweeps/index.yaml",
    )
    parser.add_argument(
        "--sweeps-root",
        default=str(default_sweeps_root()),
        help="Path to reference/system_delta_sweeps/",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    payload = summarize_sweep(
        sweep_id=None if args.sweep_id is None else str(args.sweep_id),
        include_screened=bool(args.include_screened),
        index_path=Path(str(args.index_path)).expanduser().resolve(),
        catalog_path=Path(str(args.catalog_path)).expanduser().resolve(),
        sweeps_root=Path(str(args.sweeps_root)).expanduser().resolve(),
    )
    if bool(args.json):
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(render_sweep_summary_table(payload))
    return 0
