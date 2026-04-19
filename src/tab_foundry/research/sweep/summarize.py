"""Sweep-result summarization helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

from .queue_loading import load_system_delta_queue_for_inspection, ordered_rows
from .objective_metrics import (
    is_classification_objective_metric,
    objective_metric_from_queue_metrics,
)
from .pareto import annotate_rows_with_pareto
from .transfer import annotate_rows_with_transfer_context


_WARN_CLIPPED_STEP_FRACTION = 0.05
_FAIL_CLIPPED_STEP_FRACTION = 0.20
_WARN_UPPER_BLOCK_SLOPE = 0.02
_FAIL_UPPER_BLOCK_SLOPE = 0.10


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _optional_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(value)


def _optional_text(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return str(value).strip()


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


def _runtime_summary_excerpt(metrics: Mapping[str, Any]) -> dict[str, Any] | None:
    payload = {
        "end_to_end_wall_seconds": _optional_float(metrics.get("end_to_end_wall_seconds")),
        "loader_setup_seconds": _optional_float(metrics.get("loader_setup_seconds")),
        "peak_vram_allocated": _optional_int(metrics.get("peak_vram_allocated")),
        "peak_vram_reserved": _optional_int(metrics.get("peak_vram_reserved")),
        "throughput_examples_per_second": _optional_float(
            metrics.get("throughput_examples_per_second")
        ),
        "throughput_tokens_per_second": _optional_float(metrics.get("throughput_tokens_per_second")),
        "non_train_overhead_seconds": _optional_float(metrics.get("non_train_overhead_seconds")),
        "loader_effective_num_workers": _optional_int(metrics.get("loader_effective_num_workers")),
        "loader_effective_prefetch_factor": _optional_int(
            metrics.get("loader_effective_prefetch_factor")
        ),
        "loader_task_batch_cache_mode": _optional_text(metrics.get("loader_task_batch_cache_mode")),
        "compile_shape_dispatch_mode": _optional_text(metrics.get("compile_shape_dispatch_mode")),
        "compile_shape_dispatch_max_families": _optional_int(
            metrics.get("compile_shape_dispatch_max_families")
        ),
        "compile_dispatch_compiled_family_count": _optional_int(
            metrics.get("compile_dispatch_compiled_family_count")
        ),
        "compile_dispatch_family_switch_count": _optional_int(
            metrics.get("compile_dispatch_family_switch_count")
        ),
    }
    return payload if any(value is not None for value in payload.values()) else None


def _regime_budget_excerpt(metrics: Mapping[str, Any]) -> dict[str, Any] | None:
    payload = {
        "tokens_per_step": _optional_float(metrics.get("tokens_per_step")),
        "tokens_seen": _optional_int(metrics.get("tokens_seen")),
        "token_budget": _optional_int(metrics.get("token_budget")),
        "unique_task_budget": _optional_int(metrics.get("unique_task_budget")),
        "objective_metric": _optional_text(metrics.get("objective_metric")),
        "curriculum_id": _optional_text(metrics.get("curriculum_id")),
    }
    return payload if any(value is not None for value in payload.values()) else None


def build_sweep_summary_payload(
    *,
    queue: Mapping[str, Any],
    include_screened: bool = False,
) -> dict[str, Any]:
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
                "model": None if not isinstance(row.get("model"), Mapping) else dict(cast(Mapping[str, Any], row["model"])),
                "transfer_context": (
                    dict(cast(Mapping[str, Any], row["transfer_context"]))
                    if isinstance(row.get("transfer_context"), Mapping)
                    else None
                ),
                "transfer_resolution": (
                    dict(cast(Mapping[str, Any], row["transfer_resolution"]))
                    if isinstance(row.get("transfer_resolution"), Mapping)
                    else None
                ),
                "imported_baseline_provenance": (
                    dict(cast(Mapping[str, Any], row["imported_baseline_provenance"]))
                    if isinstance(row.get("imported_baseline_provenance"), Mapping)
                    else None
                ),
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
                "final_train_loss": _optional_float(metrics.get("final_train_loss")),
                "final_train_loss_ema": _optional_float(metrics.get("final_train_loss_ema")),
                "final_tail_mean_train_loss": _optional_float(
                    metrics.get("final_tail_mean_train_loss")
                ),
                "final_tail_mean_train_loss_ema": _optional_float(
                    metrics.get("final_tail_mean_train_loss_ema")
                ),
                "final_tail_record_count": _optional_int(metrics.get("final_tail_record_count")),
                "train_elapsed_seconds": _optional_float(metrics.get("train_elapsed_seconds")),
                "wall_elapsed_seconds": _optional_float(metrics.get("wall_elapsed_seconds")),
                "end_to_end_wall_seconds": _optional_float(
                    metrics.get("end_to_end_wall_seconds")
                ),
                "loader_setup_seconds": _optional_float(metrics.get("loader_setup_seconds")),
                "compile_dispatch_compiled_family_count": _optional_int(
                    metrics.get("compile_dispatch_compiled_family_count")
                ),
                "compile_dispatch_family_switch_count": _optional_int(
                    metrics.get("compile_dispatch_family_switch_count")
                ),
                "one_family_step_count": _optional_int(metrics.get("one_family_step_count")),
                "mixed_family_step_count": _optional_int(metrics.get("mixed_family_step_count")),
                "consecutive_repeated_family_step_count": _optional_int(
                    metrics.get("consecutive_repeated_family_step_count")
                ),
                "consecutive_switched_family_step_count": _optional_int(
                    metrics.get("consecutive_switched_family_step_count")
                ),
                "family_block_count": _optional_int(metrics.get("family_block_count")),
                "estimated_family_switch_count": _optional_int(
                    metrics.get("estimated_family_switch_count")
                ),
                "peak_vram_reserved": _optional_int(metrics.get("peak_vram_reserved")),
                "throughput_tokens_per_second": _optional_float(
                    metrics.get("throughput_tokens_per_second")
                ),
                "tokens_per_step": _optional_float(metrics.get("tokens_per_step")),
                "runtime_summary": _runtime_summary_excerpt(metrics),
                "regime_budget": _regime_budget_excerpt(metrics),
            }
        )
    selector_summary = annotate_rows_with_pareto(
        rows=rows_payload,
        surface_role=(str(queue.get("surface_role")) if queue.get("surface_role") is not None else None),
    )
    transfer_summary = annotate_rows_with_transfer_context(
        rows=rows_payload,
        surface_role=(str(queue.get("surface_role")) if queue.get("surface_role") is not None else None),
    )
    return {
        "sweep_id": str(queue["sweep_id"]),
        "row_count": len(rows_payload),
        "include_screened": bool(include_screened),
        "surface_role": str(queue.get("surface_role") or ""),
        "rows": rows_payload,
        "selector_summary": selector_summary,
        "transfer_summary": transfer_summary,
    }


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
    return build_sweep_summary_payload(queue=queue, include_screened=include_screened)


def _format_float(value: float | None, *, signed: bool = False) -> str:
    if value is None:
        return "n/a"
    return f"{value:+.4f}" if signed else f"{value:.4f}"


def _format_bytes(value: int | None) -> str:
    if value is None:
        return "n/a"
    numeric = float(value)
    for scale, suffix in (
        (1024.0**3, "GiB"),
        (1024.0**2, "MiB"),
        (1024.0, "KiB"),
    ):
        if numeric >= scale:
            return f"{numeric / scale:.1f}{suffix}"
    return f"{int(numeric)}B"


def render_sweep_summary_table(payload: Mapping[str, Any]) -> str:
    rows = cast(list[dict[str, Any]], payload["rows"])
    headers = [
        "ord",
        "delta_id",
        "status",
        "decision",
        "stability",
        "pareto",
        "d_primary",
        "d_roc_auc",
        "tok/s",
        "vram_rsv",
        "tok/step",
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
                (
                    "Y"
                    if row.get("pareto_admissible") is True
                    else ("N" if row.get("pareto_admissible") is False else "n/a")
                ),
                _format_float(primary_delta, signed=True),
                _format_float(cast(float | None, row["delta_final_roc_auc"]), signed=True),
                _format_float(cast(float | None, row.get("throughput_tokens_per_second"))),
                _format_bytes(cast(int | None, row.get("peak_vram_reserved"))),
                _format_float(cast(float | None, row.get("tokens_per_step"))),
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
    selector_summary = payload.get("selector_summary")
    if isinstance(selector_summary, Mapping):
        global_frontier_orders = selector_summary.get("global_frontier_orders")
        if isinstance(global_frontier_orders, list) and global_frontier_orders:
            lines.extend(
                [
                    "",
                    "Pareto frontier:",
                    "- global quality/time admissible rows: "
                    + ", ".join(f"{int(value):02d}" for value in global_frontier_orders),
                ]
            )
        best_row = selector_summary.get("best_row")
        if isinstance(best_row, Mapping):
            lines.append(
                "- best row: "
                f"order {int(best_row['order']):02d}, "
                f"geometry={best_row.get('geometry_label') or 'n/a'}, "
                f"prescription={best_row.get('prescription_label') or 'n/a'}, "
                f"log_loss={float(best_row['final_log_loss']):.6f}, "
                f"wall={float(best_row['end_to_end_wall_seconds']):.1f}s"
            )
        kept_contract = selector_summary.get("kept_contract")
        if isinstance(kept_contract, Mapping):
            lines.append(
                "- kept contract: "
                f"{kept_contract['prescription_label']} "
                f"(frontier_geometries={int(kept_contract['geometry_count'])}, "
                f"mean_wall={float(kept_contract['mean_end_to_end_wall_seconds']):.1f}s, "
                f"mean_log_loss={float(kept_contract['mean_benchmark_log_loss']):.6f})"
            )
        elif selector_summary.get("no_universal_kept_contract") is True:
            lines.append("- kept contract: none (no prescription reached majority frontier coverage)")
        per_geometry_frontiers = selector_summary.get("per_geometry_frontiers")
        if isinstance(per_geometry_frontiers, Mapping) and per_geometry_frontiers:
            for geometry_label, geometry_rows in sorted(per_geometry_frontiers.items()):
                if not isinstance(geometry_rows, list) or not geometry_rows:
                    continue
                lines.append(
                    f"- {geometry_label} frontier: "
                    + ", ".join(
                        f"{cast(Mapping[str, Any], geometry_row).get('prescription_label')}@{int(cast(Mapping[str, Any], geometry_row)['order']):02d}"
                        for geometry_row in geometry_rows
                        if isinstance(geometry_row, Mapping)
                    )
                )
    transfer_summary = payload.get("transfer_summary")
    if isinstance(transfer_summary, Mapping):
        best_row = transfer_summary.get("best_row")
        fastest_row = transfer_summary.get("fastest_row")
        leaderboard = transfer_summary.get("regime_leaderboard")
        imported_orders = transfer_summary.get("imported_baseline_orders")
        if best_row or fastest_row or leaderboard or imported_orders:
            lines.extend(["", "Transfer summary:"])
        if isinstance(best_row, Mapping):
            lines.append(
                "- best transfer row: "
                f"order {int(best_row['order']):02d}, "
                f"regime={best_row.get('regime_label') or 'n/a'}, "
                f"log_loss={float(best_row['final_log_loss']):.6f}, "
                f"budget={best_row.get('target_budget_label') or 'n/a'}"
            )
        if isinstance(fastest_row, Mapping):
            lines.append(
                "- fastest transfer row: "
                f"order {int(fastest_row['order']):02d}, "
                f"regime={fastest_row.get('regime_label') or 'n/a'}, "
                f"wall={float(fastest_row['end_to_end_wall_seconds']):.1f}s, "
                f"log_loss={float(fastest_row['final_log_loss']):.6f}"
            )
        if isinstance(leaderboard, list) and leaderboard:
            rendered_leaderboard = ", ".join(
                (
                    f"{str(cast(Mapping[str, Any], item)['regime_label'])} "
                    f"(mean_log_loss={float(cast(Mapping[str, Any], item)['mean_benchmark_log_loss']):.6f}"
                    + (
                        ""
                        if cast(Mapping[str, Any], item).get("mean_end_to_end_wall_seconds") is None
                        else f", mean_wall={float(cast(Mapping[str, Any], item)['mean_end_to_end_wall_seconds']):.1f}s"
                    )
                    + ")"
                )
                for item in leaderboard
                if isinstance(item, Mapping)
            )
            if rendered_leaderboard:
                lines.append(f"- regime leaderboard: {rendered_leaderboard}")
        kept_regime = transfer_summary.get("kept_regime")
        if isinstance(kept_regime, Mapping):
            lines.append(
                "- kept regime (T2 then T1): "
                f"{kept_regime['regime_label']} "
                f"(T2 order {int(kept_regime['t2_order']):02d}, "
                f"log_loss={float(kept_regime['t2_log_loss']):.6f})"
            )
        t2_vs_highbatch = transfer_summary.get("t2_vs_carried_highbatch")
        if isinstance(t2_vs_highbatch, Mapping):
            lines.append(
                "- T2 vs carried high-batch: "
                f"{t2_vs_highbatch['winning_regime_label']} "
                f"delta_log_loss={float(t2_vs_highbatch['delta_log_loss']):+.6f}"
            )
        if isinstance(imported_orders, list) and imported_orders:
            lines.append(
                "- imported baseline orders: "
                + ", ".join(f"{int(value):02d}" for value in imported_orders)
            )
    return "\n".join(lines)
