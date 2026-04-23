"""Report formatting and research package helpers for system-delta execution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, cast

from tab_foundry.benchmark_registry import load_benchmark_run_registry, resolve_registry_path_value
from tab_foundry.external_benchmarks import (
    EXTERNAL_BENCHMARK_LABELS,
    EXTERNAL_BENCHMARK_NANOTABPFN,
    EXTERNAL_BENCHMARK_TABICLV2,
)
from tab_foundry.research.lane_contract import (
    ARCHITECTURE_SCREEN_SURFACE,
    HYBRID_DIAGNOSTIC_LANE_LABEL,
    PFN_CONTROL_LANE_LABEL,
    TrainingSurfaceContext,
)

from .artifacts import write_yaml
from .objective_metrics import (
    display_metric_label,
    is_classification_objective_metric,
    objective_metric_from_queue_metrics,
)
from .paths_io import default_registry_path, repo_root
from .queue_loading import ordered_rows


def research_card_text(
    *,
    row: Mapping[str, Any],
    sweep_id: str,
    anchor_run_id: str | None,
    sweep_meta: Mapping[str, Any],
    training_surface: TrainingSurfaceContext,
) -> str:
    plan = cast(list[str], row.get("parameter_adequacy_plan", []))
    plan_lines = "\n".join(f"- {item}" for item in plan) if plan else "- No extra adequacy plan recorded."
    anchor_display = anchor_run_id or "none"
    external_benchmarks = sweep_meta.get("external_benchmarks", [])
    external_benchmarks_display = (
        ", ".join(f"`{str(value)}`" for value in external_benchmarks)
        if isinstance(external_benchmarks, list) and external_benchmarks
        else ("`none`" if isinstance(external_benchmarks, list) else "`nanotabpfn`")
    )
    return "\n".join(
        [
            "# Research Card",
            "",
            "## Delta",
            "",
            f"- `delta_id`: `{row['delta_id']}`",
            f"- `sweep_id`: `{sweep_id}`",
            f"- `dimension_family`: `{row['dimension_family']}`",
            f"- `family`: `{row['family']}`",
            f"- `anchor_run_id`: `{anchor_display}`",
            "- `comparison_policy`: `anchor_only`",
            f"- `locked_manifest_path`: `{sweep_meta['benchmark_manifest_path']}`",
            f"- `locked_control_baseline_id`: `{sweep_meta['control_baseline_id']}`",
            f"- `external_benchmarks`: {external_benchmarks_display}",
            f"- `training_experiment`: `{training_surface.training_experiment}`",
            f"- `training_config_profile`: `{training_surface.training_config_profile}`",
            f"- `surface_role`: `{training_surface.surface_role}`",
            "",
            "## What Changes",
            "",
            f"- {row['description']}",
            f"- Anchor delta: {row['anchor_delta']}",
            "",
            "## Why This Row Is Informative",
            "",
            f"- {row['rationale']}",
            f"- Hypothesis: {row['hypothesis']}",
            f"- PFN control lane: {PFN_CONTROL_LANE_LABEL}",
            f"- Hybrid diagnostic lane: {HYBRID_DIAGNOSTIC_LANE_LABEL}",
            f"- Canonical architecture-screen surface: `{ARCHITECTURE_SCREEN_SURFACE}`",
            "",
            "## Adequacy Plan",
            "",
            plan_lines,
            "",
        ]
    )


def campaign_payload(
    *,
    queue_row: Mapping[str, Any],
    materialized_row: Mapping[str, Any],
    sweep_meta: Mapping[str, Any],
    sweep_id: str,
    anchor_run_id: str | None,
    device: str,
    training_surface: TrainingSurfaceContext,
) -> dict[str, Any]:
    changed_settings: dict[str, Any] = {
        "model": cast(dict[str, Any], queue_row.get("model", {})),
        "data": cast(dict[str, Any], queue_row.get("data", {})),
        "preprocessing": cast(dict[str, Any], queue_row.get("preprocessing", {})),
        "training": cast(dict[str, Any], queue_row.get("training", {})),
    }
    raw_external_benchmarks = sweep_meta.get("external_benchmarks")
    resolved_external_benchmarks = (
        [str(value) for value in raw_external_benchmarks]
        if isinstance(raw_external_benchmarks, list) and raw_external_benchmarks
        else [EXTERNAL_BENCHMARK_NANOTABPFN]
    )
    return {
        "sweep_id": sweep_id,
        "delta_id": materialized_row["delta_id"],
        "dimension_family": materialized_row["dimension_family"],
        "family": materialized_row["family"],
        "comparison_policy": str(sweep_meta.get("comparison_policy", "anchor_only")),
        "anchor_run_id": anchor_run_id,
        "locked_manifest_path": str(sweep_meta["benchmark_manifest_path"]),
        "locked_control_baseline_id": str(sweep_meta["control_baseline_id"]),
        "external_benchmarks": resolved_external_benchmarks,
        **training_surface.to_payload_dict(),
        "control_lane": PFN_CONTROL_LANE_LABEL,
        "hybrid_diagnostic_lane": HYBRID_DIAGNOSTIC_LANE_LABEL,
        "canonical_architecture_screen_surface": ARCHITECTURE_SCREEN_SURFACE,
        "preserved_settings": {
            "queue_ref": f"reference/system_delta_sweeps/{sweep_id}/queue.yaml",
            "runtime.device": str(device),
            "logging.use_wandb": True,
        },
        "changed_settings": changed_settings,
        "adequacy_knobs": cast(list[str], materialized_row.get("adequacy_knobs", [])),
        "decision_hypothesis": "needs_followup",
    }


def write_research_package(
    *,
    delta_root: Path,
    materialized_row: Mapping[str, Any],
    queue_row: Mapping[str, Any],
    sweep_meta: Mapping[str, Any],
    sweep_id: str,
    anchor_run_id: str | None,
    device: str,
    training_surface: TrainingSurfaceContext,
) -> None:
    delta_root.mkdir(parents=True, exist_ok=True)
    (delta_root / "research_card.md").write_text(
        research_card_text(
            row=materialized_row,
            sweep_id=sweep_id,
            anchor_run_id=anchor_run_id,
            sweep_meta=sweep_meta,
            training_surface=training_surface,
        ),
        encoding="utf-8",
    )
    write_yaml(
        delta_root / "campaign.yaml",
        campaign_payload(
            queue_row=queue_row,
            materialized_row=materialized_row,
            sweep_meta=sweep_meta,
            sweep_id=sweep_id,
            anchor_run_id=anchor_run_id,
            device=device,
            training_surface=training_surface,
        ),
    )


def format_metric(value: Any, *, signed: bool = False) -> str:
    numeric = float(value)
    return f"{numeric:+.4f}" if signed else f"{numeric:.4f}"


def append_metric_line(
    lines: list[str],
    *,
    label: str,
    value: Any,
    signed: bool = False,
) -> None:
    if value is None:
        return
    lines.append(f"- {label}: `{format_metric(value, signed=signed)}`")


def append_scalar_line(lines: list[str], *, label: str, value: Any) -> None:
    if value is None:
        return
    lines.append(f"- {label}: `{value}`")


def _stage_local_stability_lines(queue_metrics: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    for stage_label, grad_key, activation_key in (
        (
            "Column stage",
            "column_encoder_final_window_mean_grad_norm",
            "column_activation_early_to_final_mean_delta",
        ),
        (
            "Row stage",
            "row_pool_final_window_mean_grad_norm",
            "row_activation_early_to_final_mean_delta",
        ),
        (
            "Context stage",
            "context_encoder_final_window_mean_grad_norm",
            "context_activation_early_to_final_mean_delta",
        ),
    ):
        parts: list[str] = []
        grad_value = queue_metrics.get(grad_key)
        if grad_value is not None:
            parts.append(f"final-window mean grad norm `{format_metric(grad_value)}`")
        activation_value = queue_metrics.get(activation_key)
        if activation_value is not None:
            parts.append(f"activation early-to-final mean delta `{format_metric(activation_value, signed=True)}`")
        if parts:
            lines.append(f"- {stage_label}: {', '.join(parts)}")
    return lines


def _primary_external_summary(summary: Mapping[str, Any]) -> tuple[str | None, Mapping[str, Any] | None]:
    raw_primary = summary.get("primary_external_benchmark")
    if isinstance(raw_primary, str) and raw_primary.strip():
        primary_name = str(raw_primary).strip()
        primary_payload = summary.get(primary_name)
        if isinstance(primary_payload, Mapping):
            return primary_name, cast(Mapping[str, Any], primary_payload)
        return primary_name, None
    for candidate_name in (EXTERNAL_BENCHMARK_NANOTABPFN, EXTERNAL_BENCHMARK_TABICLV2):
        candidate_payload = summary.get(candidate_name)
        if isinstance(candidate_payload, Mapping):
            return candidate_name, cast(Mapping[str, Any], candidate_payload)
    return None, None


def _runtime_and_regime_lines(queue_metrics: Mapping[str, Any]) -> list[str]:
    lines: list[str] = []
    append_metric_line(
        lines,
        label="Throughput examples/sec",
        value=queue_metrics.get("throughput_examples_per_second"),
    )
    append_metric_line(
        lines,
        label="Throughput tokens/sec",
        value=queue_metrics.get("throughput_tokens_per_second"),
    )
    append_scalar_line(lines, label="Peak VRAM allocated", value=queue_metrics.get("peak_vram_allocated"))
    append_scalar_line(lines, label="Peak VRAM reserved", value=queue_metrics.get("peak_vram_reserved"))
    append_metric_line(
        lines,
        label="Peak VRAM allocated fraction",
        value=queue_metrics.get("peak_vram_allocated_fraction"),
    )
    append_metric_line(
        lines,
        label="Peak VRAM reserved fraction",
        value=queue_metrics.get("peak_vram_reserved_fraction"),
    )
    append_metric_line(
        lines,
        label="Non-train overhead seconds",
        value=queue_metrics.get("non_train_overhead_seconds"),
    )
    append_metric_line(
        lines,
        label="Non-train overhead fraction",
        value=queue_metrics.get("non_train_overhead_fraction"),
    )
    append_metric_line(
        lines,
        label="Achieved train TFLOP/s",
        value=queue_metrics.get("achieved_train_tflops_per_second"),
    )
    append_metric_line(
        lines,
        label="Theoretical peak TFLOP/s",
        value=queue_metrics.get("theoretical_peak_tflops_per_second"),
    )
    append_metric_line(
        lines,
        label="Compute utilization fraction",
        value=queue_metrics.get("compute_utilization_fraction"),
    )
    append_metric_line(
        lines,
        label="Theoretical HBM bandwidth (GB/s)",
        value=queue_metrics.get("theoretical_hbm_bandwidth_gbps"),
    )
    append_metric_line(
        lines,
        label="Roofline knee (FLOP/byte)",
        value=queue_metrics.get("roofline_knee_flops_per_byte"),
    )
    append_scalar_line(lines, label="Peak compute basis", value=queue_metrics.get("peak_compute_basis"))
    append_metric_line(lines, label="Tokens per step", value=queue_metrics.get("tokens_per_step"))
    append_scalar_line(lines, label="Token budget", value=queue_metrics.get("token_budget"))
    append_scalar_line(lines, label="Unique task budget", value=queue_metrics.get("unique_task_budget"))
    append_scalar_line(lines, label="Objective metric", value=queue_metrics.get("objective_metric"))
    append_scalar_line(lines, label="Curriculum id", value=queue_metrics.get("curriculum_id"))
    return lines


def _selector_interpretation_lines(
    selector_context: Mapping[str, Any] | None,
) -> list[str]:
    if not isinstance(selector_context, Mapping):
        return []
    row_summary = selector_context.get("row_summary")
    selector_summary = selector_context.get("summary")
    if not isinstance(row_summary, Mapping) and not isinstance(selector_summary, Mapping):
        return []
    lines = ["", "## Selector interpretation", ""]
    if isinstance(row_summary, Mapping):
        pareto_status = (
            "yes"
            if row_summary.get("pareto_admissible") is True
            else ("no" if row_summary.get("pareto_admissible") is False else "n/a")
        )
        geometry_pareto_status = (
            "yes"
            if row_summary.get("geometry_pareto_admissible") is True
            else (
                "no"
                if row_summary.get("geometry_pareto_admissible") is False
                else "n/a"
            )
        )
        lines.append(f"- Quality/time Pareto admissible: `{pareto_status}`")
        lines.append(f"- Geometry-local Pareto admissible: `{geometry_pareto_status}`")
        if row_summary.get("selector_geometry_label") is not None:
            lines.append(
                f"- Selector geometry: `{row_summary['selector_geometry_label']}`"
            )
        if row_summary.get("selector_prescription_label") is not None:
            lines.append(
                f"- Selector prescription: `{row_summary['selector_prescription_label']}`"
            )
    if isinstance(selector_summary, Mapping):
        best_row = selector_summary.get("best_row")
        if isinstance(best_row, Mapping):
            lines.append(
                "- Best single selector row: "
                f"order `{int(best_row['order']):02d}` "
                f"({best_row.get('geometry_label') or 'n/a'}, "
                f"{best_row.get('prescription_label') or 'n/a'}, "
                f"log loss `{float(best_row['final_log_loss']):.6f}`, "
                f"wall `{float(best_row['end_to_end_wall_seconds']):.1f}s`)"
            )
        kept_contract = selector_summary.get("kept_contract")
        if isinstance(kept_contract, Mapping):
            lines.append(
                "- Kept contract: "
                f"`{kept_contract['prescription_label']}` "
                f"(frontier geometries `{int(kept_contract['geometry_count'])}`, "
                f"mean wall `{float(kept_contract['mean_end_to_end_wall_seconds']):.1f}s`, "
                f"mean log loss `{float(kept_contract['mean_benchmark_log_loss']):.6f}`)"
            )
        elif selector_summary.get("no_universal_kept_contract") is True:
            lines.append(
                "- Kept contract: none; no prescription reached the required majority frontier coverage."
            )
    return lines


def _transfer_interpretation_lines(
    transfer_context: Mapping[str, Any] | None,
) -> list[str]:
    if not isinstance(transfer_context, Mapping):
        return []
    row_summary = transfer_context.get("row_summary")
    transfer_summary = transfer_context.get("summary")
    if not isinstance(row_summary, Mapping) and not isinstance(transfer_summary, Mapping):
        return []
    lines = ["", "## Transfer interpretation", ""]
    if isinstance(row_summary, Mapping):
        if row_summary.get("transfer_regime_label") is not None:
            lines.append(f"- Transfer regime: `{row_summary['transfer_regime_label']}`")
        if row_summary.get("transfer_phase") is not None:
            lines.append(f"- Transfer phase: `{row_summary['transfer_phase']}`")
        if row_summary.get("transfer_formula_label") is not None:
            lines.append(f"- Transfer formula: `{row_summary['transfer_formula_label']}`")
        if row_summary.get("transfer_target_budget_label") is not None:
            lines.append(
                f"- Target budget label: `{row_summary['transfer_target_budget_label']}`"
            )
        if row_summary.get("target_effective_batch") is not None:
            lines.append(
                f"- Target effective batch: `{float(row_summary['target_effective_batch']):.1f}`"
            )
        if row_summary.get("realized_effective_batch") is not None:
            lines.append(
                f"- Realized effective batch: `{int(row_summary['realized_effective_batch'])}`"
            )
        if row_summary.get("target_effective_budget") is not None:
            lines.append(
                f"- Target effective budget: `{int(row_summary['target_effective_budget'])}`"
            )
        if row_summary.get("realized_effective_budget") is not None:
            lines.append(
                f"- Realized effective budget: `{int(row_summary['realized_effective_budget'])}`"
            )
        if row_summary.get("budget_drift") is not None:
            lines.append(f"- Budget drift: `{float(row_summary['budget_drift']):+.6f}`")
        if row_summary.get("batch_drift") is not None:
            lines.append(f"- Batch drift: `{float(row_summary['batch_drift']):+.6f}`")
        imported_baseline = row_summary.get("imported_baseline_provenance")
        if isinstance(imported_baseline, Mapping):
            source_sweep_id = imported_baseline.get("source_sweep_id")
            source_order = imported_baseline.get("source_order")
            source_run_id = imported_baseline.get("source_run_id")
            rendered_parts = []
            if source_sweep_id is not None:
                rendered_parts.append(f"sweep `{source_sweep_id}`")
            if source_order is not None:
                rendered_parts.append(f"order `{int(source_order):02d}`")
            if source_run_id is not None:
                rendered_parts.append(f"run `{source_run_id}`")
            if rendered_parts:
                lines.append("- Imported baseline provenance: " + ", ".join(rendered_parts))
        shared_anchor = row_summary.get("shared_anchor_provenance")
        if isinstance(shared_anchor, Mapping):
            anchor_sweep_id = shared_anchor.get("anchor_sweep_id")
            anchor_order = shared_anchor.get("anchor_order")
            anchor_run_dir = shared_anchor.get("anchor_run_dir")
            rendered_parts = []
            if anchor_sweep_id is not None:
                rendered_parts.append(f"sweep `{anchor_sweep_id}`")
            if anchor_order is not None:
                rendered_parts.append(f"order `{int(anchor_order):02d}`")
            if anchor_run_dir is not None:
                rendered_parts.append(f"run dir `{anchor_run_dir}`")
            if rendered_parts:
                lines.append("- Shared anchor provenance: " + ", ".join(rendered_parts))
    if isinstance(transfer_summary, Mapping):
        best_row = transfer_summary.get("best_row")
        if isinstance(best_row, Mapping):
            lines.append(
                "- Best transfer row: "
                f"order `{int(best_row['order']):02d}` "
                f"(regime `{best_row.get('regime_label') or 'n/a'}`, "
                f"log loss `{float(best_row['final_log_loss']):.6f}`, "
                f"budget `{best_row.get('target_budget_label') or 'n/a'}`)"
            )
        fastest_row = transfer_summary.get("fastest_row")
        if isinstance(fastest_row, Mapping):
            lines.append(
                "- Fastest transfer row: "
                f"order `{int(fastest_row['order']):02d}` "
                f"(regime `{fastest_row.get('regime_label') or 'n/a'}`, "
                f"wall `{float(fastest_row['end_to_end_wall_seconds']):.1f}s`)"
            )
        leaderboard = transfer_summary.get("regime_leaderboard")
        if isinstance(leaderboard, list) and leaderboard:
            rendered = "; ".join(
                (
                    f"{str(cast(Mapping[str, Any], item)['regime_label'])}: "
                    f"mean log loss `{float(cast(Mapping[str, Any], item)['mean_benchmark_log_loss']):.6f}`"
                )
                for item in leaderboard
                if isinstance(item, Mapping)
            )
            if rendered:
                lines.append(f"- Regime leaderboard: {rendered}")
        kept_regime = transfer_summary.get("kept_regime")
        if isinstance(kept_regime, Mapping):
            lines.append(
                "- Kept regime (`T2` then `T1`): "
                f"`{kept_regime['regime_label']}` "
                f"(T2 order `{int(kept_regime['t2_order']):02d}`, "
                f"log loss `{float(kept_regime['t2_log_loss']):.6f}`)"
            )
        t2_vs_highbatch = transfer_summary.get("t2_vs_carried_highbatch")
        if isinstance(t2_vs_highbatch, Mapping):
            lines.append(
                "- T2 vs carried high-batch: "
                f"winning regime `{t2_vs_highbatch['winning_regime_label']}` "
                f"minus carried high-batch delta log loss "
                f"`{float(t2_vs_highbatch['delta_log_loss']):+.6f}`"
            )
    return lines


def result_card_text(
    *,
    row: Mapping[str, Any],
    run_id: str,
    anchor_run_id: str | None,
    summary: Mapping[str, Any],
    queue_metrics: Mapping[str, Any],
    decision: str,
    conclusion: str,
    selector_context: Mapping[str, Any] | None = None,
    transfer_context: Mapping[str, Any] | None = None,
) -> str:
    primary_external_name, primary_external_summary = _primary_external_summary(summary)
    objective_metric = objective_metric_from_queue_metrics(queue_metrics)
    primary_external_label = (
        queue_metrics.get("primary_external_label")
        if isinstance(queue_metrics.get("primary_external_label"), str)
        else EXTERNAL_BENCHMARK_LABELS.get(primary_external_name or "", "External benchmark")
    )
    anchor_display = anchor_run_id or "none"
    best_step = queue_metrics.get("best_step")
    lines = [
        "# Result Card",
        "",
        "## What changed",
        "",
        f"- `delta_id`: `{row['delta_id']}`",
        f"- `run_id`: `{run_id}`",
        f"- `anchor_run_id`: `{anchor_display}`",
        f"- `description`: {row['description']}",
        f"- `anchor_delta`: {row['anchor_delta']}",
        "",
        "## Measured metrics versus the anchor",
        "",
    ]

    def _append_best_metric(label: str, key: str) -> None:
        value = queue_metrics.get(key)
        if value is None:
            return
        rendered_label = display_metric_label(
            label,
            metric_key=key,
            objective_metric=objective_metric,
        )
        if best_step is None:
            append_metric_line(lines, label=rendered_label, value=value)
            return
        lines.append(
            f"- {rendered_label}: `{format_metric(value)}` at step `{int(float(best_step))}`"
        )

    def _append_metric_for_key(label: str, key: str, *, signed: bool = False) -> None:
        append_metric_line(
            lines,
            label=display_metric_label(
                label,
                metric_key=key,
                objective_metric=objective_metric,
            ),
            value=queue_metrics.get(key),
            signed=signed,
        )

    has_bpc_metrics = queue_metrics.get("best_bpc") is not None or queue_metrics.get("final_bpc") is not None
    has_classification_metrics = (
        queue_metrics.get("best_log_loss") is not None or queue_metrics.get("final_log_loss") is not None
    )
    has_regression_metrics = queue_metrics.get("best_crps") is not None or queue_metrics.get("final_crps") is not None

    if is_classification_objective_metric(objective_metric) and has_classification_metrics:
        _append_best_metric("Best log loss", "best_log_loss")
        _append_metric_for_key("Final log loss", "final_log_loss")
        _append_metric_for_key(
            "Final minus best log loss",
            "final_minus_best_log_loss",
            signed=True,
        )
        _append_metric_for_key(
            "Delta final log loss vs anchor",
            "delta_final_log_loss",
            signed=True,
        )
        _append_metric_for_key(
            f"{primary_external_label} best log loss",
            "primary_external_best_log_loss",
        )
        _append_metric_for_key(
            f"{primary_external_label} final log loss",
            "primary_external_final_log_loss",
        )
        _append_metric_for_key("Final Brier score", "final_brier_score")
        _append_metric_for_key(
            "Final minus best Brier score",
            "final_minus_best_brier_score",
            signed=True,
        )
        _append_metric_for_key(
            "Delta final Brier score vs anchor",
            "delta_final_brier_score",
            signed=True,
        )
        _append_best_metric("Best ROC AUC", "best_roc_auc")
        _append_metric_for_key("Final ROC AUC", "final_roc_auc")
        _append_metric_for_key(
            "Final minus best ROC AUC",
            "final_minus_best_roc_auc",
            signed=True,
        )
        _append_metric_for_key(
            "Delta final ROC AUC vs anchor",
            "delta_final_roc_auc",
            signed=True,
        )
        _append_metric_for_key(
            f"{primary_external_label} best ROC AUC",
            "primary_external_best_roc_auc",
        )
        _append_metric_for_key(
            f"{primary_external_label} final ROC AUC",
            "primary_external_final_roc_auc",
        )
        if has_bpc_metrics:
            lines.append(
                "- Legacy feature-cell diagnostics use normalized benchmark inputs and remain secondary to log loss on classification-objective rows."
            )
            _append_best_metric("Best BPC", "best_bpc")
            _append_metric_for_key("Final BPC", "final_bpc")
            _append_metric_for_key(
                "Final minus best BPC",
                "final_minus_best_bpc",
                signed=True,
            )
            _append_metric_for_key(
                "Delta final BPC vs anchor",
                "delta_final_bpc",
                signed=True,
            )
            _append_best_metric("Best BPF", "best_bpf")
            _append_metric_for_key("Final BPF", "final_bpf")
            _append_metric_for_key(
                "Final minus best BPF",
                "final_minus_best_bpf",
                signed=True,
            )
            _append_metric_for_key(
                "Delta final BPF vs anchor",
                "delta_final_bpf",
                signed=True,
            )
    elif has_bpc_metrics:
        _append_best_metric("Best BPC", "best_bpc")
        _append_metric_for_key("Final BPC", "final_bpc")
        _append_metric_for_key("Final minus best BPC", "final_minus_best_bpc", signed=True)
        _append_metric_for_key("Delta final BPC vs anchor", "delta_final_bpc", signed=True)
        _append_best_metric("Best BPF", "best_bpf")
        _append_metric_for_key("Final BPF", "final_bpf")
        _append_metric_for_key("Final minus best BPF", "final_minus_best_bpf", signed=True)
        _append_metric_for_key("Delta final BPF vs anchor", "delta_final_bpf", signed=True)
        if has_classification_metrics:
            _append_best_metric("Best log loss", "best_log_loss")
            _append_metric_for_key("Final log loss", "final_log_loss")
            _append_metric_for_key(
                "Final minus best log loss",
                "final_minus_best_log_loss",
                signed=True,
            )
            _append_metric_for_key(
                "Delta final log loss vs anchor",
                "delta_final_log_loss",
                signed=True,
            )
            _append_metric_for_key("Final Brier score", "final_brier_score")
            _append_metric_for_key(
                "Final minus best Brier score",
                "final_minus_best_brier_score",
                signed=True,
            )
            _append_metric_for_key(
                "Delta final Brier score vs anchor",
                "delta_final_brier_score",
                signed=True,
            )
            _append_best_metric("Best ROC AUC", "best_roc_auc")
            _append_metric_for_key("Final ROC AUC", "final_roc_auc")
            _append_metric_for_key(
                "Final minus best ROC AUC",
                "final_minus_best_roc_auc",
                signed=True,
            )
            _append_metric_for_key(
                "Delta final ROC AUC vs anchor",
                "delta_final_roc_auc",
                signed=True,
            )
            _append_metric_for_key(
                f"{primary_external_label} best log loss",
                "primary_external_best_log_loss",
            )
            _append_metric_for_key(
                f"{primary_external_label} final log loss",
                "primary_external_final_log_loss",
            )
            _append_metric_for_key(
                f"{primary_external_label} best ROC AUC",
                "primary_external_best_roc_auc",
            )
            _append_metric_for_key(
                f"{primary_external_label} final ROC AUC",
                "primary_external_final_roc_auc",
            )
    elif has_classification_metrics:
        _append_best_metric("Best log loss", "best_log_loss")
        _append_metric_for_key("Final log loss", "final_log_loss")
        _append_metric_for_key(
            "Final minus best log loss",
            "final_minus_best_log_loss",
            signed=True,
        )
        _append_metric_for_key(
            "Delta final log loss vs anchor",
            "delta_final_log_loss",
            signed=True,
        )
        _append_metric_for_key(
            f"{primary_external_label} best log loss",
            "primary_external_best_log_loss",
        )
        _append_metric_for_key(
            f"{primary_external_label} final log loss",
            "primary_external_final_log_loss",
        )
        _append_metric_for_key("Final Brier score", "final_brier_score")
        _append_metric_for_key(
            "Final minus best Brier score",
            "final_minus_best_brier_score",
            signed=True,
        )
        _append_metric_for_key(
            "Delta final Brier score vs anchor",
            "delta_final_brier_score",
            signed=True,
        )
        _append_best_metric("Best ROC AUC", "best_roc_auc")
        _append_metric_for_key("Final ROC AUC", "final_roc_auc")
        _append_metric_for_key(
            "Final minus best ROC AUC",
            "final_minus_best_roc_auc",
            signed=True,
        )
        _append_metric_for_key(
            "Delta final ROC AUC vs anchor",
            "delta_final_roc_auc",
            signed=True,
        )
        _append_metric_for_key(
            f"{primary_external_label} best ROC AUC",
            "primary_external_best_roc_auc",
        )
        _append_metric_for_key(
            f"{primary_external_label} final ROC AUC",
            "primary_external_final_roc_auc",
        )
    elif has_regression_metrics:
        _append_best_metric("Best CRPS", "best_crps")
        _append_metric_for_key("Final CRPS", "final_crps")
        _append_metric_for_key("Final minus best CRPS", "final_minus_best_crps", signed=True)
        _append_metric_for_key("Delta final CRPS vs anchor", "delta_final_crps", signed=True)
        _append_metric_for_key(f"{primary_external_label} best CRPS", "primary_external_best_crps")
        _append_metric_for_key(
            f"{primary_external_label} final CRPS",
            "primary_external_final_crps",
        )
        _append_metric_for_key("Final avg pinball loss", "final_avg_pinball_loss")
        _append_metric_for_key(
            "Final minus best avg pinball loss",
            "final_minus_best_avg_pinball_loss",
            signed=True,
        )
        _append_metric_for_key(
            "Delta final avg pinball loss vs anchor",
            "delta_final_avg_pinball_loss",
            signed=True,
        )
        _append_metric_for_key("Final PICP 90", "final_picp_90")
        _append_metric_for_key(
            "Final minus best PICP 90",
            "final_minus_best_picp_90",
            signed=True,
        )
        _append_metric_for_key(
            "Delta final PICP 90 vs anchor",
            "delta_final_picp_90",
            signed=True,
        )
    else:
        _append_best_metric("Best ROC AUC", "best_roc_auc")
        _append_metric_for_key("Final ROC AUC", "final_roc_auc")
        _append_metric_for_key(
            "Final minus best ROC AUC",
            "final_minus_best_roc_auc",
            signed=True,
        )
        _append_metric_for_key(
            "Delta final ROC AUC vs anchor",
            "delta_final_roc_auc",
            signed=True,
        )
        _append_metric_for_key(
            f"{primary_external_label} best ROC AUC",
            "primary_external_best_roc_auc",
        )
        _append_metric_for_key(
            f"{primary_external_label} final ROC AUC",
            "primary_external_final_roc_auc",
        )

    if isinstance(primary_external_summary, Mapping):
        curve_source_mode = primary_external_summary.get("curve_source_mode")
        if isinstance(curve_source_mode, str) and curve_source_mode.strip():
            lines.append(f"- {primary_external_label} curve source: `{curve_source_mode}`")
        reused_curve_path = primary_external_summary.get("reused_curve_path")
        if isinstance(reused_curve_path, str) and reused_curve_path.strip():
            lines.append(f"- Reused {primary_external_label} curve path: `{reused_curve_path}`")

    append_metric_line(lines, label="max_grad_norm", value=queue_metrics.get("max_grad_norm"))
    append_metric_line(lines, label="clipped_step_fraction", value=queue_metrics.get("clipped_step_fraction"))
    runtime_lines = _runtime_and_regime_lines(queue_metrics)
    if runtime_lines:
        lines.extend(["", "## Runtime and regime budget", ""])
        lines.extend(runtime_lines)
    stage_local_lines = _stage_local_stability_lines(queue_metrics)
    if stage_local_lines:
        lines.extend(["", "## Stage-local stability", ""])
        lines.extend(stage_local_lines)
    lines.extend(_selector_interpretation_lines(selector_context))
    lines.extend(_transfer_interpretation_lines(transfer_context))

    lines.extend(
        [
            "",
            "## Was the change actually isolated?",
            "",
            "- The run used the queue row as the only source of model/data/preprocessing/training overrides.",
            "- Bundle, control baseline, experiment family, and schedule budget stayed on the locked sweep surface.",
            "",
            "## Hyperparameter adequacy",
            "",
            "- The row preserved the queue-declared short-run budget unless the row explicitly changed it.",
            "- No extra tuning beyond the queue row was introduced during this execution.",
            "",
            "## Why this may have helped or hurt",
            "",
            f"- Decision recorded in the registry: `{decision}`.",
            f"- Conclusion: {conclusion}",
            "",
            "## Remaining confounders",
            "",
            "- This auto-generated card is intentionally conservative; deeper interpretation still belongs in the sweep review.",
            "",
        ]
    )
    return "\n".join(lines)


def refresh_result_cards_for_queue(
    *,
    queue: Mapping[str, Any],
    registry_path: Path | None = None,
) -> None:
    from .summarize import build_sweep_summary_payload

    resolved_registry_path = registry_path or default_registry_path()
    registry = load_benchmark_run_registry(resolved_registry_path)
    runs = registry.get("runs")
    if not isinstance(runs, Mapping):
        return
    sweep_id = str(queue["sweep_id"])
    anchor_run_id = (
        str(queue["anchor_run_id"])
        if isinstance(queue.get("anchor_run_id"), str) and str(queue["anchor_run_id"]).strip()
        else None
    )
    summary_payload = build_sweep_summary_payload(queue=queue, include_screened=True)
    row_summaries = {
        int(cast(Mapping[str, Any], row)["order"]): cast(Mapping[str, Any], row)
        for row in cast(list[dict[str, Any]], summary_payload["rows"])
    }
    selector_context_summary = cast(
        Mapping[str, Any] | None,
        summary_payload.get("selector_summary"),
    )
    transfer_context_summary = cast(
        Mapping[str, Any] | None,
        summary_payload.get("transfer_summary"),
    )
    for row in ordered_rows(queue):
        if str(row.get("status", "")).strip().lower() != "completed":
            continue
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            continue
        run = runs.get(run_id)
        if not isinstance(run, Mapping):
            continue
        artifacts = run.get("artifacts")
        if not isinstance(artifacts, Mapping):
            continue
        raw_summary_path = artifacts.get("comparison_summary_path")
        if not isinstance(raw_summary_path, str) or not raw_summary_path.strip():
            continue
        summary_path = resolve_registry_path_value(raw_summary_path)
        if not summary_path.exists():
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if not isinstance(summary, dict):
            continue
        queue_metrics = row.get("benchmark_metrics")
        if not isinstance(queue_metrics, Mapping):
            queue_metrics = row.get("screen_metrics")
        if not isinstance(queue_metrics, Mapping):
            continue
        delta_root = (
            repo_root()
            / "outputs"
            / "staged_ladder"
            / "research"
            / sweep_id
            / str(row["delta_id"])
        )
        delta_root.mkdir(parents=True, exist_ok=True)
        (delta_root / "result_card.md").write_text(
            result_card_text(
                row=row,
                run_id=run_id,
                anchor_run_id=anchor_run_id,
                summary=summary,
                queue_metrics=cast(Mapping[str, Any], queue_metrics),
                decision=str(row.get("decision") or "defer"),
                conclusion=str(row.get("conclusion") or "pending"),
                selector_context={
                    "row_summary": row_summaries.get(int(row["order"])),
                    "summary": selector_context_summary,
                },
                transfer_context={
                    "row_summary": row_summaries.get(int(row["order"])),
                    "summary": transfer_context_summary,
                },
            ),
            encoding="utf-8",
        )
