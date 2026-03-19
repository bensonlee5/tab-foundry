"""Artifact and report helpers for system-delta execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

from omegaconf import OmegaConf

from tab_foundry.research.lane_contract import (
    ARCHITECTURE_SCREEN_SURFACE,
    HYBRID_DIAGNOSTIC_LANE_LABEL,
    PFN_CONTROL_LANE_LABEL,
)

from .paths_io import _write_yaml as _write_yaml_file, default_catalog_path, default_registry_path, default_sweep_index_path, default_sweeps_root, repo_root as _repo_root


@dataclass(frozen=True)
class ExecutionPaths:
    repo_root: Path
    index_path: Path
    catalog_path: Path
    sweeps_root: Path
    registry_path: Path
    program_path: Path
    control_baseline_registry_path: Path

    @classmethod
    def default(cls) -> "ExecutionPaths":
        repo_root = _repo_root()
        return cls(
            repo_root=repo_root,
            index_path=default_sweep_index_path(),
            catalog_path=default_catalog_path(),
            sweeps_root=default_sweeps_root(),
            registry_path=default_registry_path(),
            program_path=repo_root / "program.md",
            control_baseline_registry_path=repo_root / "src" / "tab_foundry" / "bench" / "control_baselines_v1.json",
        )

    def promotion_paths(self) -> Any:
        from tab_foundry.research.system_delta_promote import PromotionPaths

        return PromotionPaths(
            index_path=self.index_path,
            catalog_path=self.catalog_path,
            sweeps_root=self.sweeps_root,
            registry_path=self.registry_path,
            program_path=self.program_path,
        )


def read_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected YAML mapping at {path}")
    return cast(dict[str, Any], payload)


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    _write_yaml_file(path, payload)


def research_card_text(
    *,
    row: Mapping[str, Any],
    sweep_id: str,
    anchor_run_id: str | None,
    sweep_meta: Mapping[str, Any],
    training_experiment: str,
    training_config_profile: str,
    surface_role: str,
) -> str:
    plan = cast(list[str], row.get("parameter_adequacy_plan", []))
    plan_lines = "\n".join(f"- {item}" for item in plan) if plan else "- No extra adequacy plan recorded."
    anchor_display = anchor_run_id or "none"
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
            f"- `locked_bundle_path`: `{sweep_meta['benchmark_bundle_path']}`",
            f"- `locked_control_baseline_id`: `{sweep_meta['control_baseline_id']}`",
            f"- `training_experiment`: `{training_experiment}`",
            f"- `training_config_profile`: `{training_config_profile}`",
            f"- `surface_role`: `{surface_role}`",
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
    training_experiment: str,
    training_config_profile: str,
    surface_role: str,
) -> dict[str, Any]:
    changed_settings: dict[str, Any] = {
        "model": cast(dict[str, Any], queue_row.get("model", {})),
        "data": cast(dict[str, Any], queue_row.get("data", {})),
        "preprocessing": cast(dict[str, Any], queue_row.get("preprocessing", {})),
        "training": cast(dict[str, Any], queue_row.get("training", {})),
    }
    return {
        "sweep_id": sweep_id,
        "delta_id": materialized_row["delta_id"],
        "dimension_family": materialized_row["dimension_family"],
        "family": materialized_row["family"],
        "comparison_policy": str(sweep_meta.get("comparison_policy", "anchor_only")),
        "anchor_run_id": anchor_run_id,
        "locked_bundle_path": str(sweep_meta["benchmark_bundle_path"]),
        "locked_control_baseline_id": str(sweep_meta["control_baseline_id"]),
        "training_experiment": training_experiment,
        "training_config_profile": training_config_profile,
        "surface_role": surface_role,
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
    training_experiment: str,
    training_config_profile: str,
    surface_role: str,
) -> None:
    delta_root.mkdir(parents=True, exist_ok=True)
    (delta_root / "research_card.md").write_text(
        research_card_text(
            row=materialized_row,
            sweep_id=sweep_id,
            anchor_run_id=anchor_run_id,
            sweep_meta=sweep_meta,
            training_experiment=training_experiment,
            training_config_profile=training_config_profile,
            surface_role=surface_role,
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
            training_experiment=training_experiment,
            training_config_profile=training_config_profile,
            surface_role=surface_role,
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


def result_card_text(
    *,
    row: Mapping[str, Any],
    run_id: str,
    anchor_run_id: str | None,
    summary: Mapping[str, Any],
    queue_metrics: Mapping[str, Any],
    decision: str,
    conclusion: str,
) -> str:
    _ = summary
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
        if best_step is None:
            append_metric_line(lines, label=label, value=value)
            return
        lines.append(f"- {label}: `{format_metric(value)}` at step `{int(float(best_step))}`")

    has_classification_metrics = (
        queue_metrics.get("best_log_loss") is not None
        or queue_metrics.get("final_log_loss") is not None
    )
    has_regression_metrics = (
        queue_metrics.get("best_crps") is not None or queue_metrics.get("final_crps") is not None
    )

    if has_classification_metrics:
        _append_best_metric("Best log loss", "best_log_loss")
        append_metric_line(lines, label="Final log loss", value=queue_metrics.get("final_log_loss"))
        append_metric_line(lines, label="Final minus best log loss", value=queue_metrics.get("final_minus_best_log_loss"), signed=True)
        append_metric_line(lines, label="Delta final log loss vs anchor", value=queue_metrics.get("delta_final_log_loss"), signed=True)
        append_metric_line(lines, label="nanoTabPFN best log loss", value=queue_metrics.get("nanotabpfn_best_log_loss"))
        append_metric_line(lines, label="nanoTabPFN final log loss", value=queue_metrics.get("nanotabpfn_final_log_loss"))
        append_metric_line(lines, label="Final Brier score", value=queue_metrics.get("final_brier_score"))
        append_metric_line(lines, label="Final minus best Brier score", value=queue_metrics.get("final_minus_best_brier_score"), signed=True)
        append_metric_line(lines, label="Delta final Brier score vs anchor", value=queue_metrics.get("delta_final_brier_score"), signed=True)
        _append_best_metric("Best ROC AUC", "best_roc_auc")
        append_metric_line(lines, label="Final ROC AUC", value=queue_metrics.get("final_roc_auc"))
        append_metric_line(lines, label="Final minus best ROC AUC", value=queue_metrics.get("final_minus_best_roc_auc"), signed=True)
        append_metric_line(lines, label="Delta final ROC AUC vs anchor", value=queue_metrics.get("delta_final_roc_auc"), signed=True)
        append_metric_line(lines, label="nanoTabPFN best ROC AUC", value=queue_metrics.get("nanotabpfn_best_roc_auc"))
        append_metric_line(lines, label="nanoTabPFN final ROC AUC", value=queue_metrics.get("nanotabpfn_final_roc_auc"))
    elif has_regression_metrics:
        _append_best_metric("Best CRPS", "best_crps")
        append_metric_line(lines, label="Final CRPS", value=queue_metrics.get("final_crps"))
        append_metric_line(lines, label="Final minus best CRPS", value=queue_metrics.get("final_minus_best_crps"), signed=True)
        append_metric_line(lines, label="Delta final CRPS vs anchor", value=queue_metrics.get("delta_final_crps"), signed=True)
        append_metric_line(lines, label="nanoTabPFN best CRPS", value=queue_metrics.get("nanotabpfn_best_crps"))
        append_metric_line(lines, label="nanoTabPFN final CRPS", value=queue_metrics.get("nanotabpfn_final_crps"))
        append_metric_line(lines, label="Final avg pinball loss", value=queue_metrics.get("final_avg_pinball_loss"))
        append_metric_line(lines, label="Final minus best avg pinball loss", value=queue_metrics.get("final_minus_best_avg_pinball_loss"), signed=True)
        append_metric_line(lines, label="Delta final avg pinball loss vs anchor", value=queue_metrics.get("delta_final_avg_pinball_loss"), signed=True)
        append_metric_line(lines, label="Final PICP 90", value=queue_metrics.get("final_picp_90"))
        append_metric_line(lines, label="Final minus best PICP 90", value=queue_metrics.get("final_minus_best_picp_90"), signed=True)
        append_metric_line(lines, label="Delta final PICP 90 vs anchor", value=queue_metrics.get("delta_final_picp_90"), signed=True)
    else:
        _append_best_metric("Best ROC AUC", "best_roc_auc")
        append_metric_line(lines, label="Final ROC AUC", value=queue_metrics.get("final_roc_auc"))
        append_metric_line(lines, label="Final minus best ROC AUC", value=queue_metrics.get("final_minus_best_roc_auc"), signed=True)
        append_metric_line(lines, label="Delta final ROC AUC vs anchor", value=queue_metrics.get("delta_final_roc_auc"), signed=True)
        append_metric_line(lines, label="nanoTabPFN best ROC AUC", value=queue_metrics.get("nanotabpfn_best_roc_auc"))
        append_metric_line(lines, label="nanoTabPFN final ROC AUC", value=queue_metrics.get("nanotabpfn_final_roc_auc"))

    append_metric_line(lines, label="max_grad_norm", value=queue_metrics.get("max_grad_norm"))
    append_metric_line(lines, label="clipped_step_fraction", value=queue_metrics.get("clipped_step_fraction"))
    stage_local_lines = _stage_local_stability_lines(queue_metrics)
    if stage_local_lines:
        lines.extend(
            [
                "",
                "## Stage-local stability",
                "",
            ]
        )
        lines.extend(stage_local_lines)

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
