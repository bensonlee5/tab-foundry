"""Matrix rendering and validation helpers for system-delta sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

from tab_foundry.bench.benchmark_run_registry import load_benchmark_run_registry, resolve_registry_path_value

from .materialize import load_system_delta_queue, ordered_rows
from .paths_io import _render_path, _write_text, default_catalog_path, default_registry_path, repo_root, sweep_matrix_path, sweep_queue_path
from .validation import ensure_non_empty_string


def render_model_change_payload(model_payload: Mapping[str, Any]) -> dict[str, Any]:
    rendered: dict[str, Any] = {}
    module_overrides = model_payload.get("module_overrides")
    if isinstance(module_overrides, dict) and module_overrides:
        rendered["module_overrides"] = module_overrides
    for key, value in model_payload.items():
        if key in {"stage_label", "module_overrides"}:
            continue
        if value in (None, {}, []):
            continue
        rendered[str(key)] = value
    return rendered


def metric_summary(run: dict[str, Any], anchor: dict[str, Any]) -> dict[str, str]:
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        return float(value)

    def _format(value: float | None, *, suffix: str = "", signed: bool = False) -> str:
        if value is None:
            return "n/a"
        return f"{value:+.4f}{suffix}" if signed else f"{value:.4f}{suffix}"

    metrics = cast(dict[str, Any], run["tab_foundry_metrics"])
    anchor_metrics = cast(dict[str, Any], anchor["tab_foundry_metrics"])
    best = _optional_float(metrics.get("best_roc_auc"))
    final = _optional_float(metrics.get("final_roc_auc"))
    final_log_loss = _optional_float(metrics.get("final_log_loss"))
    anchor_final_log_loss = _optional_float(anchor_metrics.get("final_log_loss"))
    final_brier_score = _optional_float(metrics.get("final_brier_score"))
    anchor_final_brier_score = _optional_float(anchor_metrics.get("final_brier_score"))
    final_crps = _optional_float(metrics.get("final_crps"))
    anchor_final_crps = _optional_float(anchor_metrics.get("final_crps"))
    final_avg_pinball_loss = _optional_float(metrics.get("final_avg_pinball_loss"))
    anchor_final_avg_pinball_loss = _optional_float(anchor_metrics.get("final_avg_pinball_loss"))
    final_picp_90 = _optional_float(metrics.get("final_picp_90"))
    anchor_final_picp_90 = _optional_float(anchor_metrics.get("final_picp_90"))
    best_time = float(metrics["best_training_time"])
    final_time = float(metrics["final_training_time"])
    anchor_best = _optional_float(anchor_metrics.get("best_roc_auc"))
    anchor_final = _optional_float(anchor_metrics.get("final_roc_auc"))
    anchor_best_time = float(anchor_metrics["best_training_time"])
    anchor_final_time = float(anchor_metrics["final_training_time"])
    drift = None if best is None or final is None else final - best
    anchor_drift = None if anchor_best is None or anchor_final is None else anchor_final - anchor_best
    return {
        "best_roc_auc": _format(best),
        "final_roc_auc": _format(final),
        "final_minus_best": _format(drift, signed=True),
        "delta_best_roc_auc": "n/a" if best is None or anchor_best is None else f"{best - anchor_best:+.4f}",
        "delta_final_roc_auc": "n/a" if final is None or anchor_final is None else f"{final - anchor_final:+.4f}",
        "delta_drift": "n/a" if drift is None or anchor_drift is None else f"{drift - anchor_drift:+.4f}",
        "delta_training_time": f"{final_time - anchor_final_time:+.1f}s",
        "final_training_time": f"{final_time:.1f}s",
        "best_training_time": f"{best_time:.1f}s",
        "delta_best_training_time": f"{best_time - anchor_best_time:+.1f}s",
        "final_log_loss": _format(final_log_loss),
        "delta_final_log_loss": "n/a" if final_log_loss is None or anchor_final_log_loss is None else f"{final_log_loss - anchor_final_log_loss:+.4f}",
        "final_brier_score": _format(final_brier_score),
        "delta_final_brier_score": "n/a" if final_brier_score is None or anchor_final_brier_score is None else f"{final_brier_score - anchor_final_brier_score:+.4f}",
        "final_crps": _format(final_crps),
        "delta_final_crps": "n/a" if final_crps is None or anchor_final_crps is None else f"{final_crps - anchor_final_crps:+.4f}",
        "final_avg_pinball_loss": _format(final_avg_pinball_loss),
        "delta_final_avg_pinball_loss": "n/a" if final_avg_pinball_loss is None or anchor_final_avg_pinball_loss is None else f"{final_avg_pinball_loss - anchor_final_avg_pinball_loss:+.4f}",
        "final_picp_90": _format(final_picp_90),
        "delta_final_picp_90": "n/a" if final_picp_90 is None or anchor_final_picp_90 is None else f"{final_picp_90 - anchor_final_picp_90:+.4f}",
    }


def result_card_path(*, sweep_id: str, delta_id: str) -> Path:
    return repo_root() / "outputs" / "staged_ladder" / "research" / sweep_id / delta_id / "result_card.md"


def validate_system_delta_queue(
    queue: Mapping[str, Any],
    *,
    registry_path: Path | None = None,
) -> list[str]:
    issues: list[str] = []
    registry = load_benchmark_run_registry(registry_path or default_registry_path())
    runs = cast(dict[str, dict[str, Any]], registry["runs"])
    sweep_id = ensure_non_empty_string(queue.get("sweep_id"), context="materialized queue sweep_id")
    for row in ordered_rows(queue):
        status = str(row.get("status", "")).strip().lower()
        if status != "completed":
            continue
        delta_id = str(row["delta_id"])
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            issues.append(f"{delta_id}: completed rows must include run_id")
            continue
        run = runs.get(run_id)
        if run is None:
            issues.append(f"{delta_id}: run_id {run_id!r} is missing from the benchmark registry")
            continue
        card_path = result_card_path(sweep_id=sweep_id, delta_id=delta_id)
        if not card_path.exists():
            issues.append(f"{delta_id}: missing result card at {card_path}")
        training_surface_record_path = cast(dict[str, Any], run["artifacts"]).get("training_surface_record_path")
        if not isinstance(training_surface_record_path, str) or not training_surface_record_path.strip():
            issues.append(f"{delta_id}: run {run_id!r} is missing artifacts.training_surface_record_path")
        else:
            resolved = resolve_registry_path_value(training_surface_record_path)
            if not resolved.exists():
                issues.append(f"{delta_id}: training surface artifact does not exist at {resolved}")
    return issues


def render_system_delta_matrix(
    queue: Mapping[str, Any],
    *,
    registry_path: Path | None = None,
) -> str:
    registry = load_benchmark_run_registry(registry_path or default_registry_path())
    runs = cast(dict[str, dict[str, Any]], registry["runs"])
    sweep_id = ensure_non_empty_string(queue.get("sweep_id"), context="materialized queue sweep_id")
    anchor_run_id = str(queue["anchor_run_id"])
    anchor = runs.get(anchor_run_id)
    if anchor is None:
        raise RuntimeError(f"anchor_run_id {anchor_run_id!r} is missing from the benchmark registry")
    anchor_metrics = cast(dict[str, Any], anchor["tab_foundry_metrics"])
    upstream = cast(dict[str, Any], queue["upstream_reference"])
    anchor_surface = cast(dict[str, Any], queue["anchor_surface"])
    catalog_path = str(queue.get("catalog_path", _render_path(default_catalog_path())))
    canonical_queue_path = str(queue.get("canonical_queue_path", _render_path(sweep_queue_path(sweep_id))))

    lines: list[str] = []
    lines.append("# System Delta Matrix")
    lines.append("")
    lines.append(
        f"This file is rendered from `{canonical_queue_path}` plus `{catalog_path}` and the canonical benchmark registry."
    )
    lines.append("")
    lines.append("## Sweep")
    lines.append("")
    lines.append(f"- Sweep id: `{sweep_id}`")
    lines.append(f"- Sweep status: `{queue.get('sweep_status')}`")
    lines.append(f"- Parent sweep id: `{queue.get('parent_sweep_id')}`")
    lines.append(f"- Complexity level: `{queue.get('complexity_level')}`")
    lines.append("")
    lines.append("## Locked Surface")
    lines.append("")
    lines.append(f"- Anchor run id: `{anchor_run_id}`")
    lines.append(f"- Benchmark bundle: `{queue['benchmark_bundle_path']}`")
    lines.append(f"- Control baseline id: `{queue['control_baseline_id']}`")
    lines.append(f"- Comparison policy: `{queue['comparison_policy']}`")
    anchor_metric_parts: list[str] = []
    for label, key in (
        ("final log loss", "final_log_loss"),
        ("final Brier score", "final_brier_score"),
        ("best ROC AUC", "best_roc_auc"),
        ("final ROC AUC", "final_roc_auc"),
        ("final CRPS", "final_crps"),
        ("final avg pinball loss", "final_avg_pinball_loss"),
        ("final PICP 90", "final_picp_90"),
    ):
        raw_value = anchor_metrics.get(key)
        if raw_value is not None:
            anchor_metric_parts.append(f"{label} `{float(raw_value):.4f}`")
    anchor_metric_parts.append(f"final training time `{float(anchor_metrics['final_training_time']):.1f}s`")
    lines.append(f"- Anchor metrics: {', '.join(anchor_metric_parts)}")
    lines.append("")
    lines.append("## Anchor Comparison")
    lines.append("")
    lines.append(f"Upstream reference: `{upstream['name']}` from `{upstream['model_source']}`.")
    lines.append("")
    lines.append("| Dimension | Upstream nanoTabPFN | Locked anchor | Interpretation |")
    lines.append("| --- | --- | --- | --- |")
    for dimension_row in cast(list[dict[str, Any]], anchor_surface["dimension_table"]):
        lines.append(
            f"| {dimension_row['dimension']} | {dimension_row['upstream']} | "
            f"{dimension_row['anchor']} | {dimension_row['interpretation']} |"
        )
    lines.append("")
    lines.append("## Queue Summary")
    lines.append("")
    lines.append("| Order | Delta | Family | Binary | Status | Recipe alias | Effective change | Next action |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for queue_row in ordered_rows(queue):
        lines.append(
            f"| {queue_row['order']} | `{queue_row['delta_id']}` | {queue_row['family']} | "
            f"{'yes' if queue_row.get('binary_applicable', False) else 'no'} | {queue_row['status']} | "
            f"{queue_row.get('entangled_legacy_stage', 'none')} | "
            f"{queue_row['description']} | {queue_row['next_action']} |"
        )
    lines.append("")
    lines.append("## Detailed Rows")
    lines.append("")
    for queue_row in ordered_rows(queue):
        delta_id = str(queue_row["delta_id"])
        run_id = queue_row.get("run_id")
        run = runs.get(run_id) if isinstance(run_id, str) else None
        lines.append(f"### {queue_row['order']}. `{delta_id}`")
        lines.append("")
        lines.append(f"- Dimension family: `{queue_row['dimension_family']}`")
        lines.append(f"- Status: `{queue_row['status']}`")
        lines.append(f"- Binary applicable: `{queue_row.get('binary_applicable', False)}`")
        lines.append(f"- Recipe alias: `{queue_row.get('entangled_legacy_stage', 'none')}`")
        lines.append(f"- Description: {queue_row['description']}")
        lines.append(f"- Rationale: {queue_row['rationale']}")
        lines.append(f"- Hypothesis: {queue_row['hypothesis']}")
        lines.append(f"- Upstream delta: {queue_row['upstream_delta']}")
        lines.append(f"- Anchor delta: {queue_row['anchor_delta']}")
        lines.append(f"- Expected effect: {queue_row['expected_effect']}")
        lines.append(
            f"- Effective labels: model=`{queue_row['model']['stage_label']}`, "
            f"data=`{queue_row['data']['surface_label']}`, "
            f"preprocessing=`{queue_row['preprocessing']['surface_label']}`, "
            f"training=`{queue_row['training']['surface_label']}`"
        )
        if queue_row["dimension_family"] == "model":
            lines.append(f"- Model overrides: `{render_model_change_payload(cast(Mapping[str, Any], queue_row['model']))}`")
            dynamic_model_overrides = queue_row.get("dynamic_model_overrides")
            if isinstance(dynamic_model_overrides, Mapping) and dynamic_model_overrides:
                lines.append(f"- Dynamic model overrides: `{dict(dynamic_model_overrides)}`")
        elif queue_row["dimension_family"] == "data":
            lines.append(f"- Data overrides: `{queue_row['data'].get('surface_overrides', {})}`")
        elif queue_row["dimension_family"] == "training":
            lines.append(f"- Training overrides: `{queue_row['training'].get('overrides', {})}`")
        else:
            lines.append(f"- Preprocessing overrides: `{queue_row['preprocessing'].get('overrides', {})}`")
        lines.append("- Parameter adequacy plan:")
        for plan_item in cast(list[str], queue_row.get("parameter_adequacy_plan", [])):
            lines.append(f"  - {plan_item}")
        if cast(list[str], queue_row.get("adequacy_knobs", [])):
            lines.append("- Adequacy knobs to dimension explicitly:")
            for adequacy_knob in cast(list[str], queue_row["adequacy_knobs"]):
                lines.append(f"  - {adequacy_knob}")
        lines.append(f"- Execution policy: `{queue_row.get('execution_policy', 'benchmark_full')}`")
        lines.append(f"- Interpretation status: `{queue_row.get('interpretation_status', 'pending')}`")
        lines.append(f"- Decision: `{queue_row.get('decision')}`")
        if cast(list[str], queue_row.get("confounders", [])):
            lines.append("- Confounders:")
            for confounder in cast(list[str], queue_row["confounders"]):
                lines.append(f"  - {confounder}")
        if cast(list[str], queue_row.get("notes", [])):
            lines.append("- Notes:")
            for note in cast(list[str], queue_row["notes"]):
                lines.append(f"  - {note}")
        lines.append(f"- Follow-up run ids: `{queue_row.get('followup_run_ids', [])}`")
        lines.append(f"- Result card path: `{_render_path(result_card_path(sweep_id=sweep_id, delta_id=delta_id))}`")
        if run is None:
            screen_metrics = queue_row.get("screen_metrics")
            if isinstance(screen_metrics, Mapping):
                lines.append("- Screen metrics:")
                upper_mean = screen_metrics.get("upper_block_final_window_mean")
                if upper_mean is not None:
                    lines.append(
                        f"  - Upper-block final-window mean: `{float(upper_mean):.4f}`"
                    )
                upper_slope = screen_metrics.get("upper_block_post_warmup_mean_slope")
                if upper_slope is not None:
                    lines.append(
                        f"  - Upper-block post-warmup mean slope: `{float(upper_slope):.6f}`"
                    )
                clip_fraction = screen_metrics.get("clipped_step_fraction")
                if clip_fraction is not None:
                    lines.append(
                        f"  - Clipped-step fraction: `{float(clip_fraction):.4f}`"
                    )
                final_loss_ema = screen_metrics.get("final_train_loss_ema")
                if final_loss_ema is not None:
                    lines.append(f"  - Final train-loss EMA: `{float(final_loss_ema):.4f}`")
            inline_metrics = queue_row.get("benchmark_metrics")
            if inline_metrics:
                best = float(inline_metrics["best_roc_auc"])
                step = inline_metrics.get("best_step", "?")
                final = float(inline_metrics["final_roc_auc"])
                drift = float(inline_metrics["drift"])
                lines.append("- Benchmark metrics:")
                lines.append(f"  - Best ROC AUC: `{best:.4f}` (step {step})")
                lines.append(f"  - Final ROC AUC: `{final:.4f}`")
                lines.append(f"  - Drift (final − best): `{drift:.4f}`")
                if "nanotabpfn_best" in inline_metrics:
                    lines.append(f"  - NanoTabPFN control: `{float(inline_metrics['nanotabpfn_best']):.4f}`")
                if "max_grad_norm" in inline_metrics:
                    lines.append(f"  - max_grad_norm: `{float(inline_metrics['max_grad_norm']):.3f}`")
            else:
                lines.append("- Benchmark metrics: pending")
        else:
            metrics = metric_summary(run, anchor)
            metric_parts = [
                f"final log loss `{metrics['final_log_loss']}`",
                f"delta final log loss `{metrics['delta_final_log_loss']}`",
                f"final Brier score `{metrics['final_brier_score']}`",
                f"delta final Brier score `{metrics['delta_final_brier_score']}`",
                f"best ROC AUC `{metrics['best_roc_auc']}`",
                f"final ROC AUC `{metrics['final_roc_auc']}`",
                f"final-minus-best `{metrics['final_minus_best']}`",
                f"delta final ROC AUC `{metrics['delta_final_roc_auc']}`",
                f"delta drift `{metrics['delta_drift']}`",
                f"final CRPS `{metrics['final_crps']}`",
                f"delta final CRPS `{metrics['delta_final_crps']}`",
                f"final avg pinball loss `{metrics['final_avg_pinball_loss']}`",
                f"delta final avg pinball loss `{metrics['delta_final_avg_pinball_loss']}`",
                f"final PICP 90 `{metrics['final_picp_90']}`",
                f"delta final PICP 90 `{metrics['delta_final_picp_90']}`",
                f"delta final training time `{metrics['delta_training_time']}`",
            ]
            filtered_metric_parts = [part for part in metric_parts if not part.endswith("`n/a`")]
            lines.append(f"- Registered run: `{run_id}` with {', '.join(filtered_metric_parts)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_and_write_system_delta_matrix(
    *,
    sweep_id: str | None = None,
    queue: Mapping[str, Any] | None = None,
    registry_path: Path | None = None,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    resolved_queue = (
        queue
        if queue is not None
        else load_system_delta_queue(
            sweep_id=sweep_id,
            index_path=index_path,
            catalog_path=catalog_path,
            sweeps_root=sweeps_root,
        )
    )
    resolved_sweep_id = ensure_non_empty_string(
        sweep_id if sweep_id is not None else resolved_queue.get("sweep_id"),
        context="sweep_id",
    )
    resolved_out_path = (
        sweep_matrix_path(resolved_sweep_id, sweeps_root=sweeps_root)
        if out_path is None
        else Path(out_path).expanduser().resolve()
    )
    contents = render_system_delta_matrix(resolved_queue, registry_path=registry_path)
    _write_text(resolved_out_path, contents)
    return resolved_out_path
