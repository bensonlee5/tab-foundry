"""Matrix rendering and validation helpers for system-delta sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

from tab_foundry.benchmark_registry import load_benchmark_run_registry, resolve_registry_path_value
from tab_foundry.external_benchmarks import EXTERNAL_BENCHMARK_LABELS

from .inspection_artifacts import queue_metadata_payload
from .queue_loading import load_system_delta_queue, ordered_rows, write_resolved_system_delta_queue
from .objective_metrics import (
    display_metric_label,
    first_present_metric_key,
    is_classification_objective_metric,
    objective_metric_from_queue_metrics,
    objective_metric_from_run,
    preferred_drift_metric_keys,
    preferred_final_metric_keys,
)
from .paths_io import (
    _render_path,
    default_catalog_path,
    default_registry_path,
    repo_root,
    sweep_matrix_path,
    sweep_queue_path,
    write_text,
)
from .queue_state import completed_queue_metrics_from_registry_run


def _require_non_empty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{context} must be a non-empty string")
    return str(value)


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _queue_metric_matches_expected(*, actual_raw: Any, expected_value: Any) -> bool:
    if isinstance(expected_value, bool):
        return actual_raw is expected_value
    if isinstance(expected_value, (int, float)):
        actual_value = _optional_float(actual_raw)
        return actual_value is not None and abs(actual_value - float(expected_value)) <= 1.0e-12
    return actual_raw == expected_value


def _render_locked_benchmark_surface(value: Any) -> tuple[str, str]:
    benchmark_surface_path = str(value)
    if benchmark_surface_path.endswith(".json"):
        return "Benchmark bundle", f"`{benchmark_surface_path}`"
    manifest_prefix = "data/manifests/bench/"
    manifest_suffix = "/manifest.parquet"
    if benchmark_surface_path.startswith(manifest_prefix) and benchmark_surface_path.endswith(
        manifest_suffix
    ):
        manifest_id = benchmark_surface_path[len(manifest_prefix) : -len(manifest_suffix)]
        return "Benchmark manifest", f"local benchmark-manifest id `{manifest_id}`"
    return "Benchmark manifest", f"`{benchmark_surface_path}`"


def _tracked_matrix_path(path_text: str | None) -> str | None:
    if path_text is None:
        return None
    candidate = Path(path_text)
    if not candidate.is_absolute():
        candidate = repo_root() / candidate
    if not candidate.exists():
        return None
    return path_text


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


def render_data_change_payload(data_payload: Mapping[str, Any]) -> dict[str, Any]:
    rendered: dict[str, Any] = {}
    surface_overrides = data_payload.get("surface_overrides")
    if isinstance(surface_overrides, Mapping) and surface_overrides:
        rendered.update(dict(surface_overrides))
    for key, value in data_payload.items():
        if key in {"surface_label", "surface_overrides"}:
            continue
        if value in (None, {}, []):
            continue
        rendered[str(key)] = value
    return rendered


def effective_model_label(*, queue: Mapping[str, Any], queue_row: Mapping[str, Any]) -> str:
    resolved_surface = queue_row.get("resolved_surface")
    if isinstance(resolved_surface, Mapping):
        labels = resolved_surface.get("labels")
        if isinstance(labels, Mapping):
            model_label = labels.get("model")
            if isinstance(model_label, str) and model_label.strip():
                return str(model_label)
    model_payload = queue_row.get("model")
    if isinstance(model_payload, Mapping):
        stage_label = model_payload.get("stage_label")
        if isinstance(stage_label, str) and stage_label.strip():
            return str(stage_label)
    training_experiment = queue.get("training_experiment")
    if isinstance(training_experiment, str) and training_experiment.strip():
        return str(training_experiment)
    return "none"


def metric_summary(run: dict[str, Any], anchor: dict[str, Any]) -> dict[str, str]:
    def _format(value: float | None, *, suffix: str = "", signed: bool = False) -> str:
        if value is None:
            return "n/a"
        return f"{value:+.4f}{suffix}" if signed else f"{value:.4f}{suffix}"

    metrics = cast(dict[str, Any], run["tab_foundry_metrics"])
    anchor_metrics = cast(dict[str, Any], anchor["tab_foundry_metrics"])
    best_bpc = _optional_float(metrics.get("best_bpc"))
    final_bpc = _optional_float(metrics.get("final_bpc"))
    anchor_best_bpc = _optional_float(anchor_metrics.get("best_bpc"))
    anchor_final_bpc = _optional_float(anchor_metrics.get("final_bpc"))
    best_bpf = _optional_float(metrics.get("best_bpf"))
    final_bpf = _optional_float(metrics.get("final_bpf"))
    anchor_best_bpf = _optional_float(anchor_metrics.get("best_bpf"))
    anchor_final_bpf = _optional_float(anchor_metrics.get("final_bpf"))
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
    objective_metric = objective_metric_from_run(run)
    anchor_objective_metric = objective_metric_from_run(anchor)
    drift_key = first_present_metric_key(metrics, preferred_drift_metric_keys(objective_metric))
    anchor_drift_key = first_present_metric_key(
        anchor_metrics,
        preferred_drift_metric_keys(anchor_objective_metric),
    )
    drift = None if drift_key is None else _optional_float(metrics.get(drift_key))
    anchor_drift = None if anchor_drift_key is None else _optional_float(anchor_metrics.get(anchor_drift_key))
    return {
        "best_bpc": _format(best_bpc),
        "final_bpc": _format(final_bpc),
        "best_bpf": _format(best_bpf),
        "final_bpf": _format(final_bpf),
        "best_roc_auc": _format(best),
        "final_roc_auc": _format(final),
        "final_minus_best": _format(drift, signed=True),
        "delta_best_bpc": "n/a" if best_bpc is None or anchor_best_bpc is None else f"{best_bpc - anchor_best_bpc:+.4f}",
        "delta_final_bpc": "n/a" if final_bpc is None or anchor_final_bpc is None else f"{final_bpc - anchor_final_bpc:+.4f}",
        "delta_best_bpf": "n/a" if best_bpf is None or anchor_best_bpf is None else f"{best_bpf - anchor_best_bpf:+.4f}",
        "delta_final_bpf": "n/a" if final_bpf is None or anchor_final_bpf is None else f"{final_bpf - anchor_final_bpf:+.4f}",
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


def _run_metric_parts_without_anchor(run: dict[str, Any]) -> list[str]:
    metrics = cast(dict[str, Any], run["tab_foundry_metrics"])
    parts: list[str] = []
    objective_metric = objective_metric_from_run(run)
    key_labels = {
        "final_bpc": "final BPC",
        "final_bpf": "final BPF",
        "final_log_loss": "final log loss",
        "final_brier_score": "final Brier score",
        "best_roc_auc": "best ROC AUC",
        "final_roc_auc": "final ROC AUC",
        "final_crps": "final CRPS",
        "final_avg_pinball_loss": "final avg pinball loss",
        "final_picp_90": "final PICP 90",
    }
    ordered_keys: list[str] = list(preferred_final_metric_keys(objective_metric))
    ordered_keys.extend(
        key
        for key in (
            "best_roc_auc",
            "final_roc_auc",
            "final_crps",
            "final_avg_pinball_loss",
            "final_picp_90",
            "final_bpc",
            "final_bpf",
            "final_log_loss",
            "final_brier_score",
        )
        if key not in ordered_keys
    )
    for key in ordered_keys:
        label = key_labels.get(key)
        if label is None:
            continue
        raw_value = metrics.get(key)
        if raw_value is not None:
            parts.append(
                f"{display_metric_label(label, metric_key=key, objective_metric=objective_metric)} "
                f"`{float(raw_value):.4f}`"
            )
    final_training_time = metrics.get("final_training_time")
    if final_training_time is not None:
        parts.append(f"final training time `{float(final_training_time):.1f}s`")
    return parts


def _stage_local_stability_summary(queue_metrics: Mapping[str, Any] | None) -> str | None:
    if not isinstance(queue_metrics, Mapping):
        return None
    parts: list[str] = []
    for stage_label, grad_key, activation_key in (
        (
            "column",
            "column_encoder_final_window_mean_grad_norm",
            "column_activation_early_to_final_mean_delta",
        ),
        (
            "row",
            "row_pool_final_window_mean_grad_norm",
            "row_activation_early_to_final_mean_delta",
        ),
        (
            "context",
            "context_encoder_final_window_mean_grad_norm",
            "context_activation_early_to_final_mean_delta",
        ),
    ):
        stage_parts: list[str] = []
        grad_value = queue_metrics.get(grad_key)
        if grad_value is not None:
            stage_parts.append(f"grad `{float(grad_value):.4f}`")
        activation_value = queue_metrics.get(activation_key)
        if activation_value is not None:
            stage_parts.append(f"act delta `{float(activation_value):+.4f}`")
        if stage_parts:
            parts.append(f"{stage_label} ({', '.join(stage_parts)})")
    if not parts:
        return None
    return "; ".join(parts)


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
    sweep_id = _require_non_empty_string(queue.get("sweep_id"), context="materialized queue sweep_id")
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
        benchmark_metrics = row.get("benchmark_metrics")
        if not isinstance(benchmark_metrics, Mapping):
            issues.append(f"{delta_id}: completed rows must include benchmark_metrics")
            continue
        expected_metrics = completed_queue_metrics_from_registry_run(run)
        for metric_key, expected_value in expected_metrics.items():
            if metric_key not in benchmark_metrics:
                continue
            actual_raw = benchmark_metrics.get(metric_key)
            if not _queue_metric_matches_expected(
                actual_raw=actual_raw,
                expected_value=expected_value,
            ):
                issues.append(
                    f"{delta_id}: benchmark_metrics.{metric_key} mismatch "
                    f"(queue={actual_raw!r}, registry={expected_value!r})"
                )
    return issues


def render_system_delta_matrix(
    queue: Mapping[str, Any],
    *,
    registry_path: Path | None = None,
) -> str:
    metadata = queue_metadata_payload(queue)
    registry = load_benchmark_run_registry(registry_path or default_registry_path())
    runs = cast(dict[str, dict[str, Any]], registry["runs"])
    sweep_id = _require_non_empty_string(metadata.get("sweep_id"), context="materialized queue sweep_id")
    raw_anchor_run_id = metadata.get("anchor_run_id")
    anchor_run_id = (
        None
        if raw_anchor_run_id is None or not str(raw_anchor_run_id).strip()
        else str(raw_anchor_run_id).strip()
    )
    anchor = None if anchor_run_id is None else runs.get(anchor_run_id)
    if anchor_run_id is not None and anchor is None:
        raise RuntimeError(f"anchor_run_id {anchor_run_id!r} is missing from the benchmark registry")
    anchor_metrics = None if anchor is None else cast(dict[str, Any], anchor["tab_foundry_metrics"])
    upstream = cast(dict[str, Any], queue["upstream_reference"])
    anchor_surface = cast(dict[str, Any], queue["anchor_surface"])
    catalog_path = str(queue.get("catalog_path", _render_path(default_catalog_path())))
    canonical_queue_path = str(
        metadata.get("canonical_queue_path", queue.get("canonical_queue_path", _render_path(sweep_queue_path(sweep_id))))
    )
    raw_resolved_queue_path = queue.get("canonical_resolved_queue_path")
    canonical_resolved_queue_path = _tracked_matrix_path(
        str(raw_resolved_queue_path)
        if isinstance(raw_resolved_queue_path, str) and raw_resolved_queue_path.strip()
        else None
    )
    raw_inputs_fingerprint = queue.get("inputs_fingerprint")
    inputs_fingerprint = (
        str(raw_inputs_fingerprint)
        if isinstance(raw_inputs_fingerprint, str) and raw_inputs_fingerprint.strip()
        else None
    )

    lines: list[str] = []
    lines.append("# System Delta Matrix")
    lines.append("")
    if canonical_resolved_queue_path is None:
        lines.append(
            f"This file is rendered from `{canonical_queue_path}` plus `{catalog_path}` and the canonical benchmark registry."
        )
    else:
        lines.append(
            f"This file is rendered from `{canonical_resolved_queue_path}` "
            f"(derived from `{canonical_queue_path}` plus `{catalog_path}`) and the canonical benchmark registry."
        )
    lines.append("")
    lines.append("## Sweep")
    lines.append("")
    lines.append(f"- Sweep id: `{sweep_id}`")
    lines.append(f"- Sweep status: `{queue.get('sweep_status')}`")
    lines.append(f"- Parent sweep id: `{queue.get('parent_sweep_id')}`")
    lines.append(f"- Complexity level: `{queue.get('complexity_level')}`")
    if canonical_resolved_queue_path is not None:
        lines.append(f"- Resolved queue path: `{canonical_resolved_queue_path}`")
    if inputs_fingerprint is not None:
        lines.append(f"- Resolved queue inputs fingerprint: `{inputs_fingerprint}`")
    lines.append("")
    lines.append("## Locked Surface")
    lines.append("")
    lines.append(f"- Anchor run id: `{anchor_run_id if anchor_run_id is not None else 'null'}`")
    benchmark_surface_label, benchmark_surface_text = _render_locked_benchmark_surface(
        metadata["benchmark_manifest_path"]
    )
    lines.append(f"- {benchmark_surface_label}: {benchmark_surface_text}")
    lines.append(f"- Control baseline id: `{metadata['control_baseline_id']}`")
    raw_external_benchmarks = metadata.get("external_benchmarks", [])
    external_benchmarks = (
        [str(value) for value in raw_external_benchmarks]
        if isinstance(raw_external_benchmarks, list) and raw_external_benchmarks
        else ([] if isinstance(raw_external_benchmarks, list) else ["nanotabpfn"])
    )
    lines.append(
        "- External benchmarks: "
        + (
            ", ".join(f"`{benchmark_id}`" for benchmark_id in external_benchmarks)
            if external_benchmarks
            else "`none`"
        )
    )
    lines.append(f"- Training experiment: `{metadata.get('training_experiment')}`")
    lines.append(f"- Training config profile: `{metadata.get('training_config_profile')}`")
    lines.append(f"- Surface role: `{metadata.get('surface_role')}`")
    lines.append(f"- Comparison policy: `{metadata['comparison_policy']}`")
    if anchor_metrics is None:
        lines.append("- Anchor metrics: `pending trusted rerun`")
    else:
        anchor_metric_parts: list[str] = []
        for label, key in (
            ("final BPC", "final_bpc"),
            ("final BPF", "final_bpf"),
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
    upstream_name = str(upstream.get("name", "unknown"))
    upstream_source = str(upstream.get("model_source", "unknown"))
    lines.append(f"Upstream reference: `{upstream_name}` from `{upstream_source}`.")
    if anchor is None:
        lines.append("")
        lines.append(
            "Pending trusted rerun: no anchor is registered yet, so this matrix records the locked benchmark surface and queue state before the first anchor promotion."
        )
    lines.append("")
    lines.append(f"| Dimension | Upstream {upstream_name} | Locked anchor | Interpretation |")
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
        stage_local_stability = _stage_local_stability_summary(
            cast(Mapping[str, Any] | None, queue_row.get("benchmark_metrics"))
        )
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
        resolved_surface = (
            cast(Mapping[str, Any], queue_row.get("resolved_surface"))
            if isinstance(queue_row.get("resolved_surface"), Mapping)
            else {}
        )
        resolved_labels = (
            cast(Mapping[str, Any], resolved_surface.get("labels"))
            if isinstance(resolved_surface.get("labels"), Mapping)
            else {}
        )
        resolved_runtime = (
            cast(Mapping[str, Any], resolved_surface.get("runtime"))
            if isinstance(resolved_surface.get("runtime"), Mapping)
            else {}
        )
        lines.append(
            f"- Effective labels: model=`{effective_model_label(queue=queue, queue_row=queue_row)}`, "
            f"data=`{resolved_labels.get('data', queue_row['data']['surface_label'])}`, "
            f"preprocessing=`{resolved_labels.get('preprocessing', queue_row['preprocessing']['surface_label'])}`, "
            f"training=`{resolved_labels.get('training', queue_row['training']['surface_label'])}`"
        )
        fingerprint = queue_row.get("resolved_surface_fingerprint")
        if isinstance(fingerprint, str) and fingerprint.strip():
            lines.append(f"- Resolved surface fingerprint: `{fingerprint}`")
        if resolved_runtime:
            lines.append(f"- Resolved runtime surface: `{dict(resolved_runtime)}`")
        if stage_local_stability is not None:
            lines.append(f"- Stage-local stability: {stage_local_stability}")
        if queue_row["dimension_family"] == "model":
            lines.append(f"- Model overrides: `{render_model_change_payload(cast(Mapping[str, Any], queue_row['model']))}`")
            dynamic_model_overrides = queue_row.get("dynamic_model_overrides")
            if isinstance(dynamic_model_overrides, Mapping) and dynamic_model_overrides:
                lines.append(f"- Dynamic model overrides: `{dict(dynamic_model_overrides)}`")
        elif queue_row["dimension_family"] == "data":
            lines.append(
                f"- Data overrides: `{render_data_change_payload(cast(Mapping[str, Any], queue_row['data']))}`"
            )
        elif queue_row["dimension_family"] == "training":
            lines.append(f"- Training overrides: `{queue_row['training'].get('overrides', {})}`")
        else:
            lines.append(f"- Preprocessing overrides: `{queue_row['preprocessing'].get('overrides', {})}`")
        reuse_train_artifact = queue_row.get("reuse_train_artifact")
        if isinstance(reuse_train_artifact, Mapping):
            lines.append(f"- Reuse train artifact: `{reuse_train_artifact.get('run_dir')}`")
            lines.append(
                "- Reuse training surface fingerprint: "
                f"`{reuse_train_artifact.get('training_surface_fingerprint')}`"
            )
        lines.append("- Parameter adequacy plan:")
        for plan_item in cast(list[str], queue_row.get("parameter_adequacy_plan", [])):
            lines.append(f"  - {plan_item}")
        if cast(list[str], queue_row.get("adequacy_knobs", [])):
            lines.append("- Adequacy knobs to dimension explicitly:")
            for adequacy_knob in cast(list[str], queue_row["adequacy_knobs"]):
                lines.append(f"  - {adequacy_knob}")
        lines.append(f"- Execution policy: `{queue_row.get('execution_policy', 'benchmark_full')}`")
        lines.append(
            "- Benchmark checkpoint selection: "
            f"`{queue_row.get('benchmark_checkpoint_selection', 'all')}`"
        )
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
                lines.append("- Benchmark metrics:")
                inline_objective_metric = objective_metric_from_queue_metrics(
                    cast(Mapping[str, Any], inline_metrics)
                )
                if (
                    not is_classification_objective_metric(inline_objective_metric)
                    and inline_metrics.get("best_bpc") is not None
                    and inline_metrics.get("final_bpc") is not None
                ):
                    best_bpc = float(inline_metrics["best_bpc"])
                    step = inline_metrics.get("best_step", "?")
                    final_bpc = float(inline_metrics["final_bpc"])
                    drift = float(inline_metrics["drift"])
                    lines.append(
                        "  - "
                        f"{display_metric_label('Best BPC', metric_key='best_bpc', objective_metric=inline_objective_metric)}: "
                        f"`{best_bpc:.4f}` (step {step})"
                    )
                    lines.append(
                        "  - "
                        f"{display_metric_label('Final BPC', metric_key='final_bpc', objective_metric=inline_objective_metric)}: "
                        f"`{final_bpc:.4f}`"
                    )
                    if inline_metrics.get("final_bpf") is not None:
                        lines.append(
                            "  - "
                            f"{display_metric_label('Final BPF', metric_key='final_bpf', objective_metric=inline_objective_metric)}: "
                            f"`{float(inline_metrics['final_bpf']):.4f}`"
                        )
                    lines.append(f"  - Drift (final − best): `{drift:.4f}`")
                elif inline_metrics.get("best_log_loss") is not None and inline_metrics.get("final_log_loss") is not None:
                    best_log_loss = float(inline_metrics["best_log_loss"])
                    step = inline_metrics.get("best_step", "?")
                    final_log_loss = float(inline_metrics["final_log_loss"])
                    drift = float(inline_metrics["drift"])
                    lines.append(f"  - Best log loss: `{best_log_loss:.4f}` (step {step})")
                    lines.append(f"  - Final log loss: `{final_log_loss:.4f}`")
                    if inline_metrics.get("final_brier_score") is not None:
                        lines.append(
                            f"  - Final Brier score: `{float(inline_metrics['final_brier_score']):.4f}`"
                        )
                    if inline_metrics.get("final_roc_auc") is not None:
                        lines.append(
                            f"  - Final ROC AUC: `{float(inline_metrics['final_roc_auc']):.4f}`"
                        )
                    lines.append(f"  - Drift (final − best): `{drift:.4f}`")
                    if is_classification_objective_metric(inline_objective_metric):
                        if inline_metrics.get("final_bpc") is not None or inline_metrics.get("final_bpf") is not None:
                            lines.append(
                                "  - Legacy feature-cell diagnostics remain secondary to log loss on classification-objective rows."
                            )
                        if inline_metrics.get("final_bpc") is not None:
                            lines.append(
                                "  - "
                                f"{display_metric_label('Final BPC', metric_key='final_bpc', objective_metric=inline_objective_metric)}: "
                                f"`{float(inline_metrics['final_bpc']):.4f}`"
                            )
                        if inline_metrics.get("final_bpf") is not None:
                            lines.append(
                                "  - "
                                f"{display_metric_label('Final BPF', metric_key='final_bpf', objective_metric=inline_objective_metric)}: "
                                f"`{float(inline_metrics['final_bpf']):.4f}`"
                            )
                else:
                    best = float(inline_metrics["best_roc_auc"])
                    step = inline_metrics.get("best_step", "?")
                    final = float(inline_metrics["final_roc_auc"])
                    drift = float(inline_metrics["drift"])
                    lines.append(f"  - Best ROC AUC: `{best:.4f}` (step {step})")
                    lines.append(f"  - Final ROC AUC: `{final:.4f}`")
                    lines.append(f"  - Drift (final − best): `{drift:.4f}`")
                primary_external_best = inline_metrics.get("primary_external_best")
                if primary_external_best is not None:
                    primary_external_label = str(
                        inline_metrics.get("primary_external_label", "External benchmark")
                    )
                    control_metric_label = "control"
                    if is_classification_objective_metric(inline_objective_metric):
                        control_metric_label = "control log loss"
                    lines.append(
                        f"  - {primary_external_label} {control_metric_label}: "
                        f"`{float(primary_external_best):.4f}`"
                    )
                elif "nanotabpfn_best" in inline_metrics:
                    nanotabpfn_control_label = "control"
                    if is_classification_objective_metric(inline_objective_metric):
                        nanotabpfn_control_label = "control log loss"
                    lines.append(
                        f"  - {EXTERNAL_BENCHMARK_LABELS['nanotabpfn']} {nanotabpfn_control_label}: "
                        f"`{float(inline_metrics['nanotabpfn_best']):.4f}`"
                    )
                if "max_grad_norm" in inline_metrics:
                    lines.append(f"  - max_grad_norm: `{float(inline_metrics['max_grad_norm']):.3f}`")
            else:
                lines.append("- Benchmark metrics: pending")
        else:
            if anchor is None:
                lines.append(
                    f"- Registered run: `{run_id}` with {', '.join(_run_metric_parts_without_anchor(run))}"
                )
                lines.append("")
                continue
            metrics = metric_summary(run, anchor)
            objective_metric = objective_metric_from_run(run)
            ordered_metric_keys = list(preferred_final_metric_keys(objective_metric))
            ordered_metric_keys.extend(
                key
                for key in (
                    "final_bpc",
                    "final_bpf",
                    "final_log_loss",
                    "final_brier_score",
                    "best_roc_auc",
                    "final_roc_auc",
                    "final_crps",
                    "final_avg_pinball_loss",
                    "final_picp_90",
                )
                if key not in ordered_metric_keys
            )
            metric_part_specs = {
                "final_bpc": ("final BPC", "delta_final_bpc"),
                "final_bpf": ("final BPF", "delta_final_bpf"),
                "final_log_loss": ("final log loss", "delta_final_log_loss"),
                "final_brier_score": ("final Brier score", "delta_final_brier_score"),
                "best_roc_auc": ("best ROC AUC", None),
                "final_roc_auc": ("final ROC AUC", "delta_final_roc_auc"),
                "final_crps": ("final CRPS", "delta_final_crps"),
                "final_avg_pinball_loss": ("final avg pinball loss", "delta_final_avg_pinball_loss"),
                "final_picp_90": ("final PICP 90", "delta_final_picp_90"),
            }
            metric_parts: list[str] = []
            for metric_key in ordered_metric_keys:
                spec = metric_part_specs.get(metric_key)
                if spec is None:
                    continue
                label, delta_key = spec
                rendered_label = display_metric_label(
                    label,
                    metric_key=metric_key,
                    objective_metric=objective_metric,
                )
                metric_parts.append(f"{rendered_label} `{metrics[metric_key]}`")
                if delta_key is not None:
                    metric_parts.append(
                        f"{display_metric_label(f'delta {label.lower()}', metric_key=delta_key, objective_metric=objective_metric)} "
                        f"`{metrics[delta_key]}`"
                    )
            metric_parts.extend(
                [
                    f"final-minus-best `{metrics['final_minus_best']}`",
                    f"delta drift `{metrics['delta_drift']}`",
                    f"delta final training time `{metrics['delta_training_time']}`",
                ]
            )
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
            path=write_resolved_system_delta_queue(
                sweep_id=sweep_id,
                index_path=index_path,
                catalog_path=catalog_path,
                sweeps_root=sweeps_root,
            ),
            sweep_id=sweep_id,
            index_path=index_path,
            catalog_path=catalog_path,
            sweeps_root=sweeps_root,
        )
    )
    resolved_sweep_id = _require_non_empty_string(
        sweep_id if sweep_id is not None else resolved_queue.get("sweep_id"),
        context="sweep_id",
    )
    resolved_out_path = (
        sweep_matrix_path(resolved_sweep_id, sweeps_root=sweeps_root)
        if out_path is None
        else Path(out_path).expanduser().resolve()
    )
    contents = render_system_delta_matrix(resolved_queue, registry_path=registry_path)
    write_text(resolved_out_path, contents)
    return resolved_out_path
