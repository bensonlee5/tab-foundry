"""Queue-update helpers for system-delta execution."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, cast

from tab_foundry.external_benchmarks import (
    EXTERNAL_BENCHMARK_LABELS,
    EXTERNAL_BENCHMARK_NANOTABPFN,
    EXTERNAL_BENCHMARK_TABICLV2,
)

from .objective_metrics import objective_metric_from_run, preferred_drift_metric_keys
from .objective_metrics import is_classification_objective_metric


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if not path.exists():
        return records
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            records.append(cast(dict[str, Any], payload))
    return records


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected JSON mapping at {path}")
    return cast(dict[str, Any], payload)


def clipped_step_fraction(records: list[dict[str, Any]]) -> float:
    ordered_records = sorted(records, key=lambda record: int(record.get("step", 0)))
    if not ordered_records:
        return 0.0
    clipped_steps = sum(1 for record in ordered_records if bool(record.get("grad_clip_triggered", False)))
    return float(clipped_steps / float(len(ordered_records)))


def optional_metric(payload: Mapping[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        raise RuntimeError(f"{key} must be finite when present")
    return numeric


def comparison_metric(run_entry: Mapping[str, Any], key: str) -> float | None:
    comparisons = run_entry.get("comparisons")
    if not isinstance(comparisons, Mapping):
        return None
    vs_anchor = comparisons.get("vs_anchor")
    if not isinstance(vs_anchor, Mapping):
        return None
    return optional_metric(cast(Mapping[str, Any], vs_anchor), key)


def _nested_mapping_value(payload: Mapping[str, Any], *keys: str) -> Mapping[str, Any] | None:
    current: Mapping[str, Any] | None = payload
    for key in keys:
        if current is None:
            return None
        next_value = current.get(key)
        if not isinstance(next_value, Mapping):
            return None
        current = cast(Mapping[str, Any], next_value)
    return current


def _primary_external_benchmark(summary: Mapping[str, Any]) -> tuple[str | None, Mapping[str, Any] | None]:
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


def stage_local_telemetry_metrics(run_dir: Path) -> dict[str, Any]:
    telemetry_payload = read_json(run_dir / "telemetry.json")
    if telemetry_payload is None:
        return {}
    diagnostics = telemetry_payload.get("diagnostics")
    if not isinstance(diagnostics, Mapping):
        return {}

    metrics: dict[str, Any] = {}
    stage_local_gradients = diagnostics.get("stage_local_gradients")
    if isinstance(stage_local_gradients, Mapping):
        modules = stage_local_gradients.get("modules")
        if isinstance(modules, Mapping):
            for module_name in ("column_encoder", "row_pool", "context_encoder"):
                final_window = _nested_mapping_value(
                    cast(Mapping[str, Any], modules),
                    module_name,
                    "windows",
                    "final_10pct",
                )
                final_window_mean = (
                    None
                    if final_window is None
                    else optional_metric(final_window, "mean_grad_norm")
                )
                if final_window_mean is not None:
                    metrics[f"{module_name}_final_window_mean_grad_norm"] = final_window_mean

    activation_windows = diagnostics.get("activation_windows")
    tracked_activations = (
        activation_windows.get("tracked_activations")
        if isinstance(activation_windows, Mapping)
        else None
    )
    if isinstance(tracked_activations, Mapping):
        for activation_name, prefix in (
            ("post_column_encoder", "column"),
            ("post_row_pool", "row"),
            ("post_context_encoder", "context"),
        ):
            activation_payload = tracked_activations.get(activation_name)
            if not isinstance(activation_payload, Mapping):
                continue
            early_to_final_mean_delta = optional_metric(
                cast(Mapping[str, Any], activation_payload),
                "early_to_final_mean_delta",
            )
            if early_to_final_mean_delta is not None:
                metrics[f"{prefix}_activation_early_to_final_mean_delta"] = early_to_final_mean_delta
            final_window = _nested_mapping_value(
                cast(Mapping[str, Any], activation_payload),
                "windows",
                "final_10pct",
            )
            final_window_mean = (
                None
                if final_window is None
                else optional_metric(final_window, "mean")
            )
            if final_window_mean is not None:
                metrics[f"{prefix}_activation_final_window_mean"] = final_window_mean

    return metrics


def queue_metrics(
    summary: Mapping[str, Any],
    *,
    run_dir: Path,
    run_entry: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    tab_foundry = cast(dict[str, Any], summary["tab_foundry"])
    primary_external_name, primary_external = _primary_external_benchmark(summary)
    raw_nanotabpfn = summary.get("nanotabpfn")
    nanotabpfn = (
        cast(dict[str, Any], raw_nanotabpfn)
        if isinstance(raw_nanotabpfn, Mapping)
        else {}
    )
    gradient_records = read_jsonl(run_dir / "gradient_history.jsonl")
    max_grad_norm = optional_metric(
        cast(dict[str, Any], tab_foundry["training_diagnostics"]),
        "max_grad_norm",
    )
    if max_grad_norm is None:
        raise RuntimeError("benchmark summary omitted training_diagnostics.max_grad_norm")
    best_step = optional_metric(tab_foundry, "best_step")
    if best_step is None:
        raise RuntimeError("benchmark summary omitted tab_foundry.best_step")

    metrics: dict[str, Any] = {
        "best_step": int(best_step),
        "max_grad_norm": max_grad_norm,
        "clipped_step_fraction": clipped_step_fraction(gradient_records),
    }
    objective_metric = objective_metric_from_run(run_entry)
    if objective_metric is not None:
        metrics["objective_metric"] = objective_metric
    if primary_external_name is not None:
        metrics["primary_external_benchmark"] = primary_external_name
        metrics["primary_external_label"] = EXTERNAL_BENCHMARK_LABELS.get(
            primary_external_name,
            primary_external_name,
        )

    if is_classification_objective_metric(objective_metric):
        metric_keys = (
            "best_log_loss",
            "final_log_loss",
            "best_brier_score",
            "final_brier_score",
            "best_roc_auc",
            "final_roc_auc",
            "best_bpc",
            "final_bpc",
            "best_bpf",
            "final_bpf",
            "best_crps",
            "final_crps",
            "best_avg_pinball_loss",
            "final_avg_pinball_loss",
            "best_picp_90",
            "final_picp_90",
        )
    else:
        metric_keys = (
            "best_bpc",
            "final_bpc",
            "best_bpf",
            "final_bpf",
            "best_log_loss",
            "final_log_loss",
            "best_brier_score",
            "final_brier_score",
            "best_roc_auc",
            "final_roc_auc",
            "best_crps",
            "final_crps",
            "best_avg_pinball_loss",
            "final_avg_pinball_loss",
            "best_picp_90",
            "final_picp_90",
        )
    for metric_key in metric_keys:
        tab_foundry_value = optional_metric(tab_foundry, metric_key)
        if tab_foundry_value is not None:
            metrics[metric_key] = tab_foundry_value
        if primary_external is not None:
            primary_external_value = optional_metric(primary_external, metric_key)
            if primary_external_value is not None:
                metrics[f"primary_external_{metric_key}"] = primary_external_value
        nanotabpfn_value = optional_metric(nanotabpfn, metric_key)
        if nanotabpfn_value is not None:
            metrics[f"nanotabpfn_{metric_key}"] = nanotabpfn_value

    delta_keys = {
        "best_to_final_bpc_delta": "final_minus_best_bpc",
        "best_to_final_bpf_delta": "final_minus_best_bpf",
        "best_to_final_log_loss_delta": "final_minus_best_log_loss",
        "best_to_final_brier_score_delta": "final_minus_best_brier_score",
        "best_to_final_roc_auc_delta": "final_minus_best_roc_auc",
        "best_to_final_crps_delta": "final_minus_best_crps",
        "best_to_final_avg_pinball_loss_delta": "final_minus_best_avg_pinball_loss",
        "best_to_final_picp_90_delta": "final_minus_best_picp_90",
    }
    for summary_key, queue_key in delta_keys.items():
        value = optional_metric(tab_foundry, summary_key)
        if value is not None:
            metrics[queue_key] = value

    for drift_key in preferred_drift_metric_keys(objective_metric):
        drift_value = metrics.get(drift_key)
        if drift_value is not None:
            metrics["drift"] = drift_value
            break
    preferred_external_metric_pairs = (
        ("primary_external_best_log_loss", "primary_external_final_log_loss", "primary_external_best", "primary_external_final"),
        ("primary_external_best_roc_auc", "primary_external_final_roc_auc", "primary_external_best", "primary_external_final"),
        ("primary_external_best_crps", "primary_external_final_crps", "primary_external_best", "primary_external_final"),
    )
    if not is_classification_objective_metric(objective_metric):
        preferred_external_metric_pairs = (
            ("primary_external_best_roc_auc", "primary_external_final_roc_auc", "primary_external_best", "primary_external_final"),
            ("primary_external_best_log_loss", "primary_external_final_log_loss", "primary_external_best", "primary_external_final"),
            ("primary_external_best_crps", "primary_external_final_crps", "primary_external_best", "primary_external_final"),
        )
    for best_key, final_key, generic_best_key, generic_final_key in preferred_external_metric_pairs:
        if metrics.get(best_key) is not None:
            metrics[generic_best_key] = metrics[best_key]
        if metrics.get(final_key) is not None:
            metrics[generic_final_key] = metrics[final_key]
        if metrics.get(generic_best_key) is not None or metrics.get(generic_final_key) is not None:
            break
    preferred_nanotabpfn_metric_pairs = (
        ("nanotabpfn_best_log_loss", "nanotabpfn_final_log_loss", "nanotabpfn_best", "nanotabpfn_final"),
        ("nanotabpfn_best_roc_auc", "nanotabpfn_final_roc_auc", "nanotabpfn_best", "nanotabpfn_final"),
        ("nanotabpfn_best_crps", "nanotabpfn_final_crps", "nanotabpfn_best", "nanotabpfn_final"),
    )
    if not is_classification_objective_metric(objective_metric):
        preferred_nanotabpfn_metric_pairs = (
            ("nanotabpfn_best_roc_auc", "nanotabpfn_final_roc_auc", "nanotabpfn_best", "nanotabpfn_final"),
            ("nanotabpfn_best_log_loss", "nanotabpfn_final_log_loss", "nanotabpfn_best", "nanotabpfn_final"),
            ("nanotabpfn_best_crps", "nanotabpfn_final_crps", "nanotabpfn_best", "nanotabpfn_final"),
        )
    for best_key, final_key, generic_best_key, generic_final_key in preferred_nanotabpfn_metric_pairs:
        if metrics.get(best_key) is not None:
            metrics[generic_best_key] = metrics[best_key]
        if metrics.get(final_key) is not None:
            metrics[generic_final_key] = metrics[final_key]
        if metrics.get(generic_best_key) is not None or metrics.get(generic_final_key) is not None:
            break

    if run_entry is not None:
        if is_classification_objective_metric(objective_metric):
            comparison_keys = {
                "final_log_loss_delta": "delta_final_log_loss",
                "final_brier_score_delta": "delta_final_brier_score",
                "final_roc_auc_delta": "delta_final_roc_auc",
                "final_bpc_delta": "delta_final_bpc",
                "final_bpf_delta": "delta_final_bpf",
                "final_crps_delta": "delta_final_crps",
                "final_avg_pinball_loss_delta": "delta_final_avg_pinball_loss",
                "final_picp_90_delta": "delta_final_picp_90",
            }
        else:
            comparison_keys = {
                "final_bpc_delta": "delta_final_bpc",
                "final_bpf_delta": "delta_final_bpf",
                "final_log_loss_delta": "delta_final_log_loss",
                "final_brier_score_delta": "delta_final_brier_score",
                "final_roc_auc_delta": "delta_final_roc_auc",
                "final_crps_delta": "delta_final_crps",
                "final_avg_pinball_loss_delta": "delta_final_avg_pinball_loss",
                "final_picp_90_delta": "delta_final_picp_90",
            }
        for comparison_key, queue_key in comparison_keys.items():
            value = comparison_metric(run_entry, comparison_key)
            if value is not None:
                metrics[queue_key] = value

    metrics.update(stage_local_telemetry_metrics(run_dir))

    return metrics


def append_note(notes: list[str], note: str) -> list[str]:
    updated = list(notes)
    if note not in updated:
        updated.append(note)
    return updated


def update_queue_row(
    *,
    queue_row: dict[str, Any],
    run_id: str,
    queue_metrics: Mapping[str, Any],
    decision: str,
    conclusion: str,
) -> None:
    original_run_id = queue_row.get("run_id")
    queue_row["status"] = "completed"
    queue_row["run_id"] = run_id
    queue_row["followup_run_ids"] = []
    queue_row["decision"] = decision
    queue_row["interpretation_status"] = "completed"
    queue_row["benchmark_metrics"] = dict(queue_metrics)
    queue_row["confounders"] = []
    notes = cast(list[str], queue_row.get("notes", []))
    if isinstance(original_run_id, str) and original_run_id.strip() and original_run_id != run_id:
        notes = append_note(
            notes,
            f"Supersedes historical queue run `{original_run_id}`; that registry entry is retained as history only.",
        )
    notes = append_note(notes, f"Canonical rerun registered as `{run_id}`.")
    notes = append_note(notes, conclusion)
    queue_row["notes"] = notes


def update_screened_queue_row(
    *,
    queue_row: dict[str, Any],
    run_id: str,
    screen_metrics: Mapping[str, Any],
    conclusion: str,
) -> None:
    original_run_id = queue_row.get("run_id")
    queue_row["status"] = "screened"
    queue_row["run_id"] = run_id
    queue_row["followup_run_ids"] = []
    queue_row["decision"] = "defer"
    queue_row["interpretation_status"] = "screened"
    queue_row["screen_metrics"] = dict(screen_metrics)
    queue_row["benchmark_metrics"] = None
    queue_row["confounders"] = []
    notes = cast(list[str], queue_row.get("notes", []))
    if isinstance(original_run_id, str) and original_run_id.strip() and original_run_id != run_id:
        notes = append_note(
            notes,
            f"Supersedes historical queue run `{original_run_id}`; that train-only screen is retained as history only.",
        )
    notes = append_note(notes, f"Train-only screen recorded as `{run_id}`.")
    notes = append_note(notes, conclusion)
    queue_row["notes"] = notes
