"""Queue-update helpers for system-delta execution."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, cast


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


def _stage_local_telemetry_metrics(run_dir: Path) -> dict[str, Any]:
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
    nanotabpfn = cast(dict[str, Any], summary["nanotabpfn"])
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

    metric_keys = (
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
        nanotabpfn_value = optional_metric(nanotabpfn, metric_key)
        if nanotabpfn_value is not None:
            metrics[f"nanotabpfn_{metric_key}"] = nanotabpfn_value

    delta_keys = {
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

    if metrics.get("final_minus_best_roc_auc") is not None:
        metrics["drift"] = metrics["final_minus_best_roc_auc"]
    if metrics.get("nanotabpfn_best_roc_auc") is not None:
        metrics["nanotabpfn_best"] = metrics["nanotabpfn_best_roc_auc"]
    if metrics.get("nanotabpfn_final_roc_auc") is not None:
        metrics["nanotabpfn_final"] = metrics["nanotabpfn_final_roc_auc"]

    if run_entry is not None:
        comparison_keys = {
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

    metrics.update(_stage_local_telemetry_metrics(run_dir))

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
