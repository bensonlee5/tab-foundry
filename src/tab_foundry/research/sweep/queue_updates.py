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
