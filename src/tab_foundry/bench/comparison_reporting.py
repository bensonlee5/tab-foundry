"""Summary shaping and artifact publishing for benchmark comparison runs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

from tab_foundry.external_benchmarks import (
    EXTERNAL_BENCHMARK_NANOTABPFN,
    EXTERNAL_BENCHMARK_TABICLV2,
)
from tab_foundry.training.instability import gradient_history_path, telemetry_path

from .artifacts import write_json

_BENCHMARK_METRIC_KEYS = (
    "best_step",
    "best_training_time",
    "final_step",
    "final_training_time",
    "best_bpc",
    "final_bpc",
    "best_bpf",
    "final_bpf",
    "best_roc_auc",
    "final_roc_auc",
    "best_log_loss",
    "final_log_loss",
    "best_brier_score",
    "final_brier_score",
    "best_crps",
    "final_crps",
    "best_avg_pinball_loss",
    "final_avg_pinball_loss",
    "best_picp_90",
    "final_picp_90",
    "best_to_final_roc_auc_delta",
    "best_to_final_bpc_delta",
    "best_to_final_bpf_delta",
    "best_to_final_log_loss_delta",
    "best_to_final_brier_score_delta",
    "best_to_final_crps_delta",
    "best_to_final_avg_pinball_loss_delta",
    "best_to_final_picp_90_delta",
)


def _mapping_value(payload: Mapping[str, Any], key: str) -> Mapping[str, Any] | None:
    raw_value = payload.get(key)
    if not isinstance(raw_value, Mapping):
        return None
    return cast(Mapping[str, Any], raw_value)


def optional_non_empty_string(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return str(value).strip()


def _compact_metric_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in _BENCHMARK_METRIC_KEYS:
        if key in payload:
            metrics[key] = payload[key]
    return metrics


def benchmark_wandb_summary_payload(summary: Mapping[str, Any]) -> dict[str, Any]:
    benchmark_payload: dict[str, Any] = {"benchmark": {}}
    primary_external_benchmark = optional_non_empty_string(summary.get("primary_external_benchmark"))
    if primary_external_benchmark is not None:
        benchmark_payload["benchmark"]["primary_external_benchmark"] = primary_external_benchmark
    external_benchmarks = summary.get("external_benchmarks")
    if isinstance(external_benchmarks, list) and external_benchmarks:
        benchmark_payload["benchmark"]["external_benchmarks"] = [
            str(value)
            for value in external_benchmarks
            if isinstance(value, str) and value.strip()
        ]
    tab_foundry = _mapping_value(summary, "tab_foundry")
    if tab_foundry is not None:
        tab_foundry_payload = _compact_metric_payload(tab_foundry)
        training_diagnostics = _mapping_value(tab_foundry, "training_diagnostics")
        if training_diagnostics:
            tab_foundry_payload["training_diagnostics"] = dict(training_diagnostics)
        if tab_foundry_payload:
            benchmark_payload["benchmark"]["tab_foundry"] = tab_foundry_payload
        model_size = _mapping_value(tab_foundry, "model_size")
        if model_size:
            benchmark_payload["benchmark"]["model_size"] = dict(model_size)

    nanotabpfn = _mapping_value(summary, "nanotabpfn")
    if nanotabpfn is not None:
        nanotabpfn_payload = _compact_metric_payload(nanotabpfn)
        for key in ("num_seeds",):
            if key in nanotabpfn:
                nanotabpfn_payload[key] = nanotabpfn[key]
        if nanotabpfn_payload:
            benchmark_payload["benchmark"]["nanotabpfn"] = nanotabpfn_payload
    tabiclv2 = _mapping_value(summary, "tabiclv2")
    if tabiclv2 is not None:
        tabiclv2_payload = _compact_metric_payload(tabiclv2)
        for key in ("checkpoint_version",):
            if key in tabiclv2:
                tabiclv2_payload[key] = tabiclv2[key]
        if tabiclv2_payload:
            benchmark_payload["benchmark"]["tabiclv2"] = tabiclv2_payload
    return benchmark_payload if benchmark_payload["benchmark"] else {}


def finalize_benchmark_summary(
    *,
    summary: dict[str, Any],
    requested_external_benchmarks: Sequence[str],
    primary_external_benchmark: str | None,
    nanotabpfn_records: Sequence[Mapping[str, Any]],
    tabiclv2_records: Sequence[Mapping[str, Any]],
    benchmark_tasks_path: Path,
    tab_foundry_curve_path: Path,
    nanotabpfn_curve_path: Path,
    tabiclv2_curve_path: Path,
    comparison_curve_path: Path,
    benchmark_manifest_path: Path,
    comparison_summary_path: Path,
    benchmark_run_record_path: Path,
    training_surface_record_path: Path,
    tab_foundry_run_dir: Path,
    derive_benchmark_run_record_fn: Callable[..., dict[str, Any]],
    posthoc_update_wandb_summary_fn: Callable[..., Any],
) -> dict[str, Any]:
    summary["external_benchmarks"] = list(requested_external_benchmarks)
    if primary_external_benchmark is not None:
        summary["primary_external_benchmark"] = primary_external_benchmark
    gradient_history_jsonl = gradient_history_path(tab_foundry_run_dir)
    telemetry_json = telemetry_path(tab_foundry_run_dir)
    primary_external_curve_jsonl: str | None
    if primary_external_benchmark == EXTERNAL_BENCHMARK_NANOTABPFN and nanotabpfn_records:
        primary_external_curve_jsonl = str(nanotabpfn_curve_path)
    elif primary_external_benchmark == EXTERNAL_BENCHMARK_TABICLV2 and tabiclv2_records:
        primary_external_curve_jsonl = str(tabiclv2_curve_path)
    else:
        primary_external_curve_jsonl = None
    summary["artifacts"] = {
        "benchmark_tasks_json": str(benchmark_tasks_path),
        "tab_foundry_curve_jsonl": str(tab_foundry_curve_path),
        "primary_external_curve_jsonl": primary_external_curve_jsonl,
        "nanotabpfn_curve_jsonl": (
            str(nanotabpfn_curve_path)
            if EXTERNAL_BENCHMARK_NANOTABPFN in requested_external_benchmarks and nanotabpfn_records
            else None
        ),
        "tabiclv2_curve_jsonl": (
            str(tabiclv2_curve_path)
            if EXTERNAL_BENCHMARK_TABICLV2 in requested_external_benchmarks and tabiclv2_records
            else None
        ),
        "comparison_curve_png": str(comparison_curve_path),
        "benchmark_manifest": str(benchmark_manifest_path),
        "gradient_history_jsonl": (
            str(gradient_history_jsonl.resolve()) if gradient_history_jsonl.exists() else None
        ),
        "telemetry_json": str(telemetry_json.resolve()) if telemetry_json.exists() else None,
        "benchmark_run_record_json": str(benchmark_run_record_path),
        "training_surface_record_json": str(training_surface_record_path),
    }
    write_json(comparison_summary_path, summary)
    benchmark_run_record = derive_benchmark_run_record_fn(
        run_dir=tab_foundry_run_dir,
        comparison_summary_path=comparison_summary_path,
        benchmark_run_record_path=benchmark_run_record_path,
    )
    tab_foundry_summary = cast(dict[str, Any], summary["tab_foundry"])
    tab_foundry_summary["manifest_path"] = str(benchmark_run_record["manifest_path"])
    tab_foundry_summary["seed_set"] = list(benchmark_run_record["seed_set"])
    tab_foundry_summary["training_diagnostics"] = dict(benchmark_run_record["training_diagnostics"])
    tab_foundry_summary["model_size"] = dict(benchmark_run_record["model_size"])
    summary["artifacts"]["training_surface_record_json"] = cast(
        dict[str, Any],
        benchmark_run_record["artifacts"],
    ).get("training_surface_record_path")
    if benchmark_run_record.get("surface_labels") is not None:
        tab_foundry_summary["surface_labels"] = dict(benchmark_run_record["surface_labels"])
    write_json(comparison_summary_path, summary)
    write_json(benchmark_run_record_path, benchmark_run_record)
    _ = posthoc_update_wandb_summary_fn(
        telemetry_path=telemetry_json,
        payload=benchmark_wandb_summary_payload(summary),
    )
    return summary
