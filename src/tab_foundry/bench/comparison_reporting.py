"""Summary shaping helpers for benchmark comparison runs."""

from __future__ import annotations

from typing import Any, Mapping, cast

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
