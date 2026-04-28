"""Notebook-style nanoTabPFN comparison helpers."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .artifacts import (
    benchmark_host_fingerprint,
    collect_checkpoint_snapshots,
    evaluate_tab_foundry_run,
    resolve_device,
    resolve_tab_foundry_best_checkpoint,
    resolve_tab_foundry_run_artifact_paths,
)
from .bundle import (
    BENCHMARK_BUNDLE_FILENAME,
    benchmark_bundle_allows_missing_values,
    benchmark_bundle_summary,
    benchmark_bundle_task_type,
    default_anchor_benchmark_summary,
    default_anchor_control_baseline_id,
    default_benchmark_bundle_path,
    default_benchmark_manifest_path,
    load_benchmark_bundle,
    load_benchmark_bundle_for_execution,
    normalize_benchmark_bundle,
    validate_default_anchor_benchmark_summary,
)
from .curves import (
    DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_CONFIDENCE,
    DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_SAMPLES,
    DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_SEED,
    annotate_curve_records_with_task_statistics,
    curve_adjacent_ci_overlap_fraction,
    curve_summary,
    summarize_checkpoint_curve,
    task_bootstrap_roc_auc_interval,
)
from .dataset_common import (
    BenchmarkDatasetEvaluationError,
    load_dataset_cache,
    save_dataset_cache,
)
from .metrics import (
    dataset_avg_pinball_loss_metrics,
    dataset_bpc_metrics,
    dataset_bpf_metrics,
    dataset_brier_score_metrics,
    dataset_crps_metrics,
    dataset_log_loss_metrics,
    dataset_picp_90_metrics,
    dataset_roc_auc_metrics,
    evaluate_classifier,
    evaluate_regressor,
)

_LAZY_EXPORTS = {
    "TaskType": ("openml.tasks", "TaskType"),
    "aggregate_curve": (".summary", "aggregate_curve"),
    "build_comparison_summary": (".summary", "build_comparison_summary"),
    "plot_comparison_curve": (".summary", "plot_comparison_curve"),
    "validate_openml_missingness_manifest": (
        ".missingness_validation",
        "validate_openml_missingness_manifest",
    ),
}
_DATASET_EXPORTS = {
    "PreparedOpenMLBenchmarkTask",
    "benchmark_manifest_allows_missing_values",
    "benchmark_manifest_bundle_summary",
    "benchmark_manifest_task_type",
    "load_benchmark_manifest_datasets",
    "get_feature_preprocessor",
    "prepare_openml_benchmark_task",
    "read_required_openml_quality",
}


__all__ = [
    "BENCHMARK_BUNDLE_FILENAME",
    "BenchmarkDatasetEvaluationError",
    "DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_CONFIDENCE",
    "DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_SAMPLES",
    "DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_SEED",
    "PreparedOpenMLBenchmarkTask",
    "TaskType",
    "aggregate_curve",
    "annotate_curve_records_with_task_statistics",
    "benchmark_host_fingerprint",
    "benchmark_bundle_allows_missing_values",
    "benchmark_bundle_summary",
    "benchmark_bundle_task_type",
    "benchmark_manifest_allows_missing_values",
    "benchmark_manifest_bundle_summary",
    "benchmark_manifest_task_type",
    "build_comparison_summary",
    "collect_checkpoint_snapshots",
    "curve_adjacent_ci_overlap_fraction",
    "curve_summary",
    "dataset_avg_pinball_loss_metrics",
    "dataset_bpc_metrics",
    "dataset_bpf_metrics",
    "dataset_brier_score_metrics",
    "dataset_crps_metrics",
    "dataset_log_loss_metrics",
    "dataset_picp_90_metrics",
    "dataset_roc_auc_metrics",
    "default_anchor_benchmark_summary",
    "default_anchor_control_baseline_id",
    "default_benchmark_bundle_path",
    "default_benchmark_manifest_path",
    "evaluate_classifier",
    "evaluate_regressor",
    "evaluate_tab_foundry_run",
    "get_feature_preprocessor",
    "load_benchmark_bundle",
    "load_benchmark_manifest_datasets",
    "load_benchmark_bundle_for_execution",
    "load_dataset_cache",
    "normalize_benchmark_bundle",
    "openml",
    "plot_comparison_curve",
    "prepare_openml_benchmark_task",
    "read_required_openml_quality",
    "resolve_device",
    "resolve_tab_foundry_best_checkpoint",
    "resolve_tab_foundry_run_artifact_paths",
    "save_dataset_cache",
    "summarize_checkpoint_curve",
    "task_bootstrap_roc_auc_interval",
    "validate_default_anchor_benchmark_summary",
    "validate_openml_missingness_manifest",
]


def __getattr__(name: str) -> Any:
    if name == "openml":
        return import_module("openml")
    export = _LAZY_EXPORTS.get(name)
    if export is not None:
        module_name, attribute_name = export
        module = (
            import_module(module_name, __name__)
            if module_name.startswith(".")
            else import_module(module_name)
        )
        return getattr(module, attribute_name)
    if name in _DATASET_EXPORTS:
        module = import_module(".datasets", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
