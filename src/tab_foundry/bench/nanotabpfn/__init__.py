"""Notebook-style nanoTabPFN comparison helpers."""

from __future__ import annotations

import openml
from openml.tasks import TaskType

from .artifacts import (
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
    default_benchmark_bundle_path,
    load_benchmark_bundle,
    load_benchmark_bundle_for_execution,
    normalize_benchmark_bundle,
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
from .datasets import (
    BenchmarkDatasetEvaluationError,
    PreparedOpenMLBenchmarkTask,
    get_feature_preprocessor,
    load_dataset_cache,
    load_openml_benchmark_datasets,
    prepare_openml_benchmark_task,
    read_required_openml_quality,
    save_dataset_cache,
)
from .metrics import (
    dataset_avg_pinball_loss_metrics,
    dataset_brier_score_metrics,
    dataset_crps_metrics,
    dataset_log_loss_metrics,
    dataset_picp_90_metrics,
    dataset_roc_auc_metrics,
    evaluate_classifier,
    evaluate_regressor,
)
from .summary import aggregate_curve, build_comparison_summary, plot_comparison_curve


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
    "benchmark_bundle_allows_missing_values",
    "benchmark_bundle_summary",
    "benchmark_bundle_task_type",
    "build_comparison_summary",
    "collect_checkpoint_snapshots",
    "curve_adjacent_ci_overlap_fraction",
    "curve_summary",
    "dataset_avg_pinball_loss_metrics",
    "dataset_brier_score_metrics",
    "dataset_crps_metrics",
    "dataset_log_loss_metrics",
    "dataset_picp_90_metrics",
    "dataset_roc_auc_metrics",
    "default_benchmark_bundle_path",
    "evaluate_classifier",
    "evaluate_regressor",
    "evaluate_tab_foundry_run",
    "get_feature_preprocessor",
    "load_benchmark_bundle",
    "load_benchmark_bundle_for_execution",
    "load_dataset_cache",
    "load_openml_benchmark_datasets",
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
]
