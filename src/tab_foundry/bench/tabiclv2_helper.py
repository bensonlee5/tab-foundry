"""Library helper for external TabICLv2 benchmark execution."""

from __future__ import annotations

from pathlib import Path
import sys
import time
from typing import Any

import numpy as np


DEFAULT_TABICLV2_QUANTILE_LEVELS = np.asarray(
    [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95],
    dtype=np.float64,
)


class TabICLv2QuantileRegressorAdapter:
    """Expose the predict_quantiles surface expected by the shared evaluator."""

    def __init__(self, regressor: Any, *, quantile_levels: np.ndarray | None = None) -> None:
        self._regressor = regressor
        self._quantile_levels = np.asarray(
            DEFAULT_TABICLV2_QUANTILE_LEVELS if quantile_levels is None else quantile_levels,
            dtype=np.float64,
        )

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "TabICLv2QuantileRegressorAdapter":
        self._regressor.fit(x_train, y_train)
        return self

    def predict_quantiles(self, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        quantiles = self._regressor.predict(
            x_test,
            output_type="quantiles",
            alphas=self._quantile_levels.tolist(),
        )
        return np.asarray(quantiles, dtype=np.float64), np.asarray(
            self._quantile_levels,
            dtype=np.float64,
        )


def run_tabiclv2_helper(
    *,
    tab_foundry_src: Path,
    benchmark_manifest: Path,
    out_path: Path,
    task_type: str,
    checkpoint_version: str,
    device: str = "auto",
    allow_missing_values: bool = False,
    tab_realdata_hub_root: Path | None = None,
    helper_root: Path | None = None,
) -> int:
    """Evaluate TabICLv2 on a manifest-backed benchmark surface."""

    src_root = tab_foundry_src.expanduser().resolve()
    tabicl_root = Path.cwd().resolve() if helper_root is None else helper_root.expanduser().resolve()
    if str(tabicl_root) not in sys.path:
        sys.path.insert(0, str(tabicl_root))
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))
    from tab_foundry.bench.helper_imports import prepend_explicit_tab_realdata_hub_src

    prepend_explicit_tab_realdata_hub_src(
        sys.path,
        tab_realdata_hub_root=tab_realdata_hub_root,
    )

    try:
        from tabicl import TabICLClassifier, TabICLRegressor
    except ImportError as exc:
        raise RuntimeError(
            f"tabicl import unavailable in helper env: {tabicl_root}; "
            "ensure the sibling TabICLv2 environment is bootstrapped"
        ) from exc

    try:
        from tab_realdata_hub.manifest import load_manifest_datasets
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "tab-realdata-hub import unavailable in the TabICLv2 helper env; "
            "run `tab-foundry bench env bootstrap` first"
        ) from exc
    from tab_foundry.bench.artifacts import write_jsonl
    from tab_foundry.bench.openml_benchmark.metrics import (
        dataset_avg_pinball_loss_metrics,
        dataset_brier_score_metrics,
        dataset_crps_metrics,
        dataset_log_loss_metrics,
        dataset_picp_90_metrics,
        dataset_roc_auc_metrics,
        evaluate_classifier,
        evaluate_regressor,
    )

    resolved_device = None if str(device).strip().lower() == "auto" else str(device).strip()
    resolved_checkpoint_version = str(checkpoint_version).strip()
    resolved_task_type = str(task_type).strip()
    benchmark_manifest_path = benchmark_manifest.expanduser().resolve()
    out_path = out_path.expanduser().resolve()
    if not resolved_checkpoint_version:
        raise RuntimeError("checkpoint_version must be a non-empty string")
    if resolved_task_type not in {"supervised_classification", "supervised_regression"}:
        raise RuntimeError(f"unsupported task_type: {resolved_task_type!r}")
    datasets = load_manifest_datasets(
        benchmark_manifest_path,
        allow_missing_values=bool(allow_missing_values),
    ).datasets
    allow_missing_values = bool(allow_missing_values)
    started_at = time.perf_counter()

    if resolved_task_type == "supervised_classification":
        classifier = TabICLClassifier(
            kv_cache=False,
            checkpoint_version=resolved_checkpoint_version,
            device=resolved_device,
        )
        metrics = evaluate_classifier(
            classifier,
            datasets,
            allow_missing_values=allow_missing_values,
        )
        records = [
            {
                "seed": 0,
                "step": 0,
                "training_time": float(time.perf_counter() - started_at),
                "roc_auc": float(metrics["ROC AUC"]),
                "log_loss": float(metrics["Log Loss"]),
                "brier_score": float(metrics["Brier Score"]),
                "dataset_roc_auc": dataset_roc_auc_metrics(metrics),
                "dataset_log_loss": dataset_log_loss_metrics(metrics),
                "dataset_brier_score": dataset_brier_score_metrics(metrics),
            }
        ]
    else:
        regressor = TabICLv2QuantileRegressorAdapter(
            TabICLRegressor(
                kv_cache=False,
                checkpoint_version=resolved_checkpoint_version,
                device=resolved_device,
            )
        )
        metrics = evaluate_regressor(
            regressor,
            datasets,
            allow_missing_values=allow_missing_values,
        )
        records = [
            {
                "seed": 0,
                "step": 0,
                "training_time": float(time.perf_counter() - started_at),
                "crps": float(metrics["CRPS"]),
                "avg_pinball_loss": float(metrics["Average Pinball Loss"]),
                "picp_90": float(metrics["PICP 90"]),
                "dataset_crps": dataset_crps_metrics(metrics),
                "dataset_avg_pinball_loss": dataset_avg_pinball_loss_metrics(metrics),
                "dataset_picp_90": dataset_picp_90_metrics(metrics),
            }
        ]

    write_jsonl(out_path, records)
    return 0
