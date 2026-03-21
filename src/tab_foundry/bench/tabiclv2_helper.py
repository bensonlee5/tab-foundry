"""External TabICLv2 benchmark helper entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any, Sequence

import numpy as np


DEFAULT_TABICLV2_QUANTILE_LEVELS = np.asarray(
    [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate TabICLv2 on cached benchmark datasets")
    parser.add_argument("--tab-foundry-src", required=True, help="tab-foundry src directory for shared helpers")
    parser.add_argument("--dataset-cache", required=True, help="Path to cached benchmark datasets (.npz)")
    parser.add_argument("--out-path", required=True, help="Output JSONL path")
    parser.add_argument(
        "--task-type",
        required=True,
        choices=("supervised_classification", "supervised_regression"),
        help="Benchmark task type",
    )
    parser.add_argument("--checkpoint-version", required=True, help="TabICLv2 checkpoint version")
    parser.add_argument("--device", default="auto", help="Device override")
    parser.add_argument(
        "--allow-missing-values",
        action="store_true",
        help="Permit missing-valued benchmark inputs when the bundle explicitly allows them",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    src_root = Path(str(args.tab_foundry_src)).expanduser().resolve()
    tabicl_root = Path.cwd().resolve()
    if str(tabicl_root) not in sys.path:
        sys.path.insert(0, str(tabicl_root))
    if str(src_root) not in sys.path:
        sys.path.insert(0, str(src_root))

    try:
        from tabicl import TabICLClassifier, TabICLRegressor
    except ImportError as exc:
        raise RuntimeError(
            f"tabicl import unavailable in helper env: {tabicl_root}; "
            "ensure the sibling TabICLv2 environment is bootstrapped"
        ) from exc

    from tab_foundry.bench.artifacts import write_jsonl
    from tab_foundry.bench.nanotabpfn import (
        dataset_avg_pinball_loss_metrics,
        dataset_brier_score_metrics,
        dataset_crps_metrics,
        dataset_log_loss_metrics,
        dataset_picp_90_metrics,
        dataset_roc_auc_metrics,
        evaluate_classifier,
        evaluate_regressor,
        load_dataset_cache,
    )

    device = None if str(args.device).strip().lower() == "auto" else str(args.device).strip()
    checkpoint_version = str(args.checkpoint_version).strip()
    if not checkpoint_version:
        raise RuntimeError("checkpoint_version must be a non-empty string")
    datasets = load_dataset_cache(Path(str(args.dataset_cache)).expanduser().resolve())
    allow_missing_values = bool(args.allow_missing_values)
    started_at = time.perf_counter()

    if str(args.task_type) == "supervised_classification":
        classifier = TabICLClassifier(
            kv_cache=False,
            checkpoint_version=checkpoint_version,
            device=device,
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
                checkpoint_version=checkpoint_version,
                device=device,
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

    write_jsonl(Path(str(args.out_path)).expanduser().resolve(), records)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
