from __future__ import annotations

import numpy as np
import pytest

import tab_foundry.bench.nanotabpfn as benchmark_module
import tab_foundry.bench.nanotabpfn.metrics as benchmark_metrics_module


class _PerfectClassifier:
    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "_PerfectClassifier":
        self.classes_ = np.unique(np.asarray(y_train, dtype=np.int64))
        return self

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        labels = np.asarray(x_test[:, 0], dtype=np.int64)
        probabilities = np.zeros((labels.shape[0], int(self.classes_.size)), dtype=np.float64)
        class_to_index = {int(label): index for index, label in enumerate(self.classes_.tolist())}
        for row_index, label in enumerate(labels.tolist()):
            probabilities[row_index, class_to_index[int(label)]] = 1.0
        return probabilities


class _PerfectRegressor:
    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "_PerfectRegressor":
        _ = (x_train, y_train)
        return self

    def predict_quantiles(self, x_test: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        target = np.asarray(x_test[:, 0], dtype=np.float64)
        quantiles = np.repeat(target[:, None], 5, axis=1)
        levels = np.asarray([0.1, 0.3, 0.5, 0.7, 0.9], dtype=np.float64)
        return quantiles, levels


def test_evaluate_classifier_reports_brier_for_binary_and_multiclass() -> None:
    binary_datasets = {
        "binary": (
            np.asarray([[0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0], [0.0], [1.0]], dtype=np.float32),
            np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64),
        )
    }
    binary_metrics = benchmark_module.evaluate_classifier(_PerfectClassifier(), binary_datasets)
    assert binary_metrics["ROC AUC"] == pytest.approx(1.0)
    assert binary_metrics["Log Loss"] < 1.0e-10
    assert binary_metrics["Brier Score"] < 1.0e-10

    multiclass_x = np.asarray(
        [[0.0], [1.0], [2.0], [0.0], [1.0], [2.0], [0.0], [1.0], [2.0], [0.0], [1.0], [2.0], [0.0], [1.0], [2.0]],
        dtype=np.float32,
    )
    multiclass_y = np.asarray([0, 1, 2] * 5, dtype=np.int64)
    multiclass_metrics = benchmark_module.evaluate_classifier(
        _PerfectClassifier(),
        {"multi": (multiclass_x, multiclass_y)},
    )
    assert multiclass_metrics["ROC AUC"] == pytest.approx(1.0)
    assert multiclass_metrics["Log Loss"] < 1.0e-10
    assert multiclass_metrics["Brier Score"] < 1.0e-10


def test_classification_brier_score_matches_expected_binary_value() -> None:
    targets = np.asarray([0, 1], dtype=np.int64)
    probabilities = np.asarray(
        [
            [0.8, 0.2],
            [0.3, 0.7],
        ],
        dtype=np.float64,
    )

    observed = benchmark_metrics_module._classification_brier_score(targets, probabilities)

    assert observed == pytest.approx(0.13)


def test_classification_brier_score_matches_expected_multiclass_value() -> None:
    targets = np.asarray([0, 2], dtype=np.int64)
    probabilities = np.asarray(
        [
            [0.7, 0.2, 0.1],
            [0.1, 0.3, 0.6],
        ],
        dtype=np.float64,
    )

    observed = benchmark_metrics_module._classification_brier_score(targets, probabilities)

    assert observed == pytest.approx(0.2)


def test_evaluate_regressor_reports_crps_pinball_and_picp() -> None:
    x = np.linspace(-1.0, 1.0, num=20, dtype=np.float32)[:, None]
    y = np.asarray(x[:, 0], dtype=np.float32)
    metrics = benchmark_module.evaluate_regressor(
        _PerfectRegressor(),
        {"reg": (x, y)},
    )

    assert metrics["CRPS"] == pytest.approx(0.0, abs=1.0e-12)
    assert metrics["Average Pinball Loss"] == pytest.approx(0.0, abs=1.0e-12)
    assert metrics["PICP 90"] == pytest.approx(1.0)


def test_normalize_benchmark_bundle_accepts_regression_tasks() -> None:
    bundle = benchmark_module.normalize_benchmark_bundle(
        {
            "name": "regression_bundle",
            "version": 1,
            "selection": {
                "new_instances": 64,
                "task_type": "supervised_regression",
                "max_features": 10,
                "max_missing_pct": 0.0,
            },
            "task_ids": [1],
            "tasks": [
                {
                    "task_id": 1,
                    "dataset_name": "toy_regression",
                    "n_rows": 64,
                    "n_features": 4,
                }
            ],
        }
    )

    assert benchmark_module.benchmark_bundle_task_type(bundle) == "supervised_regression"
    assert "n_classes" not in bundle["tasks"][0]


def test_build_comparison_summary_supports_regression_without_nanotabpfn(tmp_path) -> None:
    summary = benchmark_module.build_comparison_summary(
        tab_foundry_records=[
            {
                "checkpoint_path": "/tmp/step_000025.pt",
                "step": 25,
                "training_time": 1.0,
                "crps": 0.18,
                "avg_pinball_loss": 0.09,
                "picp_90": 0.88,
                "dataset_crps": {"toy": 0.18},
                "dataset_avg_pinball_loss": {"toy": 0.09},
                "dataset_picp_90": {"toy": 0.88},
                "model_arch": "tabfoundry_staged",
            },
            {
                "checkpoint_path": "/tmp/step_000050.pt",
                "step": 50,
                "training_time": 2.0,
                "crps": 0.16,
                "avg_pinball_loss": 0.08,
                "picp_90": 0.9,
                "dataset_crps": {"toy": 0.16},
                "dataset_avg_pinball_loss": {"toy": 0.08},
                "dataset_picp_90": {"toy": 0.9},
                "model_arch": "tabfoundry_staged",
            },
        ],
        nanotabpfn_records=[],
        benchmark_tasks=[
            {"task_id": 1, "dataset_name": "toy", "n_rows": 16, "n_features": 4}
        ],
        benchmark_bundle={
            "name": "toy_regression",
            "version": 1,
            "selection": {
                "new_instances": 16,
                "task_type": "supervised_regression",
                "max_features": 4,
                "max_missing_pct": 0.0,
            },
            "task_ids": [1],
            "tasks": [
                {"task_id": 1, "dataset_name": "toy", "n_rows": 16, "n_features": 4}
            ],
        },
        benchmark_bundle_path=tmp_path / "bundle.json",
        tab_foundry_run_dir=tmp_path / "run",
        task_type="supervised_regression",
        nanotabpfn_root=None,
        nanotabpfn_python=None,
    )

    assert summary["tab_foundry"]["final_crps"] == pytest.approx(0.16)
    assert summary["tab_foundry"]["final_avg_pinball_loss"] == pytest.approx(0.08)
    assert summary["tab_foundry"]["final_picp_90"] == pytest.approx(0.9)
    assert "nanotabpfn" not in summary
