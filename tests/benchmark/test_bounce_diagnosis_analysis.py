from __future__ import annotations

from pathlib import Path

import pytest

import tab_foundry.bench.nanotabpfn as benchmark_module


def test_mainline_bundle_loader_rejects_missing_permitting_large_bundle_by_default() -> None:
    large_bundle_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "tab_foundry"
        / "bench"
        / "nanotabpfn_openml_binary_large_v1.json"
    )

    with pytest.raises(RuntimeError, match="permits missing-valued inputs"):
        _ = benchmark_module.load_benchmark_bundle(large_bundle_path)


def test_annotate_curve_records_with_task_statistics_adds_dataset_count_and_ci() -> None:
    records = [
        {
            "step": 25,
            "training_time": 1.0,
            "roc_auc": 0.72,
            "dataset_roc_auc": {"a": 0.70, "b": 0.74, "c": 0.72},
        }
    ]

    annotated = benchmark_module.annotate_curve_records_with_task_statistics(
        records,
        bootstrap_samples=128,
        bootstrap_confidence=0.9,
        bootstrap_seed=7,
    )

    assert annotated[0]["dataset_count"] == 3
    ci = annotated[0]["roc_auc_task_bootstrap_ci"]
    assert ci["samples"] == 128
    assert ci["confidence"] == pytest.approx(0.9)
    assert float(ci["lower"]) <= float(records[0]["roc_auc"]) <= float(ci["upper"])
