from __future__ import annotations

from pathlib import Path

import pytest

import tab_foundry.bench.openml_benchmark as benchmark_module


def test_checked_in_multiclass_bundle_loads() -> None:
    bundle_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "tab_foundry"
        / "bench"
        / "openml_classification_small_v1.json"
    )

    bundle = benchmark_module.load_benchmark_bundle(bundle_path)

    assert bundle["name"] == "openml_classification_small"
    assert bundle["version"] == 1
    assert bundle["selection"]["max_classes"] == 3
    assert bundle["task_ids"] == [363613, 363621, 363629, 363685, 363707]


def test_checked_in_binary_medium_bundle_loads() -> None:
    bundle_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "tab_foundry"
        / "bench"
        / "openml_binary_medium_v1.json"
    )

    bundle = benchmark_module.load_benchmark_bundle(bundle_path)

    assert bundle["name"] == "openml_binary_medium"
    assert bundle["version"] == 1
    assert len(bundle["task_ids"]) == 10
    assert all(int(task["n_classes"]) == 2 for task in bundle["tasks"])


def test_checked_in_binary_large_no_missing_bundle_loads() -> None:
    bundle_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "tab_foundry"
        / "bench"
        / "openml_binary_large_no_missing_v1.json"
    )

    bundle = benchmark_module.load_benchmark_bundle(bundle_path)

    assert bundle["name"] == "openml_binary_large_no_missing"
    assert bundle["version"] == 1
    assert len(bundle["task_ids"]) == 64
    assert bundle["selection"]["max_missing_pct"] == 0.0
    assert all(int(task["n_classes"]) == 2 for task in bundle["tasks"])


def test_checked_in_binary_large_bundle_requires_explicit_missing_value_opt_in() -> None:
    bundle_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "tab_foundry"
        / "bench"
        / "openml_binary_large_v1.json"
    )

    with pytest.raises(RuntimeError, match="permits missing-valued inputs"):
        _ = benchmark_module.load_benchmark_bundle(bundle_path)

    bundle = benchmark_module.load_benchmark_bundle(bundle_path, allow_missing_values=True)

    assert bundle["name"] == "openml_binary_large"


def test_load_benchmark_manifest_datasets_rejects_checked_in_multiclass_bundle_json() -> None:
    bundle_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "tab_foundry"
        / "bench"
        / "openml_classification_small_v1.json"
    )
    with pytest.raises(RuntimeError, match="materialized manifest parquet"):
        benchmark_module.load_benchmark_manifest_datasets(
            benchmark_manifest_path=bundle_path,
        )


def test_load_benchmark_manifest_datasets_rejects_checked_in_binary_medium_bundle_json() -> None:
    bundle_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "tab_foundry"
        / "bench"
        / "openml_binary_medium_v1.json"
    )
    with pytest.raises(RuntimeError, match="materialized manifest parquet"):
        benchmark_module.load_benchmark_manifest_datasets(
            benchmark_manifest_path=bundle_path,
        )
