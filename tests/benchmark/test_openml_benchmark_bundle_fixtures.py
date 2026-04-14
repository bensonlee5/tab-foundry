from __future__ import annotations

from pathlib import Path

import pyarrow.parquet as pq
import pytest

import tab_foundry.bench.openml_benchmark as benchmark_module


def _checked_in_benchmark_manifest_path(surface_name: str) -> Path:
    return (
        Path(__file__).resolve().parents[2]
        / "data"
        / "manifests"
        / "bench"
        / surface_name
        / "manifest.parquet"
    )


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


@pytest.mark.parametrize(
    "surface_name",
    [
        "openml_classification_medium_v1",
        "openml_classification_large_v1",
    ],
)
def test_checked_in_benchmark_manifest_rows_use_v3_catalog_locator(surface_name: str) -> None:
    manifest_path = _checked_in_benchmark_manifest_path(surface_name)
    row = pq.read_table(manifest_path).slice(0, 1).to_pylist()[0]

    assert row["catalog_path"].endswith("dataset_catalog.parquet")
    assert int(row["catalog_dataset_index"]) >= 0
    assert len(str(row["catalog_record_sha256"])) == 64
    assert "catalog_offset_bytes" not in row
    assert "catalog_size_bytes" not in row
    assert "catalog_sha256" not in row


@pytest.mark.integration
@pytest.mark.parametrize(
    "surface_name",
    [
        "openml_classification_medium_v1",
        "openml_classification_large_v1",
    ],
)
def test_checked_in_benchmark_manifest_surface_loads_when_materialized(surface_name: str) -> None:
    manifest_path = _checked_in_benchmark_manifest_path(surface_name)
    row = pq.read_table(manifest_path).slice(0, 1).to_pylist()[0]
    required_paths = [
        (manifest_path.parent / str(row["train_path"])).resolve(),
        (manifest_path.parent / str(row["test_path"])).resolve(),
        (manifest_path.parent / str(row["catalog_path"])).resolve(),
    ]
    if not all(path.exists() for path in required_paths):
        pytest.skip(f"checked-in benchmark surface is not materialized locally: {surface_name}")

    datasets, task_records, benchmark_surface = benchmark_module.load_benchmark_manifest_datasets(
        benchmark_manifest_path=manifest_path,
    )

    assert datasets
    assert task_records
    assert benchmark_surface["manifest_path"] == str(manifest_path.resolve())
