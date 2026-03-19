from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tab_foundry.bench.nanotabpfn as benchmark_module

from tests.benchmark.openml_bundle_fakes import FakeDataset, FakeTask


def test_checked_in_multiclass_bundle_loads() -> None:
    bundle_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "tab_foundry"
        / "bench"
        / "nanotabpfn_openml_classification_small_v1.json"
    )

    bundle = benchmark_module.load_benchmark_bundle(bundle_path)

    assert bundle["name"] == "nanotabpfn_openml_classification_small"
    assert bundle["version"] == 1
    assert bundle["selection"]["max_classes"] == 3
    assert bundle["task_ids"] == [363613, 363621, 363629, 363685, 363707]


def test_checked_in_binary_medium_bundle_loads() -> None:
    bundle_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "tab_foundry"
        / "bench"
        / "nanotabpfn_openml_binary_medium_v1.json"
    )

    bundle = benchmark_module.load_benchmark_bundle(bundle_path)

    assert bundle["name"] == "nanotabpfn_openml_binary_medium"
    assert bundle["version"] == 1
    assert len(bundle["task_ids"]) == 10
    assert all(int(task["n_classes"]) == 2 for task in bundle["tasks"])


def test_checked_in_binary_large_no_missing_bundle_loads() -> None:
    bundle_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "tab_foundry"
        / "bench"
        / "nanotabpfn_openml_binary_large_no_missing_v1.json"
    )

    bundle = benchmark_module.load_benchmark_bundle(bundle_path)

    assert bundle["name"] == "nanotabpfn_openml_binary_large_no_missing"
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
        / "nanotabpfn_openml_binary_large_v1.json"
    )

    with pytest.raises(RuntimeError, match="permits missing-valued inputs"):
        _ = benchmark_module.load_benchmark_bundle(bundle_path)

    bundle = benchmark_module.load_benchmark_bundle(bundle_path, allow_missing_values=True)

    assert bundle["name"] == "nanotabpfn_openml_binary_large"


def test_load_openml_benchmark_datasets_accepts_checked_in_multiclass_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "tab_foundry"
        / "bench"
        / "nanotabpfn_openml_classification_small_v1.json"
    )
    bundle = benchmark_module.load_benchmark_bundle(bundle_path)

    fake_tasks: dict[int, FakeTask] = {}
    for task_payload in bundle["tasks"]:
        n_rows = int(task_payload["n_rows"])
        n_features = int(task_payload["n_features"])
        n_classes = int(task_payload["n_classes"])
        frame = pd.DataFrame(
            {
                f"f{column_idx}": np.arange(n_rows, dtype=np.float32) + float(column_idx)
                for column_idx in range(n_features)
            }
        )
        target = pd.Series([f"class_{index % n_classes}" for index in range(n_rows)])
        fake_tasks[int(task_payload["task_id"])] = FakeTask(
            FakeDataset(
                name=str(task_payload["dataset_name"]),
                qualities={
                    "NumberOfFeatures": float(n_features),
                    "NumberOfClasses": float(n_classes),
                    "PercentageOfInstancesWithMissingValues": 0.0,
                    "MinorityClassPercentage": 25.0,
                },
                frame=frame,
                target=target,
            )
        )

    monkeypatch.setattr(
        benchmark_module.openml.tasks,
        "get_task",
        lambda task_id, **_kwargs: fake_tasks[int(task_id)],
    )

    datasets, metadata = benchmark_module.load_openml_benchmark_datasets(
        new_instances=int(bundle["selection"]["new_instances"]),
        benchmark_bundle_path=bundle_path,
    )

    assert list(datasets) == [str(task["dataset_name"]) for task in bundle["tasks"]]
    assert metadata == bundle["tasks"]


def test_load_openml_benchmark_datasets_accepts_checked_in_binary_medium_bundle(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "tab_foundry"
        / "bench"
        / "nanotabpfn_openml_binary_medium_v1.json"
    )
    bundle = benchmark_module.load_benchmark_bundle(bundle_path)

    fake_tasks: dict[int, FakeTask] = {}
    for task_payload in bundle["tasks"]:
        n_rows = int(task_payload["n_rows"])
        n_features = int(task_payload["n_features"])
        n_classes = int(task_payload["n_classes"])
        frame = pd.DataFrame(
            {
                f"f{column_idx}": np.arange(n_rows, dtype=np.float32) + float(column_idx)
                for column_idx in range(n_features)
            }
        )
        target = pd.Series([f"class_{index % n_classes}" for index in range(n_rows)])
        fake_tasks[int(task_payload["task_id"])] = FakeTask(
            FakeDataset(
                name=str(task_payload["dataset_name"]),
                qualities={
                    "NumberOfFeatures": float(n_features),
                    "NumberOfClasses": float(n_classes),
                    "PercentageOfInstancesWithMissingValues": 0.0,
                    "MinorityClassPercentage": 25.0,
                },
                frame=frame,
                target=target,
            )
        )

    monkeypatch.setattr(
        benchmark_module.openml.tasks,
        "get_task",
        lambda task_id, **_kwargs: fake_tasks[int(task_id)],
    )

    datasets, metadata = benchmark_module.load_openml_benchmark_datasets(
        new_instances=int(bundle["selection"]["new_instances"]),
        benchmark_bundle_path=bundle_path,
    )

    assert list(datasets) == [str(task["dataset_name"]) for task in bundle["tasks"]]
    assert metadata == bundle["tasks"]
