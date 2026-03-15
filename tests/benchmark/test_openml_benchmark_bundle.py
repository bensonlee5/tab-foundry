from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import tab_foundry.bench.nanotabpfn as benchmark_module
import tab_foundry.bench.openml_benchmark_bundle as bundle_module


class _FakeDataset:
    def __init__(self, *, name: str, qualities: dict[str, float], frame: pd.DataFrame, target: pd.Series) -> None:
        self.name = name
        self.qualities = qualities
        self._frame = frame
        self._target = target

    def get_data(self, *, target: str, dataset_format: str) -> tuple[pd.DataFrame, pd.Series, list[bool], list[str]]:
        assert target == "target"
        assert dataset_format == "dataframe"
        return self._frame, self._target, [False] * self._frame.shape[1], list(self._frame.columns)


class _FakeTask:
    def __init__(self, dataset: _FakeDataset) -> None:
        self.task_type_id = benchmark_module.TaskType.SUPERVISED_CLASSIFICATION
        self.target_name = "target"
        self._dataset = dataset

    def get_dataset(self, *, download_data: bool) -> _FakeDataset:
        assert download_data is False
        return self._dataset


def _prepared_task(
    *,
    task_id: int,
    dataset_name: str,
    n_rows: int,
    n_features: int,
    n_classes: int,
    raw_feature_count: int | None = None,
    missing_pct: float = 0.0,
    minority_class_pct: float = 25.0,
) -> benchmark_module.PreparedOpenMLBenchmarkTask:
    x = np.arange(n_rows * n_features, dtype=np.float32).reshape(n_rows, n_features)
    y = (np.arange(n_rows, dtype=np.int64) % n_classes).astype(np.int64, copy=False)
    return benchmark_module.PreparedOpenMLBenchmarkTask(
        task_id=task_id,
        dataset_name=dataset_name,
        x=x,
        y=y,
        observed_task={
            "task_id": int(task_id),
            "dataset_name": dataset_name,
            "n_rows": int(n_rows),
            "n_features": int(n_features),
            "n_classes": int(n_classes),
        },
        qualities={
            "NumberOfFeatures": float(n_features if raw_feature_count is None else raw_feature_count),
            "NumberOfClasses": float(n_classes),
            "PercentageOfInstancesWithMissingValues": float(missing_pct),
            "MinorityClassPercentage": float(minority_class_pct),
        },
    )


def test_build_openml_benchmark_bundle_preserves_binary_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_by_task_id = {
        20: _prepared_task(task_id=20, dataset_name="multi", n_rows=200, n_features=4, n_classes=3),
        10: _prepared_task(task_id=10, dataset_name="binary", n_rows=200, n_features=3, n_classes=2),
    }
    monkeypatch.setattr(
        bundle_module,
        "prepare_openml_benchmark_task",
        lambda task_id, *, new_instances: prepared_by_task_id[int(task_id)],
    )
    monkeypatch.setattr(
        bundle_module.openml.tasks,
        "get_task",
        lambda task_id, download_splits=False: _FakeTask(
            _FakeDataset(
                name=prepared_by_task_id[int(task_id)].dataset_name,
                qualities={
                    "NumberOfFeatures": prepared_by_task_id[int(task_id)].qualities["NumberOfFeatures"],
                    "NumberOfClasses": prepared_by_task_id[int(task_id)].qualities["NumberOfClasses"],
                    "PercentageOfInstancesWithMissingValues": prepared_by_task_id[int(task_id)].qualities[
                        "PercentageOfInstancesWithMissingValues"
                    ],
                    "MinorityClassPercentage": prepared_by_task_id[int(task_id)].qualities[
                        "MinorityClassPercentage"
                    ],
                },
                frame=pd.DataFrame({"placeholder": [0, 1]}),
                target=pd.Series(["no", "yes"]),
            )
        ),
    )

    bundle = bundle_module.build_openml_benchmark_bundle(
        bundle_module.OpenMLBenchmarkBundleConfig(
            bundle_name="binary_small",
            version=1,
            max_classes=2,
            task_ids=(20, 10),
        )
    )

    assert bundle["selection"]["max_classes"] == 2
    assert bundle["task_ids"] == [10]
    assert bundle["tasks"] == [
        {
            "task_id": 10,
            "dataset_name": "binary",
            "n_rows": 200,
            "n_features": 3,
            "n_classes": 2,
        }
    ]


def test_build_openml_benchmark_bundle_auto_max_classes_widens_and_sorts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_by_task_id = {
        20: _prepared_task(task_id=20, dataset_name="multi", n_rows=200, n_features=4, n_classes=3),
        10: _prepared_task(task_id=10, dataset_name="binary", n_rows=200, n_features=3, n_classes=2),
    }
    monkeypatch.setattr(
        bundle_module,
        "prepare_openml_benchmark_task",
        lambda task_id, *, new_instances: prepared_by_task_id[int(task_id)],
    )
    monkeypatch.setattr(
        bundle_module.openml.tasks,
        "get_task",
        lambda task_id, download_splits=False: _FakeTask(
            _FakeDataset(
                name=prepared_by_task_id[int(task_id)].dataset_name,
                qualities={
                    "NumberOfFeatures": prepared_by_task_id[int(task_id)].qualities["NumberOfFeatures"],
                    "NumberOfClasses": prepared_by_task_id[int(task_id)].qualities["NumberOfClasses"],
                    "PercentageOfInstancesWithMissingValues": prepared_by_task_id[int(task_id)].qualities[
                        "PercentageOfInstancesWithMissingValues"
                    ],
                    "MinorityClassPercentage": prepared_by_task_id[int(task_id)].qualities[
                        "MinorityClassPercentage"
                    ],
                },
                frame=pd.DataFrame({"placeholder": [0, 1]}),
                target=pd.Series(["no", "yes"]),
            )
        ),
    )

    bundle = bundle_module.build_openml_benchmark_bundle(
        bundle_module.OpenMLBenchmarkBundleConfig(
            bundle_name="classification_small",
            version=1,
            max_classes=None,
            task_ids=(20, 10),
        )
    )

    assert bundle["selection"]["max_classes"] == 3
    assert bundle["task_ids"] == [10, 20]
    assert [task["task_id"] for task in bundle["tasks"]] == [10, 20]


def test_build_openml_benchmark_bundle_uses_named_task_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_by_task_id = {
        30: _prepared_task(task_id=30, dataset_name="source_high", n_rows=200, n_features=4, n_classes=2),
        10: _prepared_task(task_id=10, dataset_name="source_low", n_rows=200, n_features=3, n_classes=2),
    }
    monkeypatch.setattr(bundle_module, "task_ids_for_source", lambda task_source: (30, 10))
    monkeypatch.setattr(
        bundle_module,
        "prepare_openml_benchmark_task",
        lambda task_id, *, new_instances: prepared_by_task_id[int(task_id)],
    )
    monkeypatch.setattr(
        bundle_module.openml.tasks,
        "get_task",
        lambda task_id, download_splits=False: _FakeTask(
            _FakeDataset(
                name=prepared_by_task_id[int(task_id)].dataset_name,
                qualities={
                    "NumberOfFeatures": prepared_by_task_id[int(task_id)].qualities["NumberOfFeatures"],
                    "NumberOfClasses": prepared_by_task_id[int(task_id)].qualities["NumberOfClasses"],
                    "PercentageOfInstancesWithMissingValues": prepared_by_task_id[int(task_id)].qualities[
                        "PercentageOfInstancesWithMissingValues"
                    ],
                    "MinorityClassPercentage": prepared_by_task_id[int(task_id)].qualities[
                        "MinorityClassPercentage"
                    ],
                },
                frame=pd.DataFrame({"placeholder": [0, 1]}),
                target=pd.Series(["no", "yes"]),
            )
        ),
    )

    bundle = bundle_module.build_openml_benchmark_bundle(
        bundle_module.OpenMLBenchmarkBundleConfig(
            bundle_name="binary_medium",
            version=1,
            task_source="binary_expanded_v1",
        )
    )

    assert bundle["task_ids"] == [10, 30]
    assert [task["dataset_name"] for task in bundle["tasks"]] == ["source_low", "source_high"]


def test_build_openml_benchmark_bundle_rejects_empty_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_by_task_id = {
        10: _prepared_task(
            task_id=10,
            dataset_name="too_wide",
            n_rows=200,
            n_features=3,
            n_classes=2,
            missing_pct=1.0,
        )
    }
    monkeypatch.setattr(
        bundle_module,
        "prepare_openml_benchmark_task",
        lambda task_id, *, new_instances: prepared_by_task_id[int(task_id)],
    )
    monkeypatch.setattr(
        bundle_module.openml.tasks,
        "get_task",
        lambda task_id, download_splits=False: _FakeTask(
            _FakeDataset(
                name=prepared_by_task_id[int(task_id)].dataset_name,
                qualities={
                    "NumberOfFeatures": prepared_by_task_id[int(task_id)].qualities["NumberOfFeatures"],
                    "NumberOfClasses": prepared_by_task_id[int(task_id)].qualities["NumberOfClasses"],
                    "PercentageOfInstancesWithMissingValues": prepared_by_task_id[int(task_id)].qualities[
                        "PercentageOfInstancesWithMissingValues"
                    ],
                    "MinorityClassPercentage": prepared_by_task_id[int(task_id)].qualities[
                        "MinorityClassPercentage"
                    ],
                },
                frame=pd.DataFrame({"placeholder": [0, 1]}),
                target=pd.Series(["no", "yes"]),
            )
        ),
    )

    with pytest.raises(RuntimeError, match="produced no eligible tasks"):
        bundle_module.build_openml_benchmark_bundle(
            bundle_module.OpenMLBenchmarkBundleConfig(
                bundle_name="empty",
                version=1,
                task_ids=(10,),
            )
        )


def test_build_openml_benchmark_bundle_main_parses_task_source_flag(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def _fake_write(path: Path, config: bundle_module.OpenMLBenchmarkBundleConfig) -> Path:
        captured["path"] = path
        captured["config"] = config
        return path

    monkeypatch.setattr(bundle_module, "write_openml_benchmark_bundle", _fake_write)

    exit_code = bundle_module.main(
        [
            "--out-path",
            str(tmp_path / "bundle.json"),
            "--bundle-name",
            "binary_medium",
            "--version",
            "1",
            "--task-source",
            "binary_expanded_v1",
        ]
    )

    assert exit_code == 0
    config = captured["config"]
    assert isinstance(config, bundle_module.OpenMLBenchmarkBundleConfig)
    assert config.task_source == "binary_expanded_v1"
    assert "wrote benchmark bundle:" in capsys.readouterr().out


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

    fake_tasks: dict[int, _FakeTask] = {}
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
        fake_tasks[int(task_payload["task_id"])] = _FakeTask(
            _FakeDataset(
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
        lambda task_id, download_splits=False: fake_tasks[int(task_id)],
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

    fake_tasks: dict[int, _FakeTask] = {}
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
        fake_tasks[int(task_payload["task_id"])] = _FakeTask(
            _FakeDataset(
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
        lambda task_id, download_splits=False: fake_tasks[int(task_id)],
    )

    datasets, metadata = benchmark_module.load_openml_benchmark_datasets(
        new_instances=int(bundle["selection"]["new_instances"]),
        benchmark_bundle_path=bundle_path,
    )

    assert list(datasets) == [str(task["dataset_name"]) for task in bundle["tasks"]]
    assert metadata == bundle["tasks"]
