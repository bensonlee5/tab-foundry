from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import tab_foundry.bench.compare as compare_module
import tab_foundry.bench.nanotabpfn as benchmark_module


DEFAULT_BENCHMARK_SELECTION = {
    "new_instances": 200,
    "task_type": "supervised_classification",
    "max_features": 10,
    "max_classes": 2,
    "max_missing_pct": 0.0,
    "min_minority_class_pct": 2.5,
}


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


def _write_benchmark_bundle(
    path: Path,
    *,
    tasks: list[dict[str, Any]],
    selection_overrides: dict[str, Any] | None = None,
) -> Path:
    selection = dict(DEFAULT_BENCHMARK_SELECTION)
    if tasks:
        selection["new_instances"] = int(tasks[0]["n_rows"])
    if selection_overrides is not None:
        selection.update(selection_overrides)
    payload = {
        "name": "test_bundle",
        "version": 1,
        "selection": selection,
        "task_ids": [int(task["task_id"]) for task in tasks],
        "tasks": tasks,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_load_openml_benchmark_datasets_matches_notebook_filters(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    keep_frame = pd.DataFrame(
        {
            "num": [0, 1, 2, 3, 4, 5],
            "cat": ["a", "b", "a", "b", "a", "b"],
            "constant": [1, 1, 1, 1, 1, 1],
        }
    )
    keep_target = pd.Series(["no", "yes", "no", "yes", "no", "yes"])
    keep_dataset = _FakeDataset(
        name="keep_me",
        qualities={
            "NumberOfFeatures": 3,
            "NumberOfClasses": 2,
            "PercentageOfInstancesWithMissingValues": 0,
            "MinorityClassPercentage": 50.0,
        },
        frame=keep_frame,
        target=keep_target,
    )
    drop_dataset = _FakeDataset(
        name="drop_me",
        qualities={
            "NumberOfFeatures": 11,
            "NumberOfClasses": 2,
            "PercentageOfInstancesWithMissingValues": 0,
            "MinorityClassPercentage": 50.0,
        },
        frame=keep_frame,
        target=keep_target,
    )
    fake_tasks = {
        1: _FakeTask(keep_dataset),
        2: _FakeTask(drop_dataset),
    }
    bundle_path = _write_benchmark_bundle(
        tmp_path / "benchmark_bundle.json",
        tasks=[
            {
                "task_id": 1,
                "dataset_name": "keep_me",
                "n_rows": 6,
                "n_features": 2,
                "n_classes": 2,
            }
        ],
    )
    monkeypatch.setattr(
        benchmark_module.openml.tasks,
        "get_task",
        lambda task_id, download_splits=False: fake_tasks[int(task_id)],
    )

    datasets, metadata = benchmark_module.load_openml_benchmark_datasets(
        new_instances=6,
        benchmark_bundle_path=bundle_path,
    )

    bundle = benchmark_module.load_benchmark_bundle(bundle_path)

    assert list(datasets) == ["keep_me"]
    x, y = datasets["keep_me"]
    assert x.shape == (6, 2)
    assert y.tolist() == [0, 1, 0, 1, 0, 1]
    assert metadata[0]["dataset_name"] == "keep_me"
    assert bundle["selection"]["max_missing_pct"] == pytest.approx(0.0)
    assert bundle["selection"]["min_minority_class_pct"] == pytest.approx(2.5)


def test_load_openml_benchmark_datasets_fails_on_bundle_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame({"num": [0, 1, 2, 3], "cat": ["a", "b", "a", "b"]})
    target = pd.Series(["no", "yes", "no", "yes"])
    fake_tasks = {
        1: _FakeTask(
            _FakeDataset(
                name="keep_me",
                qualities={
                    "NumberOfFeatures": 2,
                    "NumberOfClasses": 2,
                    "PercentageOfInstancesWithMissingValues": 0.0,
                    "MinorityClassPercentage": 50.0,
                },
                frame=frame,
                target=target,
            )
        )
    }
    bundle_path = _write_benchmark_bundle(
        tmp_path / "benchmark_bundle.json",
        tasks=[
            {
                "task_id": 1,
                "dataset_name": "wrong_name",
                "n_rows": 4,
                "n_features": 2,
                "n_classes": 2,
            }
        ],
    )
    monkeypatch.setattr(
        benchmark_module.openml.tasks,
        "get_task",
        lambda task_id, download_splits=False: fake_tasks[int(task_id)],
    )

    with pytest.raises(RuntimeError, match="benchmark bundle drift"):
        benchmark_module.load_openml_benchmark_datasets(
            new_instances=4,
            benchmark_bundle_path=bundle_path,
        )


def test_load_benchmark_bundle_requires_full_selection(tmp_path: Path) -> None:
    path = tmp_path / "benchmark_bundle.json"
    path.write_text(
        json.dumps(
            {
                "name": "test_bundle",
                "version": 1,
                "selection": {
                    "new_instances": 4,
                    "task_type": "supervised_classification",
                    "max_features": 10,
                    "max_classes": 2,
                    "max_missing_pct": 0.0,
                },
                "task_ids": [1],
                "tasks": [
                    {
                        "task_id": 1,
                        "dataset_name": "toy",
                        "n_rows": 4,
                        "n_features": 2,
                        "n_classes": 2,
                    }
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="selection keys mismatch"):
        benchmark_module.load_benchmark_bundle(path)


def test_load_openml_benchmark_datasets_requires_bundle_new_instances_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    frame = pd.DataFrame({"num": [0, 1, 2, 3], "cat": ["a", "b", "a", "b"]})
    target = pd.Series(["no", "yes", "no", "yes"])
    fake_tasks = {
        1: _FakeTask(
            _FakeDataset(
                name="keep_me",
                qualities={
                    "NumberOfFeatures": 2,
                    "NumberOfClasses": 2,
                    "PercentageOfInstancesWithMissingValues": 0.0,
                    "MinorityClassPercentage": 50.0,
                },
                frame=frame,
                target=target,
            )
        )
    }
    bundle_path = _write_benchmark_bundle(
        tmp_path / "benchmark_bundle.json",
        tasks=[
            {
                "task_id": 1,
                "dataset_name": "keep_me",
                "n_rows": 4,
                "n_features": 2,
                "n_classes": 2,
            }
        ],
        selection_overrides={"new_instances": 4},
    )
    monkeypatch.setattr(
        benchmark_module.openml.tasks,
        "get_task",
        lambda task_id, download_splits=False: fake_tasks[int(task_id)],
    )

    with pytest.raises(RuntimeError, match="selection mismatch"):
        benchmark_module.load_openml_benchmark_datasets(
            new_instances=3,
            benchmark_bundle_path=bundle_path,
        )


@pytest.mark.parametrize(
    ("qualities", "message"),
    [
        (
            {
                "NumberOfFeatures": 11,
                "NumberOfClasses": 2,
                "PercentageOfInstancesWithMissingValues": 0.0,
                "MinorityClassPercentage": 50.0,
            },
            "max_features",
        ),
        (
            {
                "NumberOfFeatures": 2,
                "NumberOfClasses": 3,
                "PercentageOfInstancesWithMissingValues": 0.0,
                "MinorityClassPercentage": 50.0,
            },
            "max_classes",
        ),
        (
            {
                "NumberOfFeatures": 2,
                "NumberOfClasses": 2,
                "PercentageOfInstancesWithMissingValues": 5.0,
                "MinorityClassPercentage": 50.0,
            },
            "max_missing_pct",
        ),
        (
            {
                "NumberOfFeatures": 2,
                "NumberOfClasses": 2,
                "PercentageOfInstancesWithMissingValues": 0.0,
                "MinorityClassPercentage": 2.0,
            },
            "min_minority_class_pct",
        ),
    ],
)
def test_load_openml_benchmark_datasets_fails_on_selection_drift(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    qualities: dict[str, float],
    message: str,
) -> None:
    frame = pd.DataFrame({"num": [0, 1, 2, 3], "cat": ["a", "b", "a", "b"]})
    target = pd.Series(["no", "yes", "no", "yes"])
    fake_tasks = {
        1: _FakeTask(
            _FakeDataset(
                name="keep_me",
                qualities=qualities,
                frame=frame,
                target=target,
            )
        )
    }
    bundle_path = _write_benchmark_bundle(
        tmp_path / "benchmark_bundle.json",
        tasks=[
            {
                "task_id": 1,
                "dataset_name": "keep_me",
                "n_rows": 4,
                "n_features": 2,
                "n_classes": 2,
            }
        ],
    )
    monkeypatch.setattr(
        benchmark_module.openml.tasks,
        "get_task",
        lambda task_id, download_splits=False: fake_tasks[int(task_id)],
    )

    with pytest.raises(RuntimeError, match=message):
        benchmark_module.load_openml_benchmark_datasets(
            new_instances=4,
            benchmark_bundle_path=bundle_path,
        )


def test_run_nanotabpfn_benchmark_orchestrates_external_helper(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    smoke_run_dir = tmp_path / "smoke_run"
    smoke_run_dir.mkdir()
    nanotab_root = tmp_path / "nano"
    (nanotab_root / ".venv" / "bin").mkdir(parents=True)
    nanotab_python = nanotab_root / ".venv" / "bin" / "python"
    nanotab_python.write_text("#!/bin/sh\n", encoding="utf-8")
    prior_dump = nanotab_root / "300k_150x5_2.h5"
    prior_dump.write_bytes(b"prior")
    out_root = tmp_path / "benchmark_out"
    source_bundle_path = tmp_path / "source_bundle.json"
    benchmark_bundle = {
        "name": "test_bundle",
        "version": 1,
        "selection": {
            "new_instances": 6,
            "task_type": "supervised_classification",
            "max_features": 10,
            "max_classes": 2,
            "max_missing_pct": 0.0,
            "min_minority_class_pct": 2.5,
        },
        "task_ids": [1],
        "tasks": [
            {
                "task_id": 1,
                "dataset_name": "toy",
                "n_rows": 6,
                "n_features": 2,
                "n_classes": 2,
            }
        ],
    }
    source_bundle_path.write_text(
        json.dumps(benchmark_bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        compare_module,
        "load_openml_benchmark_datasets",
        lambda benchmark_bundle_path=None: (
            {
                "toy": (
                    np.zeros((6, 2), dtype=np.float32),
                    np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
                )
            },
            [{"task_id": 1, "dataset_name": "toy", "n_rows": 6, "n_features": 2, "n_classes": 2}],
        ),
    )
    monkeypatch.setattr(compare_module, "default_benchmark_bundle_path", lambda: source_bundle_path)
    monkeypatch.setattr(compare_module, "load_benchmark_bundle", lambda path=None: benchmark_bundle)
    monkeypatch.setattr(
        compare_module,
        "evaluate_tab_foundry_run",
        lambda *_args, **_kwargs: [
            {
                "checkpoint_path": "/tmp/step_000025.pt",
                "step": 25,
                "training_time": 1.2,
                "roc_auc": 0.81,
                "dataset_roc_auc": {"toy": 0.81},
            }
        ],
    )

    captured: dict[str, Any] = {}

    def _fake_run(cmd: list[str], *, cwd: Path, check: bool) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["check"] = check
        out_index = cmd.index("--out-path") + 1
        out_path = Path(cmd[out_index])
        out_path.write_text(
            json.dumps(
                {
                    "seed": 0,
                    "step": 25,
                    "training_time": 2.0,
                    "roc_auc": 0.78,
                    "dataset_roc_auc": {"toy": 0.78},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(compare_module, "subprocess", SimpleNamespace(run=_fake_run))

    summary = compare_module.run_nanotabpfn_benchmark(
        compare_module.NanoTabPFNBenchmarkConfig(
            tab_foundry_run_dir=smoke_run_dir,
            out_root=out_root,
            nanotabpfn_root=nanotab_root,
            nanotab_prior_dump=prior_dump,
        )
    )

    assert captured["cwd"] == nanotab_root.resolve()
    assert captured["check"] is True
    assert Path(captured["cmd"][0]) == nanotab_python.resolve()
    assert Path(captured["cmd"][1]) == Path(compare_module.__file__).resolve().with_name("nanotabpfn_helper.py")
    assert summary["dataset_count"] == 1
    assert summary["tab_foundry"]["best_step"] == pytest.approx(25.0)
    assert summary["tab_foundry"]["best_roc_auc"] == pytest.approx(0.81)
    assert summary["tab_foundry"]["final_dataset_roc_auc"] == {"toy": pytest.approx(0.81)}
    assert summary["nanotabpfn"]["best_step"] == pytest.approx(25.0)
    assert summary["nanotabpfn"]["final_roc_auc"] == pytest.approx(0.78)
    assert summary["nanotabpfn"]["final_dataset_roc_auc"] == {"toy": pytest.approx(0.78)}
    assert summary["benchmark_bundle"]["name"] == "test_bundle"
    assert summary["benchmark_bundle"]["version"] == 1
    assert summary["benchmark_bundle"]["task_count"] == 1
    assert summary["benchmark_bundle"]["task_ids"] == [1]
    assert summary["benchmark_bundle"]["source_path"] == str(source_bundle_path.resolve())
    assert (out_root / "comparison_summary.json").exists()
    assert (out_root / "comparison_curve.png").exists()
    written_bundle = json.loads((out_root / "benchmark_tasks.json").read_text(encoding="utf-8"))
    assert written_bundle == benchmark_bundle


def test_collect_checkpoint_snapshots_prefers_train_elapsed_seconds(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "train_outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "step_000025.pt").write_bytes(b"step25")
    history_path = run_dir / "train_outputs" / "train_history.jsonl"
    history_path.write_text(
        json.dumps(
            {
                "step": 25,
                "stage": "stage1",
                "train_loss": 0.5,
                "lr": 1.0e-3,
                "elapsed_seconds": 9.0,
                "train_elapsed_seconds": 3.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    snapshots = benchmark_module.collect_checkpoint_snapshots(run_dir)

    assert snapshots[0]["elapsed_seconds"] == pytest.approx(3.0)


def test_collect_checkpoint_snapshots_supports_plain_training_output(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "best.pt").write_bytes(b"best")
    (checkpoint_dir / "step_000025.pt").write_bytes(b"step25")
    history_path = run_dir / "train_history.jsonl"
    history_path.write_text(
        json.dumps(
            {
                "step": 25,
                "stage": "stage1",
                "train_loss": 0.5,
                "lr": 1.0e-3,
                "elapsed_seconds": 9.0,
                "train_elapsed_seconds": 3.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    snapshots = benchmark_module.collect_checkpoint_snapshots(run_dir)

    assert snapshots[0]["step"] == 25
    assert snapshots[0]["elapsed_seconds"] == pytest.approx(3.0)


def test_run_nanotabpfn_benchmark_includes_control_baseline_annotation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    smoke_run_dir = tmp_path / "smoke_run"
    smoke_run_dir.mkdir()
    nanotab_root = tmp_path / "nano"
    (nanotab_root / ".venv" / "bin").mkdir(parents=True)
    nanotab_python = nanotab_root / ".venv" / "bin" / "python"
    nanotab_python.write_text("#!/bin/sh\n", encoding="utf-8")
    prior_dump = nanotab_root / "300k_150x5_2.h5"
    prior_dump.write_bytes(b"prior")
    out_root = tmp_path / "benchmark_out"
    source_bundle_path = tmp_path / "source_bundle.json"
    benchmark_bundle = {
        "name": "test_bundle",
        "version": 1,
        "selection": {
            "new_instances": 6,
            "task_type": "supervised_classification",
            "max_features": 10,
            "max_classes": 2,
            "max_missing_pct": 0.0,
            "min_minority_class_pct": 2.5,
        },
        "task_ids": [1],
        "tasks": [
            {
                "task_id": 1,
                "dataset_name": "toy",
                "n_rows": 6,
                "n_features": 2,
                "n_classes": 2,
            }
        ],
    }
    source_bundle_path.write_text(
        json.dumps(benchmark_bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    registry_path = tmp_path / "control_baselines_v1.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema": "tab-foundry-control-baselines-v1",
                "version": 1,
                "baselines": {
                    "cls_benchmark_linear_v1": {
                        "baseline_id": "cls_benchmark_linear_v1",
                        "experiment": "cls_benchmark_linear",
                        "config_profile": "cls_benchmark_linear",
                        "budget_class": "short-run",
                        "manifest_path": "data/manifests/default.parquet",
                        "seed_set": [1],
                        "run_dir": "outputs/control_baselines/cls_benchmark_linear_v1/train",
                        "comparison_summary_path": "outputs/control_baselines/cls_benchmark_linear_v1/benchmark/comparison_summary.json",
                        "benchmark_bundle": {
                            "name": "test_bundle",
                            "version": 1,
                            "source_path": str(source_bundle_path.resolve()),
                            "task_count": 1,
                            "task_ids": [1],
                        },
                        "tab_foundry_metrics": {
                            "best_step": 25.0,
                            "best_training_time": 1.2,
                            "best_roc_auc": 0.81,
                            "final_step": 25.0,
                            "final_training_time": 1.2,
                            "final_roc_auc": 0.81,
                        },
                    }
                },
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        compare_module,
        "load_openml_benchmark_datasets",
        lambda benchmark_bundle_path=None: (
            {
                "toy": (
                    np.zeros((6, 2), dtype=np.float32),
                    np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
                )
            },
            [{"task_id": 1, "dataset_name": "toy", "n_rows": 6, "n_features": 2, "n_classes": 2}],
        ),
    )
    monkeypatch.setattr(compare_module, "default_benchmark_bundle_path", lambda: source_bundle_path)
    monkeypatch.setattr(compare_module, "load_benchmark_bundle", lambda path=None: benchmark_bundle)
    monkeypatch.setattr(
        compare_module,
        "evaluate_tab_foundry_run",
        lambda *_args, **_kwargs: [
            {"checkpoint_path": "/tmp/step_000025.pt", "step": 25, "training_time": 1.2, "roc_auc": 0.81}
        ],
    )

    def _fake_run(cmd: list[str], *, cwd: Path, check: bool) -> subprocess.CompletedProcess[str]:
        out_index = cmd.index("--out-path") + 1
        out_path = Path(cmd[out_index])
        out_path.write_text(
            json.dumps({"seed": 0, "step": 25, "training_time": 2.0, "roc_auc": 0.78}) + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(compare_module, "subprocess", SimpleNamespace(run=_fake_run))

    summary = compare_module.run_nanotabpfn_benchmark(
        compare_module.NanoTabPFNBenchmarkConfig(
            tab_foundry_run_dir=smoke_run_dir,
            out_root=out_root,
            nanotabpfn_root=nanotab_root,
            nanotab_prior_dump=prior_dump,
            control_baseline_id="cls_benchmark_linear_v1",
            control_baseline_registry=registry_path,
        )
    )

    assert summary["control_baseline"]["baseline_id"] == "cls_benchmark_linear_v1"
    written_summary = json.loads((out_root / "comparison_summary.json").read_text(encoding="utf-8"))
    assert written_summary["control_baseline"]["budget_class"] == "short-run"


def test_run_nanotabpfn_benchmark_rejects_unknown_control_baseline(tmp_path: Path) -> None:
    smoke_run_dir = tmp_path / "smoke_run"
    smoke_run_dir.mkdir()
    nanotab_root = tmp_path / "nano"
    (nanotab_root / ".venv" / "bin").mkdir(parents=True)
    nanotab_python = nanotab_root / ".venv" / "bin" / "python"
    nanotab_python.write_text("#!/bin/sh\n", encoding="utf-8")
    prior_dump = nanotab_root / "300k_150x5_2.h5"
    prior_dump.write_bytes(b"prior")
    registry_path = tmp_path / "control_baselines_v1.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema": "tab-foundry-control-baselines-v1",
                "version": 1,
                "baselines": {},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="unknown control baseline id"):
        compare_module.run_nanotabpfn_benchmark(
            compare_module.NanoTabPFNBenchmarkConfig(
                tab_foundry_run_dir=smoke_run_dir,
                out_root=tmp_path / "benchmark_out",
                nanotabpfn_root=nanotab_root,
                nanotab_prior_dump=prior_dump,
                control_baseline_id="missing",
                control_baseline_registry=registry_path,
            )
        )
