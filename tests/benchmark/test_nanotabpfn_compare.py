from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

import numpy as np
import pandas as pd
import pytest

import tab_foundry.bench.compare as compare_module
import tab_foundry.bench.nanotabpfn as benchmark_module


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


def test_load_openml_benchmark_datasets_matches_notebook_filters(
    monkeypatch: pytest.MonkeyPatch,
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
    tasks = {
        1: _FakeTask(keep_dataset),
        2: _FakeTask(drop_dataset),
    }
    monkeypatch.setattr(benchmark_module, "NANOTABPFN_TASK_IDS", (1, 2))
    monkeypatch.setattr(
        benchmark_module.openml.tasks,
        "get_task",
        lambda task_id, download_splits=False: tasks[int(task_id)],
    )

    datasets, metadata = benchmark_module.load_openml_benchmark_datasets(new_instances=6)

    assert list(datasets) == ["keep_me"]
    x, y = datasets["keep_me"]
    assert x.shape == (6, 2)
    assert y.tolist() == [0, 1, 0, 1, 0, 1]
    assert metadata[0]["dataset_name"] == "keep_me"


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

    monkeypatch.setattr(
        compare_module,
        "load_openml_benchmark_datasets",
        lambda: ({"toy": (np.zeros((6, 2), dtype=np.float32), np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64))}, [{"dataset_name": "toy"}]),
    )
    monkeypatch.setattr(
        compare_module,
        "evaluate_tab_foundry_run",
        lambda *_args, **_kwargs: [
            {"checkpoint_path": "/tmp/step_000025.pt", "step": 25, "training_time": 1.2, "roc_auc": 0.81}
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
            json.dumps({"seed": 0, "step": 25, "training_time": 2.0, "roc_auc": 0.78}) + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(compare_module.subprocess, "run", _fake_run)

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
    assert summary["dataset_count"] == 1
    assert summary["tab_foundry"]["best_step"] == pytest.approx(25.0)
    assert summary["tab_foundry"]["best_roc_auc"] == pytest.approx(0.81)
    assert summary["nanotabpfn"]["best_step"] == pytest.approx(25.0)
    assert summary["nanotabpfn"]["final_roc_auc"] == pytest.approx(0.78)
    assert (out_root / "comparison_summary.json").exists()
    assert (out_root / "comparison_curve.png").exists()


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
