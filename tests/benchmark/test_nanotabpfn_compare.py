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


def _create_nanotab_env(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    smoke_run_dir = tmp_path / "smoke_run"
    smoke_run_dir.mkdir()
    nanotab_root = tmp_path / "nano"
    (nanotab_root / ".venv" / "bin").mkdir(parents=True)
    nanotab_python = nanotab_root / ".venv" / "bin" / "python"
    nanotab_python.write_text("#!/bin/sh\n", encoding="utf-8")
    prior_dump = nanotab_root / "300k_150x5_2.h5"
    prior_dump.write_bytes(b"prior")
    return smoke_run_dir, nanotab_root, nanotab_python, prior_dump


def _fake_benchmark_bundle() -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    datasets = {
        "toy_alpha": (
            np.asarray(
                [[0.0, 1.0], [1.0, 0.0], [0.0, 2.0], [1.0, 1.0], [0.0, 3.0], [1.0, 2.0]],
                dtype=np.float32,
            ),
            np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
        ),
        "toy_beta": (
            np.asarray(
                [[2.0, 0.0], [3.0, 1.0], [2.0, 1.0], [3.0, 0.0], [2.0, 2.0], [3.0, 2.0]],
                dtype=np.float32,
            ),
            np.asarray([1, 0, 1, 0, 1, 0], dtype=np.int64),
        ),
    }
    benchmark_tasks = [
        {"task_id": 1, "dataset_name": "toy_alpha", "n_rows": 6, "n_features": 2, "n_classes": 2},
        {"task_id": 2, "dataset_name": "toy_beta", "n_rows": 6, "n_features": 2, "n_classes": 2},
    ]
    return datasets, benchmark_tasks


def _fake_tab_foundry_records() -> list[dict[str, Any]]:
    return [
        {"checkpoint_path": "/tmp/step_000025.pt", "step": 25, "training_time": 1.2, "roc_auc": 0.81},
        {"checkpoint_path": "/tmp/step_000050.pt", "step": 50, "training_time": 2.4, "roc_auc": 0.79},
    ]


def _install_fake_loader(
    monkeypatch: pytest.MonkeyPatch,
    *,
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    benchmark_tasks: list[dict[str, Any]],
    call_counter: dict[str, int] | None = None,
) -> None:
    def _fake_loader(**_kwargs: Any) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
        if call_counter is not None:
            call_counter["count"] = call_counter.get("count", 0) + 1
        return datasets, benchmark_tasks

    monkeypatch.setattr(benchmark_module, "load_openml_benchmark_datasets", _fake_loader)


def _install_fake_compare_execution(
    monkeypatch: pytest.MonkeyPatch,
    *,
    records: list[dict[str, Any]] | None = None,
    captured: dict[str, Any] | None = None,
) -> None:
    monkeypatch.setattr(
        compare_module,
        "evaluate_tab_foundry_run",
        lambda *_args, **_kwargs: _fake_tab_foundry_records() if records is None else records,
    )

    def _fake_run(cmd: list[str], *, cwd: Path, check: bool) -> subprocess.CompletedProcess[str]:
        if captured is not None:
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

    monkeypatch.setattr(compare_module, "subprocess", SimpleNamespace(run=_fake_run))


def _assert_not_called(*_args: Any, **_kwargs: Any) -> Any:
    raise AssertionError("unexpected execution path")


def _build_config(
    *,
    smoke_run_dir: Path,
    out_root: Path,
    nanotab_root: Path,
    prior_dump: Path,
    benchmark_bundle_dir: Path | None = None,
) -> compare_module.NanoTabPFNBenchmarkConfig:
    return compare_module.NanoTabPFNBenchmarkConfig(
        tab_foundry_run_dir=smoke_run_dir,
        out_root=out_root,
        benchmark_bundle_dir=benchmark_bundle_dir,
        nanotabpfn_root=nanotab_root,
        nanotab_prior_dump=prior_dump,
    )


def _write_bundle_inputs(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


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


def test_run_nanotabpfn_benchmark_creates_bundle_and_summary_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    smoke_run_dir, nanotab_root, nanotab_python, prior_dump = _create_nanotab_env(tmp_path)
    datasets, benchmark_tasks = _fake_benchmark_bundle()
    load_counter = {"count": 0}
    captured: dict[str, Any] = {}
    out_root = tmp_path / "benchmark_out"

    monkeypatch.setattr(benchmark_module, "NANOTABPFN_TASK_IDS", (1, 2))
    _install_fake_loader(monkeypatch, datasets=datasets, benchmark_tasks=benchmark_tasks, call_counter=load_counter)
    _install_fake_compare_execution(monkeypatch, captured=captured)

    summary = compare_module.run_nanotabpfn_benchmark(
        _build_config(
            smoke_run_dir=smoke_run_dir,
            out_root=out_root,
            nanotab_root=nanotab_root,
            prior_dump=prior_dump,
        )
    )

    benchmark_inputs_path = out_root / benchmark_module.BENCHMARK_INPUTS_FILENAME
    dataset_cache_path = out_root / benchmark_module.BENCHMARK_DATASET_CACHE_FILENAME
    benchmark_tasks_path = out_root / "benchmark_tasks.json"
    comparison_summary_path = out_root / "comparison_summary.json"

    assert load_counter["count"] == 1
    assert captured["cwd"] == nanotab_root.resolve()
    assert captured["check"] is True
    assert Path(captured["cmd"][0]) == nanotab_python.resolve()
    assert Path(captured["cmd"][1]) == Path(compare_module.__file__).resolve().with_name("nanotabpfn_helper.py")
    assert summary["dataset_count"] == 2
    assert summary["benchmark_inputs"]["source"] == "created"
    assert summary["benchmark_inputs"]["task_ids"] == [1, 2]
    assert summary["benchmark_inputs"]["task_count"] == 2
    assert summary["benchmark_inputs"]["dataset_names"] == ["toy_alpha", "toy_beta"]
    assert summary["benchmark_inputs"]["bundle_dir"] == str(out_root.resolve())
    assert summary["artifacts"]["benchmark_inputs_json"] == str(benchmark_inputs_path.resolve())
    assert summary["artifacts"]["benchmark_dataset_cache"] == str(dataset_cache_path.resolve())
    assert summary["tab_foundry"]["best_step"] == pytest.approx(25.0)
    assert summary["tab_foundry"]["best_roc_auc"] == pytest.approx(0.81)
    assert summary["nanotabpfn"]["best_step"] == pytest.approx(25.0)
    assert summary["nanotabpfn"]["final_roc_auc"] == pytest.approx(0.78)
    assert comparison_summary_path.exists()

    bundle_payload = json.loads(benchmark_inputs_path.read_text(encoding="utf-8"))
    assert bundle_payload["task_ids"] == [1, 2]
    assert bundle_payload["benchmark_tasks"] == benchmark_tasks
    assert bundle_payload["dataset_names"] == ["toy_alpha", "toy_beta"]
    assert bundle_payload["dataset_cache_sha256"] == summary["benchmark_inputs"]["dataset_cache_sha256"]
    assert json.loads(benchmark_tasks_path.read_text(encoding="utf-8")) == benchmark_tasks
    assert json.loads(comparison_summary_path.read_text(encoding="utf-8"))["benchmark_inputs"]["source"] == "created"


def test_run_nanotabpfn_benchmark_reuses_bundle_without_reloading_openml(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    smoke_run_dir, nanotab_root, _nanotab_python, prior_dump = _create_nanotab_env(tmp_path)
    datasets, benchmark_tasks = _fake_benchmark_bundle()
    load_counter = {"count": 0}
    bundle_dir = tmp_path / "bundle"
    first_out_root = tmp_path / "benchmark_out_first"
    second_out_root = tmp_path / "benchmark_out_second"

    monkeypatch.setattr(benchmark_module, "NANOTABPFN_TASK_IDS", (1, 2))
    _install_fake_loader(monkeypatch, datasets=datasets, benchmark_tasks=benchmark_tasks, call_counter=load_counter)
    _install_fake_compare_execution(monkeypatch)

    first_summary = compare_module.run_nanotabpfn_benchmark(
        _build_config(
            smoke_run_dir=smoke_run_dir,
            out_root=first_out_root,
            nanotab_root=nanotab_root,
            prior_dump=prior_dump,
            benchmark_bundle_dir=bundle_dir,
        )
    )
    assert first_summary["benchmark_inputs"]["source"] == "created"
    assert load_counter["count"] == 1

    monkeypatch.setattr(benchmark_module, "load_openml_benchmark_datasets", _assert_not_called)

    second_summary = compare_module.run_nanotabpfn_benchmark(
        _build_config(
            smoke_run_dir=smoke_run_dir,
            out_root=second_out_root,
            nanotab_root=nanotab_root,
            prior_dump=prior_dump,
            benchmark_bundle_dir=bundle_dir,
        )
    )

    assert second_summary["benchmark_inputs"]["source"] == "reused"
    assert second_summary["benchmark_inputs"]["bundle_dir"] == str(bundle_dir.resolve())
    assert second_summary["artifacts"]["benchmark_dataset_cache"] == str(
        (bundle_dir / benchmark_module.BENCHMARK_DATASET_CACHE_FILENAME).resolve()
    )
    assert json.loads((second_out_root / "benchmark_tasks.json").read_text(encoding="utf-8")) == benchmark_tasks


def test_run_nanotabpfn_benchmark_rerun_with_same_out_root_rematerializes_bundle(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    smoke_run_dir, nanotab_root, _nanotab_python, prior_dump = _create_nanotab_env(tmp_path)
    datasets, benchmark_tasks = _fake_benchmark_bundle()
    load_counter = {"count": 0}
    out_root = tmp_path / "benchmark_out"

    monkeypatch.setattr(benchmark_module, "NANOTABPFN_TASK_IDS", (1, 2))
    _install_fake_loader(monkeypatch, datasets=datasets, benchmark_tasks=benchmark_tasks, call_counter=load_counter)
    _install_fake_compare_execution(monkeypatch)

    first_summary = compare_module.run_nanotabpfn_benchmark(
        _build_config(
            smoke_run_dir=smoke_run_dir,
            out_root=out_root,
            nanotab_root=nanotab_root,
            prior_dump=prior_dump,
        )
    )
    second_summary = compare_module.run_nanotabpfn_benchmark(
        _build_config(
            smoke_run_dir=smoke_run_dir,
            out_root=out_root,
            nanotab_root=nanotab_root,
            prior_dump=prior_dump,
        )
    )

    assert load_counter["count"] == 2
    assert first_summary["benchmark_inputs"]["source"] == "created"
    assert second_summary["benchmark_inputs"]["source"] == "created"
    assert second_summary["benchmark_inputs"]["bundle_dir"] == str(out_root.resolve())
    assert json.loads((out_root / "benchmark_tasks.json").read_text(encoding="utf-8")) == benchmark_tasks


def test_run_nanotabpfn_benchmark_recreates_legacy_out_root_bundle_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    smoke_run_dir, nanotab_root, _nanotab_python, prior_dump = _create_nanotab_env(tmp_path)
    datasets, benchmark_tasks = _fake_benchmark_bundle()
    load_counter = {"count": 0}
    out_root = tmp_path / "benchmark_out"
    legacy_datasets = {
        "legacy": (
            np.asarray([[9.0, 9.0], [8.0, 8.0]], dtype=np.float32),
            np.asarray([0, 1], dtype=np.int64),
        )
    }

    benchmark_module.save_dataset_cache(out_root / benchmark_module.BENCHMARK_DATASET_CACHE_FILENAME, legacy_datasets)
    (out_root / "benchmark_tasks.json").write_text(
        json.dumps([{"task_id": 999, "dataset_name": "legacy"}], indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(benchmark_module, "NANOTABPFN_TASK_IDS", (1, 2))
    _install_fake_loader(monkeypatch, datasets=datasets, benchmark_tasks=benchmark_tasks, call_counter=load_counter)
    _install_fake_compare_execution(monkeypatch)

    summary = compare_module.run_nanotabpfn_benchmark(
        _build_config(
            smoke_run_dir=smoke_run_dir,
            out_root=out_root,
            nanotab_root=nanotab_root,
            prior_dump=prior_dump,
        )
    )

    assert load_counter["count"] == 1
    assert summary["benchmark_inputs"]["source"] == "created"
    assert json.loads((out_root / benchmark_module.BENCHMARK_INPUTS_FILENAME).read_text(encoding="utf-8"))[
        "benchmark_tasks"
    ] == benchmark_tasks
    assert list(benchmark_module.load_dataset_cache(out_root / benchmark_module.BENCHMARK_DATASET_CACHE_FILENAME)) == [
        "toy_alpha",
        "toy_beta",
    ]


@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (
            lambda payload: payload.__setitem__("task_ids", [999]),
            "task-id drift",
        ),
        (
            lambda payload: payload.__setitem__("task_count", int(payload["task_count"]) + 1),
            "task-count drift",
        ),
        (
            lambda payload: (
                payload.__setitem__("benchmark_tasks", list(reversed(payload["benchmark_tasks"]))),
                payload.__setitem__("dataset_names", list(reversed(payload["dataset_names"]))),
            ),
            "task-list drift",
        ),
    ],
)
def test_run_nanotabpfn_benchmark_fails_on_bundle_drift_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mutator: Any,
    message: str,
) -> None:
    smoke_run_dir, nanotab_root, _nanotab_python, prior_dump = _create_nanotab_env(tmp_path)
    datasets, benchmark_tasks = _fake_benchmark_bundle()
    bundle_dir = tmp_path / "bundle"

    monkeypatch.setattr(benchmark_module, "NANOTABPFN_TASK_IDS", (1, 2))
    _install_fake_loader(monkeypatch, datasets=datasets, benchmark_tasks=benchmark_tasks)
    _install_fake_compare_execution(monkeypatch)

    compare_module.run_nanotabpfn_benchmark(
        _build_config(
            smoke_run_dir=smoke_run_dir,
            out_root=tmp_path / "benchmark_out_initial",
            nanotab_root=nanotab_root,
            prior_dump=prior_dump,
            benchmark_bundle_dir=bundle_dir,
        )
    )

    benchmark_inputs_path = bundle_dir / benchmark_module.BENCHMARK_INPUTS_FILENAME
    payload = json.loads(benchmark_inputs_path.read_text(encoding="utf-8"))
    _ = mutator(payload)
    _write_bundle_inputs(benchmark_inputs_path, payload)

    monkeypatch.setattr(benchmark_module, "load_openml_benchmark_datasets", _assert_not_called)
    monkeypatch.setattr(compare_module, "evaluate_tab_foundry_run", _assert_not_called)
    monkeypatch.setattr(compare_module, "subprocess", SimpleNamespace(run=_assert_not_called))

    with pytest.raises(RuntimeError, match=message):
        compare_module.run_nanotabpfn_benchmark(
            _build_config(
                smoke_run_dir=smoke_run_dir,
                out_root=tmp_path / "benchmark_out_retry",
                nanotab_root=nanotab_root,
                prior_dump=prior_dump,
                benchmark_bundle_dir=bundle_dir,
            )
        )


def test_run_nanotabpfn_benchmark_fails_on_cache_metadata_mismatch_before_execution(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    smoke_run_dir, nanotab_root, _nanotab_python, prior_dump = _create_nanotab_env(tmp_path)
    datasets, benchmark_tasks = _fake_benchmark_bundle()
    bundle_dir = tmp_path / "bundle"

    monkeypatch.setattr(benchmark_module, "NANOTABPFN_TASK_IDS", (1, 2))
    _install_fake_loader(monkeypatch, datasets=datasets, benchmark_tasks=benchmark_tasks)
    _install_fake_compare_execution(monkeypatch)

    compare_module.run_nanotabpfn_benchmark(
        _build_config(
            smoke_run_dir=smoke_run_dir,
            out_root=tmp_path / "benchmark_out_initial",
            nanotab_root=nanotab_root,
            prior_dump=prior_dump,
            benchmark_bundle_dir=bundle_dir,
        )
    )

    benchmark_inputs_path = bundle_dir / benchmark_module.BENCHMARK_INPUTS_FILENAME
    payload = json.loads(benchmark_inputs_path.read_text(encoding="utf-8"))
    payload["benchmark_tasks"][0]["dataset_name"] = "renamed_alpha"
    payload["benchmark_tasks"][1]["dataset_name"] = "renamed_beta"
    payload["dataset_names"] = ["renamed_alpha", "renamed_beta"]
    _write_bundle_inputs(benchmark_inputs_path, payload)

    monkeypatch.setattr(benchmark_module, "load_openml_benchmark_datasets", _assert_not_called)
    monkeypatch.setattr(compare_module, "evaluate_tab_foundry_run", _assert_not_called)
    monkeypatch.setattr(compare_module, "subprocess", SimpleNamespace(run=_assert_not_called))

    with pytest.raises(RuntimeError, match="cache metadata mismatch"):
        compare_module.run_nanotabpfn_benchmark(
            _build_config(
                smoke_run_dir=smoke_run_dir,
                out_root=tmp_path / "benchmark_out_retry",
                nanotab_root=nanotab_root,
                prior_dump=prior_dump,
                benchmark_bundle_dir=bundle_dir,
            )
        )


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
