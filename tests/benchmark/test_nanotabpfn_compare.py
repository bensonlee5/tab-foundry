from __future__ import annotations

import json
from pathlib import Path
import runpy
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

import tab_foundry.bench.compare as compare_module
import tab_foundry.bench.nanotabpfn as benchmark_module

REPO_ROOT = Path(__file__).resolve().parents[2]


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
        lambda task_id, **_kwargs: fake_tasks[int(task_id)],
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


def test_compare_main_parses_cli_invocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def _fake_run(config):
        captured["config"] = config
        return {
            "dataset_count": 3,
            "tab_foundry": {"best_roc_auc": 0.71, "final_roc_auc": 0.70},
            "nanotabpfn": {"best_roc_auc": 0.72, "final_roc_auc": 0.71},
            "artifacts": {"comparison_curve_png": "/tmp/comparison_curve.png"},
        }

    monkeypatch.setattr(compare_module, "run_nanotabpfn_benchmark", _fake_run)

    exit_code = compare_module.main(
        [
            "--tab-foundry-run-dir",
            str(tmp_path / "run"),
            "--out-root",
            str(tmp_path / "bench"),
            "--nanotabpfn-root",
            str(tmp_path / "nano"),
            "--nanotab-prior-dump",
            str(tmp_path / "prior.h5"),
            "--device",
            "cpu",
            "--nanotabpfn-steps",
            "125",
            "--nanotabpfn-seeds",
            "3",
            "--control-baseline-id",
            "cls_benchmark_linear_v1",
            "--control-baseline-registry",
            str(tmp_path / "control_baselines.json"),
            "--benchmark-bundle-path",
            str(tmp_path / "bundle.json"),
        ]
    )

    assert exit_code == 0
    config = captured["config"]
    assert config.tab_foundry_run_dir == tmp_path / "run"
    assert config.out_root == tmp_path / "bench"
    assert config.nanotabpfn_root == tmp_path / "nano"
    assert config.nanotab_prior_dump == tmp_path / "prior.h5"
    assert config.device == "cpu"
    assert config.nanotabpfn_steps == 125
    assert config.nanotabpfn_seeds == 3
    assert config.control_baseline_id == "cls_benchmark_linear_v1"
    assert config.control_baseline_registry == tmp_path / "control_baselines.json"
    assert config.benchmark_bundle_path == tmp_path / "bundle.json"
    assert "nanoTabPFN comparison complete:" in capsys.readouterr().out


def test_benchmark_nanotabpfn_script_delegates_to_compare_main(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_main(argv=None):
        captured["argv"] = argv
        return 0

    monkeypatch.setattr(compare_module, "main", _fake_main)
    monkeypatch.setattr(sys, "argv", ["benchmark_nanotabpfn.py"])

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(REPO_ROOT / "scripts" / "benchmark_nanotabpfn.py"), run_name="__main__")

    assert exc_info.value.code == 0
    assert captured["argv"] is None


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
        lambda task_id, **_kwargs: fake_tasks[int(task_id)],
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
        lambda task_id, **_kwargs: fake_tasks[int(task_id)],
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
        lambda task_id, **_kwargs: fake_tasks[int(task_id)],
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
    (smoke_run_dir / "gradient_history.jsonl").write_text("{}\n", encoding="utf-8")
    (smoke_run_dir / "telemetry.json").write_text("{}\n", encoding="utf-8")
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
    policy_calls: dict[str, list[bool]] = {"load": [], "datasets": [], "evaluate": []}

    monkeypatch.setattr(
        compare_module,
        "load_openml_benchmark_datasets",
        lambda *, new_instances=200, benchmark_bundle_path=None, allow_missing_values=False: (
            policy_calls["datasets"].append(bool(allow_missing_values)) or {
                "toy": (
                    np.zeros((6, 2), dtype=np.float32),
                    np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
                )
            },
            [{"task_id": 1, "dataset_name": "toy", "n_rows": 6, "n_features": 2, "n_classes": 2}],
        ),
    )
    monkeypatch.setattr(compare_module, "default_benchmark_bundle_path", lambda: source_bundle_path)
    monkeypatch.setattr(
        compare_module,
        "load_benchmark_bundle_for_execution",
        lambda path=None: (
            policy_calls["load"].append(False) or benchmark_bundle,
            False,
        ),
    )
    monkeypatch.setattr(
        compare_module,
        "evaluate_tab_foundry_run",
        lambda *_args, **_kwargs: (
            policy_calls["evaluate"].append(bool(_kwargs["allow_missing_values"])) or [
                {
                    "checkpoint_path": "/tmp/step_000025.pt",
                    "step": 25,
                    "training_time": 1.2,
                    "roc_auc": 0.81,
                    "dataset_roc_auc": {"toy": 0.81},
                },
                {
                    "checkpoint_path": "/tmp/step_000050.pt",
                    "step": 50,
                    "training_time": 2.4,
                    "evaluation_error": "benchmark evaluation failed for dataset 'toy': Input contains NaN.",
                    "evaluation_error_type": "ValueError",
                    "failed_dataset": "toy",
                },
            ]
        ),
    )
    monkeypatch.setattr(
        compare_module,
        "derive_benchmark_run_record",
        lambda **_kwargs: {
            "manifest_path": "data/manifests/binary.parquet",
            "seed_set": [1],
            "model": {
                "arch": "tabfoundry_staged",
                "stage": "nano_exact",
                "benchmark_profile": "nano_exact",
                "d_icl": 96,
                "tficl_n_heads": 4,
                "tficl_n_layers": 3,
                "head_hidden_dim": 192,
                "input_normalization": "train_zscore_clip",
                "many_class_base": 2,
            },
            "benchmark_bundle": {
                "name": "test_bundle",
                "version": 1,
                "source_path": str(source_bundle_path.resolve()),
                "task_count": 1,
                "task_ids": [1],
            },
            "artifacts": {
                "run_dir": str(smoke_run_dir.resolve()),
                "benchmark_dir": str(out_root.resolve()),
                "prior_dir": None,
                "history_path": str((smoke_run_dir / "train_history.jsonl").resolve()),
                "best_checkpoint_path": str((smoke_run_dir / "checkpoints" / "best.pt").resolve()),
                "comparison_summary_path": str((out_root / "comparison_summary.json").resolve()),
                "comparison_curve_path": str((out_root / "comparison_curve.png").resolve()),
                "benchmark_run_record_path": str((out_root / "benchmark_run_record.json").resolve()),
                "training_surface_record_path": str(
                    (smoke_run_dir / "training_surface_record.json").resolve()
                ),
            },
            "tab_foundry_metrics": {
                "best_step": 25.0,
                "best_training_time": 1.2,
                "best_roc_auc": 0.81,
                "final_step": 25.0,
                "final_training_time": 1.2,
                "final_roc_auc": 0.81,
            },
            "training_diagnostics": {
                "best_val_loss": 0.2,
                "final_val_loss": 0.21,
                "best_val_step": 25.0,
                "post_warmup_train_loss_var": 0.01,
                "mean_grad_norm": 0.4,
                "max_grad_norm": 0.5,
                "final_grad_norm": 0.45,
                "train_elapsed_seconds": 1.2,
                "wall_elapsed_seconds": 1.3,
            },
            "model_size": {"total_params": 1234, "trainable_params": 1234},
            "generated_at_utc": "2026-03-13T00:00:00Z",
        },
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
    assert policy_calls == {"load": [False], "datasets": [False], "evaluate": [False]}
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
    assert summary["benchmark_bundle"]["allow_missing_values"] is False
    assert summary["benchmark_bundle"]["all_tasks_no_missing"] is True
    assert summary["tab_foundry"]["manifest_path"] == "data/manifests/binary.parquet"
    assert summary["tab_foundry"]["model_size"]["total_params"] == 1234
    assert summary["tab_foundry"]["training_diagnostics"]["mean_grad_norm"] == pytest.approx(0.4)
    assert summary["tab_foundry"]["best_to_final_roc_auc_delta"] == pytest.approx(0.0)
    assert summary["tab_foundry"]["best_to_final_dataset_roc_auc_delta"] == {
        "toy": pytest.approx(0.0)
    }
    diagnostics = summary["tab_foundry"]["checkpoint_diagnostics"]
    assert diagnostics["checkpoint_count"] == 2
    assert diagnostics["successful_checkpoint_count"] == 1
    assert diagnostics["failed_checkpoint_count"] == 1
    assert diagnostics["task_count"] == 1
    assert diagnostics["best_checkpoint_path"] == "/tmp/step_000025.pt"
    assert diagnostics["final_checkpoint_path"] == "/tmp/step_000025.pt"
    assert diagnostics["last_attempted_step"] == 50
    assert diagnostics["last_attempted_checkpoint_path"] == "/tmp/step_000050.pt"
    assert diagnostics["bootstrap"]["samples"] == benchmark_module.DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_SAMPLES
    assert diagnostics["best_checkpoint"]["roc_auc_task_bootstrap_ci"]["confidence"] == pytest.approx(
        benchmark_module.DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_CONFIDENCE
    )
    assert diagnostics["checkpoints"][0]["is_best_checkpoint"] is True
    assert diagnostics["checkpoints"][0]["is_final_checkpoint"] is True
    assert diagnostics["checkpoints"][1]["evaluation_error_type"] == "ValueError"
    assert diagnostics["checkpoints"][1]["failed_dataset"] == "toy"
    assert diagnostics["failed_checkpoints"][0]["failed_dataset"] == "toy"
    assert summary["artifacts"]["training_surface_record_json"] == str(
        (smoke_run_dir / "training_surface_record.json").resolve()
    )
    assert summary["artifacts"]["gradient_history_jsonl"] == str(
        (smoke_run_dir / "gradient_history.jsonl").resolve()
    )
    assert summary["artifacts"]["telemetry_json"] == str(
        (smoke_run_dir / "telemetry.json").resolve()
    )
    assert (out_root / "comparison_summary.json").exists()
    assert (out_root / "comparison_curve.png").exists()
    assert (out_root / "benchmark_run_record.json").exists()
    written_summary = json.loads((out_root / "comparison_summary.json").read_text(encoding="utf-8"))
    assert written_summary["artifacts"]["training_surface_record_json"] == str(
        (smoke_run_dir / "training_surface_record.json").resolve()
    )
    assert written_summary["artifacts"]["gradient_history_jsonl"] == str(
        (smoke_run_dir / "gradient_history.jsonl").resolve()
    )
    assert written_summary["artifacts"]["telemetry_json"] == str(
        (smoke_run_dir / "telemetry.json").resolve()
    )
    assert written_summary["tab_foundry"]["checkpoint_diagnostics"]["failed_checkpoint_count"] == 1
    written_bundle = json.loads((out_root / "benchmark_tasks.json").read_text(encoding="utf-8"))
    assert written_bundle == benchmark_bundle


def test_run_nanotabpfn_benchmark_explicit_large_bundle_allows_missing_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    smoke_run_dir = tmp_path / "smoke_run"
    smoke_run_dir.mkdir()
    nanotab_root = tmp_path / "nano"
    (nanotab_root / ".venv" / "bin").mkdir(parents=True)
    (nanotab_root / ".venv" / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    prior_dump = nanotab_root / "300k_150x5_2.h5"
    prior_dump.write_bytes(b"prior")
    out_root = tmp_path / "benchmark_out"
    large_bundle_path = tmp_path / "large_bundle.json"
    large_bundle_path.write_text("{}", encoding="utf-8")
    reuse_curve_path = tmp_path / "reuse_curve.jsonl"
    reuse_curve_path.write_text(
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

    policy_calls: dict[str, list[bool]] = {"load": [], "datasets": [], "evaluate": []}
    benchmark_bundle = {
        "name": "large_bundle",
        "version": 1,
        "selection": {
            "new_instances": 6,
            "task_type": "supervised_classification",
            "max_features": 10,
            "max_classes": 2,
            "max_missing_pct": 5.0,
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

    monkeypatch.setattr(
        compare_module,
        "load_benchmark_bundle_for_execution",
        lambda path=None: (
            policy_calls["load"].append(True) or benchmark_bundle,
            True,
        ),
    )
    monkeypatch.setattr(
        compare_module,
        "load_openml_benchmark_datasets",
        lambda *, new_instances=200, benchmark_bundle_path=None, allow_missing_values=False: (
            policy_calls["datasets"].append(bool(allow_missing_values)) or {
                "toy": (
                    np.zeros((6, 2), dtype=np.float32),
                    np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
                )
            },
            [{"task_id": 1, "dataset_name": "toy", "n_rows": 6, "n_features": 2, "n_classes": 2}],
        ),
    )
    monkeypatch.setattr(
        compare_module,
        "evaluate_tab_foundry_run",
        lambda *_args, **_kwargs: (
            policy_calls["evaluate"].append(bool(_kwargs["allow_missing_values"])) or [
                {
                    "checkpoint_path": "/tmp/step_000025.pt",
                    "step": 25,
                    "training_time": 1.2,
                    "roc_auc": 0.81,
                    "dataset_roc_auc": {"toy": 0.81},
                }
            ]
        ),
    )
    monkeypatch.setattr(
        compare_module,
        "summarize_checkpoint_curve",
        lambda records, **_kwargs: {"records": records},
    )
    monkeypatch.setattr(compare_module, "plot_comparison_curve", lambda **_kwargs: None)
    monkeypatch.setattr(
        compare_module,
        "build_comparison_summary",
        lambda **_kwargs: {
            "dataset_count": 1,
            "benchmark_bundle": {"name": "large_bundle", "allow_missing_values": True},
            "tab_foundry": {},
            "nanotabpfn": {},
        },
    )
    monkeypatch.setattr(
        compare_module,
        "derive_benchmark_run_record",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError(
                "checkpoint config must include explicit model.arch metadata for benchmark "
                "registration; legacy checkpoints without persisted model.arch cannot be "
                "registered"
            )
        ),
    )

    summary = compare_module.run_nanotabpfn_benchmark(
        compare_module.NanoTabPFNBenchmarkConfig(
            tab_foundry_run_dir=smoke_run_dir,
            out_root=out_root,
            nanotabpfn_root=nanotab_root,
            nanotab_prior_dump=prior_dump,
            benchmark_bundle_path=large_bundle_path,
            reuse_nanotabpfn_curve_path=reuse_curve_path,
        )
    )

    assert policy_calls == {"load": [True], "datasets": [True], "evaluate": [True]}
    assert summary["benchmark_bundle"]["allow_missing_values"] is True


def test_run_nanotabpfn_benchmark_honors_nondefault_bundle_path(
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
    source_bundle_path = tmp_path / "custom_bundle.json"
    default_bundle_path = tmp_path / "default_bundle.json"
    benchmark_bundle = {
        "name": "custom_bundle",
        "version": 1,
        "selection": {
            "new_instances": 6,
            "task_type": "supervised_classification",
            "max_features": 10,
            "max_classes": 3,
            "max_missing_pct": 0.0,
            "min_minority_class_pct": 2.5,
        },
        "task_ids": [7],
        "tasks": [
            {
                "task_id": 7,
                "dataset_name": "toy_multi",
                "n_rows": 6,
                "n_features": 2,
                "n_classes": 3,
            }
        ],
    }
    source_bundle_path.write_text(
        json.dumps(benchmark_bundle, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    default_bundle_path.write_text(
        json.dumps({"name": "unused", "version": 1, "selection": dict(DEFAULT_BENCHMARK_SELECTION), "task_ids": [1], "tasks": []}, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )

    captured: dict[str, Any] = {}

    def _fake_load_bundle(path: Path | None = None) -> tuple[dict[str, Any], bool]:
        captured["bundle_path"] = None if path is None else str(Path(path).resolve())
        return benchmark_bundle, False

    def _fake_load_datasets(
        *,
        new_instances: int = 200,
        benchmark_bundle_path: Path | None = None,
        allow_missing_values: bool = False,
    ) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
        captured["dataset_new_instances"] = int(new_instances)
        captured["dataset_bundle_path"] = None if benchmark_bundle_path is None else str(Path(benchmark_bundle_path).resolve())
        captured["dataset_allow_missing_values"] = bool(allow_missing_values)
        return (
            {
                "toy_multi": (
                    np.zeros((6, 2), dtype=np.float32),
                    np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64),
                )
            },
            [{"task_id": 7, "dataset_name": "toy_multi", "n_rows": 6, "n_features": 2, "n_classes": 3}],
        )

    monkeypatch.setattr(compare_module, "default_benchmark_bundle_path", lambda: default_bundle_path)
    monkeypatch.setattr(compare_module, "load_benchmark_bundle_for_execution", _fake_load_bundle)
    monkeypatch.setattr(compare_module, "load_openml_benchmark_datasets", _fake_load_datasets)
    monkeypatch.setattr(
        compare_module,
        "evaluate_tab_foundry_run",
        lambda *_args, **_kwargs: [
            {
                "checkpoint_path": "/tmp/step_000025.pt",
                "step": 25,
                "training_time": 1.2,
                "roc_auc": 0.81,
                "dataset_roc_auc": {"toy_multi": 0.81},
            }
        ],
    )
    monkeypatch.setattr(
        compare_module,
        "derive_benchmark_run_record",
        lambda **_kwargs: {
            "manifest_path": "data/manifests/multiclass.parquet",
            "seed_set": [1],
            "model": {
                "arch": "tabfoundry_staged",
                "stage": "many_class",
                "benchmark_profile": "many_class",
                "d_icl": 96,
                "tficl_n_heads": 4,
                "tficl_n_layers": 3,
                "head_hidden_dim": 192,
                "input_normalization": "train_zscore_clip",
                "many_class_base": 10,
            },
            "benchmark_bundle": {
                "name": "custom_bundle",
                "version": 1,
                "source_path": str(source_bundle_path.resolve()),
                "task_count": 1,
                "task_ids": [7],
            },
            "artifacts": {
                "run_dir": str(smoke_run_dir.resolve()),
                "benchmark_dir": str(out_root.resolve()),
                "prior_dir": None,
                "history_path": str((smoke_run_dir / "train_history.jsonl").resolve()),
                "best_checkpoint_path": str((smoke_run_dir / "checkpoints" / "best.pt").resolve()),
                "comparison_summary_path": str((out_root / "comparison_summary.json").resolve()),
                "comparison_curve_path": str((out_root / "comparison_curve.png").resolve()),
                "benchmark_run_record_path": str((out_root / "benchmark_run_record.json").resolve()),
                "training_surface_record_path": str(
                    (smoke_run_dir / "training_surface_record.json").resolve()
                ),
            },
            "tab_foundry_metrics": {
                "best_step": 25.0,
                "best_training_time": 1.2,
                "best_roc_auc": 0.81,
                "final_step": 25.0,
                "final_training_time": 1.2,
                "final_roc_auc": 0.81,
            },
            "training_diagnostics": {
                "best_val_loss": 0.2,
                "final_val_loss": 0.21,
                "best_val_step": 25.0,
                "post_warmup_train_loss_var": 0.01,
                "mean_grad_norm": 0.4,
                "max_grad_norm": 0.5,
                "final_grad_norm": 0.45,
                "train_elapsed_seconds": 1.2,
                "wall_elapsed_seconds": 1.3,
            },
            "model_size": {"total_params": 1234, "trainable_params": 1234},
            "generated_at_utc": "2026-03-13T00:00:00Z",
        },
    )

    def _fake_run(cmd: list[str], *, cwd: Path, check: bool) -> subprocess.CompletedProcess[str]:
        out_index = cmd.index("--out-path") + 1
        out_path = Path(cmd[out_index])
        out_path.write_text(
            json.dumps(
                {
                    "seed": 0,
                    "step": 25,
                    "training_time": 2.0,
                    "roc_auc": 0.78,
                    "dataset_roc_auc": {"toy_multi": 0.78},
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
            benchmark_bundle_path=source_bundle_path,
        )
    )

    assert captured["bundle_path"] == str(source_bundle_path.resolve())
    assert captured["dataset_new_instances"] == 6
    assert captured["dataset_bundle_path"] == str(source_bundle_path.resolve())
    assert captured["dataset_allow_missing_values"] is False
    assert summary["benchmark_bundle"]["source_path"] == str(source_bundle_path.resolve())
    assert summary["artifacts"]["training_surface_record_json"] == str(
        (smoke_run_dir / "training_surface_record.json").resolve()
    )
    written_summary = json.loads((out_root / "comparison_summary.json").read_text(encoding="utf-8"))
    assert written_summary["benchmark_bundle"]["source_path"] == str(source_bundle_path.resolve())
    assert written_summary["artifacts"]["training_surface_record_json"] == str(
        (smoke_run_dir / "training_surface_record.json").resolve()
    )
    written_bundle = json.loads((out_root / "benchmark_tasks.json").read_text(encoding="utf-8"))
    assert written_bundle == benchmark_bundle


def test_run_nanotabpfn_benchmark_skips_legacy_record_derivation_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    smoke_run_dir = tmp_path / "smoke_run"
    smoke_run_dir.mkdir()
    nanotab_root = tmp_path / "nano"
    (nanotab_root / ".venv" / "bin").mkdir(parents=True)
    (nanotab_root / ".venv" / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    prior_dump = nanotab_root / "300k_150x5_2.h5"
    prior_dump.write_bytes(b"prior")
    out_root = tmp_path / "benchmark_out"
    source_bundle_path = _write_benchmark_bundle(
        tmp_path / "legacy_bundle.json",
        tasks=[
            {
                "task_id": 1,
                "dataset_name": "toy",
                "n_rows": 6,
                "n_features": 2,
                "n_classes": 2,
            }
        ],
    )
    benchmark_bundle = json.loads(source_bundle_path.read_text(encoding="utf-8"))

    monkeypatch.setattr(compare_module, "default_benchmark_bundle_path", lambda: source_bundle_path)
    monkeypatch.setattr(
        compare_module,
        "load_benchmark_bundle_for_execution",
        lambda path=None: (benchmark_bundle, False),
    )
    monkeypatch.setattr(
        compare_module,
        "load_openml_benchmark_datasets",
        lambda *, new_instances=200, benchmark_bundle_path=None, allow_missing_values=False: (
            {
                "toy": (
                    np.zeros((6, 2), dtype=np.float32),
                    np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
                )
            },
            [{"task_id": 1, "dataset_name": "toy", "n_rows": 6, "n_features": 2, "n_classes": 2}],
        ),
    )
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
    monkeypatch.setattr(
        compare_module,
        "derive_benchmark_run_record",
        lambda **_kwargs: (_ for _ in ()).throw(
            RuntimeError(
                "checkpoint config must include explicit model.arch metadata for benchmark "
                "registration; legacy checkpoints without persisted model.arch cannot be "
                "registered"
            )
        ),
    )

    def _fake_run(cmd: list[str], *, cwd: Path, check: bool) -> subprocess.CompletedProcess[str]:
        out_index = cmd.index("--out-path") + 1
        Path(cmd[out_index]).write_text(
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

    written_summary = json.loads((out_root / "comparison_summary.json").read_text(encoding="utf-8"))
    assert summary["artifacts"]["benchmark_run_record_json"] is None
    assert written_summary["artifacts"]["benchmark_run_record_json"] is None
    assert summary["artifacts"]["training_surface_record_json"] is None
    assert written_summary["artifacts"]["training_surface_record_json"] is None
    assert "persisted model.arch" in summary["tab_foundry"]["benchmark_run_record_warning"]
    assert "persisted model.arch" in written_summary["tab_foundry"]["benchmark_run_record_warning"]
    assert not (out_root / "benchmark_run_record.json").exists()
    assert "Skipping benchmark_run_record.json derivation" in capsys.readouterr().err


def test_explicit_benchmark_bundle_paths_accept_checked_in_legacy_and_medium_binary_bundles() -> None:
    legacy_bundle_path = (
        REPO_ROOT / "src" / "tab_foundry" / "bench" / "nanotabpfn_openml_benchmark_v1.json"
    )
    medium_bundle_path = (
        REPO_ROOT / "src" / "tab_foundry" / "bench" / "nanotabpfn_openml_binary_medium_v1.json"
    )

    legacy_bundle = benchmark_module.load_benchmark_bundle(legacy_bundle_path)
    medium_bundle = benchmark_module.load_benchmark_bundle(medium_bundle_path)

    assert legacy_bundle["name"] == "nanotabpfn_openml_binary_small"
    assert legacy_bundle["task_ids"] == [363613, 363621, 363629]
    assert medium_bundle["name"] == "nanotabpfn_openml_binary_medium"
    assert len(medium_bundle["task_ids"]) == 10
    assert all(int(task["n_classes"]) == 2 for task in medium_bundle["tasks"])


def test_default_benchmark_bundle_path_resolves_to_medium_binary_bundle() -> None:
    bundle_path = compare_module.default_benchmark_bundle_path()

    assert bundle_path == (
        REPO_ROOT / "src" / "tab_foundry" / "bench" / "nanotabpfn_openml_binary_medium_v1.json"
    )
    assert benchmark_module.load_benchmark_bundle(bundle_path)["name"] == "nanotabpfn_openml_binary_medium"


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
        lambda *, new_instances=200, benchmark_bundle_path=None, allow_missing_values=False: (
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
    monkeypatch.setattr(
        compare_module,
        "load_benchmark_bundle_for_execution",
        lambda path=None: (benchmark_bundle, False),
    )
    monkeypatch.setattr(
        compare_module,
        "evaluate_tab_foundry_run",
        lambda *_args, **_kwargs: [
            {"checkpoint_path": "/tmp/step_000025.pt", "step": 25, "training_time": 1.2, "roc_auc": 0.81}
        ],
    )
    monkeypatch.setattr(
        compare_module,
        "derive_benchmark_run_record",
        lambda **_kwargs: {
            "manifest_path": "data/manifests/default.parquet",
            "seed_set": [1],
            "model": {
                "arch": "tabfoundry_staged",
                "stage": "nano_exact",
                "benchmark_profile": "nano_exact",
                "d_icl": 96,
                "tficl_n_heads": 4,
                "tficl_n_layers": 3,
                "head_hidden_dim": 192,
                "input_normalization": "train_zscore_clip",
                "many_class_base": 2,
            },
            "benchmark_bundle": {
                "name": "test_bundle",
                "version": 1,
                "source_path": str(source_bundle_path.resolve()),
                "task_count": 1,
                "task_ids": [1],
            },
            "artifacts": {
                "run_dir": str(smoke_run_dir.resolve()),
                "benchmark_dir": str(out_root.resolve()),
                "prior_dir": None,
                "history_path": str((smoke_run_dir / "train_history.jsonl").resolve()),
                "best_checkpoint_path": str((smoke_run_dir / "checkpoints" / "best.pt").resolve()),
                "comparison_summary_path": str((out_root / "comparison_summary.json").resolve()),
                "comparison_curve_path": str((out_root / "comparison_curve.png").resolve()),
                "benchmark_run_record_path": str((out_root / "benchmark_run_record.json").resolve()),
                "training_surface_record_path": str(
                    (smoke_run_dir / "training_surface_record.json").resolve()
                ),
            },
            "tab_foundry_metrics": {
                "best_step": 25.0,
                "best_training_time": 1.2,
                "best_roc_auc": 0.81,
                "final_step": 25.0,
                "final_training_time": 1.2,
                "final_roc_auc": 0.81,
            },
            "training_diagnostics": {
                "best_val_loss": 0.2,
                "final_val_loss": 0.21,
                "best_val_step": 25.0,
                "post_warmup_train_loss_var": 0.01,
                "mean_grad_norm": 0.4,
                "max_grad_norm": 0.5,
                "final_grad_norm": 0.45,
                "train_elapsed_seconds": 1.2,
                "wall_elapsed_seconds": 1.3,
            },
            "model_size": {"total_params": 1234, "trainable_params": 1234},
            "generated_at_utc": "2026-03-13T00:00:00Z",
        },
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
    assert summary["artifacts"]["training_surface_record_json"] == str(
        (smoke_run_dir / "training_surface_record.json").resolve()
    )
    written_summary = json.loads((out_root / "comparison_summary.json").read_text(encoding="utf-8"))
    assert written_summary["control_baseline"]["budget_class"] == "short-run"
    assert written_summary["artifacts"]["training_surface_record_json"] == str(
        (smoke_run_dir / "training_surface_record.json").resolve()
    )


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


def test_build_comparison_summary_preserves_model_identity_metadata(tmp_path: Path) -> None:
    summary = benchmark_module.build_comparison_summary(
        tab_foundry_records=[
            {
                "checkpoint_path": "/tmp/step_000025.pt",
                "step": 25,
                "training_time": 1.2,
                "roc_auc": 0.81,
                "model_arch": "tabfoundry_staged",
                "model_stage": "nano_exact",
                "benchmark_profile": "nano_exact",
            }
        ],
        nanotabpfn_records=[
            {"seed": 0, "step": 25, "training_time": 2.0, "roc_auc": 0.78}
        ],
        benchmark_tasks=[
            {"task_id": 1, "dataset_name": "toy", "n_rows": 6, "n_features": 2, "n_classes": 2}
        ],
        benchmark_bundle={
            "name": "toy_bundle",
            "version": 1,
            "selection": dict(DEFAULT_BENCHMARK_SELECTION),
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
        },
        benchmark_bundle_path=tmp_path / "bundle.json",
        tab_foundry_run_dir=tmp_path / "run",
        nanotabpfn_root=tmp_path / "nano",
        nanotabpfn_python=tmp_path / "nano" / ".venv" / "bin" / "python",
    )

    assert summary["tab_foundry"]["model_arch"] == "tabfoundry_staged"
    assert summary["tab_foundry"]["model_stage"] == "nano_exact"
    assert summary["tab_foundry"]["benchmark_profile"] == "nano_exact"
    assert summary["benchmark_bundle"]["allow_missing_values"] is False
    assert summary["benchmark_bundle"]["all_tasks_no_missing"] is True
    assert summary["tab_foundry"]["checkpoint_diagnostics"]["checkpoint_count"] == 1
    assert summary["tab_foundry"]["checkpoint_diagnostics"]["failed_checkpoint_count"] == 0
