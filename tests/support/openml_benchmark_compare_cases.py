from __future__ import annotations

import json
from pathlib import Path
import subprocess
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest
import torch

import tab_foundry.bench.comparison_runtime as compare_module
import tab_foundry.bench.openml_benchmark as benchmark_module
import tab_foundry.bench.openml_benchmark.artifacts as benchmark_artifacts_module
import tab_foundry.cli.bench_compare as compare_cli_module
from tests.support import manifest_and_dataset_cases as data_cases
from tests.support.paths import REPO_ROOT


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


def _runtime_benchmark_surface(
    *,
    benchmark_manifest_path: Path,
    source_bundle_path: Path,
    benchmark_bundle: dict[str, Any],
    datasets: dict[str, tuple[np.ndarray, np.ndarray]],
    task_records: list[dict[str, Any]],
    allow_missing_values: bool,
) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, Any]], dict[str, Any]]:
    return (
        datasets,
        task_records,
        {
            "manifest_path": str(benchmark_manifest_path.resolve()),
            "contract_version": 2,
            "manifest_sha256": "0" * 64,
            "task_type": str(benchmark_bundle["selection"]["task_type"]),
            "allow_missing_values": allow_missing_values,
            "benchmark_bundle": {
                "name": str(benchmark_bundle["name"]),
                "version": int(benchmark_bundle["version"]),
                "source_path": str(source_bundle_path.resolve()),
                "task_count": int(len(task_records)),
                "task_ids": [int(task["task_id"]) for task in task_records],
                "selection": dict(benchmark_bundle["selection"]),
                "allow_missing_values": allow_missing_values,
                "all_tasks_no_missing": not allow_missing_values,
            },
            "persisted_summary": {},
        },
    )


def test_load_benchmark_manifest_datasets_matches_notebook_filters(tmp_path: Path) -> None:
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
    with pytest.raises(RuntimeError, match="materialized manifest parquet"):
        benchmark_module.load_benchmark_manifest_datasets(
            benchmark_manifest_path=bundle_path,
        )


def test_load_benchmark_manifest_datasets_allows_missing_when_manifest_provenance_allows_it(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    x_train, y_train, x_test, y_test = data_cases._classification_arrays(seed=17, n_classes=2)
    x_train[0, 0] = float("nan")
    x_test[0, 1] = float("inf")
    selection = dict(DEFAULT_BENCHMARK_SELECTION)
    selection["max_missing_pct"] = 10.0
    _ = data_cases._write_packed_shard(
        root / "shard_00000",
        datasets=[
            {
                "dataset_index": 0,
                "x_train": x_train,
                "y_train": y_train,
                "x_test": x_test,
                "y_test": y_test,
                "feature_types": ["floating"] * int(x_train.shape[1]),
                "metadata": {
                    **data_cases._classification_metadata(
                        n_features=x_train.shape[1],
                        n_classes=2,
                        filter_status="accepted",
                        filter_accepted=True,
                    ),
                    "observed_task": {"dataset_name": "missing_case"},
                    "openml": {"task_id": 101, "dataset_name": "missing_case"},
                    "benchmark_bundle": {
                        "name": "missing_bundle",
                        "version": 1,
                        "source_path": str((tmp_path / "bundle.json").resolve()),
                        "selection": selection,
                        "task_id": 101,
                        "allow_missing_values": True,
                    },
                },
            }
        ],
    )
    manifest_path = tmp_path / "manifest.parquet"
    _ = data_cases.build_manifest(
        [root],
        manifest_path,
        filter_policy="accepted_only",
        missing_value_policy="allow_any",
    )

    datasets, task_records, benchmark_surface = benchmark_module.load_benchmark_manifest_datasets(
        benchmark_manifest_path=manifest_path,
    )

    assert list(datasets) == ["missing_case"]
    assert not np.isfinite(datasets["missing_case"][0]).all()
    assert datasets["missing_case"][2] == ["floating"] * int(x_train.shape[1])
    assert task_records[0]["dataset_name"] == "missing_case"
    assert benchmark_surface["allow_missing_values"] is True
    assert benchmark_surface["benchmark_bundle"]["allow_missing_values"] is True


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
            "primary_external_benchmark": compare_cli_module.EXTERNAL_BENCHMARK_TABICLV2,
            "artifacts": {"comparison_curve_png": "/tmp/comparison_curve.png"},
        }

    monkeypatch.setattr(compare_cli_module, "run_nanotabpfn_benchmark", _fake_run)

    exit_code = compare_cli_module.main(
        [
            "--tab-foundry-run-dir",
            str(tmp_path / "run"),
            "--out-root",
            str(tmp_path / "bench"),
            "--tabicl-root",
            str(tmp_path / "tabicl"),
            "--nanotabpfn-root",
            str(tmp_path / "nano"),
            "--nanotabpfn-prior-dump",
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
            "--benchmark-manifest-path",
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
    assert config.benchmark_manifest_path == tmp_path / "bundle.json"
    assert config.external_benchmarks == (compare_cli_module.EXTERNAL_BENCHMARK_TABICLV2,)
    assert config.tabicl_root == tmp_path / "tabicl"
    stdout = capsys.readouterr().out
    assert "benchmark comparison complete:" in stdout
    assert "primary_external_benchmark=" in stdout


def test_compare_main_parses_cli_invocation_with_explicit_tabiclv2(
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
            "tabiclv2": {"best_roc_auc": 0.74, "final_roc_auc": 0.73},
            "primary_external_benchmark": compare_cli_module.EXTERNAL_BENCHMARK_TABICLV2,
            "artifacts": {"comparison_curve_png": "/tmp/comparison_curve.png"},
        }

    monkeypatch.setattr(compare_cli_module, "run_nanotabpfn_benchmark", _fake_run)

    exit_code = compare_cli_module.main(
        [
            "--tab-foundry-run-dir",
            str(tmp_path / "run"),
            "--out-root",
            str(tmp_path / "bench"),
            "--external-benchmark",
            compare_cli_module.EXTERNAL_BENCHMARK_TABICLV2,
            "--tabicl-root",
            str(tmp_path / "tabicl"),
            "--tabicl-classifier-checkpoint-version",
            "classifier.ckpt",
            "--tabicl-regressor-checkpoint-version",
            "regressor.ckpt",
        ]
    )

    assert exit_code == 0
    config = captured["config"]
    assert config.external_benchmarks == (compare_cli_module.EXTERNAL_BENCHMARK_TABICLV2,)
    assert config.tabicl_root == tmp_path / "tabicl"
    assert config.tabicl_classifier_checkpoint_version == "classifier.ckpt"
    assert config.tabicl_regressor_checkpoint_version == "regressor.ckpt"
    stdout = capsys.readouterr().out
    assert "benchmark comparison complete:" in stdout
    assert "tabiclv2=" in stdout


def test_compare_main_parses_cli_invocation_with_explicit_nanotabpfn(
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
            "primary_external_benchmark": compare_module.EXTERNAL_BENCHMARK_NANOTABPFN,
            "artifacts": {"comparison_curve_png": "/tmp/comparison_curve.png"},
        }

    monkeypatch.setattr(compare_cli_module, "run_nanotabpfn_benchmark", _fake_run)

    exit_code = compare_cli_module.main(
        [
            "--tab-foundry-run-dir",
            str(tmp_path / "run"),
            "--out-root",
            str(tmp_path / "bench"),
            "--external-benchmark",
            "nanotabpfn",
            "--nanotabpfn-root",
            str(tmp_path / "nano"),
        ]
    )

    assert exit_code == 0
    config = captured["config"]
    assert config.external_benchmarks == (compare_module.EXTERNAL_BENCHMARK_NANOTABPFN,)
    stdout = capsys.readouterr().out
    assert "primary_external_benchmark=nanotabpfn" in stdout


def test_compare_main_requires_tabicl_root_for_default_tabiclv2(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        compare_cli_module.main(
            [
                "--tab-foundry-run-dir",
                str(tmp_path / "run"),
            ]
        )

    assert exc_info.value.code == 2
    assert "--tabicl-root is required" in capsys.readouterr().err


def test_compare_main_requires_nanotabpfn_root_when_nanotabpfn_is_selected(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc_info:
        compare_cli_module.main(
            [
                "--tab-foundry-run-dir",
                str(tmp_path / "run"),
                "--external-benchmark",
                "nanotabpfn",
            ]
        )

    assert exc_info.value.code == 2
    assert "--nanotabpfn-root is required" in capsys.readouterr().err


def test_load_benchmark_manifest_datasets_fails_on_bundle_drift(
    tmp_path: Path,
) -> None:
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
    with pytest.raises(RuntimeError, match="materialized manifest parquet"):
        benchmark_module.load_benchmark_manifest_datasets(
            benchmark_manifest_path=bundle_path,
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


def test_load_benchmark_manifest_datasets_requires_bundle_new_instances_match(tmp_path: Path) -> None:
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
    with pytest.raises(RuntimeError, match="materialized manifest parquet"):
        benchmark_module.load_benchmark_manifest_datasets(
            benchmark_manifest_path=bundle_path,
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
def test_load_benchmark_manifest_datasets_fails_on_selection_drift(
    tmp_path: Path,
    qualities: dict[str, float],
    message: str,
) -> None:
    del qualities, message
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
    with pytest.raises(RuntimeError, match="materialized manifest parquet"):
        benchmark_module.load_benchmark_manifest_datasets(
            benchmark_manifest_path=bundle_path,
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
    benchmark_manifest_path = tmp_path / "benchmark_manifest.parquet"
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
    policy_calls: dict[str, list[Any]] = {"datasets": [], "evaluate": [], "checkpoint_selection": []}
    captured_posthoc: dict[str, Any] = {}

    monkeypatch.setattr(
        compare_module,
        "load_benchmark_manifest_datasets",
        lambda *, new_instances=200, benchmark_manifest_path=None, allow_missing_values=False: (
            policy_calls["datasets"].append(bool(allow_missing_values))
            or _runtime_benchmark_surface(
                benchmark_manifest_path=(
                    benchmark_manifest_path
                    if benchmark_manifest_path is None
                    else Path(benchmark_manifest_path)
                ),
                source_bundle_path=source_bundle_path,
                benchmark_bundle=benchmark_bundle,
                datasets={
                    "toy": (
                        np.zeros((6, 2), dtype=np.float32),
                        np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
                    )
                },
                task_records=[
                    {
                        "task_id": 1,
                        "dataset_name": "toy",
                        "n_rows": 6,
                        "n_features": 2,
                        "n_classes": 2,
                    }
                ],
                allow_missing_values=False,
            )
        ),
    )
    monkeypatch.setattr(compare_module, "default_benchmark_manifest_path", lambda: benchmark_manifest_path)
    monkeypatch.setattr(
        compare_module,
        "evaluate_tab_foundry_run",
        lambda *_args, **_kwargs: (
            policy_calls["evaluate"].append(bool(_kwargs["allow_missing_values"])),
            policy_calls["checkpoint_selection"].append(str(_kwargs["checkpoint_selection"])),
            [
                {
                    "checkpoint_path": "/tmp/step_000025.pt",
                    "step": 25,
                    "training_time": 1.2,
                    "roc_auc": 0.81,
                    "log_loss": 0.42,
                    "dataset_roc_auc": {"toy": 0.81},
                    "dataset_log_loss": {"toy": 0.42},
                },
                {
                    "checkpoint_path": "/tmp/step_000050.pt",
                    "step": 50,
                    "training_time": 2.4,
                    "evaluation_error": "benchmark evaluation failed for dataset 'toy': Input contains NaN.",
                    "evaluation_error_type": "ValueError",
                    "failed_dataset": "toy",
                },
            ],
        )[-1]
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
                    "final_log_loss": 0.42,
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
    monkeypatch.setattr(compare_module, "resolve_device", lambda device: "cuda")
    monkeypatch.setattr(compare_module, "benchmark_host_fingerprint", lambda: "host-a")

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
                    "log_loss": 0.48,
                    "dataset_roc_auc": {"toy": 0.78},
                    "dataset_log_loss": {"toy": 0.48},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(
        compare_module,
        "subprocess",
        SimpleNamespace(run=_fake_run, CalledProcessError=subprocess.CalledProcessError),
    )
    monkeypatch.setattr(
        compare_module,
        "posthoc_update_wandb_summary",
        lambda *, telemetry_path, payload: captured_posthoc.update(
            {"telemetry_path": telemetry_path, "payload": payload}
        )
        or True,
    )

    summary = compare_module.run_nanotabpfn_benchmark(
        compare_module.BenchmarkComparisonConfig(
            tab_foundry_run_dir=smoke_run_dir,
            out_root=out_root,
            nanotabpfn_root=nanotab_root,
            nanotab_prior_dump=prior_dump,
        )
    )

    assert captured["cwd"] == nanotab_root.resolve()
    assert captured["check"] is True
    assert Path(captured["cmd"][0]) == nanotab_python.resolve()
    assert Path(captured["cmd"][1]) == REPO_ROOT / "scripts" / "bench" / "openml_benchmark_helper.py"
    assert captured["cmd"][captured["cmd"].index("--tab-foundry-src") + 1] == str(REPO_ROOT / "src")
    assert "--tab-realdata-hub-root" not in captured["cmd"]
    assert captured["cmd"][captured["cmd"].index("--eval-every") + 1] == str(
        compare_module.DEFAULT_NANOTABPFN_EVAL_EVERY
    )
    assert policy_calls == {
        "datasets": [False],
        "evaluate": [False],
        "checkpoint_selection": ["all"],
    }
    assert summary["dataset_count"] == 1
    assert summary["tab_foundry"]["best_step"] == pytest.approx(25.0)
    assert summary["tab_foundry"]["best_roc_auc"] == pytest.approx(0.81)
    assert summary["tab_foundry"]["final_log_loss"] == pytest.approx(0.42)
    assert summary["tab_foundry"]["final_dataset_roc_auc"] == {"toy": pytest.approx(0.81)}
    assert summary["tab_foundry"]["final_dataset_log_loss"] == {"toy": pytest.approx(0.42)}
    assert summary["nanotabpfn"]["best_step"] == pytest.approx(25.0)
    assert summary["nanotabpfn"]["final_roc_auc"] == pytest.approx(0.78)
    assert summary["nanotabpfn"]["final_log_loss"] == pytest.approx(0.48)
    assert summary["nanotabpfn"]["final_dataset_roc_auc"] == {"toy": pytest.approx(0.78)}
    assert summary["nanotabpfn"]["final_dataset_log_loss"] == {"toy": pytest.approx(0.48)}
    assert summary["nanotabpfn"]["device"] == "auto"
    assert summary["nanotabpfn"]["resolved_device"] == "cuda"
    assert summary["nanotabpfn"]["benchmark_host_fingerprint"] == "host-a"
    assert summary["nanotabpfn"]["prior_dump_path"] == str(prior_dump.resolve())
    assert summary["nanotabpfn"]["steps"] == compare_module.DEFAULT_NANOTABPFN_STEPS
    assert summary["nanotabpfn"]["eval_every"] == compare_module.DEFAULT_NANOTABPFN_EVAL_EVERY
    assert summary["nanotabpfn"]["batch_size"] == compare_module.DEFAULT_NANOTABPFN_BATCH_SIZE
    assert summary["nanotabpfn"]["lr"] == pytest.approx(compare_module.DEFAULT_NANOTABPFN_LR)
    assert summary["nanotabpfn"]["curve_source_mode"] == "fresh"
    assert summary["nanotabpfn"]["reused_curve_path"] is None
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
    assert captured_posthoc["telemetry_path"] == smoke_run_dir / "telemetry.json"
    assert captured_posthoc["payload"]["benchmark"]["tab_foundry"]["final_log_loss"] == pytest.approx(0.42)
    assert captured_posthoc["payload"]["benchmark"]["tab_foundry"]["training_diagnostics"]["mean_grad_norm"] == pytest.approx(0.4)
    assert captured_posthoc["payload"]["benchmark"]["model_size"]["total_params"] == 1234
    assert captured_posthoc["payload"]["benchmark"]["nanotabpfn"]["final_log_loss"] == pytest.approx(0.48)
    written_bundle = json.loads((out_root / "benchmark_tasks.json").read_text(encoding="utf-8"))
    assert written_bundle["tasks"] == benchmark_bundle["tasks"]
    assert written_bundle["manifest"]["manifest_path"] == str(benchmark_manifest_path.resolve())
    assert written_bundle["manifest"]["benchmark_bundle"]["name"] == benchmark_bundle["name"]
    assert written_bundle["manifest"]["benchmark_bundle"]["source_path"] == str(
        source_bundle_path.resolve()
    )


def test_run_nanotabpfn_benchmark_optionally_runs_tabiclv2(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    smoke_run_dir = tmp_path / "smoke_run"
    smoke_run_dir.mkdir()
    (smoke_run_dir / "gradient_history.jsonl").write_text("{}\n", encoding="utf-8")
    (smoke_run_dir / "telemetry.json").write_text("{}\n", encoding="utf-8")
    nanotab_root = tmp_path / "nano"
    (nanotab_root / ".venv" / "bin").mkdir(parents=True)
    (nanotab_root / ".venv" / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    prior_dump = nanotab_root / "300k_150x5_2.h5"
    prior_dump.write_bytes(b"prior")
    tabicl_root = tmp_path / "tabicl"
    (tabicl_root / ".venv" / "bin").mkdir(parents=True)
    tabicl_python = tabicl_root / ".venv" / "bin" / "python"
    tabicl_python.write_text("#!/bin/sh\n", encoding="utf-8")
    hub_root = tmp_path / "tab-realdata-hub"
    hub_root.mkdir()
    out_root = tmp_path / "benchmark_out"
    benchmark_manifest_path = tmp_path / "benchmark_manifest.parquet"
    source_bundle_path = _write_benchmark_bundle(
        tmp_path / "source_bundle.json",
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
    captured_posthoc: dict[str, Any] = {}
    helper_calls: list[tuple[str, Path]] = []
    helper_commands: dict[str, list[str]] = {}

    monkeypatch.setattr(compare_module, "default_benchmark_manifest_path", lambda: benchmark_manifest_path)
    monkeypatch.setattr(
        compare_module,
        "load_benchmark_manifest_datasets",
        lambda *, new_instances=200, benchmark_manifest_path=None, allow_missing_values=False: (
            _runtime_benchmark_surface(
                benchmark_manifest_path=(
                    benchmark_manifest_path
                    if benchmark_manifest_path is None
                    else Path(benchmark_manifest_path)
                ),
                source_bundle_path=source_bundle_path,
                benchmark_bundle=benchmark_bundle,
                datasets={
                    "toy": (
                        np.zeros((6, 2), dtype=np.float32),
                        np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
                    )
                },
                task_records=[
                    {
                        "task_id": 1,
                        "dataset_name": "toy",
                        "n_rows": 6,
                        "n_features": 2,
                        "n_classes": 2,
                    }
                ],
                allow_missing_values=False,
            )
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
                "log_loss": 0.42,
                "brier_score": 0.12,
                "dataset_roc_auc": {"toy": 0.81},
                "dataset_log_loss": {"toy": 0.42},
                "dataset_brier_score": {"toy": 0.12},
            }
        ],
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
                "final_log_loss": 0.42,
                "final_brier_score": 0.12,
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
    monkeypatch.setattr(compare_module, "resolve_device", lambda device: "cuda")
    monkeypatch.setattr(compare_module, "benchmark_host_fingerprint", lambda: "host-a")
    monkeypatch.setattr(
        compare_module,
        "posthoc_update_wandb_summary",
        lambda *, telemetry_path, payload: captured_posthoc.update(
            {"telemetry_path": telemetry_path, "payload": payload}
        )
        or True,
    )

    def _fake_run(cmd: list[str], *, cwd: Path, check: bool) -> subprocess.CompletedProcess[str]:
        script_name = Path(cmd[1]).name
        helper_calls.append((script_name, cwd))
        helper_commands[script_name] = list(cmd)
        out_path = Path(cmd[cmd.index("--out-path") + 1])
        if script_name == "openml_benchmark_helper.py":
            payload = {
                "seed": 0,
                "step": 25,
                "training_time": 2.0,
                "roc_auc": 0.78,
                "log_loss": 0.48,
                "brier_score": 0.16,
                "dataset_roc_auc": {"toy": 0.78},
                "dataset_log_loss": {"toy": 0.48},
                "dataset_brier_score": {"toy": 0.16},
            }
        elif script_name == "tabiclv2_helper.py":
            payload = {
                "seed": 0,
                "step": 0,
                "training_time": 3.5,
                "roc_auc": 0.84,
                "log_loss": 0.39,
                "brier_score": 0.11,
                "dataset_roc_auc": {"toy": 0.84},
                "dataset_log_loss": {"toy": 0.39},
                "dataset_brier_score": {"toy": 0.11},
            }
        else:
            raise AssertionError(f"unexpected helper script {script_name!r}")
        out_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(
        compare_module,
        "subprocess",
        SimpleNamespace(run=_fake_run, CalledProcessError=subprocess.CalledProcessError),
    )

    summary = compare_module.run_nanotabpfn_benchmark(
        compare_module.BenchmarkComparisonConfig(
            tab_foundry_run_dir=smoke_run_dir,
            out_root=out_root,
            nanotabpfn_root=nanotab_root,
            nanotab_prior_dump=prior_dump,
            external_benchmarks=(
                compare_module.EXTERNAL_BENCHMARK_NANOTABPFN,
                compare_module.EXTERNAL_BENCHMARK_TABICLV2,
            ),
            tabicl_root=tabicl_root,
            tab_realdata_hub_root=hub_root,
            tabicl_classifier_checkpoint_version="classifier.ckpt",
        )
    )

    assert helper_calls == [
        ("openml_benchmark_helper.py", nanotab_root.resolve()),
        ("tabiclv2_helper.py", tabicl_root.resolve()),
    ]
    assert helper_commands["openml_benchmark_helper.py"][
        helper_commands["openml_benchmark_helper.py"].index("--tab-realdata-hub-root") + 1
    ] == str(hub_root.resolve())
    assert helper_commands["tabiclv2_helper.py"][
        helper_commands["tabiclv2_helper.py"].index("--tab-realdata-hub-root") + 1
    ] == str(hub_root.resolve())
    assert summary["tabiclv2"]["final_roc_auc"] == pytest.approx(0.84)
    assert summary["tabiclv2"]["final_log_loss"] == pytest.approx(0.39)
    assert summary["tabiclv2"]["final_brier_score"] == pytest.approx(0.11)
    assert summary["tabiclv2"]["checkpoint_version"] == "classifier.ckpt"
    assert summary["tabiclv2"]["root"] == str(tabicl_root.resolve())
    assert summary["tabiclv2"]["python"] == str(tabicl_python.resolve())
    assert summary["tabiclv2"]["device"] == "auto"
    assert summary["tabiclv2"]["resolved_device"] == "cuda"
    assert summary["tabiclv2"]["benchmark_host_fingerprint"] == "host-a"
    assert summary["tabiclv2"]["tab_realdata_hub_root"] == str(hub_root.resolve())
    assert summary["nanotabpfn"]["tab_realdata_hub_root"] == str(hub_root.resolve())
    assert summary["external_benchmarks"] == [
        compare_module.EXTERNAL_BENCHMARK_NANOTABPFN,
        compare_module.EXTERNAL_BENCHMARK_TABICLV2,
    ]
    assert summary["primary_external_benchmark"] == compare_module.EXTERNAL_BENCHMARK_NANOTABPFN
    assert summary["artifacts"]["primary_external_curve_jsonl"] == str(
        (out_root.resolve() / "nanotabpfn_curve.jsonl")
    )
    assert summary["artifacts"]["tabiclv2_curve_jsonl"] == str(
        (out_root.resolve() / "tabiclv2_curve.jsonl")
    )
    written_summary = json.loads((out_root / "comparison_summary.json").read_text(encoding="utf-8"))
    assert written_summary["tabiclv2"]["final_dataset_log_loss"] == {"toy": pytest.approx(0.39)}
    assert written_summary["artifacts"]["tabiclv2_curve_jsonl"] == str(
        (out_root.resolve() / "tabiclv2_curve.jsonl")
    )
    assert captured_posthoc["telemetry_path"] == smoke_run_dir / "telemetry.json"
    assert captured_posthoc["payload"]["benchmark"]["tabiclv2"]["final_log_loss"] == pytest.approx(0.39)
    assert (out_root / "tabiclv2_curve.jsonl").exists()


def test_run_nanotabpfn_benchmark_with_tabiclv2_selected_fails_clear_when_env_missing(
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
    benchmark_manifest_path = tmp_path / "benchmark_manifest.parquet"
    source_bundle_path = _write_benchmark_bundle(
        tmp_path / "bundle.json",
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

    monkeypatch.setattr(compare_module, "default_benchmark_manifest_path", lambda: benchmark_manifest_path)
    monkeypatch.setattr(
        compare_module,
        "load_benchmark_manifest_datasets",
        lambda *, benchmark_manifest_path=None, allow_missing_values=None: (
            _runtime_benchmark_surface(
                benchmark_manifest_path=(
                    benchmark_manifest_path
                    if benchmark_manifest_path is None
                    else Path(benchmark_manifest_path)
                ),
                source_bundle_path=source_bundle_path,
                benchmark_bundle=benchmark_bundle,
                datasets={
                    "toy": (
                        np.zeros((6, 2), dtype=np.float32),
                        np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
                    )
                },
                task_records=[
                    {
                        "task_id": 1,
                        "dataset_name": "toy",
                        "n_rows": 6,
                        "n_features": 2,
                        "n_classes": 2,
                    }
                ],
                allow_missing_values=False,
            )
        ),
    )

    with pytest.raises(RuntimeError, match="TabICLv2 root does not exist"):
        compare_module.run_nanotabpfn_benchmark(
            compare_module.BenchmarkComparisonConfig(
                tab_foundry_run_dir=smoke_run_dir,
                out_root=tmp_path / "benchmark_out",
                nanotabpfn_root=nanotab_root,
                nanotab_prior_dump=prior_dump,
                external_benchmarks=(compare_module.EXTERNAL_BENCHMARK_TABICLV2,),
                tabicl_root=tmp_path / "missing_tabicl",
            )
        )


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
    benchmark_manifest_path = tmp_path / "large_bundle_manifest.parquet"
    source_bundle_path = tmp_path / "large_bundle.json"
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

    policy_calls: dict[str, list[bool]] = {"datasets": [], "evaluate": []}
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
        "load_benchmark_manifest_datasets",
        lambda *, new_instances=200, benchmark_manifest_path=None, allow_missing_values=False: (
            policy_calls["datasets"].append(bool(allow_missing_values))
            or _runtime_benchmark_surface(
                benchmark_manifest_path=(
                    benchmark_manifest_path
                    if benchmark_manifest_path is None
                    else Path(benchmark_manifest_path)
                ),
                source_bundle_path=source_bundle_path,
                benchmark_bundle=benchmark_bundle,
                datasets={
                    "toy": (
                        np.zeros((6, 2), dtype=np.float32),
                        np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
                    )
                },
                task_records=[
                    {
                        "task_id": 1,
                        "dataset_name": "toy",
                        "n_rows": 6,
                        "n_features": 2,
                        "n_classes": 2,
                    }
                ],
                allow_missing_values=True,
            )
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
        lambda **_kwargs: {
            "manifest_path": str(benchmark_manifest_path),
            "seed_set": [1],
            "model": {"arch": "tabfoundry_staged", "stage": "nano_exact"},
            "benchmark_bundle": {
                "name": "large_bundle",
                "version": 1,
                "source_path": str(source_bundle_path.resolve()),
                "task_count": 1,
                "task_ids": [1],
            },
            "artifacts": {},
            "tab_foundry_metrics": {"best_step": 25.0, "final_step": 25.0},
            "training_diagnostics": {},
            "model_size": {"total_params": 1, "trainable_params": 1},
            "generated_at_utc": "2026-03-13T00:00:00Z",
        },
    )
    monkeypatch.setattr(compare_module, "resolve_device", lambda device: "cuda")
    monkeypatch.setattr(compare_module, "benchmark_host_fingerprint", lambda: "host-a")

    summary = compare_module.run_nanotabpfn_benchmark(
        compare_module.BenchmarkComparisonConfig(
            tab_foundry_run_dir=smoke_run_dir,
            out_root=out_root,
            nanotabpfn_root=nanotab_root,
            nanotab_prior_dump=prior_dump,
            benchmark_manifest_path=benchmark_manifest_path,
            reuse_nanotabpfn_curve_path=reuse_curve_path,
        )
    )

    assert policy_calls == {"datasets": [False], "evaluate": [True]}
    assert summary["benchmark_bundle"]["allow_missing_values"] is True
    assert summary["nanotabpfn"]["curve_source_mode"] == "reused"
    assert summary["nanotabpfn"]["reused_curve_path"] == str(reuse_curve_path.resolve())
    assert summary["nanotabpfn"]["resolved_device"] == "cuda"
    assert summary["nanotabpfn"]["benchmark_host_fingerprint"] == "host-a"
    assert summary["nanotabpfn"]["prior_dump_path"] == str(prior_dump.resolve())


def test_run_nanotabpfn_benchmark_forwards_missing_bundle_policy_to_helper(
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
    bundle_path = tmp_path / "large_bundle.json"
    bundle_path.write_text("{}", encoding="utf-8")

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
        "load_benchmark_manifest_datasets",
        lambda *, new_instances=200, benchmark_manifest_path=None, allow_missing_values=False: _runtime_benchmark_surface(
            benchmark_manifest_path=(
                bundle_path
                if benchmark_manifest_path is None
                else Path(benchmark_manifest_path)
            ),
            source_bundle_path=bundle_path,
            benchmark_bundle=benchmark_bundle,
            datasets={
                "toy": (
                    np.zeros((6, 2), dtype=np.float32),
                    np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
                )
            },
            task_records=[
                {"task_id": 1, "dataset_name": "toy", "n_rows": 6, "n_features": 2, "n_classes": 2}
            ],
            allow_missing_values=True,
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
    monkeypatch.setattr(compare_module, "summarize_checkpoint_curve", lambda records, **_kwargs: {"records": records})
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
        lambda **_kwargs: {
            "manifest_path": str(bundle_path.resolve()),
            "seed_set": [0],
            "training_diagnostics": {},
            "model_size": {},
            "artifacts": {"training_surface_record_path": None},
        },
    )
    monkeypatch.setattr(compare_module, "resolve_device", lambda device: "cuda")
    monkeypatch.setattr(compare_module, "benchmark_host_fingerprint", lambda: "host-a")

    captured: dict[str, Any] = {}

    def _fake_run(cmd: list[str], *, cwd: Path, check: bool) -> subprocess.CompletedProcess[str]:
        captured["cmd"] = cmd
        captured["cwd"] = cwd
        captured["check"] = check
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

    monkeypatch.setattr(
        compare_module,
        "subprocess",
        SimpleNamespace(run=_fake_run, CalledProcessError=subprocess.CalledProcessError),
    )

    _ = compare_module.run_nanotabpfn_benchmark(
        compare_module.BenchmarkComparisonConfig(
            tab_foundry_run_dir=smoke_run_dir,
            out_root=out_root,
            nanotabpfn_root=nanotab_root,
            nanotab_prior_dump=prior_dump,
            benchmark_manifest_path=bundle_path,
        )
    )

    assert captured["cwd"] == nanotab_root.resolve()
    assert captured["check"] is True
    assert "--allow-missing-values" in captured["cmd"]


def test_run_nanotabpfn_benchmark_tolerates_missing_bundle_helper_failure(
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
    bundle_path = tmp_path / "large_bundle.json"
    bundle_path.write_text("{}", encoding="utf-8")

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
        "load_benchmark_manifest_datasets",
        lambda *, new_instances=200, benchmark_manifest_path=None, allow_missing_values=False: _runtime_benchmark_surface(
            benchmark_manifest_path=(
                bundle_path
                if benchmark_manifest_path is None
                else Path(benchmark_manifest_path)
            ),
            source_bundle_path=bundle_path,
            benchmark_bundle=benchmark_bundle,
            datasets={
                "toy": (
                    np.zeros((6, 2), dtype=np.float32),
                    np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
                )
            },
            task_records=[
                {"task_id": 1, "dataset_name": "toy", "n_rows": 6, "n_features": 2, "n_classes": 2}
            ],
            allow_missing_values=True,
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
                "log_loss": 0.42,
                "brier_score": 0.12,
                "dataset_roc_auc": {"toy": 0.81},
                "dataset_log_loss": {"toy": 0.42},
                "dataset_brier_score": {"toy": 0.12},
            }
        ],
    )
    monkeypatch.setattr(compare_module, "summarize_checkpoint_curve", lambda records, **_kwargs: {"records": records})
    monkeypatch.setattr(compare_module, "plot_comparison_curve", lambda **_kwargs: None)
    monkeypatch.setattr(
        compare_module,
        "derive_benchmark_run_record",
        lambda **_kwargs: {
            "manifest_path": str(bundle_path.resolve()),
            "seed_set": [0],
            "training_diagnostics": {},
            "model_size": {},
            "artifacts": {"training_surface_record_path": None},
        },
    )
    monkeypatch.setattr(compare_module, "resolve_device", lambda device: "cuda")
    monkeypatch.setattr(compare_module, "benchmark_host_fingerprint", lambda: "host-a")

    def _fake_run(cmd: list[str], *, cwd: Path, check: bool) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, cmd)

    monkeypatch.setattr(
        compare_module,
        "subprocess",
        SimpleNamespace(run=_fake_run, CalledProcessError=subprocess.CalledProcessError),
    )

    summary = compare_module.run_nanotabpfn_benchmark(
        compare_module.BenchmarkComparisonConfig(
            tab_foundry_run_dir=smoke_run_dir,
            out_root=out_root,
            nanotabpfn_root=nanotab_root,
            nanotab_prior_dump=prior_dump,
            benchmark_manifest_path=bundle_path,
        )
    )

    assert "nanotabpfn" not in summary
    assert summary["nanotabpfn_error"]["kind"] == "helper_failed_on_missing_bundle"
    assert summary["artifacts"]["nanotabpfn_curve_jsonl"] is None


def test_run_nanotabpfn_benchmark_falls_back_to_successful_primary_external_benchmark(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    smoke_run_dir = tmp_path / "smoke_run"
    smoke_run_dir.mkdir()
    out_root = tmp_path / "benchmark_out"
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text("{}", encoding="utf-8")
    nanotab_root = tmp_path / "nano"
    (nanotab_root / ".venv" / "bin").mkdir(parents=True)
    (nanotab_root / ".venv" / "bin" / "python").write_text("#!/bin/sh\n", encoding="utf-8")
    prior_dump = nanotab_root / "300k_150x5_2.h5"
    prior_dump.write_text("prior", encoding="utf-8")
    tabicl_root = tmp_path / "tabicl"
    tabicl_python = tabicl_root / ".venv" / "bin" / "python"
    tabicl_python.parent.mkdir(parents=True)
    tabicl_python.write_text("#!/bin/sh\n", encoding="utf-8")
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
        "load_benchmark_manifest_datasets",
        lambda *, new_instances=200, benchmark_manifest_path=None, allow_missing_values=False: _runtime_benchmark_surface(
            benchmark_manifest_path=(
                bundle_path
                if benchmark_manifest_path is None
                else Path(benchmark_manifest_path)
            ),
            source_bundle_path=bundle_path,
            benchmark_bundle=benchmark_bundle,
            datasets={
                "toy": (
                    np.zeros((6, 2), dtype=np.float32),
                    np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
                )
            },
            task_records=[
                {"task_id": 1, "dataset_name": "toy", "n_rows": 6, "n_features": 2, "n_classes": 2}
            ],
            allow_missing_values=True,
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
                "log_loss": 0.42,
                "brier_score": 0.12,
                "dataset_roc_auc": {"toy": 0.81},
                "dataset_log_loss": {"toy": 0.42},
                "dataset_brier_score": {"toy": 0.12},
            }
        ],
    )
    monkeypatch.setattr(compare_module, "summarize_checkpoint_curve", lambda records, **_kwargs: {"records": records})
    monkeypatch.setattr(compare_module, "plot_comparison_curve", lambda **_kwargs: None)
    monkeypatch.setattr(
        compare_module,
        "derive_benchmark_run_record",
        lambda **_kwargs: {
            "manifest_path": str(bundle_path.resolve()),
            "seed_set": [0],
            "training_diagnostics": {},
            "model_size": {},
            "artifacts": {"training_surface_record_path": None},
        },
    )
    monkeypatch.setattr(compare_module, "resolve_device", lambda device: "cuda")
    monkeypatch.setattr(compare_module, "benchmark_host_fingerprint", lambda: "host-a")

    def _fake_run(cmd: list[str], *, cwd: Path, check: bool) -> subprocess.CompletedProcess[str]:
        script_name = Path(cmd[1]).name
        if script_name == "openml_benchmark_helper.py":
            raise subprocess.CalledProcessError(1, cmd)
        if script_name != "tabiclv2_helper.py":
            raise AssertionError(f"unexpected helper script {script_name!r}")
        out_path = Path(cmd[cmd.index("--out-path") + 1])
        out_path.write_text(
            json.dumps(
                {
                    "seed": 0,
                    "step": 0,
                    "training_time": 3.5,
                    "roc_auc": 0.84,
                    "log_loss": 0.39,
                    "brier_score": 0.11,
                    "dataset_roc_auc": {"toy": 0.84},
                    "dataset_log_loss": {"toy": 0.39},
                    "dataset_brier_score": {"toy": 0.11},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(
        compare_module,
        "subprocess",
        SimpleNamespace(run=_fake_run, CalledProcessError=subprocess.CalledProcessError),
    )

    summary = compare_module.run_nanotabpfn_benchmark(
        compare_module.BenchmarkComparisonConfig(
            tab_foundry_run_dir=smoke_run_dir,
            out_root=out_root,
            nanotabpfn_root=nanotab_root,
            nanotab_prior_dump=prior_dump,
            benchmark_manifest_path=bundle_path,
            external_benchmarks=(
                compare_module.EXTERNAL_BENCHMARK_NANOTABPFN,
                compare_module.EXTERNAL_BENCHMARK_TABICLV2,
            ),
            tabicl_root=tabicl_root,
            tabicl_classifier_checkpoint_version="classifier.ckpt",
        )
    )

    assert "nanotabpfn" not in summary
    assert summary["nanotabpfn_error"]["kind"] == "helper_failed_on_missing_bundle"
    assert summary["tabiclv2"]["final_log_loss"] == pytest.approx(0.39)
    assert summary["primary_external_benchmark"] == compare_module.EXTERNAL_BENCHMARK_TABICLV2
    assert summary["artifacts"]["primary_external_curve_jsonl"] == str(
        out_root.resolve() / "tabiclv2_curve.jsonl"
    )
    written_summary = json.loads((out_root / "comparison_summary.json").read_text(encoding="utf-8"))
    assert written_summary["primary_external_benchmark"] == compare_module.EXTERNAL_BENCHMARK_TABICLV2


def test_run_nanotabpfn_benchmark_reuses_curve_without_local_nanotabpfn_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    smoke_run_dir = tmp_path / "smoke_run"
    smoke_run_dir.mkdir()
    out_root = tmp_path / "benchmark_out"
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text("{}", encoding="utf-8")
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
    source_nanotab_root = tmp_path / "source_nano"
    source_nanotab_python = source_nanotab_root / ".venv" / "bin" / "python"
    source_prior_dump = source_nanotab_root / "300k_150x5_2.h5"
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
        "load_benchmark_manifest_datasets",
        lambda *, new_instances=200, benchmark_manifest_path=None, allow_missing_values=False: _runtime_benchmark_surface(
            benchmark_manifest_path=(
                bundle_path
                if benchmark_manifest_path is None
                else Path(benchmark_manifest_path)
            ),
            source_bundle_path=bundle_path,
            benchmark_bundle=benchmark_bundle,
            datasets={
                "toy": (
                    np.zeros((6, 2), dtype=np.float32),
                    np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
                )
            },
            task_records=[
                {"task_id": 1, "dataset_name": "toy", "n_rows": 6, "n_features": 2, "n_classes": 2}
            ],
            allow_missing_values=True,
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
        lambda **_kwargs: {
            "manifest_path": str(bundle_path.resolve()),
            "seed_set": [0],
            "training_diagnostics": {},
            "model_size": {},
            "artifacts": {"training_surface_record_path": None},
        },
    )

    summary = compare_module.run_nanotabpfn_benchmark(
        compare_module.BenchmarkComparisonConfig(
            tab_foundry_run_dir=smoke_run_dir,
            out_root=out_root,
            nanotabpfn_root=tmp_path / "missing_nano",
            nanotab_prior_dump=tmp_path / "missing_prior.h5",
            benchmark_manifest_path=bundle_path,
            reuse_nanotabpfn_curve_path=reuse_curve_path,
            reuse_nanotabpfn_metadata={
                "root": str(source_nanotab_root.resolve()),
                "python": str(source_nanotab_python.resolve()),
                "device": "auto",
                "resolved_device": "cuda",
                "benchmark_host_fingerprint": "host-a",
                "prior_dump_path": str(source_prior_dump.resolve()),
                "num_seeds": compare_module.DEFAULT_NANOTABPFN_SEEDS,
                "steps": compare_module.DEFAULT_NANOTABPFN_STEPS,
                "eval_every": compare_module.DEFAULT_NANOTABPFN_EVAL_EVERY,
                "batch_size": compare_module.DEFAULT_NANOTABPFN_BATCH_SIZE,
                "lr": compare_module.DEFAULT_NANOTABPFN_LR,
            },
        )
    )

    assert summary["benchmark_bundle"]["allow_missing_values"] is True
    assert summary["nanotabpfn"]["curve_source_mode"] == "reused"
    assert summary["nanotabpfn"]["reused_curve_path"] == str(reuse_curve_path.resolve())
    assert summary["nanotabpfn"]["root"] == str(source_nanotab_root.resolve())
    assert summary["nanotabpfn"]["python"] == str(source_nanotab_python.resolve())
    assert summary["nanotabpfn"]["device"] == "auto"
    assert summary["nanotabpfn"]["resolved_device"] == "cuda"
    assert summary["nanotabpfn"]["benchmark_host_fingerprint"] == "host-a"
    assert summary["nanotabpfn"]["prior_dump_path"] == str(source_prior_dump.resolve())


def test_run_nanotabpfn_benchmark_reuses_error_without_local_nanotabpfn_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    smoke_run_dir = tmp_path / "smoke_run"
    smoke_run_dir.mkdir()
    out_root = tmp_path / "benchmark_out"
    bundle_path = tmp_path / "bundle.json"
    bundle_path.write_text("{}", encoding="utf-8")
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
        "load_benchmark_manifest_datasets",
        lambda *, new_instances=200, benchmark_manifest_path=None, allow_missing_values=False: _runtime_benchmark_surface(
            benchmark_manifest_path=(
                bundle_path
                if benchmark_manifest_path is None
                else Path(benchmark_manifest_path)
            ),
            source_bundle_path=bundle_path,
            benchmark_bundle=benchmark_bundle,
            datasets={
                "toy": (
                    np.zeros((6, 2), dtype=np.float32),
                    np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
                )
            },
            task_records=[
                {"task_id": 1, "dataset_name": "toy", "n_rows": 6, "n_features": 2, "n_classes": 2}
            ],
            allow_missing_values=True,
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
                "log_loss": 0.42,
                "brier_score": 0.12,
                "dataset_roc_auc": {"toy": 0.81},
                "dataset_log_loss": {"toy": 0.42},
                "dataset_brier_score": {"toy": 0.12},
            }
        ],
    )
    monkeypatch.setattr(
        compare_module,
        "summarize_checkpoint_curve",
        lambda records, **_kwargs: {"records": records},
    )
    monkeypatch.setattr(compare_module, "plot_comparison_curve", lambda **_kwargs: None)
    monkeypatch.setattr(
        compare_module,
        "derive_benchmark_run_record",
        lambda **_kwargs: {
            "manifest_path": str(bundle_path.resolve()),
            "seed_set": [0],
            "training_diagnostics": {},
            "model_size": {},
            "artifacts": {"training_surface_record_path": None},
        },
    )

    monkeypatch.setattr(
        compare_module,
        "subprocess",
        SimpleNamespace(
            run=lambda *args, **kwargs: (_ for _ in ()).throw(
                AssertionError("unexpected nanoTabPFN helper invocation")
            ),
            CalledProcessError=subprocess.CalledProcessError,
        ),
    )

    reuse_error = {
        "kind": "helper_failed_on_missing_bundle",
        "message": "helper returned non-zero exit status 1",
        "returncode": 1,
    }
    summary = compare_module.run_nanotabpfn_benchmark(
        compare_module.BenchmarkComparisonConfig(
            tab_foundry_run_dir=smoke_run_dir,
            out_root=out_root,
            nanotabpfn_root=tmp_path / "missing_nano",
            nanotab_prior_dump=tmp_path / "missing_prior.h5",
            benchmark_manifest_path=bundle_path,
            reuse_nanotabpfn_error=reuse_error,
        )
    )

    assert "nanotabpfn" not in summary
    assert summary["nanotabpfn_error"] == reuse_error
    assert summary["artifacts"]["nanotabpfn_curve_jsonl"] is None


def test_run_nanotabpfn_benchmark_honors_nondefault_manifest_path(
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
    benchmark_manifest_path = tmp_path / "custom_bundle_manifest.parquet"
    default_manifest_path = tmp_path / "default_bundle_manifest.parquet"
    source_bundle_path = tmp_path / "custom_bundle.json"
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

    captured: dict[str, Any] = {}

    def _fake_load_datasets(
        *,
        new_instances: int = 200,
        benchmark_manifest_path: Path | None = None,
        allow_missing_values: bool = False,
    ) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], list[dict[str, Any]], dict[str, Any]]:
        captured["dataset_manifest_path"] = None if benchmark_manifest_path is None else str(Path(benchmark_manifest_path).resolve())
        captured["dataset_allow_missing_values"] = bool(allow_missing_values)
        return _runtime_benchmark_surface(
            benchmark_manifest_path=(
                default_manifest_path
                if benchmark_manifest_path is None
                else Path(benchmark_manifest_path)
            ),
            source_bundle_path=source_bundle_path,
            benchmark_bundle=benchmark_bundle,
            datasets={
                "toy_multi": (
                    np.zeros((6, 2), dtype=np.float32),
                    np.asarray([0, 1, 2, 0, 1, 2], dtype=np.int64),
                )
            },
            task_records=[
                {
                    "task_id": 7,
                    "dataset_name": "toy_multi",
                    "n_rows": 6,
                    "n_features": 2,
                    "n_classes": 3,
                }
            ],
            allow_missing_values=False,
        )

    monkeypatch.setattr(compare_module, "default_benchmark_manifest_path", lambda: default_manifest_path)
    monkeypatch.setattr(compare_module, "load_benchmark_manifest_datasets", _fake_load_datasets)
    monkeypatch.setattr(
        compare_module,
        "evaluate_tab_foundry_run",
        lambda *_args, **_kwargs: [
            {
                "checkpoint_path": "/tmp/step_000025.pt",
                "step": 25,
                "training_time": 1.2,
                "log_loss": 0.42,
                "brier_score": 0.12,
                "roc_auc": 0.81,
                "dataset_log_loss": {"toy_multi": 0.42},
                "dataset_brier_score": {"toy_multi": 0.12},
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
                    "log_loss": 0.48,
                    "brier_score": 0.16,
                    "roc_auc": 0.78,
                    "dataset_log_loss": {"toy_multi": 0.48},
                    "dataset_brier_score": {"toy_multi": 0.16},
                    "dataset_roc_auc": {"toy_multi": 0.78},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(compare_module, "subprocess", SimpleNamespace(run=_fake_run))

    summary = compare_module.run_nanotabpfn_benchmark(
        compare_module.BenchmarkComparisonConfig(
            tab_foundry_run_dir=smoke_run_dir,
            out_root=out_root,
            nanotabpfn_root=nanotab_root,
            nanotab_prior_dump=prior_dump,
            benchmark_manifest_path=benchmark_manifest_path,
        )
    )

    assert captured["dataset_manifest_path"] == str(benchmark_manifest_path.resolve())
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
    assert written_bundle["tasks"] == benchmark_bundle["tasks"]
    assert written_bundle["manifest"]["manifest_path"] == str(benchmark_manifest_path.resolve())
    assert written_bundle["manifest"]["benchmark_bundle"]["name"] == benchmark_bundle["name"]
    assert written_bundle["manifest"]["benchmark_bundle"]["source_path"] == str(
        source_bundle_path.resolve()
    )


def test_run_nanotabpfn_benchmark_propagates_record_derivation_failure(
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
    benchmark_manifest_path = tmp_path / "legacy_bundle_manifest.parquet"
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

    monkeypatch.setattr(compare_module, "default_benchmark_manifest_path", lambda: benchmark_manifest_path)
    monkeypatch.setattr(
        compare_module,
        "load_benchmark_manifest_datasets",
        lambda *, new_instances=200, benchmark_manifest_path=None, allow_missing_values=False: (
            _runtime_benchmark_surface(
                benchmark_manifest_path=(
                    benchmark_manifest_path
                    if benchmark_manifest_path is None
                    else Path(benchmark_manifest_path)
                ),
                source_bundle_path=source_bundle_path,
                benchmark_bundle=benchmark_bundle,
                datasets={
                    "toy": (
                        np.zeros((6, 2), dtype=np.float32),
                        np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
                    )
                },
                task_records=[
                    {
                        "task_id": 1,
                        "dataset_name": "toy",
                        "n_rows": 6,
                        "n_features": 2,
                        "n_classes": 2,
                    }
                ],
                allow_missing_values=False,
            )
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
                "log_loss": 0.42,
                "brier_score": 0.12,
                "roc_auc": 0.81,
                "dataset_log_loss": {"toy": 0.42},
                "dataset_brier_score": {"toy": 0.12},
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
                    "log_loss": 0.48,
                    "brier_score": 0.16,
                    "roc_auc": 0.78,
                    "dataset_log_loss": {"toy": 0.48},
                    "dataset_brier_score": {"toy": 0.16},
                    "dataset_roc_auc": {"toy": 0.78},
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(compare_module, "subprocess", SimpleNamespace(run=_fake_run))

    with pytest.raises(RuntimeError, match="legacy checkpoints without persisted model.arch"):
        compare_module.run_nanotabpfn_benchmark(
            compare_module.BenchmarkComparisonConfig(
                tab_foundry_run_dir=smoke_run_dir,
                out_root=out_root,
                nanotabpfn_root=nanotab_root,
                nanotab_prior_dump=prior_dump,
                benchmark_manifest_path=benchmark_manifest_path,
            )
        )


def test_explicit_benchmark_manifest_paths_accept_checked_in_legacy_and_medium_binary_bundles() -> None:
    legacy_bundle_path = (
        REPO_ROOT / "src" / "tab_foundry" / "bench" / "openml_benchmark_v1.json"
    )
    medium_bundle_path = (
        REPO_ROOT / "src" / "tab_foundry" / "bench" / "openml_binary_medium_v1.json"
    )

    legacy_bundle = benchmark_module.load_benchmark_bundle(legacy_bundle_path)
    medium_bundle = benchmark_module.load_benchmark_bundle(medium_bundle_path)

    assert legacy_bundle["name"] == "openml_binary_small"
    assert legacy_bundle["task_ids"] == [363613, 363621, 363629]
    assert medium_bundle["name"] == "openml_binary_medium"
    assert len(medium_bundle["task_ids"]) == 10
    assert all(int(task["n_classes"]) == 2 for task in medium_bundle["tasks"])


def test_default_benchmark_manifest_path_resolves_to_medium_binary_bundle() -> None:
    bundle_path = compare_module.default_benchmark_manifest_path()

    assert bundle_path == (
        REPO_ROOT
        / "data"
        / "manifests"
        / "bench"
        / "openml_binary_medium_v1"
        / "manifest.parquet"
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


def test_collect_checkpoint_snapshots_falls_back_to_latest_checkpoint_when_no_step_snapshots(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "best.pt").write_bytes(b"best")
    (checkpoint_dir / "latest.pt").write_bytes(b"latest")
    history_path = run_dir / "train_history.jsonl"
    history_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "step": 1,
                        "stage": "stage1",
                        "train_loss": 0.8,
                        "lr": 1.0e-3,
                        "elapsed_seconds": 5.0,
                        "train_elapsed_seconds": 1.0,
                    }
                ),
                json.dumps(
                    {
                        "step": 3,
                        "stage": "stage1",
                        "train_loss": 0.4,
                        "lr": 1.0e-3,
                        "elapsed_seconds": 7.0,
                        "train_elapsed_seconds": 2.5,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    snapshots = benchmark_module.collect_checkpoint_snapshots(run_dir)

    assert snapshots == [
        {
            "step": 3,
            "path": str((checkpoint_dir / "latest.pt").resolve()),
            "elapsed_seconds": pytest.approx(2.5),
        }
    ]


def test_collect_checkpoint_snapshots_filters_missing_telemetry_steps_and_appends_latest(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "step_000025.pt").write_bytes(b"step25")
    (checkpoint_dir / "step_000600.pt").write_bytes(b"step600")
    torch.save({"model": {}, "config": {}, "global_step": 2500}, checkpoint_dir / "latest.pt")
    history_path = run_dir / "train_history.jsonl"
    history_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "step": 25,
                        "stage": "stage1",
                        "train_loss": 0.8,
                        "lr": 1.0e-3,
                        "elapsed_seconds": 5.0,
                        "train_elapsed_seconds": 1.0,
                    }
                ),
                json.dumps(
                    {
                        "step": 600,
                        "stage": "stage1",
                        "train_loss": 0.5,
                        "lr": 1.0e-3,
                        "elapsed_seconds": 12.0,
                        "train_elapsed_seconds": 8.0,
                    }
                ),
                json.dumps(
                    {
                        "step": 2500,
                        "stage": "stage1",
                        "train_loss": 0.4,
                        "lr": 1.0e-3,
                        "elapsed_seconds": 30.0,
                        "train_elapsed_seconds": 25.0,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    telemetry_path = run_dir / "telemetry.json"
    telemetry_path.write_text(
        json.dumps(
            {
                "checkpoint_snapshots": [
                    {
                        "step": 25,
                        "path": "/remote-retained-artifact/run/checkpoints/step_000025.pt",
                        "elapsed_seconds": 5.0,
                        "train_elapsed_seconds": 1.0,
                    },
                    {
                        "step": 600,
                        "path": "/remote-retained-artifact/run/checkpoints/step_000600.pt",
                        "elapsed_seconds": 12.0,
                        "train_elapsed_seconds": 8.0,
                    },
                    {
                        "step": 2500,
                        "path": "/remote-retained-artifact/run/checkpoints/step_002500.pt",
                        "elapsed_seconds": 30.0,
                        "train_elapsed_seconds": 25.0,
                    },
                ]
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    snapshots = benchmark_module.collect_checkpoint_snapshots(run_dir)

    assert [int(snapshot["step"]) for snapshot in snapshots] == [25, 600, 2500]
    assert [Path(str(snapshot["path"])).name for snapshot in snapshots] == [
        "step_000025.pt",
        "step_000600.pt",
        "latest.pt",
    ]
    assert snapshots[-1]["elapsed_seconds"] == pytest.approx(25.0)


def test_selected_checkpoint_snapshots_best_and_final_falls_back_to_latest_when_best_missing(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    torch.save({"model": {}, "config": {}, "global_step": 3}, checkpoint_dir / "latest.pt")
    history_path = run_dir / "train_history.jsonl"
    history_path.write_text(
        json.dumps(
            {
                "step": 3,
                "stage": "stage1",
                "train_loss": 0.4,
                "lr": 1.0e-3,
                "elapsed_seconds": 7.0,
                "train_elapsed_seconds": 2.5,
            }
        )
        + "\n",
        encoding="utf-8",
    )

    snapshots = benchmark_artifacts_module.selected_checkpoint_snapshots(
        run_dir,
        checkpoint_selection="best_and_final",
    )

    assert snapshots == [
        {
            "step": 3,
            "path": str((checkpoint_dir / "latest.pt").resolve()),
            "elapsed_seconds": pytest.approx(2.5),
        }
    ]


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
    benchmark_manifest_path = tmp_path / "benchmark_manifest.parquet"
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
        "load_benchmark_manifest_datasets",
        lambda *, new_instances=200, benchmark_manifest_path=None, allow_missing_values=False: (
            _runtime_benchmark_surface(
                benchmark_manifest_path=(
                    benchmark_manifest_path
                    if benchmark_manifest_path is None
                    else Path(benchmark_manifest_path)
                ),
                source_bundle_path=source_bundle_path,
                benchmark_bundle=benchmark_bundle,
                datasets={
                    "toy": (
                        np.zeros((6, 2), dtype=np.float32),
                        np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
                    )
                },
                task_records=[
                    {
                        "task_id": 1,
                        "dataset_name": "toy",
                        "n_rows": 6,
                        "n_features": 2,
                        "n_classes": 2,
                    }
                ],
                allow_missing_values=False,
            )
        ),
    )
    monkeypatch.setattr(compare_module, "default_benchmark_manifest_path", lambda: benchmark_manifest_path)
    monkeypatch.setattr(
        compare_module,
        "evaluate_tab_foundry_run",
        lambda *_args, **_kwargs: [
            {
                "checkpoint_path": "/tmp/step_000025.pt",
                "step": 25,
                "training_time": 1.2,
                "log_loss": 0.42,
                "brier_score": 0.12,
                "roc_auc": 0.81,
            }
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
            json.dumps(
                {
                    "seed": 0,
                    "step": 25,
                    "training_time": 2.0,
                    "log_loss": 0.48,
                    "brier_score": 0.16,
                    "roc_auc": 0.78,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        return subprocess.CompletedProcess(cmd, 0)

    monkeypatch.setattr(compare_module, "subprocess", SimpleNamespace(run=_fake_run))

    summary = compare_module.run_nanotabpfn_benchmark(
        compare_module.BenchmarkComparisonConfig(
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


def test_run_nanotabpfn_benchmark_rejects_unknown_control_baseline(
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

    benchmark_bundle = {
        "name": "bundle",
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
    monkeypatch.setattr(
        compare_module,
        "default_benchmark_manifest_path",
        lambda: tmp_path / "bundle_manifest.parquet",
    )
    monkeypatch.setattr(
        compare_module,
        "load_benchmark_manifest_datasets",
        lambda *, benchmark_manifest_path=None, allow_missing_values=None: (
            _runtime_benchmark_surface(
                benchmark_manifest_path=(
                    tmp_path / "bundle_manifest.parquet"
                    if benchmark_manifest_path is None
                    else Path(benchmark_manifest_path)
                ),
                source_bundle_path=tmp_path / "bundle_source.json",
                benchmark_bundle=benchmark_bundle,
                datasets={
                    "toy": (
                        np.zeros((6, 2), dtype=np.float32),
                        np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64),
                    )
                },
                task_records=[
                    {
                        "task_id": 1,
                        "dataset_name": "toy",
                        "n_rows": 6,
                        "n_features": 2,
                        "n_classes": 2,
                    }
                ],
                allow_missing_values=False,
            )
        ),
    )

    with pytest.raises(RuntimeError, match="unknown control baseline id"):
        compare_module.run_nanotabpfn_benchmark(
            compare_module.BenchmarkComparisonConfig(
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
                "log_loss": 0.42,
                "brier_score": 0.12,
                "model_arch": "tabfoundry_staged",
                "model_stage": "nano_exact",
                "benchmark_profile": "nano_exact",
            }
        ],
        nanotabpfn_records=[
            {
                "seed": 0,
                "step": 25,
                "training_time": 2.0,
                "roc_auc": 0.78,
                "log_loss": 0.48,
                "brier_score": 0.16,
            }
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
        benchmark_manifest_path=tmp_path / "bundle.json",
        tab_foundry_run_dir=tmp_path / "run",
        task_type="supervised_classification",
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


def test_build_comparison_summary_uses_log_loss_as_classification_best_step(tmp_path: Path) -> None:
    summary = benchmark_module.build_comparison_summary(
        tab_foundry_records=[
            {
                'checkpoint_path': '/tmp/step_000025.pt',
                'step': 25,
                'training_time': 1.2,
                'roc_auc': 0.83,
                'log_loss': 0.45,
                'brier_score': 0.13,
            },
            {
                'checkpoint_path': '/tmp/step_000050.pt',
                'step': 50,
                'training_time': 2.4,
                'roc_auc': 0.81,
                'log_loss': 0.40,
                'brier_score': 0.11,
            },
        ],
        nanotabpfn_records=[],
        benchmark_tasks=[
            {'task_id': 1, 'dataset_name': 'toy', 'n_rows': 6, 'n_features': 2, 'n_classes': 2}
        ],
        benchmark_bundle={
            'name': 'toy_bundle',
            'version': 1,
            'selection': dict(DEFAULT_BENCHMARK_SELECTION),
            'task_ids': [1],
            'tasks': [
                {
                    'task_id': 1,
                    'dataset_name': 'toy',
                    'n_rows': 6,
                    'n_features': 2,
                    'n_classes': 2,
                }
            ],
        },
        benchmark_manifest_path=tmp_path / 'bundle.json',
        tab_foundry_run_dir=tmp_path / 'run',
        task_type='supervised_classification',
        nanotabpfn_root=tmp_path / 'nano',
        nanotabpfn_python=tmp_path / 'nano' / '.venv' / 'bin' / 'python',
    )

    assert summary['tab_foundry']['best_step'] == pytest.approx(50.0)
    assert summary['tab_foundry']['best_log_loss'] == pytest.approx(0.40)
    assert summary['tab_foundry']['best_roc_auc'] == pytest.approx(0.81)


def test_build_comparison_summary_averages_external_scalar_metrics_across_seeds(
    tmp_path: Path,
) -> None:
    summary = benchmark_module.build_comparison_summary(
        tab_foundry_records=[
            {
                "checkpoint_path": "/tmp/step_000025.pt",
                "step": 25,
                "training_time": 1.0,
                "roc_auc": 0.80,
                "log_loss": 0.40,
                "brier_score": 0.12,
                "dataset_roc_auc": {"toy": 0.80},
                "dataset_log_loss": {"toy": 0.40},
                "dataset_brier_score": {"toy": 0.12},
            }
        ],
        nanotabpfn_records=[
            {
                "seed": 0,
                "step": 25,
                "training_time": 1.0,
                "roc_auc": 0.90,
                "log_loss": 0.30,
                "brier_score": 0.10,
                "dataset_roc_auc": {"toy": 0.90},
                "dataset_log_loss": {"toy": 0.30},
                "dataset_brier_score": {"toy": 0.10},
            },
            {
                "seed": 1,
                "step": 25,
                "training_time": 3.0,
                "roc_auc": 0.50,
                "log_loss": 0.50,
                "brier_score": 0.30,
                "dataset_roc_auc": {"toy": 0.50},
                "dataset_log_loss": {"toy": 0.50},
                "dataset_brier_score": {"toy": 0.30},
            },
            {
                "seed": 0,
                "step": 50,
                "training_time": 2.0,
                "roc_auc": 0.40,
                "log_loss": 0.60,
                "brier_score": 0.40,
                "dataset_roc_auc": {"toy": 0.40},
                "dataset_log_loss": {"toy": 0.60},
                "dataset_brier_score": {"toy": 0.40},
            },
            {
                "seed": 1,
                "step": 50,
                "training_time": 4.0,
                "roc_auc": 0.80,
                "log_loss": 0.40,
                "brier_score": 0.20,
                "dataset_roc_auc": {"toy": 0.80},
                "dataset_log_loss": {"toy": 0.40},
                "dataset_brier_score": {"toy": 0.20},
            },
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
        benchmark_manifest_path=tmp_path / "bundle.json",
        tab_foundry_run_dir=tmp_path / "run",
        task_type="supervised_classification",
        nanotabpfn_root=tmp_path / "nano",
        nanotabpfn_python=tmp_path / "nano" / ".venv" / "bin" / "python",
    )

    assert summary["nanotabpfn"]["best_step"] == pytest.approx(25.0)
    assert summary["nanotabpfn"]["best_training_time"] == pytest.approx(2.0)
    assert summary["nanotabpfn"]["best_log_loss"] == pytest.approx(0.40)
    assert summary["nanotabpfn"]["best_roc_auc"] == pytest.approx(0.70)
    assert summary["nanotabpfn"]["best_brier_score"] == pytest.approx(0.20)
    assert summary["nanotabpfn"]["final_step"] == pytest.approx(50.0)
    assert summary["nanotabpfn"]["final_training_time"] == pytest.approx(3.0)
    assert summary["nanotabpfn"]["final_log_loss"] == pytest.approx(0.50)
    assert summary["nanotabpfn"]["final_roc_auc"] == pytest.approx(0.60)
    assert summary["nanotabpfn"]["final_brier_score"] == pytest.approx(0.30)
    assert summary["nanotabpfn"]["best_to_final_log_loss_delta"] == pytest.approx(0.10)
    assert summary["nanotabpfn"]["best_to_final_roc_auc_delta"] == pytest.approx(-0.10)
    assert summary["nanotabpfn"]["best_to_final_brier_score_delta"] == pytest.approx(0.10)
    assert summary["nanotabpfn"]["best_dataset_log_loss"] == {"toy": pytest.approx(0.40)}
    assert summary["nanotabpfn"]["final_dataset_log_loss"] == {"toy": pytest.approx(0.50)}
    assert summary["nanotabpfn"]["best_dataset_roc_auc"] == {"toy": pytest.approx(0.70)}
    assert summary["nanotabpfn"]["final_dataset_roc_auc"] == {"toy": pytest.approx(0.60)}
