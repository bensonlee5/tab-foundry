from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import tab_foundry.bench.control_baseline as control_baseline_module


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_checkpoint(checkpoint_path: Path, *, manifest_path: str, seed: int = 1) -> Path:
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": {},
            "config": {
                "data": {"manifest_path": manifest_path},
                "runtime": {"seed": int(seed)},
            },
        },
        checkpoint_path,
    )
    return checkpoint_path


def _write_comparison_summary(
    path: Path,
    *,
    run_dir: Path,
    benchmark_bundle_source_path: str,
    final_roc_auc: float = 0.83,
    final_log_loss: float = 0.41,
    include_diagnostics: bool = False,
) -> Path:
    payload = {
        "dataset_count": 1,
        "benchmark_bundle": {
            "name": "test_bundle",
            "version": 1,
            "source_path": benchmark_bundle_source_path,
            "task_count": 1,
            "task_ids": [1],
            "selection": {
                "new_instances": 6,
                "task_type": "supervised_classification",
                "max_features": 10,
                "max_classes": 2,
                "max_missing_pct": 0.0,
                "min_minority_class_pct": 2.5,
            },
            "allow_missing_values": False,
            "all_tasks_no_missing": True,
        },
        "tab_foundry": {
            "best_step": 25.0,
            "best_training_time": 1.2,
            "best_roc_auc": 0.81,
            "final_step": 25.0,
            "final_training_time": 1.2,
            "final_roc_auc": float(final_roc_auc),
            "final_log_loss": float(final_log_loss),
            "run_dir": str(run_dir.resolve()),
        },
        "nanotabpfn": {
            "best_step": 25.0,
            "best_training_time": 2.0,
            "best_roc_auc": 0.78,
            "final_step": 25.0,
            "final_training_time": 2.0,
            "final_roc_auc": 0.78,
            "root": "/tmp/nano",
            "python": "/tmp/nano/.venv/bin/python",
            "num_seeds": 2,
        },
    }
    if include_diagnostics:
        payload["tab_foundry"]["best_to_final_roc_auc_delta"] = float(final_roc_auc) - 0.81
        payload["tab_foundry"]["checkpoint_diagnostics"] = {
            "checkpoint_count": 2,
            "successful_checkpoint_count": 1,
            "failed_checkpoint_count": 1,
            "task_count": 1,
            "adjacent_ci_overlap_fraction": None,
            "best_checkpoint_path": "/tmp/step_000025.pt",
            "final_checkpoint_path": "/tmp/step_000025.pt",
            "last_attempted_step": 50,
            "last_attempted_checkpoint_path": "/tmp/step_000050.pt",
            "bootstrap": {"samples": 2000, "confidence": 0.95, "seed": 0},
            "best_checkpoint": {
                "checkpoint_path": "/tmp/step_000025.pt",
                "step": 25,
                "training_time": 1.2,
                "roc_auc": 0.81,
                "dataset_roc_auc": {"toy": 0.81},
                "dataset_count": 1,
                "roc_auc_task_bootstrap_ci": {
                    "samples": 2000,
                    "confidence": 0.95,
                    "lower": 0.81,
                    "upper": 0.81,
                },
                "is_best_checkpoint": True,
                "is_final_checkpoint": True,
            },
            "final_checkpoint": {
                "checkpoint_path": "/tmp/step_000025.pt",
                "step": 25,
                "training_time": 1.2,
                "roc_auc": 0.81,
                "dataset_roc_auc": {"toy": 0.81},
                "dataset_count": 1,
                "roc_auc_task_bootstrap_ci": {
                    "samples": 2000,
                    "confidence": 0.95,
                    "lower": 0.81,
                    "upper": 0.81,
                },
                "is_best_checkpoint": True,
                "is_final_checkpoint": True,
            },
            "checkpoints": [
                {
                    "checkpoint_path": "/tmp/step_000025.pt",
                    "step": 25,
                    "training_time": 1.2,
                    "roc_auc": 0.81,
                    "dataset_roc_auc": {"toy": 0.81},
                    "dataset_count": 1,
                    "roc_auc_task_bootstrap_ci": {
                        "samples": 2000,
                        "confidence": 0.95,
                        "lower": 0.81,
                        "upper": 0.81,
                    },
                    "is_best_checkpoint": True,
                    "is_final_checkpoint": True,
                },
                {
                    "checkpoint_path": "/tmp/step_000050.pt",
                    "step": 50,
                    "training_time": 2.4,
                    "evaluation_error": "benchmark evaluation failed for dataset 'toy': Input contains NaN.",
                    "evaluation_error_type": "ValueError",
                    "failed_dataset": "toy",
                    "is_best_checkpoint": False,
                    "is_final_checkpoint": False,
                },
            ],
            "failed_checkpoints": [
                {
                    "checkpoint_path": "/tmp/step_000050.pt",
                    "step": 50,
                    "training_time": 2.4,
                    "evaluation_error": "benchmark evaluation failed for dataset 'toy': Input contains NaN.",
                    "evaluation_error_type": "ValueError",
                    "failed_dataset": "toy",
                }
            ],
        }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return path


def test_freeze_control_baseline_writes_repo_relative_registry_entry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    registry_path = repo_root / "src" / "tab_foundry" / "bench" / "control_baselines_v1.json"
    manifest_path = repo_root / "data" / "manifests" / "default.parquet"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(b"manifest")
    benchmark_bundle_path = (
        repo_root / "src" / "tab_foundry" / "bench" / "nanotabpfn_openml_benchmark_v1.json"
    )
    benchmark_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_bundle_path.write_text("{}\n", encoding="utf-8")
    run_dir = repo_root / "outputs" / "control_baselines" / "cls_benchmark_linear_v1" / "train"
    _ = _write_checkpoint(
        run_dir / "checkpoints" / "best.pt",
        manifest_path="data/manifests/default.parquet",
        seed=7,
    )
    summary_path = repo_root / "outputs" / "control_baselines" / "cls_benchmark_linear_v1" / "benchmark" / "comparison_summary.json"
    _ = _write_comparison_summary(
        summary_path,
        run_dir=run_dir,
        benchmark_bundle_source_path="src/tab_foundry/bench/nanotabpfn_openml_benchmark_v1.json",
    )
    monkeypatch.setattr(control_baseline_module, "project_root", lambda: repo_root)

    frozen = control_baseline_module.freeze_control_baseline(
        baseline_id="cls_benchmark_linear_v1",
        experiment="cls_benchmark_linear",
        config_profile="cls_benchmark_linear",
        budget_class="short-run",
        run_dir=run_dir,
        comparison_summary_path=summary_path,
        registry_path=registry_path,
    )

    assert frozen["registry_path"] == str(registry_path.resolve())
    assert frozen["baseline"]["manifest_path"] == "data/manifests/default.parquet"
    assert frozen["baseline"]["run_dir"] == "outputs/control_baselines/cls_benchmark_linear_v1/train"
    assert (
        frozen["baseline"]["comparison_summary_path"]
        == "outputs/control_baselines/cls_benchmark_linear_v1/benchmark/comparison_summary.json"
    )
    assert frozen["baseline"]["benchmark_bundle"]["source_path"] == (
        "src/tab_foundry/bench/nanotabpfn_openml_benchmark_v1.json"
    )
    registry = control_baseline_module.load_control_baseline_registry(registry_path)
    assert registry["baselines"]["cls_benchmark_linear_v1"]["seed_set"] == [7]

    _ = _write_comparison_summary(
        summary_path,
        run_dir=run_dir,
        benchmark_bundle_source_path="src/tab_foundry/bench/nanotabpfn_openml_benchmark_v1.json",
        final_roc_auc=0.86,
    )
    _ = control_baseline_module.freeze_control_baseline(
        baseline_id="cls_benchmark_linear_v1",
        experiment="cls_benchmark_linear",
        config_profile="cls_benchmark_linear",
        budget_class="short-run",
        run_dir=run_dir,
        comparison_summary_path=summary_path,
        registry_path=registry_path,
    )
    updated = control_baseline_module.load_control_baseline_entry(
        "cls_benchmark_linear_v1",
        registry_path=registry_path,
    )
    assert updated["tab_foundry_metrics"]["final_roc_auc"] == pytest.approx(0.86)
    assert updated["tab_foundry_metrics"]["final_log_loss"] == pytest.approx(0.41)


def test_freeze_control_baseline_validates_summary_run_dir(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    registry_path = repo_root / "src" / "tab_foundry" / "bench" / "control_baselines_v1.json"
    run_dir = repo_root / "outputs" / "control_baselines" / "cls_benchmark_linear_v1" / "train"
    other_run_dir = repo_root / "outputs" / "other_run"
    manifest_path = repo_root / "data" / "manifests" / "default.parquet"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(b"manifest")
    benchmark_bundle_path = (
        repo_root / "src" / "tab_foundry" / "bench" / "nanotabpfn_openml_benchmark_v1.json"
    )
    benchmark_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_bundle_path.write_text("{}\n", encoding="utf-8")
    _ = _write_checkpoint(
        run_dir / "checkpoints" / "best.pt",
        manifest_path="data/manifests/default.parquet",
        seed=7,
    )
    summary_path = repo_root / "comparison_summary.json"
    _ = _write_comparison_summary(
        summary_path,
        run_dir=other_run_dir,
        benchmark_bundle_source_path="src/tab_foundry/bench/nanotabpfn_openml_benchmark_v1.json",
    )
    monkeypatch.setattr(control_baseline_module, "project_root", lambda: repo_root)

    with pytest.raises(RuntimeError, match="run_dir does not match"):
        control_baseline_module.freeze_control_baseline(
            baseline_id="cls_benchmark_linear_v1",
            experiment="cls_benchmark_linear",
            config_profile="cls_benchmark_linear",
            budget_class="short-run",
            run_dir=run_dir,
            comparison_summary_path=summary_path,
            registry_path=registry_path,
        )


def test_freeze_control_baseline_accepts_richer_checkpoint_diagnostics_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root = tmp_path / "repo"
    registry_path = repo_root / "src" / "tab_foundry" / "bench" / "control_baselines_v1.json"
    run_dir = repo_root / "outputs" / "control_baselines" / "cls_benchmark_linear_v2" / "train"
    manifest_path = repo_root / "data" / "manifests" / "default.parquet"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_bytes(b"manifest")
    benchmark_bundle_path = (
        repo_root / "src" / "tab_foundry" / "bench" / "nanotabpfn_openml_binary_medium_v1.json"
    )
    benchmark_bundle_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark_bundle_path.write_text("{}\n", encoding="utf-8")
    _ = _write_checkpoint(
        run_dir / "checkpoints" / "best.pt",
        manifest_path="data/manifests/default.parquet",
        seed=11,
    )
    summary_path = repo_root / "outputs" / "control_baselines" / "cls_benchmark_linear_v2" / "benchmark" / "comparison_summary.json"
    _ = _write_comparison_summary(
        summary_path,
        run_dir=run_dir,
        benchmark_bundle_source_path="src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
        include_diagnostics=True,
    )
    monkeypatch.setattr(control_baseline_module, "project_root", lambda: repo_root)

    frozen = control_baseline_module.freeze_control_baseline(
        baseline_id="cls_benchmark_linear_v2",
        experiment="cls_benchmark_linear",
        config_profile="cls_benchmark_linear",
        budget_class="short-run",
        run_dir=run_dir,
        comparison_summary_path=summary_path,
        registry_path=registry_path,
    )

    assert frozen["baseline"]["benchmark_bundle"]["source_path"] == (
        "src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json"
    )
    assert frozen["baseline"]["seed_set"] == [11]
    assert frozen["baseline"]["tab_foundry_metrics"]["best_roc_auc"] == pytest.approx(0.81)
    assert frozen["baseline"]["tab_foundry_metrics"]["final_roc_auc"] == pytest.approx(0.83)
    assert frozen["baseline"]["tab_foundry_metrics"]["final_log_loss"] == pytest.approx(0.41)


def test_checked_in_control_baseline_registry_preserves_v1_and_adds_v2() -> None:
    registry_path = REPO_ROOT / "src" / "tab_foundry" / "bench" / "control_baselines_v1.json"

    registry = control_baseline_module.load_control_baseline_registry(registry_path)

    assert {"cls_benchmark_linear_v1", "cls_benchmark_linear_v2"} <= set(registry["baselines"])
    assert registry["baselines"]["cls_benchmark_linear_v1"]["benchmark_bundle"]["source_path"] == (
        "src/tab_foundry/bench/nanotabpfn_openml_benchmark_v1.json"
    )
    assert "final_log_loss" not in registry["baselines"]["cls_benchmark_linear_v1"]["tab_foundry_metrics"]
    assert registry["baselines"]["cls_benchmark_linear_v2"]["benchmark_bundle"]["source_path"] == (
        "src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json"
    )
