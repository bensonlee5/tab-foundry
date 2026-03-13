from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch

import tab_foundry.bench.control_baseline as control_baseline_module


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
) -> Path:
    payload = {
        "dataset_count": 1,
        "benchmark_bundle": {
            "name": "test_bundle",
            "version": 1,
            "source_path": benchmark_bundle_source_path,
            "task_count": 1,
            "task_ids": [1],
        },
        "tab_foundry": {
            "best_step": 25.0,
            "best_training_time": 1.2,
            "best_roc_auc": 0.81,
            "final_step": 25.0,
            "final_training_time": 1.2,
            "final_roc_auc": float(final_roc_auc),
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
