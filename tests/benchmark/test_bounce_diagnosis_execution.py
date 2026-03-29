from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

import tab_foundry.bench.bounce_diagnosis as diagnosis_module


def test_run_benchmark_bounce_diagnosis_writes_summary_and_flags_benchmark_noise(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "train_history.jsonl").write_text("{}\n", encoding="utf-8")
    out_root = tmp_path / "diagnosis"
    primary_bundle_path = tmp_path / "medium.json"
    confirmation_bundle_path = tmp_path / "large.json"
    primary_bundle_path.write_text("{}", encoding="utf-8")
    confirmation_bundle_path.write_text("{}", encoding="utf-8")
    policy_calls: list[tuple[str, bool]] = []

    def _fake_load_datasets(
        *,
        benchmark_manifest_path: Path | None = None,
    ) -> tuple[dict[str, tuple[list[float], list[int]]], list[dict[str, Any]], dict[str, Any]]:
        if benchmark_manifest_path is not None and Path(benchmark_manifest_path).resolve() == confirmation_bundle_path.resolve():
            policy_calls.append(("datasets_confirmation", True))
            benchmark_tasks = [
                {
                    "task_id": task_id,
                    "dataset_name": f"d{task_id}",
                    "n_rows": 4,
                    "n_features": 2,
                    "n_classes": 2,
                }
                for task_id in range(1, 13)
            ]
            return (
                {f"d{task_id}": ([0.0], [0]) for task_id in range(1, 13)},
                benchmark_tasks,
                {
                    "manifest_path": str(confirmation_bundle_path.resolve()),
                    "contract_version": 1,
                    "manifest_sha256": "confirmation",
                    "task_type": "supervised_classification",
                    "allow_missing_values": True,
                    "benchmark_bundle": {
                        "name": "large",
                        "version": 1,
                        "source_path": str(confirmation_bundle_path.resolve()),
                        "task_count": len(benchmark_tasks),
                        "task_ids": [task["task_id"] for task in benchmark_tasks],
                        "selection": {"new_instances": 4, "max_missing_pct": 5.0},
                        "allow_missing_values": True,
                    },
                    "persisted_summary": None,
                },
            )
        policy_calls.append(("datasets_primary", False))
        benchmark_tasks = [
            {
                "task_id": task_id,
                "dataset_name": f"d{task_id}",
                "n_rows": 4,
                "n_features": 2,
                "n_classes": 2,
            }
            for task_id in range(1, 11)
        ]
        return (
            {f"d{task_id}": ([0.0], [0]) for task_id in range(1, 11)},
            benchmark_tasks,
            {
                "manifest_path": str(primary_bundle_path.resolve()),
                "contract_version": 1,
                "manifest_sha256": "primary",
                "task_type": "supervised_classification",
                "allow_missing_values": False,
                "benchmark_bundle": {
                    "name": "medium",
                    "version": 1,
                    "source_path": str(primary_bundle_path.resolve()),
                    "task_count": len(benchmark_tasks),
                    "task_ids": [task["task_id"] for task in benchmark_tasks],
                    "selection": {"new_instances": 4, "max_missing_pct": 0.0},
                    "allow_missing_values": False,
                },
                "persisted_summary": None,
            },
        )

    def _fake_evaluate_run(
        _run_dir: Path,
        *,
        datasets: dict[str, tuple[list[float], list[int]]],
        task_type: str,
        device: str,
        allow_checkpoint_failures: bool = False,
        allow_missing_values: bool = False,
    ) -> list[dict[str, Any]]:
        assert task_type == "supervised_classification"
        assert device == "cpu"
        assert allow_checkpoint_failures is True
        if len(datasets) == 12:
            policy_calls.append(("evaluate_confirmation", bool(allow_missing_values)))
            return [
                {
                    "checkpoint_path": "/tmp/step_000025.pt",
                    "step": 25,
                    "training_time": 1.0,
                    "roc_auc": 0.70,
                    "dataset_roc_auc": {f"d{task_id}": 0.70 + (0.005 * (task_id % 2)) for task_id in range(1, 13)},
                },
                {
                    "checkpoint_path": "/tmp/step_000050.pt",
                    "step": 50,
                    "training_time": 2.0,
                    "roc_auc": 0.74,
                    "dataset_roc_auc": {f"d{task_id}": 0.74 + (0.005 * (task_id % 2)) for task_id in range(1, 13)},
                },
            ]
        policy_calls.append(("evaluate_primary", bool(allow_missing_values)))
        return [
            {
                "checkpoint_path": "/tmp/step_000025.pt",
                "step": 25,
                "training_time": 1.0,
                "roc_auc": 0.7102,
                "dataset_roc_auc": {
                    f"d{task_id}": 0.7102 + (0.001 * ((task_id % 4) - 1.5))
                    for task_id in range(1, 11)
                },
            },
            {
                "checkpoint_path": "/tmp/step_000050.pt",
                "step": 50,
                "training_time": 2.0,
                "roc_auc": 0.7098,
                "dataset_roc_auc": {
                    f"d{task_id}": 0.7098 + (0.001 * ((task_id % 4) - 1.5))
                    for task_id in range(1, 11)
                },
            },
        ]

    monkeypatch.setattr(diagnosis_module, "load_benchmark_manifest_datasets", _fake_load_datasets)
    monkeypatch.setattr(diagnosis_module, "evaluate_tab_foundry_run", _fake_evaluate_run)
    monkeypatch.setattr(
        diagnosis_module,
        "load_history",
        lambda _path: [
            {"step": 25, "train_loss": 0.5, "grad_norm": 1.0},
            {"step": 50, "train_loss": 0.49, "grad_norm": 1.2},
        ],
    )

    summary = diagnosis_module.run_benchmark_bounce_diagnosis(
        diagnosis_module.BenchmarkBounceDiagnosisConfig(
            run_dir=run_dir,
            out_root=out_root,
            device="cpu",
            benchmark_manifest_path=primary_bundle_path,
            confirmation_benchmark_manifest_path=confirmation_bundle_path,
            bootstrap_samples=64,
        )
    )

    assert "benchmark_noise" in summary["classification"]["primary_causes"]
    assert summary["bundle_analysis"]["best_step_changed_between_bundles"] is True
    assert (out_root / "benchmark_bounce_diagnosis.json").exists()
    written = json.loads((out_root / "benchmark_bounce_diagnosis.json").read_text(encoding="utf-8"))
    assert written["schema"] == diagnosis_module.DIAGNOSIS_SCHEMA
    assert Path(written["artifacts"]["primary_bundle_curve_jsonl"]).exists()
    assert Path(written["artifacts"]["confirmation_bundle_curve_jsonl"]).exists()
    assert policy_calls == [
        ("datasets_primary", False),
        ("evaluate_primary", False),
        ("datasets_confirmation", True),
        ("evaluate_confirmation", True),
    ]


def test_run_benchmark_bounce_diagnosis_without_confirmation_uses_primary_bundle_only(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "train_history.jsonl").write_text("{}\n", encoding="utf-8")
    out_root = tmp_path / "diagnosis"
    primary_bundle_path = tmp_path / "medium.json"
    primary_bundle_path.write_text("{}", encoding="utf-8")
    policy_calls: list[tuple[str, bool]] = []

    def _fake_load_datasets(
        *,
        benchmark_manifest_path: Path | None = None,
    ) -> tuple[dict[str, tuple[list[float], list[int]]], list[dict[str, Any]], dict[str, Any]]:
        assert benchmark_manifest_path is not None
        assert Path(benchmark_manifest_path).resolve() == primary_bundle_path.resolve()
        policy_calls.append(("datasets_primary", False))
        benchmark_tasks = [
            {"task_id": 1, "dataset_name": "d1", "n_rows": 4, "n_features": 2, "n_classes": 2},
            {"task_id": 2, "dataset_name": "d2", "n_rows": 4, "n_features": 2, "n_classes": 2},
        ]
        return (
            {"d1": ([0.0], [0]), "d2": ([0.0], [0])},
            benchmark_tasks,
            {
                "manifest_path": str(primary_bundle_path.resolve()),
                "contract_version": 1,
                "manifest_sha256": "primary",
                "task_type": "supervised_classification",
                "allow_missing_values": False,
                "benchmark_bundle": {
                    "name": "medium",
                    "version": 1,
                    "source_path": str(primary_bundle_path.resolve()),
                    "task_count": len(benchmark_tasks),
                    "task_ids": [task["task_id"] for task in benchmark_tasks],
                    "selection": {"new_instances": 4, "max_missing_pct": 0.0},
                    "allow_missing_values": False,
                },
                "persisted_summary": None,
            },
        )

    def _fake_evaluate_run(
        _run_dir: Path,
        *,
        datasets: dict[str, tuple[list[float], list[int]]],
        task_type: str,
        device: str,
        allow_checkpoint_failures: bool = False,
        allow_missing_values: bool = False,
    ) -> list[dict[str, Any]]:
        assert task_type == "supervised_classification"
        assert device == "cpu"
        assert allow_checkpoint_failures is True
        assert allow_missing_values is False
        policy_calls.append(("evaluate_primary", bool(allow_missing_values)))
        assert list(datasets) == ["d1", "d2"]
        return [
            {
                "checkpoint_path": "/tmp/step_000025.pt",
                "step": 25,
                "training_time": 1.0,
                "roc_auc": 0.71,
                "dataset_roc_auc": {"d1": 0.72, "d2": 0.70},
            },
            {
                "checkpoint_path": "/tmp/step_000050.pt",
                "step": 50,
                "training_time": 2.0,
                "roc_auc": 0.70,
                "dataset_roc_auc": {"d1": 0.71, "d2": 0.69},
            },
        ]

    monkeypatch.setattr(diagnosis_module, "load_benchmark_manifest_datasets", _fake_load_datasets)
    monkeypatch.setattr(diagnosis_module, "evaluate_tab_foundry_run", _fake_evaluate_run)
    monkeypatch.setattr(
        diagnosis_module,
        "load_history",
        lambda _path: [
            {"step": 25, "train_loss": 0.5, "grad_norm": 1.0},
            {"step": 50, "train_loss": 0.49, "grad_norm": 1.1},
        ],
    )

    summary = diagnosis_module.run_benchmark_bounce_diagnosis(
        diagnosis_module.BenchmarkBounceDiagnosisConfig(
            run_dir=run_dir,
            out_root=out_root,
            device="cpu",
            benchmark_manifest_path=primary_bundle_path,
            bootstrap_samples=64,
        )
    )

    assert summary["bundles"]["confirmation"] is None
    assert summary["artifacts"]["confirmation_bundle_curve_jsonl"] is None
    assert summary["bundle_analysis"]["confirmation"] is None
    assert policy_calls == [
        ("datasets_primary", False),
        ("evaluate_primary", False),
    ]
