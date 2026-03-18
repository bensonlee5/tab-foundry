from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
import pytest

import tab_foundry.bench.bounce_diagnosis as diagnosis_module
import tab_foundry.bench.nanotabpfn as benchmark_module


def test_mainline_bundle_loader_rejects_missing_permitting_large_bundle_by_default() -> None:
    large_bundle_path = (
        Path(__file__).resolve().parents[2]
        / "src"
        / "tab_foundry"
        / "bench"
        / "nanotabpfn_openml_binary_large_v1.json"
    )

    with pytest.raises(RuntimeError, match="permits missing-valued inputs"):
        _ = benchmark_module.load_benchmark_bundle(large_bundle_path)


def test_annotate_curve_records_with_task_statistics_adds_dataset_count_and_ci() -> None:
    records = [
        {
            "step": 25,
            "training_time": 1.0,
            "roc_auc": 0.72,
            "dataset_roc_auc": {"a": 0.70, "b": 0.74, "c": 0.72},
        }
    ]

    annotated = benchmark_module.annotate_curve_records_with_task_statistics(
        records,
        bootstrap_samples=128,
        bootstrap_confidence=0.9,
        bootstrap_seed=7,
    )

    assert annotated[0]["dataset_count"] == 3
    ci = annotated[0]["roc_auc_task_bootstrap_ci"]
    assert ci["samples"] == 128
    assert ci["confidence"] == pytest.approx(0.9)
    assert float(ci["lower"]) <= float(records[0]["roc_auc"]) <= float(ci["upper"])


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

    def _fake_load_bundle(path: Path | None = None) -> tuple[dict[str, Any], bool]:
        resolved = None if path is None else Path(path).resolve()
        if resolved == confirmation_bundle_path.resolve():
            policy_calls.append(("load_confirmation", True))
            return (
                {
                    "name": "large",
                    "version": 1,
                    "selection": {"new_instances": 4, "max_missing_pct": 5.0},
                    "task_ids": list(range(1, 13)),
                    "tasks": [
                        {
                            "task_id": task_id,
                            "dataset_name": f"d{task_id}",
                            "n_rows": 4,
                            "n_features": 2,
                            "n_classes": 2,
                        }
                        for task_id in range(1, 13)
                    ],
                },
                True,
            )
        policy_calls.append(("load_primary", False))
        return (
            {
                "name": "medium",
                "version": 1,
                "selection": {"new_instances": 4, "max_missing_pct": 0.0},
                "task_ids": list(range(1, 11)),
                "tasks": [
                    {
                        "task_id": task_id,
                        "dataset_name": f"d{task_id}",
                        "n_rows": 4,
                        "n_features": 2,
                        "n_classes": 2,
                    }
                    for task_id in range(1, 11)
                ],
            },
            False,
        )

    def _fake_load_datasets(
        *,
        new_instances: int,
        benchmark_bundle_path: Path | None = None,
        allow_missing_values: bool = False,
    ) -> tuple[dict[str, tuple[list[float], list[int]]], list[dict[str, Any]]]:
        del new_instances
        if benchmark_bundle_path is not None and Path(benchmark_bundle_path).resolve() == confirmation_bundle_path.resolve():
            policy_calls.append(("datasets_confirmation", bool(allow_missing_values)))
            return (
                {f"d{task_id}": ([0.0], [0]) for task_id in range(1, 13)},
                [
                    {
                        "task_id": task_id,
                        "dataset_name": f"d{task_id}",
                        "n_rows": 4,
                        "n_features": 2,
                        "n_classes": 2,
                    }
                    for task_id in range(1, 13)
                ],
            )
        policy_calls.append(("datasets_primary", bool(allow_missing_values)))
        return (
            {f"d{task_id}": ([0.0], [0]) for task_id in range(1, 11)},
            [
                {
                    "task_id": task_id,
                    "dataset_name": f"d{task_id}",
                    "n_rows": 4,
                    "n_features": 2,
                    "n_classes": 2,
                }
                for task_id in range(1, 11)
            ],
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

    monkeypatch.setattr(
        diagnosis_module,
        "load_benchmark_bundle_for_execution",
        _fake_load_bundle,
    )
    monkeypatch.setattr(diagnosis_module, "load_openml_benchmark_datasets", _fake_load_datasets)
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
            benchmark_bundle_path=primary_bundle_path,
            confirmation_benchmark_bundle_path=confirmation_bundle_path,
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
        ("load_primary", False),
        ("datasets_primary", False),
        ("evaluate_primary", False),
        ("load_confirmation", True),
        ("datasets_confirmation", True),
        ("evaluate_confirmation", True),
    ]


def test_run_benchmark_bounce_diagnosis_dense_rerun_uses_prior_path_and_flags_aliasing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "train_history.jsonl").write_text("{}\n", encoding="utf-8")
    out_root = tmp_path / "diagnosis"

    dense_calls: list[dict[str, Any]] = []

    monkeypatch.setattr(
        diagnosis_module,
        "_checkpoint_cfg_from_run",
        lambda _run_dir: OmegaConf.create(
            {
                "runtime": {
                    "output_dir": str(run_dir / "orig"),
                    "checkpoint_every": 25,
                    "eval_every": 25,
                    "val_batches": 0,
                },
                "optimizer": {"name": "schedulefree_adamw"},
                "training": {"surface_label": "prior_constant_lr"},
                "logging": {
                    "use_wandb": True,
                    "run_name": "orig",
                    "history_jsonl_path": str(run_dir / "train_history.jsonl"),
                },
            }
        ),
    )

    def _fake_prior_train(cfg: Any) -> None:
        dense_calls.append(
            {
                "checkpoint_every": int(cfg.runtime.checkpoint_every),
                "eval_every": int(cfg.runtime.eval_every),
                "output_dir": str(cfg.runtime.output_dir),
                "use_wandb": bool(cfg.logging.use_wandb),
            }
        )
        dense_dir = Path(str(cfg.runtime.output_dir))
        dense_dir.mkdir(parents=True, exist_ok=True)

    def _fake_evaluate_one_bundle(
        *,
        run_dir: Path,
        bundle_path: Path,
        device: str,
        out_path: Path,
        bootstrap_samples: int,
        bootstrap_confidence: float,
    ) -> dict[str, Any]:
        del bundle_path, device, bootstrap_samples, bootstrap_confidence
        if run_dir.name == "dense_checkpoint_run":
            records = [
                {
                    "checkpoint_path": "/tmp/step_000030.pt",
                    "step": 30,
                    "training_time": 1.2,
                    "roc_auc": 0.76,
                    "dataset_roc_auc": {"a": 0.76, "b": 0.76},
                    "dataset_count": 2,
                    "roc_auc_task_bootstrap_ci": {"lower": 0.74, "upper": 0.78},
                },
                {
                    "checkpoint_path": "/tmp/step_000050.pt",
                    "step": 50,
                    "training_time": 2.0,
                    "roc_auc": 0.74,
                    "dataset_roc_auc": {"a": 0.74, "b": 0.74},
                    "dataset_count": 2,
                    "roc_auc_task_bootstrap_ci": {"lower": 0.72, "upper": 0.76},
                },
            ]
        else:
            records = [
                {
                    "checkpoint_path": "/tmp/step_000025.pt",
                    "step": 25,
                    "training_time": 1.0,
                    "roc_auc": 0.73,
                    "dataset_roc_auc": {"a": 0.73, "b": 0.73},
                    "dataset_count": 2,
                    "roc_auc_task_bootstrap_ci": {"lower": 0.71, "upper": 0.75},
                },
                {
                    "checkpoint_path": "/tmp/step_000050.pt",
                    "step": 50,
                    "training_time": 2.0,
                    "roc_auc": 0.72,
                    "dataset_roc_auc": {"a": 0.72, "b": 0.72},
                    "dataset_count": 2,
                    "roc_auc_task_bootstrap_ci": {"lower": 0.70, "upper": 0.74},
                },
            ]
        diagnosis_module.write_jsonl(out_path, records)
        return {
            "bundle": {"name": "bundle", "version": 1, "source_path": str(out_path), "task_count": 2, "task_ids": [1, 2]},
            "benchmark_tasks": [],
            "records": records,
            "records_path": str(out_path.resolve()),
            "summary": diagnosis_module._curve_summary(records),
        }

    monkeypatch.setattr(diagnosis_module, "train_tabfoundry_simple_prior", _fake_prior_train)
    monkeypatch.setattr(diagnosis_module, "_evaluate_one_bundle", _fake_evaluate_one_bundle)
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
            dense_checkpoint_every=10,
            rerun_mode="auto",
        )
    )

    assert dense_calls == [
        {
            "checkpoint_every": 10,
            "eval_every": 10,
            "output_dir": str((out_root / "dense_checkpoint_run").resolve()),
            "use_wandb": False,
        }
    ]
    assert "checkpoint_aliasing" in summary["classification"]["primary_causes"]


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

    def _fake_load_bundle(path: Path | None = None) -> tuple[dict[str, Any], bool]:
        assert path is not None
        assert Path(path).resolve() == primary_bundle_path.resolve()
        policy_calls.append(("load_primary", False))
        return (
                {
                    "name": "medium",
                    "version": 1,
                    "selection": {"new_instances": 4, "max_missing_pct": 0.0},
                    "task_ids": [1, 2],
                "tasks": [
                    {"task_id": 1, "dataset_name": "d1", "n_rows": 4, "n_features": 2, "n_classes": 2},
                    {"task_id": 2, "dataset_name": "d2", "n_rows": 4, "n_features": 2, "n_classes": 2},
                ],
            },
            False,
        )

    def _fake_load_datasets(
        *,
        new_instances: int,
        benchmark_bundle_path: Path | None = None,
        allow_missing_values: bool = False,
    ) -> tuple[dict[str, tuple[list[float], list[int]]], list[dict[str, Any]]]:
        assert new_instances == 4
        assert benchmark_bundle_path is not None
        assert Path(benchmark_bundle_path).resolve() == primary_bundle_path.resolve()
        policy_calls.append(("datasets_primary", bool(allow_missing_values)))
        return (
            {"d1": ([0.0], [0]), "d2": ([0.0], [0])},
            [
                {"task_id": 1, "dataset_name": "d1", "n_rows": 4, "n_features": 2, "n_classes": 2},
                {"task_id": 2, "dataset_name": "d2", "n_rows": 4, "n_features": 2, "n_classes": 2},
            ],
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

    monkeypatch.setattr(
        diagnosis_module,
        "load_benchmark_bundle_for_execution",
        _fake_load_bundle,
    )
    monkeypatch.setattr(diagnosis_module, "load_openml_benchmark_datasets", _fake_load_datasets)
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
            benchmark_bundle_path=primary_bundle_path,
            bootstrap_samples=64,
        )
    )

    assert summary["bundles"]["confirmation"] is None
    assert summary["artifacts"]["confirmation_bundle_curve_jsonl"] is None
    assert summary["bundle_analysis"]["confirmation"] is None
    assert policy_calls == [
        ("load_primary", False),
        ("datasets_primary", False),
        ("evaluate_primary", False),
    ]


def test_run_benchmark_bounce_diagnosis_dense_confirmation_inherits_missing_value_policy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    dense_run_dir = tmp_path / "dense_run"
    dense_run_dir.mkdir()
    (run_dir / "train_history.jsonl").write_text("{}\n", encoding="utf-8")
    out_root = tmp_path / "diagnosis"
    primary_bundle_path = tmp_path / "medium.json"
    confirmation_bundle_path = tmp_path / "large.json"
    primary_bundle_path.write_text("{}", encoding="utf-8")
    confirmation_bundle_path.write_text("{}", encoding="utf-8")
    policy_calls: list[tuple[str, bool]] = []

    def _fake_load_bundle(path: Path | None = None) -> tuple[dict[str, Any], bool]:
        assert path is not None
        resolved = Path(path).resolve()
        if resolved == confirmation_bundle_path.resolve():
            policy_calls.append(("load_large", True))
            return (
                {
                    "name": "large",
                    "version": 1,
                    "selection": {"new_instances": 4, "max_missing_pct": 5.0},
                    "task_ids": [1, 2],
                    "tasks": [
                        {"task_id": 1, "dataset_name": "d1", "n_rows": 4, "n_features": 2, "n_classes": 2},
                        {"task_id": 2, "dataset_name": "d2", "n_rows": 4, "n_features": 2, "n_classes": 2},
                    ],
                },
                True,
            )
        policy_calls.append(("load_medium", False))
        return (
            {
                "name": "medium",
                "version": 1,
                "selection": {"new_instances": 4, "max_missing_pct": 0.0},
                "task_ids": [1, 2],
                "tasks": [
                    {"task_id": 1, "dataset_name": "d1", "n_rows": 4, "n_features": 2, "n_classes": 2},
                    {"task_id": 2, "dataset_name": "d2", "n_rows": 4, "n_features": 2, "n_classes": 2},
                ],
            },
            False,
        )

    def _fake_load_datasets(
        *,
        new_instances: int,
        benchmark_bundle_path: Path | None = None,
        allow_missing_values: bool = False,
    ) -> tuple[dict[str, tuple[list[float], list[int]]], list[dict[str, Any]]]:
        assert new_instances == 4
        assert benchmark_bundle_path is not None
        resolved = Path(benchmark_bundle_path).resolve()
        label = "datasets_large" if resolved == confirmation_bundle_path.resolve() else "datasets_medium"
        policy_calls.append((label, bool(allow_missing_values)))
        return (
            {"d1": ([0.0], [0]), "d2": ([0.0], [0])},
            [
                {"task_id": 1, "dataset_name": "d1", "n_rows": 4, "n_features": 2, "n_classes": 2},
                {"task_id": 2, "dataset_name": "d2", "n_rows": 4, "n_features": 2, "n_classes": 2},
            ],
        )

    def _fake_evaluate_run(
        actual_run_dir: Path,
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
        assert list(datasets) == ["d1", "d2"]
        label = "evaluate_dense" if Path(actual_run_dir).resolve() == dense_run_dir.resolve() else "evaluate_run"
        policy_calls.append((label, bool(allow_missing_values)))
        return [
            {
                "checkpoint_path": "/tmp/step_000025.pt",
                "step": 25,
                "training_time": 1.0,
                "roc_auc": 0.71 if not allow_missing_values else 0.72,
                "dataset_roc_auc": {"d1": 0.72, "d2": 0.70},
            },
            {
                "checkpoint_path": "/tmp/step_000050.pt",
                "step": 50,
                "training_time": 2.0,
                "roc_auc": 0.70 if not allow_missing_values else 0.71,
                "dataset_roc_auc": {"d1": 0.71, "d2": 0.69},
            },
        ]

    monkeypatch.setattr(diagnosis_module, "load_benchmark_bundle_for_execution", _fake_load_bundle)
    monkeypatch.setattr(diagnosis_module, "load_openml_benchmark_datasets", _fake_load_datasets)
    monkeypatch.setattr(diagnosis_module, "evaluate_tab_foundry_run", _fake_evaluate_run)
    monkeypatch.setattr(diagnosis_module, "_run_dense_checkpoint_rerun", lambda _config: dense_run_dir)
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
            benchmark_bundle_path=primary_bundle_path,
            confirmation_benchmark_bundle_path=confirmation_bundle_path,
            dense_checkpoint_every=10,
        )
    )

    assert summary["bundles"]["confirmation"]["benchmark_bundle"]["allow_missing_values"] is True
    assert policy_calls == [
        ("load_medium", False),
        ("datasets_medium", False),
        ("evaluate_run", False),
        ("load_large", True),
        ("datasets_large", True),
        ("evaluate_run", True),
        ("load_large", True),
        ("datasets_large", True),
        ("evaluate_dense", True),
    ]
