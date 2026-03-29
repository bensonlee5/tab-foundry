from __future__ import annotations

import json
from pathlib import Path

import tab_foundry.bench.comparison_runtime as compare_module
from tab_foundry.bench.comparison_runtime import BenchmarkComparisonConfig, run_nanotabpfn_benchmark


def test_run_nanotabpfn_benchmark_supports_local_only_mode(
    monkeypatch,
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    out_root = tmp_path / "benchmark"
    bundle_path = tmp_path / "bundle.json"
    bundle = {
        "name": "test_bundle",
        "selection": {
            "new_instances": 200,
            "task_type": "supervised_classification",
        },
        "task_ids": [1],
        "tasks": [
                {
                    "dataset_name": "dummy",
                    "task_id": 1,
                    "task": "classification",
                    "n_rows": 200,
                    "n_features": 1,
                    "n_classes": 2,
            }
        ],
        "version": 1,
    }
    bundle_path.write_text(json.dumps(bundle, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    record = {
        "step": 2500,
        "training_time": 12.5,
        "checkpoint_path": str((run_dir / "checkpoints" / "step_002500.pt").resolve()),
        "roc_auc": 0.73,
        "log_loss": 0.47,
        "brier_score": 0.31,
        "dataset_count": 1,
        "dataset_roc_auc": {"dummy": 0.73},
        "dataset_log_loss": {"dummy": 0.47},
        "dataset_brier_score": {"dummy": 0.31},
        "model_arch": "tabfoundry_sandwich",
        "model_stage": None,
        "benchmark_profile": None,
    }
    summarized_curve = {
        "records": [record],
        "successful_records": [record],
        "failed_records": [],
        "best_record": record,
        "final_record": record,
        "summary": {
            "best_step": 2500.0,
            "final_step": 2500.0,
            "task_count": 1,
            "adjacent_ci_overlap_fraction": 1.0,
            "best_roc_auc": 0.73,
            "final_roc_auc": 0.73,
        },
        "checkpoint_count": 1,
        "successful_checkpoint_count": 1,
        "failed_checkpoint_count": 0,
        "best_checkpoint_path": record["checkpoint_path"],
        "final_checkpoint_path": record["checkpoint_path"],
        "last_attempted_step": 2500,
        "last_attempted_checkpoint_path": record["checkpoint_path"],
        "bootstrap": {"samples": 1, "confidence": 0.95, "seed": 0},
        "best_to_final_roc_auc_delta": 0.0,
        "best_to_final_crps_delta": None,
    }

    monkeypatch.setattr(
        compare_module,
        "_validate_tab_foundry_run_dir",
        lambda path: Path(path).expanduser().resolve(),
    )
    monkeypatch.setattr(
        compare_module,
        "load_benchmark_manifest_datasets",
        lambda **_kwargs: (
            {"dummy": ([0.0], [0])},
            list(bundle["tasks"]),
            {
                "manifest_path": str(bundle_path.resolve()),
                "contract_version": 1,
                "manifest_sha256": "0" * 64,
                "task_type": "supervised_classification",
                "allow_missing_values": False,
                "benchmark_bundle": {
                    "name": str(bundle["name"]),
                    "version": int(bundle["version"]),
                    "source_path": str(bundle_path.resolve()),
                    "task_count": 1,
                    "task_ids": [1],
                    "selection": dict(bundle["selection"]),
                    "allow_missing_values": False,
                    "all_tasks_no_missing": True,
                },
                "persisted_summary": None,
            },
        ),
    )
    monkeypatch.setattr(compare_module, "evaluate_tab_foundry_run", lambda *_args, **_kwargs: [record])
    monkeypatch.setattr(compare_module, "summarize_checkpoint_curve", lambda *_args, **_kwargs: summarized_curve)
    monkeypatch.setattr(compare_module, "plot_comparison_curve", lambda **_kwargs: None)
    monkeypatch.setattr(
        compare_module,
        "derive_benchmark_run_record",
        lambda **_kwargs: {
            "manifest_path": str((tmp_path / "manifest.json").resolve()),
            "seed_set": [13],
            "training_diagnostics": {"health": "ok"},
            "model_size": {"parameter_count": 123},
            "artifacts": {
                "training_surface_record_path": str(
                    (tmp_path / "training_surface_record.json").resolve()
                )
            },
        },
    )

    summary = run_nanotabpfn_benchmark(
        BenchmarkComparisonConfig(
            tab_foundry_run_dir=run_dir,
            out_root=out_root,
            benchmark_manifest_path=bundle_path,
            external_benchmarks=(),
        )
    )

    assert summary["external_benchmarks"] == []
    assert "primary_external_benchmark" not in summary
    assert "nanotabpfn" not in summary
    written_summary = json.loads((out_root / "comparison_summary.json").read_text(encoding="utf-8"))
    assert written_summary["external_benchmarks"] == []
    assert "primary_external_benchmark" not in written_summary
