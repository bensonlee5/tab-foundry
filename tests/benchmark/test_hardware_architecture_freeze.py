from __future__ import annotations

import json
from pathlib import Path

import pytest

import tab_foundry.cli.bench_hardware_architecture_freeze as hardware_freeze_cli_module
import tab_foundry.bench.hardware_architecture_freeze as hardware_freeze_module


def _run_entry(
    run_id: str,
    *,
    delta_id: str,
    d_icl: int,
    layers: int,
    total_params: int,
    final_log_loss: float,
) -> dict[str, object]:
    reserved_gb = 6.47 + 2.36e-6 * float(total_params)
    train_wall_seconds = 8407.97 + 1.01e-4 * float(total_params)
    return {
        "run_id": run_id,
        "track": "system_delta_classification_medium_v1",
        "experiment": "cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1",
        "config_profile": "cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1",
        "budget_class": "short-run",
        "model": {
            "arch": "tabfoundry_sandwich",
            "d_icl": d_icl,
            "tficl_n_heads": 1,
            "tficl_n_layers": 12,
            "head_hidden_dim": 96,
            "input_normalization": "train_zscore_clip",
            "many_class_base": 10,
            "architecture": {"latents": 24},
            "build_spec": {
                "sandwich_layers": layers,
                "sandwich_heads": 1,
                "sandwich_latents": 24,
            },
        },
        "lineage": {
            "parent_run_id": None,
            "anchor_run_id": "sd_tf_rd_009_anchor_replay_heads1_medium_v1_01_delta_tf_rd_024_followup_cls_sandwich_heads1_v1_v2",
            "control_baseline_id": "cls_benchmark_linear_multiclass_medium_v1",
        },
        "manifest_path": "data/manifests/bench/openml_classification_medium_v1/manifest.parquet",
        "seed_set": [1],
        "benchmark_bundle": {
            "name": "medium_bundle",
            "version": 1,
            "source_path": "src/tab_foundry/bench/openml_classification_medium_v1.json",
            "task_count": 242,
            "task_ids": [1],
        },
        "artifacts": {
            "run_dir": f"outputs/{run_id}/train",
            "benchmark_dir": f"outputs/{run_id}/benchmark",
            "prior_dir": None,
            "history_path": f"outputs/{run_id}/train/train_history.jsonl",
            "best_checkpoint_path": f"outputs/{run_id}/train/checkpoints/best.pt",
            "comparison_summary_path": f"outputs/{run_id}/benchmark/comparison_summary.json",
            "comparison_curve_path": f"outputs/{run_id}/benchmark/comparison_curve.png",
            "benchmark_run_record_path": f"outputs/{run_id}/benchmark/benchmark_run_record.json",
            "training_surface_record_path": f"outputs/{run_id}/benchmark/training_surface_record.json",
        },
        "tab_foundry_metrics": {
            "best_step": 100.0,
            "best_training_time": 100.0,
            "final_step": 100.0,
            "final_training_time": 100.0,
            "final_log_loss": final_log_loss,
            "final_brier_score": 0.39,
            "final_roc_auc": 0.67,
        },
        "training_diagnostics": {
            "best_val_loss": 0.3,
            "final_val_loss": 0.3,
            "best_val_step": 100.0,
            "post_warmup_train_loss_var": 0.01,
            "mean_grad_norm": 1.0,
            "max_grad_norm": 6.0,
            "final_grad_norm": 0.8,
            "train_elapsed_seconds": train_wall_seconds,
            "wall_elapsed_seconds": train_wall_seconds,
        },
        "runtime_summary": {
            "peak_vram_allocated": int((reserved_gb - 0.75) * float(1024**3)),
            "peak_vram_reserved": int(reserved_gb * float(1024**3)),
            "throughput_examples_per_second": 18.06,
            "throughput_tokens_per_second": 107589.79,
            "non_train_overhead_seconds": 18.9,
        },
        "hardware_summary": {
            "device_type": "cuda",
            "raw_device_name": "Quadro RTX 8000",
            "gpu_class": "rtx8000",
            "total_device_vram_bytes": 47560916992,
            "vram_class_gb": 44,
            "hardware_profile_id": "rtx8000_44gb",
        },
        "regime_budget": {
            "token_budget": 917716352,
            "unique_task_budget": 143976,
            "objective_metric": "final_log_loss_at_matched_regime_budget",
            "curriculum_id": "tf_rd_010_dagzoo_medium_control",
        },
        "model_size": {"total_params": total_params, "trainable_params": total_params},
        "surface_labels": {
            "model": "tabfoundry_sandwich",
            "data": "tf_rd_010_dagzoo_medium_control",
            "preprocessing": "runtime_default",
            "training": "prior_cosine_warmup",
        },
        "sweep": {
            "sweep_id": "tf_rd_009_width_depth_medium_v1",
            "delta_id": delta_id,
            "parent_sweep_id": "tf_rd_009_width_transfer_medium_v1",
            "queue_order": 1,
            "run_kind": "primary",
        },
        "comparisons": {"vs_parent": None, "vs_anchor": None},
        "decision": "keep",
        "conclusion": "registry fixture",
        "registered_at_utc": "2026-04-09T00:00:00Z",
    }


def _write_benchmark_registry(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema": "tab-foundry-benchmark-runs-v1",
        "version": 1,
        "runs": {
            "anchor_60x2": _run_entry(
                "anchor_60x2",
                delta_id="delta_tf_rd_024_followup_cls_sandwich_heads1_v1",
                d_icl=60,
                layers=2,
                total_params=646970,
                final_log_loss=0.6620,
            ),
            "baseline_96x2": _run_entry(
                "baseline_96x2",
                delta_id="delta_tf_rd_009_cls_sandwich_dicl96_v1",
                d_icl=96,
                layers=2,
                total_params=1618286,
                final_log_loss=0.6331,
            ),
            "upper_128x2": _run_entry(
                "upper_128x2",
                delta_id="delta_tf_rd_009_cls_sandwich_dicl128_v1",
                d_icl=128,
                layers=2,
                total_params=2849422,
                final_log_loss=0.6225,
            ),
            "joint_72x1": _run_entry(
                "joint_72x1",
                delta_id="delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1",
                d_icl=72,
                layers=1,
                total_params=671809,
                final_log_loss=0.6410,
            ),
            "joint_112x3": _run_entry(
                "joint_112x3",
                delta_id="delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1",
                d_icl=112,
                layers=3,
                total_params=2798089,
                final_log_loss=0.6290,
            ),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_freeze_hardware_architecture_baseline_derives_preferred_entry(tmp_path: Path) -> None:
    registry_path = tmp_path / "benchmark_run_registry_v1.json"
    hardware_registry_path = tmp_path / "hardware_architecture_baselines_v1.json"
    _write_benchmark_registry(registry_path)

    result = hardware_freeze_module.freeze_hardware_architecture_baseline(
        baseline_id="tf_rd_009_rtx8000_44gb_classification_medium_v1",
        preferred_run_id="baseline_96x2",
        formal_anchor_run_id="anchor_60x2",
        baseline_run_id="baseline_96x2",
        evidence_run_ids=(
            "anchor_60x2",
            "baseline_96x2",
            "upper_128x2",
            "joint_72x1",
            "joint_112x3",
        ),
        rationale="RTX 8000 medium classification baseline retained at 96x2 after healthy-only width-depth comparison.",
        decision="keep",
        surface_role="classification_scaling_law",
        benchmark_registry_path=registry_path,
        registry_path=hardware_registry_path,
    )

    baseline = result["baseline"]
    assert result["registry_path"] == str(hardware_registry_path.resolve())
    assert baseline["hardware_profile_id"] == "rtx8000_44gb"
    assert baseline["gpu_class"] == "rtx8000"
    assert baseline["vram_class_gb"] == 44
    assert baseline["preferred_run_id"] == "baseline_96x2"
    assert baseline["preferred_delta_ref"] == "delta_tf_rd_009_cls_sandwich_dicl96_v1"
    assert baseline["preferred_architecture"]["d_icl"] == 96
    assert baseline["preferred_architecture"]["tficl_n_layers"] == 12
    assert baseline["preferred_architecture"]["sandwich_layers"] == 2
    assert baseline["objective_metric"] == "final_log_loss_at_matched_regime_budget"
    assert baseline["selection_rule"] == "best_loss_healthy_only"
    assert baseline["constraint_model"]["baseline_row"] == "96x2"
    assert baseline["constraint_model"]["effective_size_expression"] == "S(d, L) = L * d^2"
    parameter_formula = baseline["constraint_model"]["formulas"]["parameter_count"]
    assert parameter_formula["fit_kind"] == "affine_depth_aware_least_squares"
    assert " + " in parameter_formula["expression"]
    assert "d^2 + " in parameter_formula["expression"]
    assert [row["row"] for row in baseline["constraint_model"]["rows"]] == [
        "60x2",
        "72x1",
        "96x2",
        "112x3",
        "128x2",
    ]
    predicted_rows = {row["row"]: row["predicted"]["total_params"] for row in baseline["constraint_model"]["rows"]}
    assert predicted_rows["72x1"] == pytest.approx(670661, abs=2500)
    assert predicted_rows["112x3"] == pytest.approx(2797614, abs=2500)
    assert "89.00 * L * d^2" not in parameter_formula["expression"]

    written = json.loads(hardware_registry_path.read_text(encoding="utf-8"))
    assert "tf_rd_009_rtx8000_44gb_classification_medium_v1" in written["baselines"]


def test_fit_parameter_bridge_uses_depth_aware_affine_model_for_mixed_depth_evidence() -> None:
    entries = list(
        json.loads(
            json.dumps(
                {
                    "runs": {
                        key: value
                        for key, value in {
                            "anchor_60x2": _run_entry(
                                "anchor_60x2",
                                delta_id="delta_tf_rd_024_followup_cls_sandwich_heads1_v1",
                                d_icl=60,
                                layers=2,
                                total_params=646970,
                                final_log_loss=0.6620,
                            ),
                            "baseline_96x2": _run_entry(
                                "baseline_96x2",
                                delta_id="delta_tf_rd_009_cls_sandwich_dicl96_v1",
                                d_icl=96,
                                layers=2,
                                total_params=1618286,
                                final_log_loss=0.6331,
                            ),
                            "upper_128x2": _run_entry(
                                "upper_128x2",
                                delta_id="delta_tf_rd_009_cls_sandwich_dicl128_v1",
                                d_icl=128,
                                layers=2,
                                total_params=2849422,
                                final_log_loss=0.6225,
                            ),
                            "joint_72x1": _run_entry(
                                "joint_72x1",
                                delta_id="delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1",
                                d_icl=72,
                                layers=1,
                                total_params=671809,
                                final_log_loss=0.6410,
                            ),
                            "joint_112x3": _run_entry(
                                "joint_112x3",
                                delta_id="delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1",
                                d_icl=112,
                                layers=3,
                                total_params=2798089,
                                final_log_loss=0.6290,
                            ),
                        }.items()
                    }
                }
            )
        )["runs"].values()
    )

    formula, predictor = hardware_freeze_module._fit_parameter_bridge(  # pyright: ignore[reportPrivateUsage]
        entries,
        evidence_run_ids=("anchor_60x2", "baseline_96x2", "upper_128x2", "joint_72x1", "joint_112x3"),
    )

    assert formula["fit_kind"] == "affine_depth_aware_least_squares"
    coefficients = formula["coefficients"]
    assert coefficients["intercept"] == pytest.approx(29249.24, abs=5.0)
    assert coefficients["d_squared_coefficient"] == pytest.approx(75.25, abs=0.05)
    assert coefficients["layered_d_squared_coefficient"] == pytest.approx(48.48, abs=0.05)
    assert predictor(72, 1, 72 * 72) == pytest.approx(670661, abs=2500)
    assert predictor(112, 3, 3 * 112 * 112) == pytest.approx(2797614, abs=2500)


def test_freeze_hardware_architecture_baseline_cli_parses_expected_flags(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}

    def _fake_freeze_hardware_architecture_baseline(**kwargs):
        captured.update(kwargs)
        return {
            "registry_path": str((tmp_path / "hardware_architecture_baselines_v1.json").resolve()),
            "baseline": {"baseline_id": kwargs["baseline_id"]},
        }

    monkeypatch.setattr(
        hardware_freeze_cli_module,
        "freeze_hardware_architecture_baseline",
        _fake_freeze_hardware_architecture_baseline,
    )

    exit_code = hardware_freeze_cli_module.main(
        [
            "--baseline-id",
            "tf_rd_009_a100_classification_medium_v1",
            "--preferred-run-id",
            "baseline_96x2",
            "--formal-anchor-run-id",
            "anchor_60x2",
            "--baseline-run-id",
            "baseline_96x2",
            "--evidence-run-id",
            "anchor_60x2",
            "--evidence-run-id",
            "baseline_96x2",
            "--rationale",
            "CLI coverage",
            "--decision",
            "keep",
            "--surface-role",
            "classification_scaling_law",
            "--runtime-profile",
            "compile_eager_dynamic",
            "--selection-rule",
            "best_loss_healthy_only",
            "--benchmark-registry-path",
            str(tmp_path / "benchmark_run_registry_v1.json"),
            "--registry-path",
            str(tmp_path / "hardware_architecture_baselines_v1.json"),
        ]
    )

    assert exit_code == 0
    assert captured["baseline_id"] == "tf_rd_009_a100_classification_medium_v1"
    assert captured["preferred_run_id"] == "baseline_96x2"
    assert captured["formal_anchor_run_id"] == "anchor_60x2"
    assert captured["baseline_run_id"] == "baseline_96x2"
    assert captured["evidence_run_ids"] == ("anchor_60x2", "baseline_96x2")
    assert captured["runtime_profile"] == "compile_eager_dynamic"
    assert captured["benchmark_registry_path"] == tmp_path / "benchmark_run_registry_v1.json"
    assert captured["registry_path"] == tmp_path / "hardware_architecture_baselines_v1.json"
    assert "Hardware architecture baseline frozen:" in capsys.readouterr().out
