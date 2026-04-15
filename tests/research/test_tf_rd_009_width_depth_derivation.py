from __future__ import annotations

import json
from pathlib import Path

import pytest

from tab_foundry.research.tf_rd_009_width_depth_derivation import (
    collect_tf_rd_009_completed_measured_fit_points,
    collect_tf_rd_009_muon_completed_measured_fit_points,
    derive_tf_rd_009_muon_width_depth_family,
    derive_tf_rd_009_width_depth_family,
    fit_tf_rd_009_completed_measured_power_law,
    fit_tf_rd_009_muon_completed_measured_power_law,
)


def _run_entry(
    run_id: str,
    *,
    delta_id: str,
    d_icl: int,
    layers: int,
    total_params: int,
    final_log_loss: float,
) -> dict[str, object]:
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
            "anchor_run_id": "anchor_60x2",
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
            "train_elapsed_seconds": 100.0,
            "wall_elapsed_seconds": 100.0,
        },
        "runtime_summary": {
            "peak_vram_allocated": 1,
            "peak_vram_reserved": 1,
            "throughput_examples_per_second": 18.0,
            "throughput_tokens_per_second": 100000.0,
            "non_train_overhead_seconds": 10.0,
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
    payload = {
        "schema": "tab-foundry-benchmark-runs-v1",
        "version": 1,
        "runs": {
            "baseline_96x2": _run_entry(
                "baseline_96x2",
                delta_id="delta_tf_rd_009_cls_sandwich_dicl96_v1",
                d_icl=96,
                layers=2,
                total_params=1618286,
                final_log_loss=0.6331,
            ),
            "joint_72x1": _run_entry(
                "joint_72x1",
                delta_id="delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1",
                d_icl=72,
                layers=1,
                total_params=671809,
                final_log_loss=0.6410,
            ),
            "historical_88x1": _run_entry(
                "historical_88x1",
                delta_id="delta_tf_rd_009_cls_sandwich_dicl88_layers1_v1",
                d_icl=88,
                layers=1,
                total_params=986886,
                final_log_loss=0.6370,
            ),
            "pending_112x3": _run_entry(
                "pending_112x3",
                delta_id="delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1",
                d_icl=112,
                layers=3,
                total_params=2798089,
                final_log_loss=0.6290,
            ),
            "muon_baseline_128x2": _run_entry(
                "muon_baseline_128x2",
                delta_id="delta_tf_rd_009_cls_sandwich_dicl128_v1",
                d_icl=128,
                layers=2,
                total_params=2849422,
                final_log_loss=0.6100,
            ),
            "muon_joint_72x1": _run_entry(
                "muon_joint_72x1",
                delta_id="delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1",
                d_icl=72,
                layers=1,
                total_params=671184,
                final_log_loss=0.6310,
            ),
            "muon_joint_144x4": _run_entry(
                "muon_joint_144x4",
                delta_id="delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1",
                d_icl=144,
                layers=4,
                total_params=5610697,
                final_log_loss=0.5980,
            ),
        },
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_tf_rd_009_queue_derivation_matches_canonical_family() -> None:
    derivation = derive_tf_rd_009_width_depth_family()

    assert derivation.in_family_row_labels == (
        "72x1",
        "96x2",
        "112x3",
        "128x4",
        "152x5",
        "176x6",
    )
    assert [row.delta_id for row in derivation.queue_rows] == [
        "delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1",
        "delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1",
        "delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1",
        "delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1",
        "delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1",
    ]


def test_tf_rd_009_queue_derivation_matches_note_values() -> None:
    derivation = derive_tf_rd_009_width_depth_family()

    assert derivation.lower_seed.raw_d_icl == pytest.approx(70.59, abs=0.05)
    assert derivation.upper_seed.raw_d_icl == pytest.approx(113.03, abs=0.05)
    assert derivation.interpolated_rows[0].raw_d_icl == pytest.approx(128.37, abs=0.1)
    assert derivation.interpolated_rows[1].raw_d_icl == pytest.approx(149.46, abs=0.1)
    assert derivation.ceiling_probe.raw_d_icl == pytest.approx(173.52, abs=0.1)

    assert derivation.lower_seed.predicted_total_params == pytest.approx(671809.30, abs=1.0)
    assert derivation.upper_seed.predicted_total_params == pytest.approx(2798089.49, abs=1.0)
    assert derivation.interpolated_rows[0].predicted_total_params == pytest.approx(4438957.75, abs=1.0)
    assert derivation.interpolated_rows[1].predicted_total_params == pytest.approx(7366269.02, abs=1.0)
    assert derivation.ceiling_probe.predicted_total_params == pytest.approx(11366075.68, abs=1.0)

    assert derivation.lower_seed.predicted_reserved_vram_gb == pytest.approx(8.05, abs=0.01)
    assert derivation.upper_seed.predicted_reserved_vram_gb == pytest.approx(13.06, abs=0.01)
    assert derivation.interpolated_rows[0].predicted_reserved_vram_gb == pytest.approx(16.93, abs=0.01)
    assert derivation.interpolated_rows[1].predicted_reserved_vram_gb == pytest.approx(23.82, abs=0.01)
    assert derivation.ceiling_probe.predicted_reserved_vram_gb == pytest.approx(33.25, abs=0.01)


def test_tf_rd_009_muon_queue_derivation_matches_canonical_family() -> None:
    derivation = derive_tf_rd_009_muon_width_depth_family()

    assert derivation.formal_anchor.row_label == "60x2"
    assert derivation.formal_anchor.total_params == 646970
    assert derivation.carried_baseline.row_label == "128x2"
    assert derivation.carried_baseline.total_params == 2849422
    assert derivation.in_family_row_labels == (
        "72x1",
        "112x3",
        "128x2",
        "144x4",
        "192x5",
        "264x6",
    )
    assert [row.delta_id for row in derivation.queue_rows] == [
        "delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1",
        "delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1",
        "delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1",
        "delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1",
        "delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1",
    ]


def test_tf_rd_009_muon_queue_derivation_matches_note_values() -> None:
    derivation = derive_tf_rd_009_muon_width_depth_family()

    assert derivation.lower_seed.raw_d_icl == pytest.approx(70.65, abs=0.05)
    assert derivation.upper_seed.raw_d_icl == pytest.approx(112.99, abs=0.05)
    assert derivation.interpolated_rows[0].raw_d_icl == pytest.approx(147.01, abs=0.1)
    assert derivation.interpolated_rows[1].raw_d_icl == pytest.approx(195.92, abs=0.1)
    assert derivation.ceiling_probe.raw_d_icl == pytest.approx(264.94, abs=0.1)

    assert derivation.lower_seed.predicted_total_params == pytest.approx(671184.20, abs=1.0)
    assert derivation.upper_seed.predicted_total_params == pytest.approx(2800204.87, abs=1.0)
    assert derivation.interpolated_rows[0].predicted_total_params == pytest.approx(5610696.53, abs=1.0)
    assert derivation.interpolated_rows[1].predicted_total_params == pytest.approx(11727112.47, abs=1.0)
    assert derivation.ceiling_probe.predicted_total_params == pytest.approx(25495777.49, abs=1.0)

    assert derivation.lower_seed.predicted_reserved_vram_gb == pytest.approx(9.32, abs=0.01)
    assert derivation.upper_seed.predicted_reserved_vram_gb == pytest.approx(11.29, abs=0.01)
    assert derivation.interpolated_rows[0].predicted_reserved_vram_gb == pytest.approx(13.89, abs=0.01)
    assert derivation.interpolated_rows[1].predicted_reserved_vram_gb == pytest.approx(19.57, abs=0.01)
    assert derivation.ceiling_probe.predicted_reserved_vram_gb == pytest.approx(32.33, abs=0.01)


def test_collect_tf_rd_009_measured_fit_points_uses_completed_in_family_rows_only(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "benchmark_run_registry_v1.json"
    _write_benchmark_registry(registry_path)
    queue = {
        "anchor_run_id": "baseline_96x2",
        "rows": [
            {
                "order": 1,
                "delta_id": "delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1",
                "model": {"d_icl": 72, "sandwich_layers": 1},
                "run_id": "joint_72x1",
                "interpretation_status": "completed",
                "benchmark_metrics": {
                    "objective_metric": "final_log_loss_at_matched_regime_budget",
                    "final_log_loss": 0.7000,
                },
            },
            {
                "order": 2,
                "delta_id": "delta_tf_rd_009_cls_sandwich_dicl88_layers1_v1",
                "model": {"d_icl": 88, "sandwich_layers": 1},
                "run_id": "historical_88x1",
                "interpretation_status": "completed",
                "benchmark_metrics": {
                    "objective_metric": "final_log_loss_at_matched_regime_budget",
                    "final_log_loss": 0.6370,
                },
            },
            {
                "order": 3,
                "delta_id": "delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1",
                "model": {"d_icl": 112, "sandwich_layers": 3},
                "run_id": "pending_112x3",
                "interpretation_status": "pending",
                "benchmark_metrics": {
                    "objective_metric": "final_log_loss_at_matched_regime_budget",
                    "final_log_loss": 0.6290,
                },
            },
        ],
    }

    points = collect_tf_rd_009_completed_measured_fit_points(
        queue=queue,
        registry_path=registry_path,
    )

    assert [point.row_label for point in points] == ["72x1", "96x2"]
    assert [point.total_params for point in points] == [671809, 1618286]
    assert [point.final_log_loss for point in points] == pytest.approx([0.6410, 0.6331])


def test_fit_tf_rd_009_completed_measured_power_law_uses_registry_params_and_power_law_first(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "benchmark_run_registry_v1.json"
    _write_benchmark_registry(registry_path)
    queue = {
        "anchor_run_id": "baseline_96x2",
        "rows": [
            {
                "order": 1,
                "delta_id": "delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1",
                "model": {"d_icl": 72, "sandwich_layers": 1},
                "run_id": "joint_72x1",
                "interpretation_status": "completed",
                "benchmark_metrics": {
                    "objective_metric": "final_log_loss_at_matched_regime_budget",
                    "final_log_loss": 0.7000,
                },
            }
        ],
    }

    result = fit_tf_rd_009_completed_measured_power_law(
        queue=queue,
        registry_path=registry_path,
    )

    assert result.x_axis == "model_size.total_params"
    assert result.y_axis == "final_log_loss_at_matched_regime_budget"
    assert result.fit_family == "power_law_log_log"
    assert result.fit.fit_kind == "power_law"
    assert [point.row_label for point in result.points] == ["72x1", "96x2"]
    assert result.fit.predict(671809.0) == pytest.approx(0.6410, rel=1.0e-9)
    assert result.fit.predict(1618286.0) == pytest.approx(0.6331, rel=1.0e-9)


def test_collect_tf_rd_009_muon_measured_fit_points_use_baseline_and_completed_rows_only(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "benchmark_run_registry_v1.json"
    _write_benchmark_registry(registry_path)
    queue = {
        "anchor_run_id": "muon_baseline_128x2",
        "rows": [
            {
                "order": 1,
                "delta_id": "delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1",
                "model": {"d_icl": 72, "sandwich_layers": 1},
                "run_id": "muon_joint_72x1",
                "interpretation_status": "completed",
                "benchmark_metrics": {
                    "objective_metric": "final_log_loss_at_matched_regime_budget",
                    "final_log_loss": 0.6310,
                },
            },
            {
                "order": 2,
                "delta_id": "delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1",
                "model": {"d_icl": 112, "sandwich_layers": 3},
                "run_id": "pending_112x3",
                "interpretation_status": "pending",
                "benchmark_metrics": {
                    "objective_metric": "final_log_loss_at_matched_regime_budget",
                    "final_log_loss": 0.6290,
                },
            },
            {
                "order": 3,
                "delta_id": "delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1",
                "model": {"d_icl": 144, "sandwich_layers": 4},
                "run_id": "muon_joint_144x4",
                "interpretation_status": "completed",
                "benchmark_metrics": {
                    "objective_metric": "final_log_loss_at_matched_regime_budget",
                    "final_log_loss": 0.5980,
                },
            },
        ],
    }

    points = collect_tf_rd_009_muon_completed_measured_fit_points(
        queue=queue,
        registry_path=registry_path,
    )

    assert [point.row_label for point in points] == ["72x1", "128x2", "144x4"]
    assert [point.total_params for point in points] == [671184, 2849422, 5610697]
    assert [point.final_log_loss for point in points] == pytest.approx([0.6310, 0.6100, 0.5980])


def test_fit_tf_rd_009_muon_completed_measured_power_law_uses_muon_fit_order(
    tmp_path: Path,
) -> None:
    registry_path = tmp_path / "benchmark_run_registry_v1.json"
    _write_benchmark_registry(registry_path)
    queue = {
        "anchor_run_id": "muon_baseline_128x2",
        "rows": [
            {
                "order": 1,
                "delta_id": "delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1",
                "model": {"d_icl": 72, "sandwich_layers": 1},
                "run_id": "muon_joint_72x1",
                "interpretation_status": "completed",
                "benchmark_metrics": {
                    "objective_metric": "final_log_loss_at_matched_regime_budget",
                    "final_log_loss": 0.6310,
                },
            }
        ],
    }

    result = fit_tf_rd_009_muon_completed_measured_power_law(
        queue=queue,
        registry_path=registry_path,
    )

    assert result.fit.fit_kind == "power_law"
    assert [point.row_label for point in result.points] == ["72x1", "128x2"]
    assert result.fit.predict(671184.0) == pytest.approx(0.6310, rel=1.0e-9)
    assert result.fit.predict(2849422.0) == pytest.approx(0.6100, rel=1.0e-9)
