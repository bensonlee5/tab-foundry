from __future__ import annotations

import pytest

import tab_foundry.research.sweep.summarize as summarize_module
from tab_foundry.research.sweep.summarize import render_sweep_summary_table, summarize_sweep


def test_summarize_sweep_excludes_screened_rows_by_default() -> None:
    payload = summarize_sweep(sweep_id="cuda_stack_scale_followup")

    assert payload["sweep_id"] == "cuda_stack_scale_followup"
    assert payload["row_count"] == 1
    assert all(row["status"] != "screened" for row in payload["rows"])


def test_summarize_sweep_can_include_screened_rows() -> None:
    payload = summarize_sweep(
        sweep_id="cuda_stack_scale_followup",
        include_screened=True,
    )

    assert payload["row_count"] == 5
    assert any(row["status"] == "screened" for row in payload["rows"])
    assert any(row["stability"] == "fail" for row in payload["rows"])


def test_summarize_sweep_captures_completed_benchmark_metrics() -> None:
    payload = summarize_sweep(sweep_id="input_norm_followup")

    first_completed = next(row for row in payload["rows"] if row["status"] == "completed")

    assert first_completed["delta_id"] == "dpnb_input_norm_anchor_replay"
    assert first_completed["clipped_step_fraction"] == 0.0056
    assert first_completed["delta_final_roc_auc"] is None


def test_summarize_sweep_marks_missing_stability_metrics_as_not_available() -> None:
    payload = summarize_sweep(sweep_id="binary_md_v1")

    rows_without_stability_metrics = [
        row
        for row in payload["rows"]
        if row["status"] in {"completed", "screened"}
        and row["clipped_step_fraction"] is None
        and row["upper_block_post_warmup_mean_slope"] is None
    ]

    assert rows_without_stability_metrics
    assert all(row["stability"] == "n/a" for row in rows_without_stability_metrics)


def test_summarize_sweep_reads_archived_queue_without_live_catalog_delta() -> None:
    payload = summarize_sweep(sweep_id="missingness_followup")

    assert payload["sweep_id"] == "missingness_followup"
    assert payload["row_count"] == 2
    assert payload["rows"][0]["delta_id"] == "nan_token_no_prior_missingness"
    assert payload["rows"][0]["run_id"] is None


def test_render_sweep_summary_table_handles_empty_rows() -> None:
    rendered = render_sweep_summary_table(
        {"sweep_id": "empty_sweep", "row_count": 0, "rows": []}
    )

    assert "Sweep summary: sweep_id=empty_sweep rows=0" in rendered
    assert "delta_id" in rendered


def test_render_sweep_summary_table_uses_log_loss_delta_for_classification_objective() -> None:
    rendered = render_sweep_summary_table(
        {
            "sweep_id": "classification_sweep",
            "row_count": 1,
            "rows": [
                {
                    "order": 1,
                    "delta_id": "delta_classification",
                    "status": "completed",
                    "decision": "keep",
                    "stability": "ok",
                    "objective_metric": "final_log_loss_at_matched_regime_budget",
                    "delta_final_bpc": 12.5,
                    "delta_final_log_loss": -0.031,
                    "delta_final_roc_auc": 0.004,
                    "clipped_step_fraction": 0.01,
                    "upper_block_post_warmup_mean_slope": 0.001,
                    "run_id": "run_1",
                }
            ],
        }
    )

    assert "delta_classification" in rendered
    assert "-0.0310" in rendered
    assert "+12.5000" not in rendered


def test_summarize_sweep_preserves_runtime_and_regime_budget_fields(monkeypatch) -> None:
    queue = {
        "sweep_id": "runtime_sweep",
        "rows": [
            {
                "order": 1,
                "delta_id": "delta_runtime",
                "status": "completed",
                "decision": "keep",
                "run_id": "runtime_run",
                "benchmark_metrics": {
                    "objective_metric": "final_log_loss_at_matched_regime_budget",
                    "final_log_loss": 0.41,
                    "delta_final_log_loss": -0.02,
                    "final_train_loss": 1.43,
                    "final_train_loss_ema": 1.39,
                    "final_tail_mean_train_loss": 1.37,
                    "final_tail_mean_train_loss_ema": 1.35,
                    "final_tail_record_count": 100,
                    "clipped_step_fraction": 0.01,
                    "upper_block_post_warmup_mean_slope": 0.001,
                    "train_elapsed_seconds": 264.3,
                    "wall_elapsed_seconds": 267.4,
                    "end_to_end_wall_seconds": 273.0,
                    "loader_setup_seconds": 8.7,
                    "peak_vram_reserved": 2048,
                    "peak_vram_allocated_fraction": 0.5,
                    "peak_vram_reserved_fraction": 0.625,
                    "throughput_tokens_per_second": 6400.0,
                    "achieved_train_tflops_per_second": 4.0,
                    "theoretical_peak_tflops_per_second": 312.0,
                    "compute_utilization_fraction": 4.0 / 312.0,
                    "theoretical_hbm_bandwidth_gbps": 2039.0,
                    "roofline_knee_flops_per_byte": 153.0161844031388,
                    "peak_compute_basis": "tensorcore_bf16_dense",
                    "loader_effective_num_workers": 8,
                    "loader_effective_prefetch_factor": 4,
                    "loader_task_batch_cache_mode": "bounded_streaming",
                    "compile_shape_dispatch_mode": "signature_family",
                    "compile_shape_dispatch_max_families": 16,
                    "compile_dispatch_compiled_family_count": 16,
                    "compile_dispatch_family_switch_count": 63,
                    "one_family_step_count": 112,
                    "mixed_family_step_count": 16,
                    "consecutive_repeated_family_step_count": 48,
                    "consecutive_switched_family_step_count": 63,
                    "family_block_count": 64,
                    "estimated_family_switch_count": 63,
                    "tokens_per_step": 512.0,
                    "token_budget": 38400,
                    "unique_task_budget": 96,
                    "curriculum_id": "dagzoo_shape_aware_multi_invocation",
                },
            }
        ],
    }

    monkeypatch.setattr(
        summarize_module,
        "load_system_delta_queue_for_inspection",
        lambda **_: queue,
    )
    monkeypatch.setattr(summarize_module, "ordered_rows", lambda payload: payload["rows"])

    payload = summarize_module.summarize_sweep(sweep_id="runtime_sweep")

    assert payload["row_count"] == 1
    row = payload["rows"][0]
    assert row["final_train_loss"] == 1.43
    assert row["final_train_loss_ema"] == 1.39
    assert row["final_tail_mean_train_loss"] == 1.37
    assert row["final_tail_mean_train_loss_ema"] == 1.35
    assert row["final_tail_record_count"] == 100
    assert row["train_elapsed_seconds"] == 264.3
    assert row["wall_elapsed_seconds"] == 267.4
    assert row["end_to_end_wall_seconds"] == 273.0
    assert row["loader_setup_seconds"] == 8.7
    assert row["compile_dispatch_compiled_family_count"] == 16
    assert row["compile_dispatch_family_switch_count"] == 63
    assert row["one_family_step_count"] == 112
    assert row["mixed_family_step_count"] == 16
    assert row["consecutive_repeated_family_step_count"] == 48
    assert row["consecutive_switched_family_step_count"] == 63
    assert row["family_block_count"] == 64
    assert row["estimated_family_switch_count"] == 63
    assert row["runtime_summary"] == {
        "end_to_end_wall_seconds": 273.0,
        "loader_setup_seconds": 8.7,
        "peak_vram_allocated": None,
        "peak_vram_reserved": 2048,
        "peak_vram_allocated_fraction": 0.5,
        "peak_vram_reserved_fraction": 0.625,
        "throughput_examples_per_second": None,
        "throughput_tokens_per_second": 6400.0,
        "non_train_overhead_seconds": None,
        "non_train_overhead_fraction": None,
        "loader_effective_num_workers": 8,
        "loader_effective_prefetch_factor": 4,
        "loader_task_batch_cache_mode": "bounded_streaming",
        "compile_shape_dispatch_mode": "signature_family",
        "compile_shape_dispatch_max_families": 16,
        "compile_dispatch_compiled_family_count": 16,
        "compile_dispatch_family_switch_count": 63,
    }
    assert row["utilization_summary"] == {
        "peak_vram_allocated_fraction": 0.5,
        "peak_vram_reserved_fraction": 0.625,
        "non_train_overhead_fraction": None,
        "achieved_train_tflops_per_second": 4.0,
        "theoretical_peak_tflops_per_second": 312.0,
        "compute_utilization_fraction": 4.0 / 312.0,
        "theoretical_hbm_bandwidth_gbps": 2039.0,
        "roofline_knee_flops_per_byte": 153.0161844031388,
        "peak_compute_basis": "tensorcore_bf16_dense",
    }
    assert row["regime_budget"] == {
        "tokens_per_step": 512.0,
        "tokens_seen": None,
        "token_budget": 38400,
        "unique_task_budget": 96,
        "objective_metric": "final_log_loss_at_matched_regime_budget",
        "curriculum_id": "dagzoo_shape_aware_multi_invocation",
    }


def test_render_sweep_summary_table_includes_runtime_columns() -> None:
    rendered = render_sweep_summary_table(
        {
            "sweep_id": "runtime_sweep",
            "row_count": 1,
            "rows": [
                {
                    "order": 1,
                    "delta_id": "delta_runtime",
                    "status": "completed",
                    "decision": "keep",
                    "stability": "ok",
                    "objective_metric": "final_log_loss_at_matched_regime_budget",
                    "delta_final_log_loss": -0.031,
                    "delta_final_roc_auc": 0.004,
                    "throughput_tokens_per_second": 6400.0,
                    "peak_vram_reserved": 2048,
                    "tokens_per_step": 512.0,
                    "clipped_step_fraction": 0.01,
                    "upper_block_post_warmup_mean_slope": 0.001,
                    "run_id": "run_1",
                }
            ],
        }
    )

    assert "tok/s" in rendered
    assert "vram_rsv" in rendered
    assert "tok/step" in rendered
    assert "6400.0000" in rendered
    assert "2.0KiB" in rendered
    assert "512.0000" in rendered


def _selector_row(
    *,
    order: int,
    d_icl: int,
    layers: int,
    prescription: str,
    final_log_loss: float,
    end_to_end_wall_seconds: float | None,
) -> dict[str, object]:
    return {
        "order": order,
        "delta_id": (
            f"delta_tf_rd_009_cls_sandwich_dicl{d_icl}_layers{layers}_muon_"
            f"{prescription}_v1"
        ),
        "status": "completed",
        "decision": "defer",
        "run_id": f"run_{order}",
        "model": {
            "d_icl": d_icl,
            "sandwich_layers": layers,
        },
        "benchmark_metrics": {
            "objective_metric": "final_log_loss_at_matched_regime_budget",
            "final_log_loss": final_log_loss,
            "delta_final_log_loss": 0.0,
            "end_to_end_wall_seconds": end_to_end_wall_seconds,
            "throughput_tokens_per_second": 1000.0 + float(order),
        },
    }


def test_summarize_sweep_reports_selector_kept_contract_with_time_tiebreak(
    monkeypatch,
) -> None:
    queue = {
        "sweep_id": "selector_sweep",
        "surface_role": "classification_training_dynamics_selector",
        "rows": [
            _selector_row(order=1, d_icl=128, layers=2, prescription="carry_lowbatch", final_log_loss=0.55, end_to_end_wall_seconds=100.0),
            _selector_row(order=2, d_icl=128, layers=2, prescription="carry_highbatch", final_log_loss=0.53, end_to_end_wall_seconds=160.0),
            _selector_row(order=3, d_icl=128, layers=2, prescription="linear_lr_batch", final_log_loss=0.50, end_to_end_wall_seconds=150.0),
            _selector_row(order=4, d_icl=128, layers=2, prescription="momentum_timescale", final_log_loss=0.51, end_to_end_wall_seconds=170.0),
            _selector_row(order=5, d_icl=144, layers=4, prescription="carry_lowbatch", final_log_loss=0.53, end_to_end_wall_seconds=110.0),
            _selector_row(order=6, d_icl=144, layers=4, prescription="carry_highbatch", final_log_loss=0.50, end_to_end_wall_seconds=180.0),
            _selector_row(order=7, d_icl=144, layers=4, prescription="linear_lr_batch", final_log_loss=0.48, end_to_end_wall_seconds=170.0),
            _selector_row(order=8, d_icl=144, layers=4, prescription="momentum_timescale", final_log_loss=0.49, end_to_end_wall_seconds=190.0),
            _selector_row(order=9, d_icl=264, layers=6, prescription="carry_lowbatch", final_log_loss=0.51, end_to_end_wall_seconds=130.0),
            _selector_row(order=10, d_icl=264, layers=6, prescription="carry_highbatch", final_log_loss=0.49, end_to_end_wall_seconds=230.0),
            _selector_row(order=11, d_icl=264, layers=6, prescription="linear_lr_batch", final_log_loss=0.46, end_to_end_wall_seconds=220.0),
            _selector_row(order=12, d_icl=264, layers=6, prescription="momentum_timescale", final_log_loss=0.47, end_to_end_wall_seconds=240.0),
        ],
    }

    monkeypatch.setattr(
        summarize_module,
        "load_system_delta_queue_for_inspection",
        lambda **_: queue,
    )
    monkeypatch.setattr(summarize_module, "ordered_rows", lambda payload: payload["rows"])

    payload = summarize_module.summarize_sweep(sweep_id="selector_sweep")

    selector_summary = payload["selector_summary"]
    assert selector_summary is not None
    assert payload["rows"][0]["pareto_admissible"] is True
    assert payload["rows"][2]["pareto_admissible"] is True
    assert payload["rows"][1]["pareto_admissible"] is False
    assert selector_summary["best_row"]["order"] == 11
    assert selector_summary["kept_contract"]["prescription_label"] == "carry_lowbatch"
    assert selector_summary["kept_contract"]["geometry_count"] == 3
    assert selector_summary["prescription_coverage"][0]["prescription_label"] == "carry_lowbatch"
    rendered = summarize_module.render_sweep_summary_table(payload)
    assert "Pareto frontier:" in rendered
    assert "kept contract: carry_lowbatch" in rendered


def test_summarize_sweep_reports_no_universal_selector_contract_when_no_majority(
    monkeypatch,
) -> None:
    queue = {
        "sweep_id": "selector_sweep_fragmented",
        "surface_role": "classification_training_dynamics_selector",
        "rows": [
            _selector_row(order=1, d_icl=128, layers=2, prescription="carry_lowbatch", final_log_loss=0.55, end_to_end_wall_seconds=None),
            _selector_row(order=2, d_icl=128, layers=2, prescription="carry_highbatch", final_log_loss=0.54, end_to_end_wall_seconds=None),
            _selector_row(order=3, d_icl=128, layers=2, prescription="linear_lr_batch", final_log_loss=0.50, end_to_end_wall_seconds=150.0),
            _selector_row(order=4, d_icl=128, layers=2, prescription="momentum_timescale", final_log_loss=0.51, end_to_end_wall_seconds=None),
            _selector_row(order=5, d_icl=144, layers=4, prescription="carry_lowbatch", final_log_loss=0.53, end_to_end_wall_seconds=None),
            _selector_row(order=6, d_icl=144, layers=4, prescription="carry_highbatch", final_log_loss=0.52, end_to_end_wall_seconds=None),
            _selector_row(order=7, d_icl=144, layers=4, prescription="linear_lr_batch", final_log_loss=0.49, end_to_end_wall_seconds=None),
            _selector_row(order=8, d_icl=144, layers=4, prescription="momentum_timescale", final_log_loss=0.48, end_to_end_wall_seconds=190.0),
            _selector_row(order=9, d_icl=264, layers=6, prescription="carry_lowbatch", final_log_loss=0.51, end_to_end_wall_seconds=None),
            _selector_row(order=10, d_icl=264, layers=6, prescription="carry_highbatch", final_log_loss=0.47, end_to_end_wall_seconds=205.0),
            _selector_row(order=11, d_icl=264, layers=6, prescription="linear_lr_batch", final_log_loss=0.46, end_to_end_wall_seconds=None),
            _selector_row(order=12, d_icl=264, layers=6, prescription="momentum_timescale", final_log_loss=0.45, end_to_end_wall_seconds=None),
        ],
    }

    monkeypatch.setattr(
        summarize_module,
        "load_system_delta_queue_for_inspection",
        lambda **_: queue,
    )
    monkeypatch.setattr(summarize_module, "ordered_rows", lambda payload: payload["rows"])

    payload = summarize_module.summarize_sweep(sweep_id="selector_sweep_fragmented")

    selector_summary = payload["selector_summary"]
    assert selector_summary is not None
    assert selector_summary["kept_contract"] is None
    assert selector_summary["no_universal_kept_contract"] is True
    assert [
        coverage["prescription_label"]
        for coverage in selector_summary["prescription_coverage"]
    ] == ["linear_lr_batch", "momentum_timescale", "carry_highbatch"]


def _transfer_row(
    *,
    order: int,
    regime_label: str,
    target_budget_label: str,
    final_log_loss: float,
    end_to_end_wall_seconds: float | None,
    imported: bool = False,
    shared_anchor: bool = False,
) -> dict[str, object]:
    row: dict[str, object] = {
        "order": order,
        "delta_id": f"delta_transfer_{order}",
        "status": "completed",
        "decision": "keep",
        "run_id": f"transfer_run_{order}",
        "benchmark_metrics": {
            "objective_metric": "final_log_loss_at_matched_regime_budget",
            "final_log_loss": final_log_loss,
            "delta_final_log_loss": 0.0,
            "end_to_end_wall_seconds": end_to_end_wall_seconds,
            "throughput_tokens_per_second": 2000.0 + float(order),
        },
        "transfer_context": {
            "regime_label": regime_label,
            "phase": "validation",
            "formula_label": "Theorem 2 fixed-batch transfer" if regime_label == "B" else "baseline import",
            "base_budget_label": "T0",
            "target_budget_label": target_budget_label,
            "target_effective_batch": 64,
            "realized_effective_batch": 64,
            "target_effective_budget": {"T0": 40000, "T1": 160000, "T2": 320000}[target_budget_label],
            "realized_effective_budget": {"T0": 40000, "T1": 160000, "T2": 320000}[target_budget_label],
            "budget_drift": 0.0,
            "batch_drift": 0.0,
        },
    }
    if imported:
        row["imported_baseline_provenance"] = {
            "source_sweep_id": "tf_rd_009_muon_ns_one_epoch_medium_v1",
            "source_order": order,
        }
    if shared_anchor:
        row["transfer_resolution"] = {
            "shared_anchor_provenance": {
                "anchor_sweep_id": "tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1",
                "anchor_order": 1,
            }
        }
    return row


def test_summarize_sweep_reports_transfer_regime_leaderboard(monkeypatch) -> None:
    queue = {
        "sweep_id": "transfer_sweep",
        "surface_role": "classification_training_dynamics_transfer",
        "rows": [
            _transfer_row(order=1, regime_label="carry_lowbatch", target_budget_label="T0", final_log_loss=0.49, end_to_end_wall_seconds=5400.0, imported=True),
            _transfer_row(order=2, regime_label="carry_lowbatch", target_budget_label="T1", final_log_loss=0.47, end_to_end_wall_seconds=5500.0, imported=True),
            _transfer_row(order=4, regime_label="carry_highbatch", target_budget_label="T2", final_log_loss=0.44, end_to_end_wall_seconds=6300.0),
            _transfer_row(order=7, regime_label="B", target_budget_label="T1", final_log_loss=0.43, end_to_end_wall_seconds=7100.0, shared_anchor=True),
            _transfer_row(order=8, regime_label="B", target_budget_label="T2", final_log_loss=0.41, end_to_end_wall_seconds=7200.0, shared_anchor=True),
            _transfer_row(order=9, regime_label="D", target_budget_label="T1", final_log_loss=0.435, end_to_end_wall_seconds=6800.0, shared_anchor=True),
            _transfer_row(order=10, regime_label="D", target_budget_label="T2", final_log_loss=0.425, end_to_end_wall_seconds=6900.0, shared_anchor=True),
        ],
    }

    monkeypatch.setattr(
        summarize_module,
        "load_system_delta_queue_for_inspection",
        lambda **_: queue,
    )
    monkeypatch.setattr(summarize_module, "ordered_rows", lambda payload: payload["rows"])

    payload = summarize_module.summarize_sweep(sweep_id="transfer_sweep")

    transfer_summary = payload["transfer_summary"]
    assert transfer_summary is not None
    assert transfer_summary["best_row"]["order"] == 8
    assert transfer_summary["best_row"]["regime_label"] == "B"
    assert transfer_summary["fastest_row"]["order"] == 1
    assert transfer_summary["imported_baseline_orders"] == [1, 2]
    assert [entry["regime_label"] for entry in transfer_summary["regime_leaderboard"]] == [
        "B",
        "D",
        "carry_highbatch",
        "carry_lowbatch",
    ]
    assert transfer_summary["kept_regime"]["winner_rule"] == "T2_then_T1"
    assert transfer_summary["kept_regime"]["regime_label"] == "B"
    assert transfer_summary["kept_regime"]["t1_order"] == 7
    assert transfer_summary["kept_regime"]["t1_log_loss"] == pytest.approx(0.43)
    assert transfer_summary["kept_regime"]["t2_order"] == 8
    assert transfer_summary["kept_regime"]["t2_log_loss"] == pytest.approx(0.41)
    assert transfer_summary["kept_regime"]["runner_up_regime_label"] == "D"
    assert transfer_summary["kept_regime"]["runner_up_t1_order"] == 9
    assert transfer_summary["kept_regime"]["runner_up_t1_log_loss"] == pytest.approx(0.435)
    assert transfer_summary["kept_regime"]["runner_up_t2_order"] == 10
    assert transfer_summary["kept_regime"]["runner_up_t2_log_loss"] == pytest.approx(0.425)
    assert transfer_summary["t2_vs_carried_highbatch"]["winning_regime_label"] == "B"
    assert transfer_summary["t2_vs_carried_highbatch"]["winning_regime_order"] == 8
    assert transfer_summary["t2_vs_carried_highbatch"]["winning_regime_t2_log_loss"] == pytest.approx(0.41)
    assert transfer_summary["t2_vs_carried_highbatch"]["carried_highbatch_order"] == 4
    assert transfer_summary["t2_vs_carried_highbatch"]["carried_highbatch_t2_log_loss"] == pytest.approx(0.44)
    assert transfer_summary["t2_vs_carried_highbatch"]["delta_log_loss"] == pytest.approx(-0.03)

    rendered = summarize_module.render_sweep_summary_table(payload)
    assert "Transfer summary:" in rendered
    assert "best transfer row: order 08, regime=B" in rendered
    assert "imported baseline orders: 01, 02" in rendered
    assert "kept regime (T2 then T1): B" in rendered
    assert "T2 vs carried high-batch: B delta_log_loss=-0.030000" in rendered
