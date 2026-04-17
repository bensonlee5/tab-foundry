from __future__ import annotations

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
                    "throughput_tokens_per_second": 6400.0,
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
        "throughput_examples_per_second": None,
        "throughput_tokens_per_second": 6400.0,
        "non_train_overhead_seconds": None,
        "loader_effective_num_workers": 8,
        "loader_effective_prefetch_factor": 4,
        "loader_task_batch_cache_mode": "bounded_streaming",
        "compile_shape_dispatch_mode": "signature_family",
        "compile_shape_dispatch_max_families": 16,
        "compile_dispatch_compiled_family_count": 16,
        "compile_dispatch_family_switch_count": 63,
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
