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
                    "clipped_step_fraction": 0.01,
                    "upper_block_post_warmup_mean_slope": 0.001,
                    "peak_vram_reserved": 2048,
                    "throughput_tokens_per_second": 6400.0,
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
    assert row["runtime_summary"] == {
        "peak_vram_allocated": None,
        "peak_vram_reserved": 2048,
        "throughput_examples_per_second": None,
        "throughput_tokens_per_second": 6400.0,
        "non_train_overhead_seconds": None,
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
