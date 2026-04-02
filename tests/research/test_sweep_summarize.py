from __future__ import annotations

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
