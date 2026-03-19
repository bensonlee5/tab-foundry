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


def test_render_sweep_summary_table_handles_empty_rows() -> None:
    rendered = render_sweep_summary_table(
        {"sweep_id": "empty_sweep", "row_count": 0, "rows": []}
    )

    assert "Sweep summary: sweep_id=empty_sweep rows=0" in rendered
    assert "delta_id" in rendered
