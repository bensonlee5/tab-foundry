from __future__ import annotations

from tab_foundry.research.sweep.reporting import result_card_text


def _classification_queue_metrics() -> dict[str, float | int | str]:
    return {
        "objective_metric": "final_log_loss_at_matched_regime_budget",
        "best_step": 125,
        "primary_external_label": "TabICLv2",
        "primary_external_best_log_loss": 0.392,
        "primary_external_final_log_loss": 0.401,
        "primary_external_best_roc_auc": 0.818,
        "primary_external_final_roc_auc": 0.811,
        "best_log_loss": 0.401,
        "final_log_loss": 0.409,
        "final_minus_best_log_loss": 0.008,
        "delta_final_log_loss": -0.011,
        "final_brier_score": 0.118,
        "delta_final_brier_score": -0.004,
        "best_roc_auc": 0.812,
        "final_roc_auc": 0.804,
        "delta_final_roc_auc": -0.006,
        "best_bpc": 2.011,
        "final_bpc": 2.037,
        "final_minus_best_bpc": 0.026,
        "delta_final_bpc": 0.013,
        "best_bpf": 2.007,
        "final_bpf": 2.028,
        "final_minus_best_bpf": 0.021,
        "delta_final_bpf": 0.012,
        "peak_vram_reserved": 2048,
        "throughput_tokens_per_second": 6400.0,
        "tokens_per_step": 512.0,
        "token_budget": 38400,
        "unique_task_budget": 96,
        "curriculum_id": "dagzoo_shape_aware_multi_invocation",
    }


def _classification_summary() -> dict[str, object]:
    return {
        "primary_external_benchmark": "tabiclv2",
        "tab_foundry": {},
        "tabiclv2": {},
    }


def test_result_card_text_reports_log_loss_before_roc_for_classification_objectives() -> None:
    text = result_card_text(
        row={
            "delta_id": "delta",
            "description": "Use the refreshed benchmark surface.",
            "anchor_delta": "anchor-only comparison.",
        },
        run_id="sd_test_v1",
        anchor_run_id="anchor_v1",
        summary=_classification_summary(),
        queue_metrics=_classification_queue_metrics(),
        decision="defer",
        conclusion="Monitor log-loss deltas before promotion.",
    )

    assert "- Best log loss: `0.4010` at step `125`" in text
    assert text.index("Best log loss") < text.index("Best ROC AUC")


def test_result_card_text_marks_classification_bpc_metrics_as_legacy_diagnostics() -> None:
    text = result_card_text(
        row={
            "delta_id": "delta",
            "description": "Use the refreshed benchmark surface.",
            "anchor_delta": "anchor-only comparison.",
        },
        run_id="sd_test_v1",
        anchor_run_id="anchor_v1",
        summary=_classification_summary(),
        queue_metrics=_classification_queue_metrics(),
        decision="defer",
        conclusion="Monitor log-loss deltas before promotion.",
    )

    assert "Legacy feature-cell diagnostics use normalized benchmark inputs" in text
    assert "- Final BPC (legacy feature-cell diagnostic): `2.0370`" in text
    assert "- Final BPF (legacy feature-cell diagnostic): `2.0280`" in text
    assert text.index("Best log loss") < text.index("Final ROC AUC")
    assert text.index("Final ROC AUC") < text.index("Final BPC (legacy feature-cell diagnostic)")
    assert "## Runtime and regime budget" in text
    assert "- Throughput tokens/sec: `6400.0000`" in text
    assert "- Peak VRAM reserved: `2048`" in text
    assert "- Token budget: `38400`" in text
    assert "- Curriculum id: `dagzoo_shape_aware_multi_invocation`" in text
