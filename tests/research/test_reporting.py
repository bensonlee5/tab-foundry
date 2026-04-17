from __future__ import annotations

import json
from pathlib import Path

import tab_foundry.research.sweep.reporting as reporting_module
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


def test_result_card_text_includes_selector_interpretation_when_provided() -> None:
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
        selector_context={
            "row_summary": {
                "pareto_admissible": True,
                "geometry_pareto_admissible": False,
                "selector_geometry_label": "144x4",
                "selector_prescription_label": "linear_lr_batch",
            },
            "summary": {
                "best_row": {
                    "order": 10,
                    "geometry_label": "264x6",
                    "prescription_label": "linear_lr_batch",
                    "final_log_loss": 0.46,
                    "end_to_end_wall_seconds": 222.0,
                },
                "kept_contract": {
                    "prescription_label": "carry_lowbatch",
                    "geometry_count": 3,
                    "mean_end_to_end_wall_seconds": 115.0,
                    "mean_benchmark_log_loss": 0.52,
                },
            },
        },
    )

    assert "## Selector interpretation" in text
    assert "- Quality/time Pareto admissible: `yes`" in text
    assert "- Geometry-local Pareto admissible: `no`" in text
    assert "- Selector geometry: `144x4`" in text
    assert "- Selector prescription: `linear_lr_batch`" in text
    assert "- Kept contract: `carry_lowbatch`" in text


def test_refresh_result_cards_for_queue_rewrites_completed_cards_with_selector_context(
    tmp_path: Path,
    monkeypatch,
) -> None:
    repo_root = tmp_path / "repo"
    comparison_summary_path = tmp_path / "comparison_summary.json"
    comparison_summary_path.write_text(
        json.dumps(_classification_summary()),
        encoding="utf-8",
    )
    queue = {
        "sweep_id": "selector_sweep",
        "anchor_run_id": "anchor_v1",
        "rows": [
            {
                "order": 1,
                "delta_id": "delta_selector",
                "status": "completed",
                "run_id": "run_1",
                "decision": "keep",
                "conclusion": "Keep the carried low-batch selector contract.",
                "description": "Selector replay row.",
                "anchor_delta": "anchor-only comparison.",
                "benchmark_metrics": _classification_queue_metrics(),
            }
        ],
    }

    monkeypatch.setattr(reporting_module, "repo_root", lambda: repo_root)
    monkeypatch.setattr(
        reporting_module,
        "load_benchmark_run_registry",
        lambda _path: {
            "runs": {
                "run_1": {
                    "artifacts": {
                        "comparison_summary_path": str(comparison_summary_path),
                    }
                }
            }
        },
    )
    monkeypatch.setattr(
        reporting_module,
        "resolve_registry_path_value",
        lambda value: Path(str(value)),
    )

    import tab_foundry.research.sweep.summarize as summarize_module

    monkeypatch.setattr(
        summarize_module,
        "build_sweep_summary_payload",
        lambda **_: {
            "rows": [
                {
                    "order": 1,
                    "pareto_admissible": True,
                    "geometry_pareto_admissible": True,
                    "selector_geometry_label": "128x2",
                    "selector_prescription_label": "carry_lowbatch",
                }
            ],
            "selector_summary": {
                "best_row": {
                    "order": 1,
                    "geometry_label": "128x2",
                    "prescription_label": "carry_lowbatch",
                    "final_log_loss": 0.50,
                    "end_to_end_wall_seconds": 100.0,
                },
                "kept_contract": {
                    "prescription_label": "carry_lowbatch",
                    "geometry_count": 3,
                    "mean_end_to_end_wall_seconds": 105.0,
                    "mean_benchmark_log_loss": 0.51,
                },
            },
        },
    )

    reporting_module.refresh_result_cards_for_queue(queue=queue)

    result_card_path = (
        repo_root
        / "outputs"
        / "staged_ladder"
        / "research"
        / "selector_sweep"
        / "delta_selector"
        / "result_card.md"
    )
    text = result_card_path.read_text(encoding="utf-8")
    assert "## Selector interpretation" in text
    assert "- Selector prescription: `carry_lowbatch`" in text
    assert "- Kept contract: `carry_lowbatch`" in text
