from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.system_delta import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
ANCHOR_RUN_ID = "sd_cuda_stability_followup_01_dpnb_cuda_large_anchor_batch32_replay_v1"
EXPECTED_ROWS = [
    "dpnb_cuda_stack_scale_control",
    "dpnb_cuda_stack_scale_poststack_rms",
    "dpnb_cuda_stack_scale_poststack_ln",
    "dpnb_cuda_stack_scale_depth_scaled",
    "dpnb_cuda_stack_scale_depth_scaled_plus_norm_winner",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _row_by_ref(queue: dict[str, Any], delta_ref: str) -> dict[str, Any]:
    rows = queue["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["delta_ref"] == delta_ref)


def test_cuda_stack_scale_followup_is_registered_and_selected() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["active_sweep_id"] == "cuda_stack_scale_followup"

    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps["cuda_stack_scale_followup"] == {
        "parent_sweep_id": "cuda_stability_followup",
        "status": "completed",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "binary_md",
        "benchmark_bundle_path": "src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
        "control_baseline_id": "cls_benchmark_linear_v2",
    }


def test_cuda_stack_scale_followup_metadata_and_rows_match_the_executed_screen_results() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / "cuda_stack_scale_followup"
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == "cuda_stack_scale_followup"
    assert sweep["parent_sweep_id"] == "cuda_stability_followup"
    assert sweep["status"] == "completed"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert sweep["training_experiment"] == "cls_benchmark_staged_prior"
    assert sweep["training_config_profile"] == "cls_benchmark_staged_prior"
    assert sweep["surface_role"] == "hybrid_diagnostic"
    assert sweep["anchor_context"]["run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["model"]["stage_label"] == "dpnb_cuda_large_anchor_batch32_replay"
    assert any("train-only screens" in note for note in sweep["anchor_surface"]["notes"])
    assert any("RMSNorm winning any remaining tie" in note for note in sweep["anchor_surface"]["notes"])

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["screened", "screened", "screened", "screened", "blocked"]
    assert [row["execution_policy"] for row in rows] == [
        "screen_only",
        "screen_only",
        "screen_only",
        "screen_only",
        "benchmark_full",
    ]

    row1 = _row_by_ref(queue, "dpnb_cuda_stack_scale_control")
    assert row1["training"]["overrides"]["runtime"]["max_steps"] == 1000
    assert row1["training"]["overrides"]["schedule"]["stages"][0]["steps"] == 1000
    assert row1["interpretation_status"] == "interpreted"
    assert row1["run_id"] == "sd_cuda_stack_scale_followup_01_dpnb_cuda_stack_scale_control_v1"
    assert row1["screen_metrics"]["upper_block_final_window_mean"] == 240.46404994964598
    assert row1["notes"] == [
        "If upper-block means remain clearly upward after warmup, treat the control as reproduced even if train loss looks superficially acceptable.",
        "Train-only screen recorded as `sd_cuda_stack_scale_followup_01_dpnb_cuda_stack_scale_control_v1`.",
        "Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.",
    ]

    row2 = _row_by_ref(queue, "dpnb_cuda_stack_scale_poststack_rms")
    assert row2["model"]["module_overrides"]["post_stack_norm"] == "rmsnorm"
    assert row2["execution_policy"] == "screen_only"
    assert row2["interpretation_status"] == "interpreted"
    assert row2["screen_metrics"]["upper_block_final_window_mean"] == 176.073586807251
    assert row2["notes"] == [
        "Train-only screen recorded as `sd_cuda_stack_scale_followup_02_dpnb_cuda_stack_scale_poststack_rms_v1`.",
        "Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.",
    ]

    row3 = _row_by_ref(queue, "dpnb_cuda_stack_scale_poststack_ln")
    assert row3["model"]["module_overrides"]["post_stack_norm"] == "layernorm"
    assert row3["execution_policy"] == "screen_only"
    assert row3["interpretation_status"] == "interpreted"
    assert row3["screen_metrics"]["upper_block_final_window_mean"] == 604.592961883545
    assert row3["notes"] == [
        "Train-only screen recorded as `sd_cuda_stack_scale_followup_03_dpnb_cuda_stack_scale_poststack_ln_v1`.",
        "Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.",
    ]

    row4 = _row_by_ref(queue, "dpnb_cuda_stack_scale_depth_scaled")
    assert row4["model"]["module_overrides"]["table_block_residual_scale"] == "depth_scaled"
    assert row4["execution_policy"] == "screen_only"
    assert row4["interpretation_status"] == "interpreted"
    assert row4["screen_metrics"]["upper_block_final_window_mean"] == 77.8338779258728
    assert row4["notes"] == [
        "Train-only screen recorded as `sd_cuda_stack_scale_followup_04_dpnb_cuda_stack_scale_depth_scaled_v1`.",
        "Canonical benchmark comparison recorded against the locked sweep anchor; interpret this row in the full sweep context.",
    ]

    row5 = _row_by_ref(queue, "dpnb_cuda_stack_scale_depth_scaled_plus_norm_winner")
    assert row5["status"] == "blocked"
    assert row5["execution_policy"] == "benchmark_full"
    assert row5["training"]["overrides"]["runtime"]["max_steps"] == 2500
    assert row5["interpretation_status"] == "blocked"
    assert row5["model"]["module_overrides"]["post_stack_norm"] == "rmsnorm"
    assert row5["dynamic_model_overrides"]["post_stack_norm"] == {
        "kind": "screen_winner",
        "compare_orders": [
            {"order": 2, "value": "rmsnorm"},
            {"order": 3, "value": "layernorm"},
        ],
        "tie_break_preference": "rmsnorm",
        "resolved_value": "rmsnorm",
        "resolved_from_order": 2,
        "resolution_reason": "lower upper-block final-window mean",
    }
    assert row5["notes"] == [
        "The runner resolves `model.module_overrides.post_stack_norm` at execution time from the recorded screen metrics in rows 2-3.",
        "Resolved `post_stack_norm` to `rmsnorm` from screen row `2` (lower upper-block final-window mean).",
        "A stopped diagnostic archive exists at `sd_cuda_stack_scale_followup_05_dpnb_cuda_stack_scale_depth_scaled_plus_norm_winner_v1_stopped_user_interrupt_20260319T161805Z`.",
        "The first full-budget attempt was stopped manually at step `475` before benchmark registration after the partial trace still looked mediocre relative to the screen evidence.",
    ]

    materialized = load_system_delta_queue(
        sweep_id="cuda_stack_scale_followup",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    assert [row["delta_id"] for row in materialized["rows"]] == EXPECTED_ROWS
    assert [row["execution_policy"] for row in materialized["rows"]] == [
        "screen_only",
        "screen_only",
        "screen_only",
        "screen_only",
        "benchmark_full",
    ]


def test_cuda_stack_scale_followup_matrix_records_screen_and_benchmark_policies() -> None:
    matrix = (
        REPO_ROOT / "reference" / "system_delta_sweeps" / "cuda_stack_scale_followup" / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "# System Delta Matrix" in matrix
    assert "cuda_stack_scale_followup" in matrix
    assert ANCHOR_RUN_ID in matrix
    assert "dpnb_cuda_stack_scale_control" in matrix
    assert "dpnb_cuda_stack_scale_poststack_rms" in matrix
    assert "dpnb_cuda_stack_scale_poststack_ln" in matrix
    assert "dpnb_cuda_stack_scale_depth_scaled" in matrix
    assert "dpnb_cuda_stack_scale_depth_scaled_plus_norm_winner" in matrix
    assert "Training experiment: `cls_benchmark_staged_prior`" in matrix
    assert "Surface role: `hybrid_diagnostic`" in matrix
    assert "Execution policy: `screen_only`" in matrix
    assert "Execution policy: `benchmark_full`" in matrix
    assert "Status: `blocked`" in matrix
    assert "Upper-block final-window mean: `77.8339`" in matrix
    assert "stopped manually at step `475`" in matrix
    assert "Benchmark metrics: pending" in matrix
