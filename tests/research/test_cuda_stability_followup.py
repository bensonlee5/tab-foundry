from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.system_delta import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
ANCHOR_RUN_ID = "sd_input_norm_followup_07_dpnb_input_norm_anchor_replay_batch64_sqrt_v2"
EXPECTED_ROWS = [
    "dpnb_cuda_large_anchor_batch32_replay",
    "dpnb_cuda_large_anchor_batch32_lr3e3",
    "dpnb_cuda_large_anchor_batch32_postrms",
    "dpnb_cuda_large_anchor_batch32_postln",
    "dpnb_cuda_large_anchor_batch64_noscale",
    "dpnb_cuda_large_anchor_batch64_noscale_lr3e3",
]
EXPECTED_BRIER_METRICS = {
    "dpnb_cuda_large_anchor_batch32_replay": {
        "best_brier_score": 0.4001172415363657,
        "final_brier_score": 0.4039399822200303,
        "final_minus_best_brier_score": 0.003822740683664616,
        "delta_final_brier_score": 0.14244531565515206,
    },
    "dpnb_cuda_large_anchor_batch32_lr3e3": {
        "best_brier_score": 0.3961889646064756,
        "final_brier_score": 0.4199854947017734,
        "final_minus_best_brier_score": 0.023796530095297752,
        "delta_final_brier_score": 0.15849082813689513,
    },
    "dpnb_cuda_large_anchor_batch32_postrms": {
        "best_brier_score": 0.40743217012436317,
        "final_brier_score": 0.41010681291619344,
        "final_minus_best_brier_score": 0.002674642791830273,
        "delta_final_brier_score": 0.1486121463513152,
    },
    "dpnb_cuda_large_anchor_batch32_postln": {
        "best_brier_score": 0.4049931445893689,
        "final_brier_score": 0.40690205057655027,
        "final_minus_best_brier_score": 0.001908905987181353,
        "delta_final_brier_score": 0.14540738401167203,
    },
}


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _row_by_ref(queue: dict[str, Any], delta_ref: str) -> dict[str, Any]:
    rows = queue["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["delta_ref"] == delta_ref)


def test_cuda_stability_followup_is_registered_but_not_active() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["active_sweep_id"] == "cuda_stack_scale_followup"

    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps["cuda_stability_followup"] == {
        "parent_sweep_id": "cuda_capacity_pilot",
        "status": "completed",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "binary_md",
        "benchmark_bundle_path": "src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
        "control_baseline_id": "cls_benchmark_linear_v2",
    }


def test_cuda_stability_followup_metadata_and_rows_match_the_batch32_first_debug_ladder() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / "cuda_stability_followup"
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == "cuda_stability_followup"
    assert sweep["parent_sweep_id"] == "cuda_capacity_pilot"
    assert sweep["status"] == "completed"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["model"]["stage_label"] == "dpnb_input_norm_anchor_replay_batch64_sqrt"
    assert sweep["anchor_context"]["surface_labels"]["training"] == "prior_linear_warmup_decay"
    assert any("activation scale" in note for note in sweep["anchor_surface"]["notes"])
    assert any("batch32" in note for note in sweep["anchor_surface"]["notes"])

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == [
        "completed",
        "completed",
        "completed",
        "completed",
        "deferred_separate_workstream",
        "deferred_separate_workstream",
    ]

    row1 = _row_by_ref(queue, "dpnb_cuda_large_anchor_batch32_replay")
    assert row1["training"]["prior_dump_batch_size"] == 32
    assert row1["training"]["prior_dump_lr_scale_rule"] == "none"
    assert row1["training"]["prior_dump_batch_reference_size"] == 32
    assert row1["training"]["overrides"]["optimizer"] == {"min_lr": 0.0004}
    assert row1["training"]["overrides"]["schedule"]["stages"][0]["lr_max"] == 0.004
    assert "Run this row first" in row1["next_action"]

    row2 = _row_by_ref(queue, "dpnb_cuda_large_anchor_batch32_lr3e3")
    assert row2["training"]["prior_dump_batch_size"] == 32
    assert row2["training"]["prior_dump_lr_scale_rule"] == "none"
    assert row2["training"]["prior_dump_batch_reference_size"] == 32
    assert row2["training"]["overrides"]["optimizer"] == {"min_lr": 0.0003}
    assert row2["training"]["overrides"]["schedule"]["stages"][0]["lr_max"] == 0.003

    row3 = _row_by_ref(queue, "dpnb_cuda_large_anchor_batch32_postrms")
    assert row3["model"]["module_overrides"]["post_encoder_norm"] == "rmsnorm"
    assert row3["training"]["prior_dump_batch_size"] == 32
    assert row3["training"]["overrides"]["optimizer"] == {"min_lr": 0.0004}
    assert row3["training"]["overrides"]["schedule"]["stages"][0]["lr_max"] == 0.004

    row4 = _row_by_ref(queue, "dpnb_cuda_large_anchor_batch32_postln")
    assert row4["model"]["module_overrides"]["post_encoder_norm"] == "layernorm"
    assert row4["training"]["prior_dump_batch_size"] == 32
    assert row4["training"]["overrides"]["optimizer"] == {"min_lr": 0.0004}
    assert row4["training"]["overrides"]["schedule"]["stages"][0]["lr_max"] == 0.004

    for delta_ref, expected_metrics in EXPECTED_BRIER_METRICS.items():
        benchmark_metrics = _row_by_ref(queue, delta_ref)["benchmark_metrics"]
        for metric_key, expected_value in expected_metrics.items():
            assert benchmark_metrics[metric_key] == expected_value

    row5 = _row_by_ref(queue, "dpnb_cuda_large_anchor_batch64_noscale")
    assert row5["status"] == "deferred_separate_workstream"
    assert row5["interpretation_status"] == "blocked"
    assert row5["training"]["prior_dump_batch_size"] == 64
    assert "Leave deferred" in row5["next_action"]
    assert any("second stopped diagnostic archive" in note for note in row5["notes"])

    row6 = _row_by_ref(queue, "dpnb_cuda_large_anchor_batch64_noscale_lr3e3")
    assert row6["status"] == "deferred_separate_workstream"
    assert row6["interpretation_status"] == "blocked"
    assert row6["training"]["prior_dump_batch_size"] == 64
    assert row6["training"]["overrides"]["optimizer"] == {"min_lr": 0.0003}

    materialized = load_system_delta_queue(
        sweep_id="cuda_stability_followup",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    assert [row["delta_id"] for row in materialized["rows"]] == EXPECTED_ROWS


def test_cuda_stability_followup_matrix_records_the_batch32_first_ladder_and_deferred_batch64_backlog() -> None:
    matrix = (
        REPO_ROOT / "reference" / "system_delta_sweeps" / "cuda_stability_followup" / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "# System Delta Matrix" in matrix
    assert "cuda_stability_followup" in matrix
    assert ANCHOR_RUN_ID in matrix
    assert "dpnb_cuda_large_anchor_batch32_replay" in matrix
    assert "dpnb_cuda_large_anchor_batch32_lr3e3" in matrix
    assert "dpnb_cuda_large_anchor_batch32_postrms" in matrix
    assert "dpnb_cuda_large_anchor_batch32_postln" in matrix
    assert "dpnb_cuda_large_anchor_batch64_noscale" in matrix
    assert "dpnb_cuda_large_anchor_batch64_noscale_lr3e3" in matrix
    assert "post-encoder RMSNorm" in matrix
    assert "post-encoder LayerNorm" in matrix
    assert "batch32 replay" in matrix
    assert "deferred_separate_workstream" in matrix
