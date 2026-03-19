from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.system_delta import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
EXPECTED_ROWS = [
    "dpnb_input_norm_anchor_replay",
    "dpnb_input_norm_zscore",
    "dpnb_input_norm_winsorize_zscore",
    "dpnb_input_norm_zscore_tanh",
    "dpnb_input_norm_robust_tanh",
    "dpnb_input_norm_anchor_replay_batch16_sqrt",
    "dpnb_input_norm_anchor_replay_batch64_sqrt",
    "dpnb_input_norm_zscore_tanh_batch16_sqrt",
    "dpnb_input_norm_zscore_tanh_batch64_sqrt",
]
ANCHOR_RUN_ID = "sd_input_norm_followup_07_dpnb_input_norm_anchor_replay_batch64_sqrt_v2"
BASELINE_RUN_ID = "sd_input_norm_followup_01_dpnb_input_norm_anchor_replay_v2"
BASELINE_BEST_ROC_AUC = 0.7634285072744538
BASELINE_FINAL_ROC_AUC = 0.7566546311647561
BASELINE_DRIFT = -0.006773876109697707
BATCH16_BEST_ROC_AUC = 0.7595457209489976
BATCH16_FINAL_ROC_AUC = 0.7556520172640854
BATCH16_DRIFT = -0.003893703684912264
BATCH64_BEST_ROC_AUC = 0.763449888408113
BATCH64_FINAL_ROC_AUC = 0.763449888408113
BATCH64_DRIFT = 0.0


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _row_by_ref(queue: dict[str, Any], delta_ref: str) -> dict[str, Any]:
    rows = queue["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["delta_ref"] == delta_ref)


def test_input_norm_followup_is_registered_but_not_active() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["active_sweep_id"] == "cuda_stack_scale_followup"

    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps["input_norm_followup"] == {
        "parent_sweep_id": "stability_followup",
        "status": "completed",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "binary_md",
        "benchmark_bundle_path": "src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
        "control_baseline_id": "cls_benchmark_linear_v2",
    }


def test_input_norm_followup_metadata_and_rows_match_the_bridge_baseline_plan() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / "input_norm_followup"
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == "input_norm_followup"
    assert sweep["parent_sweep_id"] == "stability_followup"
    assert sweep["status"] == "completed"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["experiment"] == "cls_benchmark_staged_prior"
    assert sweep["anchor_context"]["config_profile"] == "cls_benchmark_staged_prior"
    assert sweep["anchor_context"]["run_id"] == ANCHOR_RUN_ID
    assert sweep["anchor_context"]["model"]["stage_label"] == "dpnb_input_norm_anchor_replay_batch64_sqrt"
    assert sweep["anchor_context"]["model"]["benchmark_profile"] == "dpnb_input_norm_anchor_replay_batch64_sqrt"
    assert sweep["anchor_context"]["model"]["module_selection"] == {
        "allow_test_self_attention": False,
        "column_encoder": "none",
        "context_encoder": "none",
        "feature_encoder": "nano",
        "head": "binary_direct",
        "post_encoder_norm": "none",
        "row_pool": "row_cls",
        "table_block_style": "prenorm",
        "target_conditioner": "mean_padded_linear",
        "tokenizer": "scalar_per_feature",
    }
    assert sweep["anchor_context"]["surface_labels"]["training"] == "prior_linear_warmup_decay"
    assert any("row 7" in note.lower() or "batch64" in note.lower() for note in sweep["anchor_surface"]["notes"])

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["completed"] * 9
    assert all(row["training"]["prior_dump_non_finite_policy"] == "skip" for row in rows)
    assert all(row["training"]["overrides"]["runtime"]["trace_activations"] is True for row in rows)

    baseline = _row_by_ref(queue, "dpnb_input_norm_anchor_replay")
    assert baseline["model"]["stage_label"] == "dpnb_row_cls_cls2_linear_warmup_decay"
    assert baseline["model"]["input_normalization"] == "train_zscore_clip"
    assert baseline["training"]["prior_dump_batch_size"] == 32
    assert baseline["training"]["prior_dump_lr_scale_rule"] == "none"
    assert baseline["training"]["prior_dump_batch_reference_size"] == 32
    assert baseline["training"]["overrides"]["optimizer"] == {"min_lr": 0.0004}
    assert baseline["training"]["overrides"]["schedule"]["stages"] == [
        {
            "name": "prior_dump",
            "steps": 2500,
            "lr_max": 0.004,
            "lr_schedule": "linear",
            "warmup_ratio": 0.05,
        }
    ]
    assert baseline["run_id"] == BASELINE_RUN_ID
    assert baseline["decision"] == "defer"
    assert baseline["interpretation_status"] == "completed"
    assert baseline["benchmark_metrics"]["best_roc_auc"] == BASELINE_BEST_ROC_AUC
    assert baseline["benchmark_metrics"]["final_roc_auc"] == BASELINE_FINAL_ROC_AUC
    assert baseline["benchmark_metrics"]["drift"] == BASELINE_DRIFT
    assert "promoted canonical anchor" in baseline["next_action"]

    zscore = _row_by_ref(queue, "dpnb_input_norm_zscore")
    assert zscore["model"]["input_normalization"] == "train_zscore"
    assert zscore["run_id"] == "sd_input_norm_followup_02_dpnb_input_norm_zscore_v2"
    assert zscore["decision"] == "defer"
    assert zscore["interpretation_status"] == "completed"
    assert zscore["benchmark_metrics"]["final_roc_auc"] == BASELINE_FINAL_ROC_AUC
    assert "normalization-family wash" in zscore["next_action"]

    winsor = _row_by_ref(queue, "dpnb_input_norm_winsorize_zscore")
    assert winsor["model"]["input_normalization"] == "train_winsorize_zscore"
    assert winsor["run_id"] == "sd_input_norm_followup_03_dpnb_input_norm_winsorize_zscore_v2"
    assert winsor["decision"] == "defer"
    assert winsor["interpretation_status"] == "completed"
    assert winsor["benchmark_metrics"]["final_roc_auc"] == BASELINE_FINAL_ROC_AUC

    zscore_tanh = _row_by_ref(queue, "dpnb_input_norm_zscore_tanh")
    assert zscore_tanh["model"]["input_normalization"] == "train_zscore_tanh"
    assert zscore_tanh["status"] == "completed"
    assert zscore_tanh["run_id"] == "sd_input_norm_followup_04_dpnb_input_norm_zscore_tanh_v1"
    assert zscore_tanh["benchmark_metrics"]["final_roc_auc"] == BASELINE_FINAL_ROC_AUC
    assert "smooth-tail comparator" in zscore_tanh["next_action"]

    robust_tanh = _row_by_ref(queue, "dpnb_input_norm_robust_tanh")
    assert robust_tanh["model"]["input_normalization"] == "train_robust_tanh"
    assert robust_tanh["status"] == "completed"
    assert robust_tanh["run_id"] == "sd_input_norm_followup_05_dpnb_input_norm_robust_tanh_v1"
    assert robust_tanh["benchmark_metrics"]["final_roc_auc"] == BASELINE_FINAL_ROC_AUC

    batch16 = _row_by_ref(queue, "dpnb_input_norm_anchor_replay_batch16_sqrt")
    assert batch16["training"]["prior_dump_batch_size"] == 16
    assert batch16["training"]["prior_dump_lr_scale_rule"] == "sqrt"
    assert batch16["training"]["prior_dump_batch_reference_size"] == 32
    assert batch16["run_id"] == "sd_input_norm_followup_06_dpnb_input_norm_anchor_replay_batch16_sqrt_v1"
    assert batch16["benchmark_metrics"]["best_roc_auc"] == BATCH16_BEST_ROC_AUC
    assert batch16["benchmark_metrics"]["final_roc_auc"] == BATCH16_FINAL_ROC_AUC
    assert batch16["benchmark_metrics"]["drift"] == BATCH16_DRIFT
    assert "0.7071" in batch16["notes"][0]

    batch64 = _row_by_ref(queue, "dpnb_input_norm_anchor_replay_batch64_sqrt")
    assert batch64["training"]["prior_dump_batch_size"] == 64
    assert batch64["training"]["prior_dump_lr_scale_rule"] == "sqrt"
    assert batch64["training"]["prior_dump_batch_reference_size"] == 32
    assert batch64["run_id"] == "sd_input_norm_followup_07_dpnb_input_norm_anchor_replay_batch64_sqrt_v2"
    assert batch64["benchmark_metrics"]["best_roc_auc"] == BATCH64_BEST_ROC_AUC
    assert batch64["benchmark_metrics"]["final_roc_auc"] == BATCH64_FINAL_ROC_AUC
    assert batch64["benchmark_metrics"]["drift"] == BATCH64_DRIFT
    assert batch64["decision"] == "defer"
    assert "canonical anchor" in batch64["next_action"]
    assert "1.4142" in batch64["notes"][0]

    batch16_tanh = _row_by_ref(queue, "dpnb_input_norm_zscore_tanh_batch16_sqrt")
    assert batch16_tanh["model"]["input_normalization"] == "train_zscore_tanh"
    assert batch16_tanh["training"]["prior_dump_batch_size"] == 16
    assert batch16_tanh["run_id"] == "sd_input_norm_followup_08_dpnb_input_norm_zscore_tanh_batch16_sqrt_v1"
    assert batch16_tanh["benchmark_metrics"]["final_roc_auc"] == BATCH16_FINAL_ROC_AUC

    batch64_tanh = _row_by_ref(queue, "dpnb_input_norm_zscore_tanh_batch64_sqrt")
    assert batch64_tanh["model"]["input_normalization"] == "train_zscore_tanh"
    assert batch64_tanh["training"]["prior_dump_batch_size"] == 64
    assert batch64_tanh["run_id"] == "sd_input_norm_followup_09_dpnb_input_norm_zscore_tanh_batch64_sqrt_v1"
    assert batch64_tanh["benchmark_metrics"]["final_roc_auc"] == BATCH64_FINAL_ROC_AUC
    assert "less invasive preprocessing" in batch64_tanh["next_action"]

    materialized = load_system_delta_queue(
        sweep_id="input_norm_followup",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    assert [row["delta_id"] for row in materialized["rows"]] == EXPECTED_ROWS


def test_input_norm_followup_matrix_records_the_bridge_norm_and_batch_queue() -> None:
    matrix = (
        REPO_ROOT
        / "reference"
        / "system_delta_sweeps"
        / "input_norm_followup"
        / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "# System Delta Matrix" in matrix
    assert ANCHOR_RUN_ID in matrix
    assert "sd_input_norm_followup_07_dpnb_input_norm_anchor_replay_batch64_sqrt_v1" in matrix
    assert "sd_input_norm_followup_09_dpnb_input_norm_zscore_tanh_batch64_sqrt_v1" in matrix
    assert "dpnb_row_cls_cls2_linear_warmup_decay" in matrix
    assert "train_zscore_tanh" in matrix
    assert "0.7071" in matrix
    assert "1.4142" in matrix
    assert "dpnb_input_norm_anchor_replay_batch16_sqrt" in matrix
    assert "dpnb_input_norm_anchor_replay_batch64_sqrt" in matrix
