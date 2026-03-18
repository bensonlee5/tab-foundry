from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.system_delta import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "input_norm_none_followup"
ANCHOR_RUN_ID = "sd_input_norm_followup_07_dpnb_input_norm_anchor_replay_batch64_sqrt_v2"
RUN_ID = "sd_input_norm_none_followup_01_dpnb_input_norm_none_batch64_sqrt_v2"
BEST_ROC_AUC = 0.7597362859021641
FINAL_ROC_AUC = 0.7586267171926775
DRIFT = -0.0011095687094866413


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def test_input_norm_none_followup_is_registered_and_completed() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["active_sweep_id"] == "cuda_capacity_pilot"
    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps[SWEEP_ID] == {
        "parent_sweep_id": "input_norm_followup",
        "status": "completed",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "binary_md",
        "benchmark_bundle_path": "src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
        "control_baseline_id": "cls_benchmark_linear_v2",
    }


def test_input_norm_none_followup_metadata_and_result_match_the_deferred_result() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == SWEEP_ID
    assert sweep["parent_sweep_id"] == "input_norm_followup"
    assert sweep["status"] == "completed"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert any("batch size was the only clear source of lift" in note for note in sweep["anchor_surface"]["notes"])

    rows = queue["rows"]
    assert isinstance(rows, list) and len(rows) == 1
    row = rows[0]
    assert row["delta_ref"] == "dpnb_input_norm_none_batch64_sqrt"
    assert row["status"] == "completed"
    assert row["model"]["input_normalization"] == "none"
    assert row["training"]["prior_dump_batch_size"] == 64
    assert row["run_id"] == RUN_ID
    assert row["decision"] == "defer"
    assert row["interpretation_status"] == "completed"
    assert row["benchmark_metrics"]["best_roc_auc"] == BEST_ROC_AUC
    assert row["benchmark_metrics"]["final_roc_auc"] == FINAL_ROC_AUC
    assert row["benchmark_metrics"]["drift"] == DRIFT
    assert any("lost 0.0048 final ROC AUC" in note for note in row["notes"])
    assert "do not reopen `input_normalization=none`" in row["next_action"]

    materialized = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    assert [row["delta_id"] for row in materialized["rows"]] == ["dpnb_input_norm_none_batch64_sqrt"]


def test_input_norm_none_followup_matrix_records_the_deferred_result() -> None:
    matrix = (REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "matrix.md").read_text(encoding="utf-8")

    assert "# System Delta Matrix" in matrix
    assert RUN_ID in matrix
    assert "delta final ROC AUC `-0.0048`" in matrix
    assert "dpnb_input_norm_none_batch64_sqrt" in matrix
