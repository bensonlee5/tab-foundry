from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _row_by_ref(queue: dict[str, Any], delta_ref: str) -> dict[str, Any]:
    rows = queue["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["delta_ref"] == delta_ref)


def test_missingness_followup_is_registered_but_not_active() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["active_sweep_id"] == "cuda_capacity_pilot"

    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps["missingness_followup"] == {
        "parent_sweep_id": "stability_ladder",
        "status": "draft",
        "anchor_run_id": None,
        "complexity_level": "binary_md",
        "benchmark_bundle_path": "src/tab_foundry/bench/nanotabpfn_openml_binary_large_v1.json",
        "control_baseline_id": "cls_benchmark_linear_v2",
    }


def test_missingness_followup_metadata_and_rows_match_the_plan() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / "missingness_followup"
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == "missingness_followup"
    assert sweep["parent_sweep_id"] == "stability_ladder"
    assert sweep["benchmark_bundle_path"] == "src/tab_foundry/bench/nanotabpfn_openml_binary_large_v1.json"
    assert sweep["anchor_context"]["model"]["stage"] == "prenorm_block"
    assert sweep["anchor_context"]["surface_labels"]["data"] == "benchmark_large_allow_missing"

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == [
        "nan_token_no_prior_missingness",
        "nan_token_prior_missingness_005",
    ]
    assert [row["status"] for row in rows] == ["ready", "ready"]

    baseline_row = _row_by_ref(queue, "nan_token_no_prior_missingness")
    assert baseline_row["model"]["module_overrides"] == {
        "feature_encoder": "shared",
        "post_encoder_norm": "layernorm",
        "table_block_style": "prenorm",
        "tokenizer": "scalar_per_feature_nan_mask",
    }
    assert baseline_row["data"]["surface_overrides"] == {"allow_missing_values": True}
    assert baseline_row["training"]["surface_label"] == "prior_cosine_warmup"
    assert baseline_row["training"]["overrides"]["apply_schedule"] is True

    prior_row = _row_by_ref(queue, "nan_token_prior_missingness_005")
    assert prior_row["model"]["module_overrides"] == baseline_row["model"]["module_overrides"]
    assert prior_row["training"]["overrides"]["prior_missingness"] == {
        "enabled": True,
        "min_rate": 0.05,
        "max_rate": 0.05,
    }


def test_missingness_followup_matrix_records_fixed_normalization_and_missingness_training() -> None:
    matrix = (
        REPO_ROOT
        / "reference"
        / "system_delta_sweeps"
        / "missingness_followup"
        / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "# Missingness Follow-Up Comparison Matrix" in matrix
    assert "not an input-normalization sweep" in matrix
    assert "train_zscore_clip" in matrix
    assert "scalar_per_feature_nan_mask" in matrix
    assert "fixed `5%` synthetic missingness" in matrix
    assert "nan_token_prior_missingness_005" in matrix
