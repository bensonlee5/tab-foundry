from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.sweep.inspection_targets import inspect_sweep_row
from tab_foundry.research.sweep.materialize import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "tf_rd_010_classification_evolution_medium_v1"
EXPECTED_ROWS = [
    "delta_data_manifest_root_tf_rd_010_dagzoo_medium_control",
    "delta_data_manifest_root_tf_rd_010_missingness_mcar",
    "delta_data_manifest_root_tf_rd_010_missingness_mar",
    "delta_data_manifest_root_tf_rd_010_missingness_mnar",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def test_tf_rd_010_classification_evolution_medium_v1_is_registered() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["schema"] == "tab-foundry-system-delta-sweep-index-v2"
    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps[SWEEP_ID] == {
        "parent_sweep_id": "tf_rd_021b_sandwich_feature_removal_v1",
        "status": "draft",
        "anchor_run_id": None,
        "complexity_level": "classification_md",
        "benchmark_manifest_path": "data/manifests/bench/nanotabpfn_openml_classification_medium_v1/manifest.parquet",
        "control_baseline_id": "cls_benchmark_linear_multiclass_medium_v1",
        "external_benchmarks": [],
    }


def test_tf_rd_010_classification_evolution_medium_v1_records_the_draft_medium_contract() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == SWEEP_ID
    assert sweep["status"] == "draft"
    assert sweep["anchor_run_id"] is None
    assert sweep["training_experiment"] == "cls_benchmark_sandwich_classification_evolution_v1"
    assert sweep["training_config_profile"] == "cls_benchmark_sandwich_classification_evolution_v1"
    assert sweep["benchmark_manifest_path"] == (
        "data/manifests/bench/nanotabpfn_openml_classification_medium_v1/manifest.parquet"
    )
    assert sweep["control_baseline_id"] == "cls_benchmark_linear_multiclass_medium_v1"
    assert sweep["external_benchmarks"] == []

    notes = sweep["anchor_surface"]["notes"]
    assert isinstance(notes, list)
    assert any("#178" in note for note in notes)
    assert any("tab-realdata-hub" in note for note in notes)
    assert any("sandwich_summary_tokens_per_axis=3" in note for note in notes)
    assert any("final_bpc_at_matched_regime_budget" in note for note in notes)
    assert any("class imbalance" in note.lower() for note in notes)
    assert any("medium rung is the clean no-missing multiclass benchmark surface" in note for note in notes)

    anchor_model = sweep["anchor_context"]["model"]
    assert anchor_model["arch"] == "tabfoundry_sandwich"
    assert anchor_model["module_selection"] == {
        "feature_encoder": "shared",
        "feature_type_conditioning": "film",
        "head": "direct_multiclass",
        "target_conditioner": "label_token",
        "tokenizer": "scalar_per_feature_missingness",
    }
    assert sweep["anchor_context"]["surface_labels"] == {
        "data": "tf_rd_010_dagzoo_medium_control",
        "model": "tabfoundry_sandwich",
        "preprocessing": "runtime_default",
        "training": "prior_cosine_warmup",
    }

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == [
        "blocked_on_validation_manifests_and_control_baselines"
    ] * len(EXPECTED_ROWS)
    assert all(row["model"]["feature_type_conditioning"] == "film" for row in rows)
    assert all(row["model"]["sandwich_summary_tokens_per_axis"] == 3 for row in rows)
    assert all(row["model"]["many_class_base"] == 10 for row in rows)
    assert all(row["training"]["surface_label"] == "prior_cosine_warmup" for row in rows)

    materialized = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    assert materialized["anchor_run_id"] is None
    assert materialized["benchmark_manifest_path"] == (
        "data/manifests/bench/nanotabpfn_openml_classification_medium_v1/manifest.parquet"
    )
    assert materialized["control_baseline_id"] == "cls_benchmark_linear_multiclass_medium_v1"
    assert [row["delta_id"] for row in materialized["rows"]] == EXPECTED_ROWS
    assert all(row["model"].get("stage_label") is None for row in materialized["rows"])


def test_tf_rd_010_classification_evolution_medium_v1_matrix_links_dagzoo_and_hub() -> None:
    matrix = (
        REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "# System Delta Matrix" in matrix
    assert SWEEP_ID in matrix
    assert "Sweep status: `draft`" in matrix
    assert "final_bpc_at_matched_regime_budget" in matrix
    assert "tab-realdata-hub" in matrix
    assert "dagzoo" in matrix
    assert "sandwich_summary_tokens_per_axis=3" in matrix
    assert "direct multiclass head" in matrix
    assert "class imbalance is addressed through explicit benchmark coverage and reporting" in matrix.lower()


def test_tf_rd_010_classification_evolution_medium_v1_inspection_resolves_sandwich_row() -> None:
    payload = inspect_sweep_row(
        sweep_id=SWEEP_ID,
        order=1,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )

    resolved_model = payload["target"]["resolved"]["model"]
    assert resolved_model["arch"] == "tabfoundry_sandwich"
    assert resolved_model.get("stage_label") is None
