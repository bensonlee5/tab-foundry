from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.sweep.inspection_targets import inspect_sweep_row
from tab_foundry.research.sweep.materialize import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "tf_rd_010_classification_evolution_medium_v1"
ANCHOR_RUN_ID = (
    "sd_tf_rd_010_classification_evolution_medium_v1_01_"
    "delta_data_manifest_root_tf_rd_010_dagzoo_medium_control_v1"
)
EXPECTED_ROWS = [
    "delta_data_manifest_root_tf_rd_010_dagzoo_medium_control",
    "delta_data_manifest_root_tf_rd_010_missingness_mcar",
    "delta_data_manifest_root_tf_rd_010_missingness_mar",
    "delta_data_manifest_root_tf_rd_010_missingness_mnar",
]
TF_RD_010_RECIPE_PATHS = [
    REPO_ROOT / "reference" / "corpus_recipes" / "tf_rd_010_dagzoo_medium_control_v1.yaml",
    REPO_ROOT / "reference" / "corpus_recipes" / "tf_rd_010_missingness_mcar_v1.yaml",
    REPO_ROOT / "reference" / "corpus_recipes" / "tf_rd_010_missingness_mar_v1.yaml",
    REPO_ROOT / "reference" / "corpus_recipes" / "tf_rd_010_missingness_mnar_v1.yaml",
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
        "status": "completed",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "classification_md",
        "benchmark_manifest_path": "data/manifests/bench/nanotabpfn_openml_classification_medium_v1/manifest.parquet",
        "control_baseline_id": "cls_benchmark_linear_multiclass_medium_v1",
        "external_benchmarks": [],
    }


def test_tf_rd_010_classification_evolution_medium_v1_records_the_completed_medium_contract() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == SWEEP_ID
    assert sweep["status"] == "completed"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
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
    assert any("broader classification pool" in note.lower() for note in notes)
    assert any("min_classes=2" in note for note in notes)
    assert any("max_missing_pct=20.0" in note for note in notes)
    assert any("All four completed rows deferred" in note for note in notes)
    assert any("stability guardrail" in note for note in notes)

    anchor_model = sweep["anchor_context"]["model"]
    assert anchor_model["arch"] == "tabfoundry_sandwich"
    assert anchor_model["module_selection"] is None
    assert sweep["anchor_context"]["surface_labels"] == {
        "data": "tf_rd_010_dagzoo_medium_control",
        "model": "tabfoundry_sandwich",
        "preprocessing": "runtime_default",
        "training": "prior_cosine_warmup",
    }

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["completed"] * len(EXPECTED_ROWS)
    assert [row["decision"] for row in rows] == ["defer"] * len(EXPECTED_ROWS)
    assert all(row["run_id"] is not None for row in rows)
    assert all(row["next_action"].startswith("Completed") for row in rows)
    assert all(
        any("stability=fail" in note for note in row["notes"]) for row in rows
    )
    assert all(row["model"]["feature_type_conditioning"] == "film" for row in rows)
    assert all(row["model"]["sandwich_summary_tokens_per_axis"] == 3 for row in rows)
    assert all(row["model"]["many_class_base"] == 10 for row in rows)
    assert all(row["training"]["surface_label"] == "prior_cosine_warmup" for row in rows)
    assert all(row["training"]["overrides"]["runtime"]["max_steps"] == 400 for row in rows)
    assert all(row["training"]["overrides"]["schedule"]["stages"][0]["steps"] == 400 for row in rows)
    assert all(row["data"]["train_row_cap"] == 64 for row in rows)
    assert all(row["data"]["test_row_cap"] == 32 for row in rows)

    materialized = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    assert materialized["anchor_run_id"] == ANCHOR_RUN_ID
    assert materialized["benchmark_manifest_path"] == (
        "data/manifests/bench/nanotabpfn_openml_classification_medium_v1/manifest.parquet"
    )
    assert materialized["control_baseline_id"] == "cls_benchmark_linear_multiclass_medium_v1"
    assert [row["delta_id"] for row in materialized["rows"]] == EXPECTED_ROWS
    assert all(row["model"].get("stage_label") is None for row in materialized["rows"])

    for recipe_path in TF_RD_010_RECIPE_PATHS:
        recipe = _load_yaml(recipe_path)
        invocations = recipe["invocations"]
        assert isinstance(invocations, list)
        assert all(
            invocation["config_overrides"]["dataset"]["n_classes_min"] == 2
            for invocation in invocations
        )
        assert all(
            invocation["config_overrides"]["dataset"]["n_classes_max"] == 10
            for invocation in invocations
        )


def test_tf_rd_010_classification_evolution_medium_v1_matrix_links_dagzoo_and_hub() -> None:
    matrix = (
        REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "# System Delta Matrix" in matrix
    assert SWEEP_ID in matrix
    assert "Sweep status: `completed`" in matrix
    assert "final_bpc_at_matched_regime_budget" in matrix
    assert "tab-realdata-hub" in matrix
    assert "dagzoo" in matrix
    assert "sandwich_summary_tokens_per_axis=3" in matrix
    assert "direct multiclass head" in matrix
    assert "min_classes=2" in matrix
    assert "max_missing_pct=20.0" in matrix
    assert "Completed as the locked medium control anchor" in matrix
    assert "Completed as mixed negative evidence" in matrix
    assert "stability=fail" in matrix


def test_tf_rd_010_medium_registry_uses_renamed_hub_bundle() -> None:
    registry_text = (
        REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json"
    ).read_text(encoding="utf-8")

    assert "openml_classification_medium_v1.json" in registry_text
    assert "nanotabpfn_openml_classification_medium_v1.json" not in registry_text


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
