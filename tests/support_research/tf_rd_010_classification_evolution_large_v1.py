from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.sweep.inspection_targets import inspect_sweep_row
from tab_foundry.research.sweep.materialize import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "tf_rd_010_classification_evolution_large_v1"
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


def test_tf_rd_010_classification_evolution_large_v1_is_registered() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    assert index["schema"] == "tab-foundry-system-delta-sweep-index-v2"
    sweeps = index["sweeps"]
    assert isinstance(sweeps, dict)
    assert sweeps[SWEEP_ID] == {
        "parent_sweep_id": "tf_rd_010_classification_evolution_medium_v1",
        "status": "ready",
        "anchor_run_id": None,
        "complexity_level": "classification_lg",
        "benchmark_manifest_path": "data/manifests/bench/nanotabpfn_openml_classification_large_v1/manifest.parquet",
        "control_baseline_id": "cls_benchmark_linear_multiclass_large_v1",
        "external_benchmarks": [],
    }


def test_tf_rd_010_classification_evolution_large_v1_records_the_reset_large_contract() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == SWEEP_ID
    assert sweep["status"] == "ready"
    assert sweep["anchor_run_id"] is None
    assert sweep["training_experiment"] == "cls_benchmark_sandwich_classification_evolution_v1"
    assert sweep["training_config_profile"] == "cls_benchmark_sandwich_classification_evolution_v1"
    assert sweep["benchmark_manifest_path"] == (
        "data/manifests/bench/nanotabpfn_openml_classification_large_v1/manifest.parquet"
    )
    assert sweep["control_baseline_id"] == "cls_benchmark_linear_multiclass_large_v1"
    assert sweep["external_benchmarks"] == []

    notes = sweep["anchor_surface"]["notes"]
    assert isinstance(notes, list)
    assert any("#202" in note for note in notes)
    assert any("#203" in note for note in notes)
    assert any("#204" in note for note in notes)
    assert any("tab-realdata-hub" in note for note in notes)
    assert any("min_classes=2" in note for note in notes)
    assert any("max_missing_pct=20.0" in note for note in notes)
    assert any("sandwich_summary_tokens_per_axis=3" in note for note in notes)
    assert any("final_bpc_at_matched_regime_budget" in note for note in notes)
    assert any("broader classification pool" in note.lower() for note in notes)
    assert any("no longer canonical" in note.lower() for note in notes)
    assert any("trusted large evidence" in note.lower() for note in notes)
    assert any("144` tasks per front" in note or "144 tasks per front" in note for note in notes)
    assert any("<=1024" in note for note in notes)
    assert any("one pass over corpus manifest records/tasks" in note for note in notes)

    anchor_model = sweep["anchor_context"]["model"]
    assert anchor_model["arch"] == "tabfoundry_sandwich"
    assert anchor_model["module_selection"] is None
    assert sweep["anchor_context"]["run_id"] is None

    rows = queue["rows"]
    assert isinstance(rows, list)
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["ready"] * len(EXPECTED_ROWS)
    assert all(row["decision"] is None for row in rows)
    assert all(row["run_id"] is None for row in rows)
    assert all(row["interpretation_status"] == "pending" for row in rows)
    assert all("issue `#203`" in row["next_action"] for row in rows)
    assert all(
        any("non-canonical artifacts" in note for note in row["notes"]) for row in rows
    )
    assert all(row["model"]["feature_type_conditioning"] == "film" for row in rows)
    assert all(row["model"]["sandwich_summary_tokens_per_axis"] == 3 for row in rows)
    assert all(row["model"]["many_class_base"] == 10 for row in rows)
    assert all(row["training"]["surface_label"] == "prior_cosine_warmup" for row in rows)
    assert all(row["training"]["prior_dump_batch_size"] == 64 for row in rows)
    assert all(row["training"]["synthetic_epoch_budget"]["epochs"] == 1 for row in rows)
    assert all(
        row["training"]["synthetic_epoch_budget"]["budget_unit"] == "corpus_manifest_records"
        for row in rows
    )
    assert all(row["training"]["synthetic_epoch_budget"]["prior_dump_batch_size"] == 64 for row in rows)
    assert all(row["training"]["synthetic_epoch_budget"]["allow_partial_final_batch"] is True for row in rows)
    assert all("max_steps" not in row["training"]["overrides"].get("runtime", {}) for row in rows)
    assert all("steps" not in row["training"]["overrides"]["schedule"]["stages"][0] for row in rows)
    assert all(row["data"]["train_row_cap"] == 64 for row in rows)
    assert all(row["data"]["test_row_cap"] == 32 for row in rows)
    assert all("benchmark_metrics" not in row for row in rows)

    materialized = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    assert materialized["anchor_run_id"] is None
    assert materialized["benchmark_manifest_path"] == (
        "data/manifests/bench/nanotabpfn_openml_classification_large_v1/manifest.parquet"
    )
    assert materialized["control_baseline_id"] == "cls_benchmark_linear_multiclass_large_v1"
    assert [row["delta_id"] for row in materialized["rows"]] == EXPECTED_ROWS
    assert all(row["model"].get("stage_label") is None for row in materialized["rows"])
    assert all(row["training"]["overrides"]["runtime"]["max_steps"] == 3 for row in materialized["rows"])
    assert all(row["training"]["overrides"]["schedule"]["stages"][0]["steps"] == 3 for row in materialized["rows"])
    assert all(row["training"]["synthetic_epoch_budget"]["resolved_task_count"] == 144 for row in materialized["rows"])
    assert all(row["training"]["synthetic_epoch_budget"]["resolved_max_steps"] == 3 for row in materialized["rows"])
    assert all(
        row["training"]["synthetic_epoch_budget"]["resolution_source"] in {"recipe_definition", "local_corpus_record"}
        for row in materialized["rows"]
    )


def test_tf_rd_010_classification_evolution_large_v1_matrix_links_dagzoo_and_hub() -> None:
    matrix = (
        REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "matrix.md"
    ).read_text(encoding="utf-8")

    assert "# System Delta Matrix" in matrix
    assert SWEEP_ID in matrix
    assert "Sweep status: `ready`" in matrix
    assert "final_bpc_at_matched_regime_budget" in matrix
    assert "tab-realdata-hub" in matrix
    assert "dagzoo" in matrix
    assert "sandwich_summary_tokens_per_axis=3" in matrix
    assert "direct multiclass head" in matrix
    assert "min_classes=2" in matrix
    assert "max_missing_pct=20.0" in matrix
    assert "larger task set" in matrix
    assert "Anchor run id: `null`" in matrix
    assert "pending trusted rerun" in matrix
    assert "issue `#203`" in matrix
    assert "144-task" in matrix
    assert "1024" in matrix
    assert "single synthetic epoch" in matrix
    assert "`3` optimizer steps" in matrix


def test_tf_rd_010_classification_evolution_large_v1_inspection_resolves_sandwich_row() -> None:
    payload = inspect_sweep_row(
        sweep_id=SWEEP_ID,
        order=1,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )

    resolved_model = payload["target"]["resolved"]["model"]
    assert resolved_model["arch"] == "tabfoundry_sandwich"
    assert resolved_model.get("stage_label") is None
    assert resolved_model["architecture"]["feature_type_encoding"] == "film"
    assert resolved_model["architecture"]["floating_likelihood"] == "single_gaussian"
    assert resolved_model["architecture"]["integer_likelihood"] == "hybrid_mixture"
