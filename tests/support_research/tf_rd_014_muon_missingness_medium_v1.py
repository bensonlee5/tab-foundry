from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.sweep.materialize import load_system_delta_queue
from tab_foundry.research.sweep.surface_resolution import inspection_raw_cfg_mapping
from tab_foundry.research.lane_contract import resolve_training_surface_context
from tests.support_research.helpers import assert_training_surface_semantics


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "tf_rd_014_muon_missingness_medium_v1"
PARENT_SWEEP_ID = "tf_rd_009_muon_width_depth_medium_v1"
ANCHOR_RUN_ID = "sd_tf_rd_009_muon_width_depth_medium_v1_05_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1"
MUON_EXPERIMENT = "cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1"
EXPECTED_ROWS = [
    "delta_data_manifest_root_tf_rd_010_dagzoo_medium_control",
    "delta_data_manifest_root_tf_rd_010_missingness_mcar",
    "delta_data_manifest_root_tf_rd_010_missingness_mar",
    "delta_data_manifest_root_tf_rd_010_missingness_mnar",
]
EXPECTED_CORPUS_REFS = [
    "tf_rd_010_dagzoo_medium_control_curated_v6",
    "tf_rd_010_missingness_mcar_v3",
    "tf_rd_010_missingness_mar_v3",
    "tf_rd_010_missingness_mnar_v3",
]


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def test_tf_rd_014_muon_missingness_medium_v1_is_registered_on_the_muon_width_depth_anchor() -> None:
    index = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")

    entry = index["sweeps"][SWEEP_ID]
    assert entry == {
        "parent_sweep_id": PARENT_SWEEP_ID,
        "status": "ready",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "classification_md",
        "benchmark_manifest_path": "data/manifests/bench/openml_classification_medium_v1/manifest.parquet",
        "control_baseline_id": "cls_benchmark_linear_multiclass_medium_v1",
        "external_benchmarks": [],
    }


def test_tf_rd_014_muon_missingness_medium_v1_queue_keeps_the_carried_264x6_surface_fixed() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == SWEEP_ID
    assert sweep["parent_sweep_id"] == PARENT_SWEEP_ID
    assert sweep["status"] == "ready"
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert_training_surface_semantics(
        sweep,
        training_experiment=MUON_EXPERIMENT,
        training_config_profile=MUON_EXPERIMENT,
        surface_role="classification_scaling_law",
        comparison_policy="anchor_only",
        external_benchmarks=[],
    )
    notes = sweep["anchor_surface"]["notes"]
    assert isinstance(notes, list)
    assert any("refreshed from the hub-owned `openml_classification_medium_v1.json`" in note for note in notes)
    assert any("multiclass allow-missing validation rung" in note for note in notes)
    assert any("exploratory TF-RD-014 evidence only" in note for note in notes)

    rows = queue["rows"]
    assert [row["delta_ref"] for row in rows] == EXPECTED_ROWS
    assert [row["status"] for row in rows] == ["ready", "ready", "ready", "ready"]
    assert [row["run_id"] for row in rows] == [None, None, None, None]
    assert [row["decision"] for row in rows] == [None, None, None, None]
    assert [row["interpretation_status"] for row in rows] == ["pending", "pending", "pending", "pending"]
    assert [row["data"]["corpus_ref"] for row in rows] == EXPECTED_CORPUS_REFS
    assert {row["model"]["d_icl"] for row in rows} == {264}
    assert {row["model"]["sandwich_layers"] for row in rows} == {6}
    assert {row["model"]["sandwich_heads"] for row in rows} == {1}
    assert {row["model"]["sandwich_latents"] for row in rows} == {24}
    assert {row["training"]["task_batch_size"] for row in rows} == {16}
    assert {row["training"]["prior_dump_batch_size"] for row in rows} == {64}
    assert {row["benchmark_checkpoint_selection"] for row in rows} == {"best_and_final"}
    assert {row["execution_policy"] for row in rows} == {"benchmark_full"}
    assert {row["reuse_train_artifact"] for row in rows} == {None}
    assert {row["training"]["overrides"]["optimizer"]["name"] for row in rows} == {"muon"}
    assert {tuple(row["training"]["overrides"]["optimizer"]["betas"]) for row in rows} == {(0.9, 0.95)}
    assert {row["training"]["overrides"]["runtime"]["mixed_precision"] for row in rows} == {"bf16"}
    assert {row["training"]["overrides"]["runtime"]["compile_dynamic"] for row in rows} == {True}
    assert {row["training"]["overrides"]["runtime"]["compile_backend"] for row in rows} == {"eager"}
    assert {row["training"]["overrides"]["runtime"]["loader_task_batch_cache_mode"] for row in rows} == {
        "bounded_streaming"
    }
    assert {row["training"]["overrides"]["runtime"]["max_steps"] for row in rows} == {2500}
    assert rows[0]["training"]["one_epoch_contract"]["enabled"] is True
    assert rows[0]["training"]["one_epoch_contract"]["scope"] == "train_manifest_unique_tasks"
    assert all("one_epoch_contract" not in row["training"] for row in rows[1:])
    assert "retrain unless that artifact is restored" in " ".join(rows[0]["notes"])
    assert "exploratory and non-promotable" in " ".join(rows[1]["notes"])
    assert "does not meet the strict no-repeat `160000`-task one-epoch contract" in " ".join(rows[1]["notes"])

    materialized = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    assert materialized["anchor_run_id"] == ANCHOR_RUN_ID
    assert [row["delta_id"] for row in materialized["rows"]] == EXPECTED_ROWS
    assert [row["data"]["corpus_ref"] for row in materialized["rows"]] == EXPECTED_CORPUS_REFS
    assert all(row["training"]["task_batch_size"] == 16 for row in materialized["rows"])
    assert all(row["training"]["overrides"]["runtime"]["max_steps"] == 2500 for row in materialized["rows"])
    assert all(row["training"]["overrides"]["runtime"]["grad_accum_steps"] == 4 for row in materialized["rows"])
    assert all(row["training"]["overrides"]["runtime"]["compile_dynamic"] is True for row in materialized["rows"])
    assert all(row["training"]["overrides"]["optimizer"]["name"] == "muon" for row in materialized["rows"])
    assert all(row["benchmark_checkpoint_selection"] == "best_and_final" for row in materialized["rows"])


def test_tf_rd_014_muon_missingness_medium_v1_generated_artifacts_render_the_exploratory_package() -> None:
    resolved = _load_yaml(
        REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "resolved_queue.yaml"
    )
    matrix = (
        REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID / "matrix.md"
    ).read_text(encoding="utf-8")

    assert resolved["sweep_id"] == SWEEP_ID
    assert resolved["anchor_run_id"] == ANCHOR_RUN_ID
    assert [row["delta_id"] for row in resolved["rows"]] == EXPECTED_ROWS
    assert [row["status"] for row in resolved["rows"]] == ["ready", "ready", "ready", "ready"]
    assert [row["run_id"] for row in resolved["rows"]] == [None, None, None, None]
    assert "tf_rd_014_muon_missingness_medium_v1" in matrix
    assert "Sweep status: `ready`" in matrix
    assert f"Anchor run id: `{ANCHOR_RUN_ID}`" in matrix
    assert "refreshed hub-owned medium classification manifest" in matrix
    assert "tf_rd_010_missingness_mcar_v3" in matrix
    assert "tf_rd_010_missingness_mar_v3" in matrix
    assert "tf_rd_010_missingness_mnar_v3" in matrix
    assert "best_and_final" in matrix


def test_tf_rd_014_muon_missingness_medium_v1_inspection_cfg_keeps_effective_corpus_overrides() -> None:
    materialized = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    training_experiment = resolve_training_surface_context(materialized).training_experiment

    for row, expected_corpus_ref in zip(materialized["rows"], EXPECTED_CORPUS_REFS, strict=True):
        raw_cfg = inspection_raw_cfg_mapping(
            row=row,
            training_experiment=training_experiment,
            sweep_id=SWEEP_ID,
        )
        data_cfg = raw_cfg["data"]
        assert data_cfg["surface_label"] == row["data"]["surface_label"]
        assert data_cfg["surface_overrides"]["corpus_ref"] == expected_corpus_ref
        assert data_cfg["surface_overrides"]["source"] == "manifest"

    assert raw_cfg["logging"]["use_wandb"] is True
    assert raw_cfg["runtime"]["compile_backend"] == "eager"
    assert raw_cfg["runtime"]["compile_dynamic"] is True
