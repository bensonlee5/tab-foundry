from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.hardware_architecture_registry import load_hardware_architecture_registry
from tab_foundry.research.scaling.study import load_scaling_study_config
from tab_foundry.research.sweep.materialize import load_system_delta_queue
from tests.support_research.helpers import assert_training_surface_semantics


REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml"
CATALOG_PATH = REPO_ROOT / "reference" / "system_delta_catalog.yaml"
MUON_EXPERIMENT = "cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1"
WIDTH_SCREEN = "tf_rd_009_muon_width_screen_medium_v1"
WIDTH_DEPTH = "tf_rd_009_muon_width_depth_medium_v1"
NS_SWEEP = "tf_rd_009_muon_ns_one_epoch_medium_v1"
BCRIT_SWEEP = "tf_rd_009_muon_batch_critical_one_epoch_medium_v1"
UPPER_GATE = "tf_rd_009_muon_width_depth_upper_extension_one_epoch_medium_v1"
UPPER_NS = "tf_rd_009_muon_ns_upper_extension_one_epoch_medium_v1"
PHASE2_STUDY = "tf_rd_009_muon_phase2_one_epoch_v1"
UPPER_STUDY = "tf_rd_009_muon_phase2_upper_extension_one_epoch_v1"
CORPUS_V6 = "tf_rd_010_dagzoo_medium_control_curated_v6"
MUON_BASELINE_ID = "tf_rd_009_rtx8000_44gb_classification_medium_muon_v1"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _queue(sweep_id: str) -> dict[str, Any]:
    return _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / sweep_id / "queue.yaml")


def test_tf_rd_009_muon_width_screen_is_registered_as_a_fresh_family() -> None:
    index = _load_yaml(INDEX_PATH)

    assert index["sweeps"][WIDTH_SCREEN] == {
        "parent_sweep_id": "tf_rd_024_classification_heads_prerow_followup_v1",
        "status": "draft",
        "anchor_run_id": None,
        "complexity_level": "classification_md",
        "benchmark_manifest_path": "data/manifests/bench/openml_classification_medium_v1/manifest.parquet",
        "control_baseline_id": "cls_benchmark_linear_multiclass_medium_v1",
        "external_benchmarks": [],
    }
    assert index["sweeps"][WIDTH_DEPTH]["parent_sweep_id"] == WIDTH_SCREEN
    assert index["sweeps"][NS_SWEEP]["parent_sweep_id"] == WIDTH_DEPTH
    assert index["sweeps"][BCRIT_SWEEP]["parent_sweep_id"] == NS_SWEEP
    assert index["sweeps"][UPPER_GATE]["parent_sweep_id"] == WIDTH_DEPTH
    assert index["sweeps"][UPPER_NS]["parent_sweep_id"] == UPPER_GATE


def test_tf_rd_009_muon_width_screen_tracks_the_bounded_48_60_96_128_family() -> None:
    sweep = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / WIDTH_SCREEN / "sweep.yaml")
    queue = _queue(WIDTH_SCREEN)

    assert sweep["anchor_run_id"] is None
    assert_training_surface_semantics(
        sweep,
        training_experiment=MUON_EXPERIMENT,
        training_config_profile=MUON_EXPERIMENT,
        surface_role="classification_scaling_law",
        comparison_policy="anchor_only",
        external_benchmarks=[],
    )

    rows = queue["rows"]
    assert [row["delta_ref"] for row in rows] == [
        "delta_tf_rd_009_cls_sandwich_dicl48_v1",
        "delta_tf_rd_024_followup_cls_sandwich_heads1_v1",
        "delta_tf_rd_009_cls_sandwich_dicl96_v1",
        "delta_tf_rd_009_cls_sandwich_dicl128_v1",
    ]
    assert Counter(f"{row['model']['d_icl']}x{row['model']['sandwich_layers']}" for row in rows) == {
        "48x2": 1,
        "60x2": 1,
        "96x2": 1,
        "128x2": 1,
    }
    assert {row["status"] for row in rows} == {"ready"}
    assert {row["run_id"] for row in rows} == {None}
    assert {row["data"]["corpus_ref"] for row in rows} == {CORPUS_V6}
    assert {row["training"]["overrides"]["optimizer"]["name"] for row in rows} == {"muon"}
    assert {row["training"]["overrides"]["optimizer"]["weight_decay"] for row in rows} == {0.01}
    assert {tuple(row["training"]["overrides"]["optimizer"]["betas"]) for row in rows} == {(0.9, 0.95)}
    assert {row["training"]["overrides"]["runtime"]["compile_dynamic"] for row in rows} == {True}
    assert {row["training"]["overrides"]["runtime"]["loader_task_batch_cache_mode"] for row in rows} == {
        "bounded_streaming"
    }
    assert {row["training"]["overrides"]["runtime"]["max_steps"] for row in rows} == {2500}
    assert {row["training"]["one_epoch_contract"]["corpus_ref"] for row in rows} == {CORPUS_V6}

    materialized = load_system_delta_queue(
        sweep_id=WIDTH_SCREEN,
        index_path=INDEX_PATH,
        catalog_path=CATALOG_PATH,
    )
    assert_training_surface_semantics(
        materialized,
        training_experiment=MUON_EXPERIMENT,
        training_config_profile=MUON_EXPERIMENT,
        surface_role="classification_scaling_law",
        comparison_policy="anchor_only",
        external_benchmarks=[],
    )
    assert [row["model"]["d_icl"] for row in materialized["rows"]] == [48, 60, 96, 128]
    assert [row["model"]["sandwich_layers"] for row in materialized["rows"]] == [2, 2, 2, 2]


def test_tf_rd_009_muon_phase1_and_phase2_scaffolds_stay_empty_until_muon_rows_land() -> None:
    for sweep_id, expected_role in (
        (WIDTH_DEPTH, "classification_scaling_law"),
        (NS_SWEEP, "classification_scaling_law_phase2_ns"),
        (BCRIT_SWEEP, "classification_scaling_law_phase2_batch"),
        (UPPER_GATE, "classification_scaling_law"),
        (UPPER_NS, "classification_scaling_law_phase2_ns"),
    ):
        sweep = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / sweep_id / "sweep.yaml")
        queue = _queue(sweep_id)
        assert_training_surface_semantics(
            sweep,
            training_experiment=MUON_EXPERIMENT,
            training_config_profile=MUON_EXPERIMENT,
            surface_role=expected_role,
            comparison_policy="anchor_only",
            external_benchmarks=[],
        )
        assert queue["rows"] == []


def test_tf_rd_009_muon_scaling_studies_reference_only_muon_sweeps() -> None:
    phase2 = _load_yaml(REPO_ROOT / "reference" / "scaling_studies" / f"{PHASE2_STUDY}.yaml")
    upper = _load_yaml(REPO_ROOT / "reference" / "scaling_studies" / f"{UPPER_STUDY}.yaml")

    assert load_scaling_study_config(study_id=PHASE2_STUDY).study_id == PHASE2_STUDY
    assert phase2["sweeps"] == [
        {"name": "ns_core", "sweep_id": NS_SWEEP, "family": "ns_core"},
        {"name": "batch_critical", "sweep_id": BCRIT_SWEEP, "family": "batch_critical"},
    ]
    assert phase2["geometry_row_labels"] == []
    assert phase2["primary_fit"] == {"law": "L(N,S)", "target": "validation_loss"}
    assert phase2["historical_context_studies"] == ["tf_rd_009_phase2", "tf_rd_009_phase2_one_epoch_v1"]

    assert load_scaling_study_config(study_id=UPPER_STUDY).study_id == UPPER_STUDY
    assert upper["sweeps"] == [{"name": "ns_core", "sweep_id": UPPER_NS, "family": "ns_core"}]
    assert upper["selection_dependency"] == PHASE2_STUDY
    assert upper["geometry_row_labels"] == []


def test_tf_rd_009_muon_hardware_baseline_placeholder_is_separate_from_historical_entry() -> None:
    registry = load_hardware_architecture_registry()
    planned = registry["baselines"][MUON_BASELINE_ID]
    historical = registry["baselines"]["tf_rd_009_rtx8000_44gb_classification_medium_v1"]

    assert planned["decision"] == "planned"
    assert planned["sweep_id"] == WIDTH_SCREEN
    assert planned["runtime_profile"] == MUON_EXPERIMENT
    assert planned["config_profile"] == MUON_EXPERIMENT
    assert planned["formal_anchor_run_id"] == "pending_tf_rd_009_muon_width_screen_60x2"
    assert planned["preferred_run_id"] == "pending_tf_rd_009_muon_width_screen_60x2"
    assert planned["preferred_architecture"]["d_icl"] == 60
    assert planned["preferred_architecture"]["sandwich_layers"] == 2
    assert planned["preferred_architecture"]["sandwich_heads"] == 1
    assert historical["decision"] == "keep"
    assert historical["sweep_id"] == "tf_rd_009_width_depth_medium_v1"
