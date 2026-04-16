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
WIDTH_DEPTH_EXPECTED_ROWS = [
    "delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1",
    "delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1",
    "delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1",
    "delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1",
    "delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1",
]
WIDTH_DEPTH_RUN_IDS = {
    "sd_tf_rd_009_muon_width_depth_medium_v1_01_delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1_v1",
    "sd_tf_rd_009_muon_width_depth_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1_v1",
    "sd_tf_rd_009_muon_width_depth_medium_v1_03_delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1_v1",
    "sd_tf_rd_009_muon_width_depth_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1_v1",
    "sd_tf_rd_009_muon_width_depth_medium_v1_05_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1",
}
WIDTH_SCREEN_RUN_IDS = {
    "sd_tf_rd_009_muon_width_screen_medium_v1_01_delta_tf_rd_024_followup_cls_sandwich_heads1_v1_v1",
    "sd_tf_rd_009_muon_width_screen_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl48_v1_v1",
    "sd_tf_rd_009_muon_width_screen_medium_v1_03_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1",
    "sd_tf_rd_009_muon_width_screen_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1",
}


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
        "status": "completed",
        "anchor_run_id": "sd_tf_rd_009_muon_width_screen_medium_v1_01_delta_tf_rd_024_followup_cls_sandwich_heads1_v1_v1",
        "complexity_level": "classification_md",
        "benchmark_manifest_path": "data/manifests/bench/openml_classification_medium_v1/manifest.parquet",
        "control_baseline_id": "cls_benchmark_linear_multiclass_medium_v1",
        "external_benchmarks": [],
    }
    assert index["sweeps"][WIDTH_DEPTH]["parent_sweep_id"] == WIDTH_SCREEN
    assert (
        index["sweeps"][WIDTH_DEPTH]["anchor_run_id"]
        == "sd_tf_rd_009_muon_width_screen_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1"
    )
    assert index["sweeps"][NS_SWEEP]["parent_sweep_id"] == WIDTH_DEPTH
    assert index["sweeps"][BCRIT_SWEEP]["parent_sweep_id"] == NS_SWEEP
    assert index["sweeps"][UPPER_GATE]["parent_sweep_id"] == WIDTH_DEPTH
    assert index["sweeps"][UPPER_NS]["parent_sweep_id"] == UPPER_GATE


def test_tf_rd_009_muon_width_screen_tracks_the_bounded_48_60_96_128_family() -> None:
    sweep = _load_yaml(REPO_ROOT / "reference" / "system_delta_sweeps" / WIDTH_SCREEN / "sweep.yaml")
    queue = _queue(WIDTH_SCREEN)

    assert (
        sweep["anchor_run_id"]
        == "sd_tf_rd_009_muon_width_screen_medium_v1_01_delta_tf_rd_024_followup_cls_sandwich_heads1_v1_v1"
    )
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
        "delta_tf_rd_024_followup_cls_sandwich_heads1_v1",
        "delta_tf_rd_009_cls_sandwich_dicl48_v1",
        "delta_tf_rd_009_cls_sandwich_dicl96_v1",
        "delta_tf_rd_009_cls_sandwich_dicl128_v1",
    ]
    assert rows[0]["model"]["d_icl"] == 60
    assert rows[0]["next_action"] == (
        "Keep `60x2` as the formal external Muon anchor, but do not carry it forward as the in-family diagonal baseline now that `128x2` has won the measured width screen."
    )
    assert rows[3]["decision"] == "keep"
    assert rows[3]["next_action"] == (
        "Carry `128x2` into `tf_rd_009_muon_width_depth_medium_v1` as the current in-family Muon baseline for the diagonal derivation."
    )
    assert Counter(f"{row['model']['d_icl']}x{row['model']['sandwich_layers']}" for row in rows) == {
        "48x2": 1,
        "60x2": 1,
        "96x2": 1,
        "128x2": 1,
    }
    assert {row["status"] for row in rows} == {"completed"}
    assert {
        row["run_id"] for row in rows
    } == {
        "sd_tf_rd_009_muon_width_screen_medium_v1_01_delta_tf_rd_024_followup_cls_sandwich_heads1_v1_v1",
        "sd_tf_rd_009_muon_width_screen_medium_v1_02_delta_tf_rd_009_cls_sandwich_dicl48_v1_v1",
        "sd_tf_rd_009_muon_width_screen_medium_v1_03_delta_tf_rd_009_cls_sandwich_dicl96_v1_v1",
        "sd_tf_rd_009_muon_width_screen_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1",
    }
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
    assert [row["model"]["d_icl"] for row in materialized["rows"]] == [60, 48, 96, 128]
    assert [row["model"]["sandwich_layers"] for row in materialized["rows"]] == [2, 2, 2, 2]


def test_tf_rd_009_muon_phase1_queue_is_materialized_while_phase2_and_upper_scaffolds_stay_empty() -> None:
    width_depth_sweep = _load_yaml(
        REPO_ROOT / "reference" / "system_delta_sweeps" / WIDTH_DEPTH / "sweep.yaml"
    )
    width_depth_queue = _queue(WIDTH_DEPTH)

    assert_training_surface_semantics(
        width_depth_sweep,
        training_experiment=MUON_EXPERIMENT,
        training_config_profile=MUON_EXPERIMENT,
        surface_role="classification_scaling_law",
        comparison_policy="anchor_only",
        external_benchmarks=[],
    )
    assert (
        width_depth_sweep["anchor_run_id"]
        == "sd_tf_rd_009_muon_width_screen_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1"
    )
    assert "72x1" in width_depth_sweep["anchor_surface"]["notes"][1]
    assert "264x6" in width_depth_sweep["anchor_surface"]["notes"][1]

    rows = width_depth_queue["rows"]
    assert [row["delta_ref"] for row in rows] == WIDTH_DEPTH_EXPECTED_ROWS
    assert Counter(f"{row['model']['d_icl']}x{row['model']['sandwich_layers']}" for row in rows) == {
        "72x1": 1,
        "112x3": 1,
        "144x4": 1,
        "192x5": 1,
        "264x6": 1,
    }
    assert {row["status"] for row in rows} == {"completed"}
    assert {row["interpretation_status"] for row in rows} == {"completed"}
    assert {row["run_id"] for row in rows} == WIDTH_DEPTH_RUN_IDS
    assert {row["decision"] for row in rows} == {"defer"}
    assert [row["benchmark_metrics"]["final_log_loss"] for row in rows] == [
        0.4134940239812736,
        0.4136881204817012,
        0.411648995053474,
        0.41458147691015207,
        0.40089922764318064,
    ]
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
    assert "historical schedulefree rows remain context only" in rows[0]["next_action"]
    assert "264x6" in rows[-1]["next_action"]

    materialized = load_system_delta_queue(
        sweep_id=WIDTH_DEPTH,
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
    assert [row["delta_id"] for row in materialized["rows"]] == WIDTH_DEPTH_EXPECTED_ROWS
    assert [row["model"]["d_icl"] for row in materialized["rows"]] == [72, 112, 144, 192, 264]
    assert [row["model"]["sandwich_layers"] for row in materialized["rows"]] == [1, 3, 4, 5, 6]

    resolved = _load_yaml(
        REPO_ROOT / "reference" / "system_delta_sweeps" / WIDTH_DEPTH / "resolved_queue.yaml"
    )
    assert [row["delta_id"] for row in resolved["rows"]] == WIDTH_DEPTH_EXPECTED_ROWS

    matrix = (REPO_ROOT / "reference" / "system_delta_sweeps" / WIDTH_DEPTH / "matrix.md").read_text(
        encoding="utf-8"
    )
    assert "delta_tf_rd_009_cls_sandwich_dicl144_layers4_v1" in matrix
    assert "delta_tf_rd_009_cls_sandwich_dicl192_layers5_v1" in matrix
    assert "delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1" in matrix
    assert "historical schedulefree rows remain context only" in matrix

    for sweep_id, expected_role in (
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
    assert planned["sweep_id"] == WIDTH_DEPTH
    assert planned["runtime_profile"] == MUON_EXPERIMENT
    assert planned["config_profile"] == MUON_EXPERIMENT
    assert (
        planned["formal_anchor_run_id"]
        == "sd_tf_rd_009_muon_width_screen_medium_v1_01_delta_tf_rd_024_followup_cls_sandwich_heads1_v1_v1"
    )
    assert (
        planned["baseline_run_id"]
        == "sd_tf_rd_009_muon_width_screen_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1"
    )
    assert (
        planned["preferred_run_id"]
        == "sd_tf_rd_009_muon_width_depth_medium_v1_05_delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1_v1"
    )
    assert set(planned["evidence_run_ids"]) == WIDTH_SCREEN_RUN_IDS | WIDTH_DEPTH_RUN_IDS
    assert planned["preferred_delta_ref"] == "delta_tf_rd_009_cls_sandwich_dicl264_layers6_v1"
    assert planned["preferred_architecture"]["d_icl"] == 264
    assert planned["preferred_architecture"]["sandwich_layers"] == 6
    assert planned["preferred_architecture"]["sandwich_heads"] == 1
    assert planned["selection_rule"] == "planned_muon_phase1_materialized_pending_benchmark_freeze"
    assert "264x6" in planned["rationale"]
    assert "large-rung Muon validation" in planned["rationale"]
    assert historical["decision"] == "keep"
    assert historical["sweep_id"] == "tf_rd_009_width_depth_medium_v1"
