from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.scaling.fit import inspect_scaling_study
from tab_foundry.research.sweep.materialize import load_system_delta_queue
from tab_foundry.research.navigation import build_sweep_navigation_payload
from tests.support_research.helpers import assert_training_surface_semantics


REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml"
CATALOG_PATH = REPO_ROOT / "reference" / "system_delta_catalog.yaml"
SWEEPS_ROOT = REPO_ROOT / "reference" / "system_delta_sweeps"
REGISTRY_PATH = REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json"

WIDTH_SCREEN = "tf_rd_009_muon_width_screen_medium_v1"
WIDTH_DEPTH = "tf_rd_009_muon_width_depth_medium_v1"
NS_SWEEP = "tf_rd_009_muon_ns_one_epoch_medium_v1"
BCRIT_SWEEP = "tf_rd_009_muon_batch_critical_one_epoch_medium_v1"
SELECTOR_SWEEP = "tf_rd_009_muon_training_dynamics_endpoint_medium_v1"
PHASE2_STUDY = "tf_rd_009_muon_phase2_one_epoch_v1"
ANCHOR_RUN_ID = "sd_tf_rd_009_muon_width_screen_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1"
MUON_EXPERIMENT = "cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1"
CORPUS_V6 = "tf_rd_010_dagzoo_medium_control_curated_v6"
BENCHMARK_MANIFEST = "data/manifests/bench/openml_classification_medium_v1/manifest.parquet"
CONTROL_BASELINE = "cls_benchmark_linear_multiclass_medium_v1"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def test_tf_rd_009_muon_training_dynamics_selector_is_registered() -> None:
    index = _load_yaml(INDEX_PATH)

    assert index["sweeps"][SELECTOR_SWEEP] == {
        "parent_sweep_id": BCRIT_SWEEP,
        "status": "draft",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "classification_md",
        "benchmark_manifest_path": BENCHMARK_MANIFEST,
        "control_baseline_id": CONTROL_BASELINE,
        "external_benchmarks": [],
    }


def test_tf_rd_009_muon_training_dynamics_selector_tracks_the_exact_12_row_endpoint_matrix() -> None:
    sweep_root = SWEEPS_ROOT / SELECTOR_SWEEP
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")

    assert sweep["sweep_id"] == SELECTOR_SWEEP
    assert sweep["parent_sweep_id"] == BCRIT_SWEEP
    assert sweep["anchor_run_id"] == ANCHOR_RUN_ID
    assert_training_surface_semantics(
        sweep,
        training_experiment=MUON_EXPERIMENT,
        training_config_profile=MUON_EXPERIMENT,
        surface_role="classification_training_dynamics_selector",
        comparison_policy="anchor_only",
        external_benchmarks=[],
    )

    rows = queue["rows"]
    assert len(rows) == 12
    assert {row["status"] for row in rows} == {"ready"}
    assert {row["interpretation_status"] for row in rows} == {"pending"}
    assert {row["benchmark_checkpoint_selection"] for row in rows} == {"best_and_final"}
    assert {row["data"]["corpus_ref"] for row in rows} == {CORPUS_V6}
    assert {row["training"]["task_batch_size"] for row in rows} == {16}
    assert {row["training"]["overrides"]["runtime"]["max_steps"] for row in rows} == {5000}
    assert Counter(f"{row['model']['d_icl']}x{row['model']['sandwich_layers']}" for row in rows) == {
        "128x2": 4,
        "144x4": 4,
        "264x6": 4,
    }
    assert Counter(row["training"]["overrides"]["runtime"]["grad_accum_steps"] for row in rows) == {
        4: 3,
        16: 9,
    }

    expected_by_suffix = {
        "carry_lowbatch_v1": {"grad_accum_steps": 4, "lr_max": 0.001, "min_lr": 1e-6, "betas": (0.9, 0.95)},
        "carry_highbatch_v1": {"grad_accum_steps": 16, "lr_max": 0.001, "min_lr": 1e-6, "betas": (0.9, 0.95)},
        "linear_lr_batch_v1": {"grad_accum_steps": 16, "lr_max": 0.004, "min_lr": 4e-6, "betas": (0.9, 0.95)},
        "momentum_timescale_v1": {"grad_accum_steps": 16, "lr_max": 0.004, "min_lr": 4e-6, "betas": (0.975, 0.95)},
    }
    for row in rows:
        delta_ref = str(row["delta_ref"])
        suffix = next(key for key in expected_by_suffix if delta_ref.endswith(key))
        expected = expected_by_suffix[suffix]
        runtime = row["training"]["overrides"]["runtime"]
        optimizer = row["training"]["overrides"]["optimizer"]
        schedule = row["training"]["overrides"]["schedule"]["stages"][0]
        assert runtime["grad_accum_steps"] == expected["grad_accum_steps"]
        assert schedule["lr_max"] == expected["lr_max"]
        assert optimizer["min_lr"] == expected["min_lr"]
        assert tuple(optimizer["betas"]) == expected["betas"]
        assert optimizer["name"] == "muon"
        assert optimizer["weight_decay"] == 0.01


def test_tf_rd_009_muon_training_dynamics_selector_materializes_and_exposes_the_active_lineage() -> None:
    materialized = load_system_delta_queue(
        sweep_id=SELECTOR_SWEEP,
        index_path=INDEX_PATH,
        catalog_path=CATALOG_PATH,
        sweeps_root=SWEEPS_ROOT,
    )
    assert_training_surface_semantics(
        materialized,
        training_experiment=MUON_EXPERIMENT,
        training_config_profile=MUON_EXPERIMENT,
        surface_role="classification_training_dynamics_selector",
        comparison_policy="anchor_only",
        external_benchmarks=[],
    )
    assert [row["delta_id"] for row in materialized["rows"]] == [
        "delta_tf_rd_009_cls_sandwich_dicl128_layers2_muon_carry_lowbatch_v1",
        "delta_tf_rd_009_cls_sandwich_dicl128_layers2_muon_carry_highbatch_v1",
        "delta_tf_rd_009_cls_sandwich_dicl128_layers2_muon_linear_lr_batch_v1",
        "delta_tf_rd_009_cls_sandwich_dicl128_layers2_muon_momentum_timescale_v1",
        "delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_carry_lowbatch_v1",
        "delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_carry_highbatch_v1",
        "delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_linear_lr_batch_v1",
        "delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_momentum_timescale_v1",
        "delta_tf_rd_009_cls_sandwich_dicl264_layers6_muon_carry_lowbatch_v1",
        "delta_tf_rd_009_cls_sandwich_dicl264_layers6_muon_carry_highbatch_v1",
        "delta_tf_rd_009_cls_sandwich_dicl264_layers6_muon_linear_lr_batch_v1",
        "delta_tf_rd_009_cls_sandwich_dicl264_layers6_muon_momentum_timescale_v1",
    ]

    navigation = build_sweep_navigation_payload(queue=materialized, index_path=INDEX_PATH)
    assert [entry["sweep_id"] for entry in navigation["lineage"]][-5:] == [
        WIDTH_SCREEN,
        WIDTH_DEPTH,
        NS_SWEEP,
        BCRIT_SWEEP,
        SELECTOR_SWEEP,
    ]
    assert navigation["contract"]["benchmark_manifest_path"] == BENCHMARK_MANIFEST
    assert navigation["contract"]["uses_default_anchor_benchmark"] is True
    assert navigation["contract"]["default_anchor_benchmark"]["benchmark_manifest_path"] == BENCHMARK_MANIFEST
    assert (
        navigation["contract"]["default_anchor_benchmark"]["benchmark_bundle"]["name"]
        == "openml_classification_medium"
    )
    assert navigation["contract"]["control_baseline_id"] == CONTROL_BASELINE
    assert navigation["contract"]["corpus_ref"] == CORPUS_V6
    assert navigation["contract"]["carried_in_family_baseline_run_id"] == ANCHOR_RUN_ID
    assert navigation["winner"] is None
    assert navigation["contract_issues"] == []

    resolved = _load_yaml(SWEEPS_ROOT / SELECTOR_SWEEP / "resolved_queue.yaml")
    assert [row["delta_id"] for row in resolved["rows"]] == [row["delta_id"] for row in materialized["rows"]]
    matrix = (SWEEPS_ROOT / SELECTOR_SWEEP / "matrix.md").read_text(encoding="utf-8")
    assert "quality/time Pareto frontier" in matrix
    assert "historical schedulefree TF-RD-009 remains preserved context only" in matrix


def test_tf_rd_009_muon_phase2_inspection_exposes_the_canonical_active_contract_and_runtime_metrics() -> None:
    payload = inspect_scaling_study(
        study_id=PHASE2_STUDY,
        registry_path=REGISTRY_PATH,
        index_path=INDEX_PATH,
        catalog_path=CATALOG_PATH,
        sweeps_root=SWEEPS_ROOT,
    )

    navigation = payload["navigation"]
    assert [entry["sweep_id"] for entry in navigation["linked_sweeps"]] == [NS_SWEEP, BCRIT_SWEEP]
    assert navigation["contract"]["benchmark_manifest_path"] == BENCHMARK_MANIFEST
    assert navigation["contract"]["uses_default_anchor_benchmark"] is True
    assert navigation["contract"]["default_anchor_benchmark"]["benchmark_manifest_path"] == BENCHMARK_MANIFEST
    assert navigation["contract"]["control_baseline_id"] == CONTROL_BASELINE
    assert navigation["contract"]["corpus_ref"] == CORPUS_V6
    assert navigation["contract"]["carried_in_family_baseline_run_id"] == ANCHOR_RUN_ID
    assert navigation["contract"]["phase1_reference_sweep_id"] == WIDTH_DEPTH
    assert navigation["contract"]["historical_context_studies"] == ["tf_rd_009_phase2", "tf_rd_009_phase2_one_epoch_v1"]
    assert navigation["winner"]["geometry_label"] == "264x6"
    assert navigation["winner"]["row_order"] == 20
    assert navigation["winner"]["benchmark_log_loss"] == 0.46464118874262356
    assert "throughput_tokens_per_second" in navigation["winner"]
    assert "end_to_end_wall_seconds" in navigation["winner"]
    assert navigation["completeness"]["all_expected_points_present"] is True
    assert navigation["fit_audit_state"]["full_scope_ready"] is True
    assert navigation["contract_issues"] == []
