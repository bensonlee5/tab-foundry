from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.sweep.materialize import load_system_delta_queue
from tests.support_research.helpers import assert_training_surface_semantics


REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml"
CATALOG_PATH = REPO_ROOT / "reference" / "system_delta_catalog.yaml"
SWEEPS_ROOT = REPO_ROOT / "reference" / "system_delta_sweeps"

SCREEN_SWEEP = "tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1"
TRANSFER_SWEEP = "tf_rd_009_muon_training_dynamics_transfer_medium_v1"
SELECTOR_SWEEP = "tf_rd_009_muon_training_dynamics_endpoint_medium_v1"
ANCHOR_RUN_ID = "sd_tf_rd_009_muon_width_screen_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1"
BENCHMARK_MANIFEST = "data/manifests/bench/openml_classification_medium_v1/manifest.parquet"
CONTROL_BASELINE = "cls_benchmark_linear_multiclass_medium_v1"
MUON_EXPERIMENT = "cls_benchmark_sandwich_classification_evolution_tf_rd_009_muon_medium_v1"
CORPUS_V6 = "tf_rd_010_dagzoo_medium_control_curated_v6"
PAPER_URL = "https://arxiv.org/abs/2603.15958"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def test_tf_rd_009_muon_transfer_screen_is_registered_and_rendered() -> None:
    index = _load_yaml(INDEX_PATH)
    sweep_root = SWEEPS_ROOT / SCREEN_SWEEP
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")
    resolved = _load_yaml(sweep_root / "resolved_queue.yaml")
    matrix = (sweep_root / "matrix.md").read_text(encoding="utf-8")

    assert index["sweeps"][SCREEN_SWEEP] == {
        "parent_sweep_id": "tf_rd_009_muon_batch_critical_one_epoch_medium_v1",
        "status": "draft",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "classification_md",
        "benchmark_manifest_path": BENCHMARK_MANIFEST,
        "control_baseline_id": CONTROL_BASELINE,
        "external_benchmarks": [],
    }
    assert sweep["upstream_reference"]["model_source"] == PAPER_URL
    assert_training_surface_semantics(
        sweep,
        training_experiment=MUON_EXPERIMENT,
        training_config_profile=MUON_EXPERIMENT,
        surface_role="classification_training_dynamics_transfer_screen",
        comparison_policy="anchor_only",
        external_benchmarks=[],
    )

    rows = queue["rows"]
    assert len(rows) == 30
    assert Counter(row["transfer_context"]["regime_label"] for row in rows) == {"B": 6, "D": 24}
    assert {row["execution_policy"] for row in rows} == {"screen_only"}
    assert {row["benchmark_checkpoint_selection"] for row in rows} == {"best_and_final"}
    assert {row["model"]["d_icl"] for row in rows} == {144}
    assert {row["model"]["sandwich_layers"] for row in rows} == {4}
    assert {row["data"]["corpus_ref"] for row in rows} == {CORPUS_V6}

    b_rows = [row for row in rows if row["transfer_context"]["regime_label"] == "B"]
    assert {row["training"]["overrides"]["runtime"]["grad_accum_steps"] for row in b_rows} == {4}
    assert {row["training"]["overrides"]["runtime"]["max_steps"] for row in b_rows} == {625}
    d_rows = [row for row in rows if row["transfer_context"]["regime_label"] == "D"]
    assert Counter(row["training"]["overrides"]["runtime"]["grad_accum_steps"] for row in d_rows) == {4: 6, 5: 6, 6: 6, 8: 6}
    assert [row["delta_id"] for row in resolved["rows"][:6]] == [
        "delta_tf_rd_009_cls_sandwich_dicl144_layers4_muon_transfer_regime_b_v1"
    ] * 6
    assert "faithful paper-derived transfer screen" in matrix.lower()


def test_tf_rd_009_muon_transfer_validation_tracks_the_logical_12_row_surface() -> None:
    index = _load_yaml(INDEX_PATH)
    sweep_root = SWEEPS_ROOT / TRANSFER_SWEEP
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")
    matrix = (sweep_root / "matrix.md").read_text(encoding="utf-8")

    assert index["sweeps"][TRANSFER_SWEEP] == {
        "parent_sweep_id": SCREEN_SWEEP,
        "status": "draft",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "classification_md",
        "benchmark_manifest_path": BENCHMARK_MANIFEST,
        "control_baseline_id": CONTROL_BASELINE,
        "external_benchmarks": [],
    }
    assert_training_surface_semantics(
        sweep,
        training_experiment=MUON_EXPERIMENT,
        training_config_profile=MUON_EXPERIMENT,
        surface_role="classification_training_dynamics_transfer",
        comparison_policy="anchor_only",
        external_benchmarks=[],
    )
    assert any("superseded" in note.lower() for note in sweep["anchor_surface"]["notes"])

    rows = queue["rows"]
    assert len(rows) == 12
    assert [row["transfer_context"]["target_budget_label"] for row in rows] == [
        "T0",
        "T1",
        "T2",
        "T0",
        "T1",
        "T2",
        "T0",
        "T1",
        "T2",
        "T0",
        "T1",
        "T2",
    ]
    lowbatch_rows = rows[:3]
    assert {row["transfer_context"]["regime_label"] for row in lowbatch_rows} == {"carry_lowbatch"}
    assert all(row["reuse_train_artifact"]["run_dir"].startswith("outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/") for row in lowbatch_rows)
    assert all(row["imported_baseline_provenance"]["source_sweep_id"] == "tf_rd_009_muon_ns_one_epoch_medium_v1" for row in lowbatch_rows)

    highbatch_rows = rows[3:6]
    assert {row["transfer_context"]["regime_label"] for row in highbatch_rows} == {"carry_highbatch"}
    assert Counter(row["training"]["overrides"]["runtime"]["max_steps"] for row in highbatch_rows) == {156: 1, 625: 1, 1250: 1}
    assert {row["training"]["overrides"]["runtime"]["grad_accum_steps"] for row in highbatch_rows} == {16}

    regime_b_rows = rows[6:9]
    assert {row["transfer_context"]["regime_label"] for row in regime_b_rows} == {"B"}
    assert all("dynamic_training_overrides" in row for row in regime_b_rows)
    assert all(row["training"]["overrides"]["runtime"]["grad_accum_steps"] == 4 for row in regime_b_rows)
    assert regime_b_rows[0]["dynamic_reuse_train_artifact"]["t0_winner"]["kind"] == "screen_winner_artifact"

    regime_d_rows = rows[9:12]
    assert {row["transfer_context"]["regime_label"] for row in regime_d_rows} == {"D"}
    assert all("dynamic_training_overrides" in row for row in regime_d_rows)
    assert regime_d_rows[0]["dynamic_reuse_train_artifact"]["t0_winner"]["kind"] == "screen_winner_artifact"

    materialized = load_system_delta_queue(
        sweep_id=TRANSFER_SWEEP,
        index_path=INDEX_PATH,
        catalog_path=CATALOG_PATH,
        sweeps_root=SWEEPS_ROOT,
    )
    assert materialized["surface_role"] == "classification_training_dynamics_transfer"
    assert len(materialized["rows"]) == 12
    assert materialized["rows"][0]["imported_baseline_provenance"]["source_order"] == 9
    assert materialized["rows"][6]["dynamic_training_overrides"]["transfer_schedule"]["regime_label"] == "B"
    assert materialized["rows"][9]["dynamic_training_overrides"]["transfer_schedule"]["regime_label"] == "D"
    assert "faithful transfer study" in matrix.lower()
