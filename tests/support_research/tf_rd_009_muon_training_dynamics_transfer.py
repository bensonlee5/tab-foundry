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

SELECTOR_SWEEP = "tf_rd_009_muon_training_dynamics_endpoint_medium_v1"
OLD_SCREEN_SWEEP = "tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1"
OLD_TRANSFER_SWEEP = "tf_rd_009_muon_training_dynamics_transfer_medium_v1"
LMO_SWEEP = "tf_rd_009_muon_training_dynamics_lmo_transfer_medium_v1"
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


def test_tf_rd_009_muon_screen_based_transfer_sweeps_are_preserved_as_superseded_context() -> None:
    index = _load_yaml(INDEX_PATH)
    screen = _load_yaml(SWEEPS_ROOT / OLD_SCREEN_SWEEP / "sweep.yaml")
    transfer = _load_yaml(SWEEPS_ROOT / OLD_TRANSFER_SWEEP / "sweep.yaml")
    selector = _load_yaml(SWEEPS_ROOT / SELECTOR_SWEEP / "sweep.yaml")

    assert index["sweeps"][OLD_SCREEN_SWEEP]["status"] == "superseded"
    assert index["sweeps"][OLD_TRANSFER_SWEEP]["status"] == "superseded"
    assert screen["status"] == "superseded"
    assert transfer["status"] == "superseded"
    assert any("superseded" in note.lower() for note in screen["anchor_surface"]["notes"])
    assert any("superseded" in note.lower() for note in transfer["anchor_surface"]["notes"])
    assert selector["status"] == "superseded"


def test_tf_rd_009_muon_lmo_transfer_is_registered_and_rendered() -> None:
    index = _load_yaml(INDEX_PATH)
    sweep_root = SWEEPS_ROOT / LMO_SWEEP
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")
    resolved = _load_yaml(sweep_root / "resolved_queue.yaml")
    matrix = (sweep_root / "matrix.md").read_text(encoding="utf-8")

    assert index["sweeps"][LMO_SWEEP] == {
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
        surface_role="classification_training_dynamics_transfer",
        comparison_policy="anchor_only",
        external_benchmarks=[],
    )
    assert any(
        "strict shared-anchor lmo transfer" in note.lower()
        for note in sweep["anchor_surface"]["notes"]
    )

    rows = queue["rows"]
    assert len(rows) == 10
    assert [row["transfer_context"]["target_budget_label"] for row in rows] == [
        "T0",
        "T1",
        "T2",
        "T0",
        "T1",
        "T2",
        "T1",
        "T2",
        "T1",
        "T2",
    ]
    assert {row["model"]["d_icl"] for row in rows} == {144}
    assert {row["model"]["sandwich_layers"] for row in rows} == {4}
    assert {row["data"]["corpus_ref"] for row in rows} == {CORPUS_V6}

    lowbatch_rows = rows[:3]
    assert {row["transfer_context"]["regime_label"] for row in lowbatch_rows} == {"carry_lowbatch"}
    assert [row["imported_baseline_provenance"]["source_order"] for row in lowbatch_rows] == [9, 11, 12]
    assert all(
        row["reuse_train_artifact"]["run_dir"].startswith(
            "outputs/staged_ladder/research/tf_rd_009_muon_ns_one_epoch_medium_v1/"
        )
        for row in lowbatch_rows
    )
    assert all(
        row["imported_baseline_provenance"]["source_sweep_id"] == "tf_rd_009_muon_ns_one_epoch_medium_v1"
        for row in lowbatch_rows
    )

    highbatch_rows = rows[3:6]
    assert {row["transfer_context"]["regime_label"] for row in highbatch_rows} == {"carry_highbatch"}
    assert Counter(
        row["training"]["overrides"]["runtime"]["max_steps"] for row in highbatch_rows
    ) == {156: 1, 625: 1, 1250: 1}
    assert {row["training"]["overrides"]["runtime"]["grad_accum_steps"] for row in highbatch_rows} == {16}

    regime_b_rows = rows[6:8]
    assert {row["transfer_context"]["regime_label"] for row in regime_b_rows} == {"B"}
    assert [row["transfer_context"]["target_budget_label"] for row in regime_b_rows] == ["T1", "T2"]
    assert all(
        row["dynamic_training_overrides"]["transfer_schedule"]["kind"] == "shared_anchor_transfer"
        for row in regime_b_rows
    )
    assert all(
        row["dynamic_training_overrides"]["transfer_schedule"]["anchor_order"] == 1
        for row in regime_b_rows
    )
    assert all(
        row["dynamic_training_overrides"]["transfer_schedule"]["anchor_sweep_id"] == LMO_SWEEP
        for row in regime_b_rows
    )
    assert all(row["dynamic_reuse_train_artifact"] is None for row in regime_b_rows)

    regime_d_rows = rows[8:10]
    assert {row["transfer_context"]["regime_label"] for row in regime_d_rows} == {"D"}
    assert [row["transfer_context"]["target_budget_label"] for row in regime_d_rows] == ["T1", "T2"]
    assert all(
        row["dynamic_training_overrides"]["transfer_schedule"]["kind"] == "shared_anchor_transfer"
        for row in regime_d_rows
    )
    assert all(
        row["dynamic_training_overrides"]["transfer_schedule"]["anchor_order"] == 1
        for row in regime_d_rows
    )
    assert all(
        row["dynamic_training_overrides"]["transfer_schedule"]["fixed_effective_batch"] is None
        for row in regime_d_rows
    )
    assert all(row["dynamic_reuse_train_artifact"] is None for row in regime_d_rows)

    materialized = load_system_delta_queue(
        sweep_id=LMO_SWEEP,
        index_path=INDEX_PATH,
        catalog_path=CATALOG_PATH,
        sweeps_root=SWEEPS_ROOT,
    )
    assert materialized["surface_role"] == "classification_training_dynamics_transfer"
    assert len(materialized["rows"]) == 10
    assert materialized["rows"][0]["imported_baseline_provenance"]["source_order"] == 9
    assert materialized["rows"][6]["training"]["overrides"]["runtime"]["grad_accum_steps"] == 4
    assert materialized["rows"][6]["training"]["overrides"]["runtime"]["max_steps"] == 2500
    assert materialized["rows"][7]["training"]["overrides"]["runtime"]["grad_accum_steps"] == 4
    assert materialized["rows"][7]["training"]["overrides"]["runtime"]["max_steps"] == 5000
    assert materialized["rows"][8]["training"]["overrides"]["runtime"]["grad_accum_steps"] == 5
    assert materialized["rows"][8]["training"]["overrides"]["runtime"]["max_steps"] == 2000
    assert materialized["rows"][9]["training"]["overrides"]["runtime"]["grad_accum_steps"] == 6
    assert materialized["rows"][9]["training"]["overrides"]["runtime"]["max_steps"] == 3333
    assert materialized["rows"][6]["transfer_resolution"]["shared_anchor_provenance"]["anchor_order"] == 1
    assert materialized["rows"][8]["transfer_resolution"]["shared_anchor_provenance"]["anchor_order"] == 1

    assert [row["order"] for row in resolved["rows"]] == list(range(1, 11))
    assert "strict shared-anchor lmo transfer" in matrix.lower()
    assert "screen_winner_transfer" not in matrix
