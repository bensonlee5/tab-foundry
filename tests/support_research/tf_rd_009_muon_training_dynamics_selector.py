from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.scaling.fit import inspect_scaling_study


REPO_ROOT = Path(__file__).resolve().parents[2]
INDEX_PATH = REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml"
SWEEPS_ROOT = REPO_ROOT / "reference" / "system_delta_sweeps"
REGISTRY_PATH = REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json"

SELECTOR_SWEEP = "tf_rd_009_muon_training_dynamics_endpoint_medium_v1"
SCREEN_SWEEP = "tf_rd_009_muon_training_dynamics_transfer_screen_medium_v1"
TRANSFER_SWEEP = "tf_rd_009_muon_training_dynamics_transfer_medium_v1"
NS_SWEEP = "tf_rd_009_muon_ns_one_epoch_medium_v1"
BCRIT_SWEEP = "tf_rd_009_muon_batch_critical_one_epoch_medium_v1"
WIDTH_DEPTH = "tf_rd_009_muon_width_depth_medium_v1"
PHASE2_STUDY = "tf_rd_009_muon_phase2_one_epoch_v1"
ANCHOR_RUN_ID = "sd_tf_rd_009_muon_width_screen_medium_v1_04_delta_tf_rd_009_cls_sandwich_dicl128_v1_v1"
BENCHMARK_MANIFEST = "data/manifests/bench/openml_classification_medium_v1/manifest.parquet"
CONTROL_BASELINE = "cls_benchmark_linear_multiclass_medium_v1"
PAPER_URL = "https://arxiv.org/abs/2603.15958"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def test_tf_rd_009_muon_endpoint_selector_is_preserved_as_superseded_context() -> None:
    index = _load_yaml(INDEX_PATH)
    sweep = _load_yaml(SWEEPS_ROOT / SELECTOR_SWEEP / "sweep.yaml")
    queue = _load_yaml(SWEEPS_ROOT / SELECTOR_SWEEP / "queue.yaml")

    assert index["sweeps"][SELECTOR_SWEEP] == {
        "parent_sweep_id": BCRIT_SWEEP,
        "status": "superseded",
        "anchor_run_id": ANCHOR_RUN_ID,
        "complexity_level": "classification_md",
        "benchmark_manifest_path": BENCHMARK_MANIFEST,
        "control_baseline_id": CONTROL_BASELINE,
        "external_benchmarks": [],
    }
    assert sweep["status"] == "superseded"
    assert sweep["surface_role"] == "classification_training_dynamics_selector"
    assert sweep["parent_sweep_id"] == BCRIT_SWEEP
    assert len(queue["rows"]) == 12
    assert any("Superseded by the faithful paper-derived transfer screen" in note for note in sweep["anchor_surface"]["notes"])


def test_tf_rd_009_muon_phase2_inspection_still_points_to_the_corrected_medium_contract() -> None:
    payload = inspect_scaling_study(
        study_id=PHASE2_STUDY,
        registry_path=REGISTRY_PATH,
        index_path=INDEX_PATH,
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
        sweeps_root=SWEEPS_ROOT,
    )

    navigation = payload["navigation"]
    assert [entry["sweep_id"] for entry in navigation["linked_sweeps"]] == [NS_SWEEP, BCRIT_SWEEP]
    assert navigation["contract"]["benchmark_manifest_path"] == BENCHMARK_MANIFEST
    assert navigation["contract"]["control_baseline_id"] == CONTROL_BASELINE
    assert navigation["contract"]["phase1_reference_sweep_id"] == WIDTH_DEPTH
    assert navigation["winner"]["geometry_label"] == "264x6"
    assert navigation["winner"]["row_order"] == 20
    assert navigation["fit_audit_state"]["full_scope_ready"] is True
    assert navigation["contract_issues"] == []
