from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import tab_foundry.research.navigation as navigation_module


def test_validate_sweep_contract_reports_missing_default_anchor_manifest(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "data" / "manifests" / "bench" / "openml_classification_medium_v1" / "manifest.parquet"
    queue = {
        "sweep_id": "test_sweep",
        "anchor_run_id": "anchor_run",
        "complexity_level": "classification_md",
        "benchmark_manifest_path": str(manifest_path),
        "control_baseline_id": "cls_benchmark_linear_multiclass_medium_v1",
    }
    index_entry = SimpleNamespace(
        anchor_run_id="anchor_run",
        complexity_level="classification_md",
        benchmark_manifest_path=str(manifest_path),
        control_baseline_id="cls_benchmark_linear_multiclass_medium_v1",
    )

    monkeypatch.setattr(navigation_module, "default_benchmark_manifest_path", lambda: manifest_path)
    monkeypatch.setattr(
        navigation_module,
        "load_system_delta_index_payload",
        lambda _index_path=None: SimpleNamespace(sweeps={"test_sweep": index_entry}),
    )

    issues = navigation_module.validate_sweep_contract(queue=queue)

    assert issues == ["test_sweep: default anchor benchmark manifest is not materialized locally"]


def test_build_sweep_navigation_payload_requires_control_baseline_for_default_anchor(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "data" / "manifests" / "bench" / "openml_classification_medium_v1" / "manifest.parquet"
    queue = {
        "sweep_id": "test_sweep",
        "rows": [],
        "benchmark_manifest_path": str(manifest_path),
        "control_baseline_id": "cls_benchmark_linear_v2",
        "training_experiment": "exp",
        "training_config_profile": "profile",
        "surface_role": "classification_training_dynamics_transfer",
        "comparison_policy": "anchor_only",
    }

    monkeypatch.setattr(navigation_module, "default_benchmark_manifest_path", lambda: manifest_path)
    monkeypatch.setattr(navigation_module, "default_anchor_benchmark_summary", lambda: {"name": "bundle"})
    monkeypatch.setattr(navigation_module, "sweep_lineage_entries", lambda **_: [])
    monkeypatch.setattr(navigation_module, "_scan_linked_scaling_studies", lambda **_: [])
    monkeypatch.setattr(navigation_module, "_winner_from_rows", lambda _rows: None)
    monkeypatch.setattr(navigation_module, "validate_sweep_contract", lambda **_: [])

    payload = navigation_module.build_sweep_navigation_payload(queue=queue)

    assert payload["contract"]["uses_default_anchor_benchmark"] is False


def test_build_scaling_navigation_payload_marks_default_anchor_only_when_baseline_matches(
    monkeypatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "data" / "manifests" / "bench" / "openml_classification_medium_v1" / "manifest.parquet"
    queue = {
        "sweep_id": "test_sweep",
        "rows": [],
        "status": "draft",
        "benchmark_manifest_path": str(manifest_path),
        "control_baseline_id": "cls_benchmark_linear_multiclass_medium_v1",
        "anchor_run_id": "anchor_run",
        "training_experiment": "exp",
        "training_config_profile": "profile",
        "upstream_reference": {"name": "paper"},
    }
    config = SimpleNamespace(
        study_id="study",
        sweeps=[SimpleNamespace(name="lane", family="transfer", sweep_id="test_sweep")],
        phase1_reference_sweep_id=None,
        geometry_row_labels=[],
        step_ladder=[],
        batch_grad_accum_ladder=[],
        historical_context_studies=[],
        primary_fit=None,
        frozen_contract=None,
        output_root_path=lambda: tmp_path / "outputs",
        validation_overlay_resolved_path=lambda: tmp_path / "overlay.json",
    )

    monkeypatch.setattr(navigation_module, "default_benchmark_manifest_path", lambda: manifest_path)
    monkeypatch.setattr(navigation_module, "default_anchor_benchmark_summary", lambda: {"name": "bundle"})
    monkeypatch.setattr(navigation_module, "load_system_delta_queue", lambda **_: queue)
    monkeypatch.setattr(navigation_module, "validate_sweep_contract", lambda **_: [])
    monkeypatch.setattr(navigation_module, "build_sweep_navigation_payload", lambda **_: {"contract": {"corpus_ref": None}})
    monkeypatch.setattr(navigation_module, "sweep_lineage_entries", lambda **_: [])
    monkeypatch.setattr(navigation_module, "_winner_from_rows", lambda _rows: None)

    payload = navigation_module.build_scaling_navigation_payload(
        config=config,
        points=[],
        index_path=tmp_path / "index.yaml",
        catalog_path=tmp_path / "catalog.yaml",
        sweeps_root=tmp_path / "sweeps",
    )

    assert payload["contract"]["uses_default_anchor_benchmark"] is True
