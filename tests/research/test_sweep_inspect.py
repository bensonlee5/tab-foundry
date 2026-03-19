from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf
import pytest
import torch

import tab_foundry.research.sweep.diff as diff_module
import tab_foundry.research.sweep.graph as graph_module
import tab_foundry.research.sweep.inspect as inspect_module
from tab_foundry.config import compose_config
from tab_foundry.model.spec import model_build_spec_from_mappings


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(OmegaConf.create(payload), resolve=True), encoding="utf-8")


def _row_model_payload(*, stage_label: str, tfrow_norm: str | None = None) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "stage": "row_cls_pool",
        "stage_label": stage_label,
        "d_icl": 32,
        "many_class_base": 4,
        "tficl_n_heads": 4,
        "tficl_n_layers": 1,
        "head_hidden_dim": 64,
        "tfrow_n_heads": 2,
        "tfrow_n_layers": 1,
        "tfrow_cls_tokens": 2,
        "tfcol_n_heads": 2,
        "tfcol_n_layers": 1,
        "tfcol_n_inducing": 8,
    }
    if tfrow_norm is not None:
        payload["tfrow_norm"] = tfrow_norm
    return payload


def _anchor_checkpoint_payload(run_dir: Path) -> dict[str, Any]:
    cfg = compose_config(
        [
            "experiment=cls_smoke",
            "logging.use_wandb=false",
            "model.stage=nano_exact",
            "model.stage_label=nano_exact_anchor_test",
            "model.d_icl=32",
            "model.many_class_base=4",
            "model.tficl_n_heads=4",
            "model.tficl_n_layers=1",
            "model.head_hidden_dim=64",
            "model.tfrow_n_heads=2",
            "model.tfrow_n_layers=1",
            "model.tfrow_cls_tokens=2",
            "model.tfcol_n_heads=2",
            "model.tfcol_n_layers=1",
            "model.tfcol_n_inducing=8",
            f"runtime.output_dir={run_dir}",
        ]
    )
    payload = OmegaConf.to_container(cfg, resolve=True)
    assert isinstance(payload, dict)
    return payload


def _training_surface_record_payload(
    *,
    stage_label: str,
    data_surface_label: str,
    preprocessing_surface_label: str,
    training_surface_label: str,
    tfrow_norm: str | None = None,
) -> dict[str, Any]:
    spec = model_build_spec_from_mappings(
        task="classification",
        primary={
            "arch": "tabfoundry_staged",
            **_row_model_payload(stage_label=stage_label, tfrow_norm=tfrow_norm),
        },
    )
    return {
        "labels": {
            "model": stage_label,
            "data": data_surface_label,
            "preprocessing": preprocessing_surface_label,
            "training": training_surface_label,
        },
        "model": {
            "build_spec": spec.to_dict(),
        },
        "data": {
            "surface_label": data_surface_label,
        },
        "preprocessing": {
            "surface_label": preprocessing_surface_label,
        },
        "training": {
            "surface_label": training_surface_label,
        },
    }


def _mini_sweep_workspace(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, dict[str, Any]]:
    reference_root = tmp_path / "reference"
    sweeps_root = reference_root / "system_delta_sweeps"
    sweep_id = "mini_sweep"
    catalog_path = reference_root / "system_delta_catalog.yaml"
    index_path = sweeps_root / "index.yaml"
    queue_path = sweeps_root / sweep_id / "queue.yaml"
    sweep_path = sweeps_root / sweep_id / "sweep.yaml"
    registry_path = tmp_path / "registry.json"

    _write_yaml(
        catalog_path,
        {
            "schema": "tab-foundry-system-delta-catalog-v1",
            "deltas": {
                "delta_row_cls_pool": {
                    "dimension_family": "row_pool",
                    "family": "row_pool",
                    "description": "Switch to row CLS pooling.",
                    "upstream_delta": "anchor",
                    "expected_effect": "changed row pooling",
                    "adequacy_knobs": ["tfrow_n_heads", "tfrow_cls_tokens"],
                    "default_effective_surface": {
                        "data": {"surface_label": "anchor_manifest_default"},
                        "preprocessing": {"surface_label": "runtime_default"},
                        "training": {"surface_label": "training_default", "overrides": {}},
                    },
                    "parameter_adequacy_policy": {"default_plan": []},
                },
                "delta_row_cls_pool_rmsnorm": {
                    "dimension_family": "row_pool",
                    "family": "row_pool",
                    "description": "Switch row CLS pooling to RMSNorm.",
                    "upstream_delta": "delta_row_cls_pool",
                    "expected_effect": "changed row pooling norm",
                    "adequacy_knobs": ["tfrow_norm"],
                    "default_effective_surface": {
                        "data": {"surface_label": "anchor_manifest_default"},
                        "preprocessing": {"surface_label": "runtime_default"},
                        "training": {"surface_label": "training_default", "overrides": {}},
                    },
                    "parameter_adequacy_policy": {"default_plan": []},
                },
            },
        },
    )
    _write_yaml(
        index_path,
        {
            "schema": "tab-foundry-system-delta-sweep-index-v1",
            "active_sweep_id": sweep_id,
            "sweeps": {
                sweep_id: {
                    "parent_sweep_id": None,
                    "status": "draft",
                    "anchor_run_id": "anchor_run",
                    "complexity_level": "binary_md",
                    "benchmark_bundle_path": "src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
                    "control_baseline_id": "cls_benchmark_linear_v2",
                }
            },
        },
    )
    _write_yaml(
        sweep_path,
        {
            "schema": "tab-foundry-system-delta-sweep-v1",
            "sweep_id": sweep_id,
            "parent_sweep_id": None,
            "status": "draft",
            "complexity_level": "binary_md",
            "anchor_run_id": "anchor_run",
            "benchmark_bundle_path": "src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
            "control_baseline_id": "cls_benchmark_linear_v2",
            "comparison_policy": "anchor_only",
            "training_experiment": "cls_smoke",
            "training_config_profile": "cls_smoke",
            "surface_role": "hybrid_diagnostic",
            "upstream_reference": {"name": "test", "model_source": "local"},
            "anchor_surface": {"notes": [], "dimension_table": []},
            "anchor_context": {
                "run_id": "anchor_run",
                "model": {
                    "arch": "tabfoundry_staged",
                    "stage": "nano_exact",
                    "stage_label": "nano_exact_anchor_test",
                    "module_selection": {
                        "feature_encoder": "nano",
                        "target_conditioner": "mean_padded_linear",
                        "tokenizer": "scalar_per_feature",
                        "column_encoder": "none",
                        "row_pool": "target_column",
                        "context_encoder": "none",
                        "head": "binary_direct",
                        "table_block_style": "nano_postnorm",
                        "allow_test_self_attention": False,
                    },
                },
                "surface_labels": {
                    "model": "nano_exact_anchor_test",
                    "data": "anchor_manifest_default",
                    "preprocessing": "runtime_default",
                    "training": "training_default",
                },
            },
        },
    )
    _write_yaml(
        queue_path,
        {
            "schema": "tab-foundry-system-delta-sweep-queue-v1",
            "sweep_id": sweep_id,
            "rows": [
                {
                    "order": 1,
                    "delta_ref": "delta_row_cls_pool",
                    "status": "completed",
                    "rationale": "Test row one.",
                    "hypothesis": "Row CLS changes pooling.",
                    "anchor_delta": "target_column -> row_cls",
                    "model": _row_model_payload(stage_label="row_cls_pool_test"),
                    "data": {"surface_label": "anchor_manifest_default"},
                    "preprocessing": {"surface_label": "runtime_default"},
                    "training": {"surface_label": "training_default", "overrides": {}},
                    "execution_policy": "benchmark_full",
                    "run_id": "row_one_run",
                    "followup_run_ids": [],
                    "decision": "defer",
                    "interpretation_status": "interpreted",
                    "confounders": [],
                    "next_action": "",
                    "notes": [],
                },
                {
                    "order": 2,
                    "delta_ref": "delta_row_cls_pool_rmsnorm",
                    "status": "completed",
                    "rationale": "Test row two.",
                    "hypothesis": "RMSNorm changes row CLS stability.",
                    "anchor_delta": "row_cls layernorm -> row_cls rmsnorm",
                    "model": _row_model_payload(
                        stage_label="row_cls_pool_rmsnorm_test",
                        tfrow_norm="rmsnorm",
                    ),
                    "data": {"surface_label": "anchor_manifest_default"},
                    "preprocessing": {"surface_label": "runtime_default"},
                    "training": {"surface_label": "training_default", "overrides": {}},
                    "execution_policy": "benchmark_full",
                    "run_id": "row_two_run",
                    "followup_run_ids": [],
                    "decision": "defer",
                    "interpretation_status": "interpreted",
                    "confounders": [],
                    "next_action": "",
                    "notes": [],
                },
            ],
        },
    )

    anchor_checkpoint_path = tmp_path / "anchor" / "checkpoints" / "best.pt"
    anchor_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    anchor_run_dir = tmp_path / "anchor"
    torch.save(
        {
            "config": _anchor_checkpoint_payload(anchor_run_dir),
            "model": {},
        },
        anchor_checkpoint_path,
    )
    registry_payload = {
        "runs": {
            "row_one_run": {
                "artifacts": {
                    "run_dir": str(tmp_path / "row_one_run" / "train"),
                }
            },
            "row_two_run": {
                "artifacts": {
                    "run_dir": str(tmp_path / "row_two_run" / "train"),
                }
            },
            "anchor_run": {
                "artifacts": {
                    "run_dir": str(anchor_run_dir),
                    "best_checkpoint_path": str(anchor_checkpoint_path),
                },
                "tab_foundry_metrics": {
                    "best_roc_auc": 0.73,
                    "final_roc_auc": 0.72,
                },
                "training_diagnostics": {
                    "final_grad_norm": 0.12,
                    "mean_grad_norm": 0.08,
                },
            },
        }
    }
    return catalog_path, index_path, sweeps_root, registry_path, registry_payload


def _patch_registry(
    monkeypatch: pytest.MonkeyPatch,
    *,
    registry_payload: dict[str, Any],
) -> None:
    monkeypatch.setattr(inspect_module, "load_benchmark_run_registry", lambda _path: registry_payload)
    monkeypatch.setattr(inspect_module, "resolve_registry_path_value", lambda value: Path(value))
    monkeypatch.setattr(graph_module, "load_benchmark_run_registry", lambda _path: registry_payload)
    monkeypatch.setattr(graph_module, "resolve_registry_path_value", lambda value: Path(value))


def test_inspect_sweep_row_reports_resolved_surfaces(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_path, index_path, sweeps_root, registry_path, registry_payload = _mini_sweep_workspace(tmp_path)
    _patch_registry(monkeypatch, registry_payload=registry_payload)

    payload = inspect_module.inspect_sweep_row(
        order=1,
        sweep_id="mini_sweep",
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
        registry_path=registry_path,
    )

    assert payload["queue"]["sweep_id"] == "mini_sweep"
    assert payload["row"]["delta_id"] == "delta_row_cls_pool"
    assert payload["target"]["identity"]["run_id"] == "row_one_run"
    assert payload["target"]["resolved"]["model"]["stage_label"] == "row_cls_pool_test"
    assert payload["target"]["resolved"]["model"]["module_selection"]["row_pool"] == "row_cls"
    assert payload["target"]["resolved"]["data"]["surface_label"] == "anchor_manifest_default"


def test_diff_sweep_row_reports_anchor_and_row_differences(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_path, index_path, sweeps_root, registry_path, registry_payload = _mini_sweep_workspace(tmp_path)
    _patch_registry(monkeypatch, registry_payload=registry_payload)

    anchor_diff = diff_module.diff_sweep_row(
        order=1,
        sweep_id="mini_sweep",
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
        registry_path=registry_path,
    )
    row_diff = diff_module.diff_sweep_row(
        order=2,
        sweep_id="mini_sweep",
        against_order=1,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
        registry_path=registry_path,
    )

    assert anchor_diff["against"]["run_id"] == "anchor_run"
    assert (
        anchor_diff["differences"]["resolved.model.module_selection.row_pool"]["target"] == "row_cls"
    )
    assert row_diff["against"]["order"] == 1
    assert row_diff["differences"]["resolved.model.build_spec.tfrow_norm"]["target"] == "rmsnorm"


def test_inspect_sweep_row_prefers_persisted_training_surface_record_for_rows(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    catalog_path, index_path, sweeps_root, registry_path, registry_payload = _mini_sweep_workspace(tmp_path)
    _patch_registry(monkeypatch, registry_payload=registry_payload)
    persisted_record_path = tmp_path / "row_one_run" / "train" / "training_surface_record.json"
    persisted_record_path.parent.mkdir(parents=True, exist_ok=True)
    persisted_record_path.write_text(
        json.dumps(
            _training_surface_record_payload(
                stage_label="persisted_row_surface",
                data_surface_label="persisted_data_surface",
                preprocessing_surface_label="persisted_preprocessing_surface",
                training_surface_label="persisted_training_surface",
                tfrow_norm="rmsnorm",
            ),
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )

    payload = inspect_module.inspect_sweep_row(
        order=1,
        sweep_id="mini_sweep",
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
        registry_path=registry_path,
    )

    resolved = payload["target"]["resolved"]
    assert resolved["model"]["stage_label"] == "persisted_row_surface"
    assert resolved["model"]["build_spec"]["tfrow_norm"] == "rmsnorm"
    assert resolved["data"]["surface_label"] == "persisted_data_surface"
    assert resolved["training"]["surface_label"] == "persisted_training_surface"
