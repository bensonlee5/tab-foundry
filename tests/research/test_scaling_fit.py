from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from click.testing import CliRunner
from omegaconf import OmegaConf
import pytest

import tab_foundry.cli.app as cli_module
import tab_foundry.cli.research_scaling as research_scaling_cli_module
import tab_foundry.research.scaling.fit as scaling_fit_module
import tab_foundry.research.scaling.validation_backfill as validation_backfill_module
from tab_foundry.research.scaling.fit import (
    ScalingStudyRunPoint,
    collect_completed_scaling_points,
    fit_bcrit,
    fit_loss_vs_nd,
    fit_loss_vs_ns,
    fit_loss_vs_scale,
    fit_scaling_study,
    inspect_scaling_study,
)
from tab_foundry.research.scaling.validation_backfill import backfill_validation_study
from tab_foundry.research.scaling.validation_backfill_schema import (
    VALIDATION_BACKFILL_FILENAME,
    VALIDATION_BACKFILL_SCHEMA,
    VALIDATION_BACKFILL_VERSION,
)
from tab_foundry.types import EvalResult


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(OmegaConf.to_yaml(OmegaConf.create(payload), resolve=True), encoding="utf-8")


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_history(path: Path, *, final_step: int, validation_loss: float | None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    records = [
        {
            "step": int(final_step // 2),
            "stage": "stage1",
            "train_loss": float((validation_loss if validation_loss is not None else 1.0) + 0.15),
            "train_acc": 0.5,
            "lr": 1.0e-3,
            "grad_norm": 0.8,
            "elapsed_seconds": 1.0,
            "train_elapsed_seconds": 1.0,
        },
        {
            "step": int(final_step),
            "stage": "stage1",
            "train_loss": float((validation_loss if validation_loss is not None else 1.0) + 0.1),
            "train_acc": 0.6,
            "lr": 1.0e-3,
            "grad_norm": 0.7,
            "elapsed_seconds": 2.0,
            "train_elapsed_seconds": 2.0,
        },
    ]
    if validation_loss is not None:
        records[0]["val_loss"] = float(validation_loss + 0.05)
        records[0]["val_acc"] = 0.55
        records[1]["val_loss"] = float(validation_loss)
        records[1]["val_acc"] = 0.65
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            json.dump(record, handle, sort_keys=True)
            handle.write("\n")


def _write_run_artifacts(
    root: Path,
    *,
    run_id: str,
    final_step: int,
    validation_loss: float,
) -> tuple[str, str, str]:
    run_dir = root / "outputs" / run_id / "train"
    history_path = run_dir / "train_history.jsonl"
    telemetry_path = run_dir / "telemetry.json"
    _write_history(history_path, final_step=final_step, validation_loss=validation_loss)
    _write_json(
        telemetry_path,
        {
            "wandb": {
                "entity": "test-entity",
                "project": "test-project",
                "run_id": run_id,
                "run_name": run_id,
            }
        },
    )
    return (
        str(run_dir.relative_to(root)),
        str(history_path.relative_to(root)),
        str(telemetry_path.relative_to(root)),
    )


def _registry_entry(
    *,
    run_id: str,
    d_icl: int,
    layers: int,
    canonical_n: int,
    benchmark_loss: float,
    validation_loss: float,
    final_step: int,
    tokens_per_step: float,
    tokens_seen: int,
    train_flops_per_token: float,
    root: Path,
) -> dict[str, Any]:
    run_dir, history_path, _telemetry_path = _write_run_artifacts(
        root,
        run_id=run_id,
        final_step=final_step,
        validation_loss=validation_loss,
    )
    return {
        "run_id": run_id,
        "model": {
            "arch": "tabfoundry_sandwich",
            "d_icl": d_icl,
            "build_spec": {"sandwich_layers": layers},
        },
        "artifacts": {
            "run_dir": run_dir,
            "history_path": history_path,
        },
        "tab_foundry_metrics": {
            "final_step": float(final_step),
            "final_log_loss": float(benchmark_loss),
        },
        "regime_budget": {
            "tokens_per_step": float(tokens_per_step),
            "tokens_seen": int(tokens_seen),
        },
        "parameter_accounting": {
            "schema": "tab-foundry-model-accounting-v1",
            "method": "inspected_parameter_partition_v1",
            "total_params": canonical_n + 2048,
            "trainable_params": canonical_n + 2048,
            "strict": {
                "embedding_params": 2048,
                "non_embedding_params": canonical_n,
            },
            "expanded": {
                "embedding_like_params": 3072,
                "non_embedding_params": canonical_n - 1024,
            },
            "canonical_non_embedding_params": canonical_n,
        },
        "compute_accounting": {
            "schema": "tab-foundry-model-accounting-v1",
            "method": "inspected_analytic_v1",
            "training_multiplier": 3.0,
            "forward_flops_per_token": train_flops_per_token / 3.0,
            "train_flops_per_token": train_flops_per_token,
            "train_flops_per_step": train_flops_per_token * float(tokens_per_step),
            "total_train_flops": train_flops_per_token * float(tokens_seen),
            "tokens_seen": int(tokens_seen),
            "tokens_per_step": float(tokens_per_step),
        },
    }


def _queue_row(
    *,
    order: int,
    delta_ref: str,
    d_icl: int,
    layers: int,
    max_steps: int,
    grad_accum_steps: int,
    run_id: str,
    benchmark_loss: float,
) -> dict[str, Any]:
    return {
        "order": order,
        "delta_ref": delta_ref,
        "status": "completed",
        "rationale": "synthetic scaling test row",
        "hypothesis": "",
        "anchor_delta": "synthetic",
        "model": {
            "arch": "tabfoundry_sandwich",
            "d_icl": d_icl,
            "input_normalization": "train_zscore_clip",
            "many_class_base": 10,
            "head_hidden_dim": 96,
            "sandwich_latents": 24,
            "sandwich_layers": layers,
            "sandwich_heads": 1,
            "sandwich_ff_expansion": 2,
            "sandwich_summary_tokens_per_axis": 3,
            "sandwich_self_attention_per_cross": 4,
        },
        "data": {
            "surface_label": "test_data",
            "source": "manifest",
            "corpus_ref": "test_recipe",
        },
        "preprocessing": {"surface_label": "runtime_default"},
        "training": {
            "surface_label": "prior_cosine_warmup",
            "task_batch_size": 16,
            "overrides": {
                "runtime": {
                    "grad_accum_steps": grad_accum_steps,
                    "max_steps": max_steps,
                },
                "schedule": {
                    "stages": [{"name": "prior_dump", "steps": max_steps}],
                },
            },
        },
        "parameter_adequacy_plan": [],
        "execution_policy": "benchmark_full",
        "benchmark_checkpoint_selection": "all",
        "run_id": run_id,
        "followup_run_ids": [],
        "decision": "keep",
        "interpretation_status": "completed",
        "confounders": [],
        "next_action": "",
        "notes": [],
        "dynamic_model_overrides": None,
        "reuse_train_artifact": None,
        "screen_metrics": None,
        "benchmark_metrics": {
            "final_log_loss": float(benchmark_loss),
            "objective_metric": "final_log_loss_at_matched_regime_budget",
        },
        "parent_delta_ref": None,
    }


def _study_workspace(tmp_path: Path) -> tuple[Path, Path, Path, Path, Path]:
    root = tmp_path
    reference_root = root / "reference"
    sweeps_root = reference_root / "system_delta_sweeps"
    scaling_root = reference_root / "scaling_studies"
    registry_path = root / "benchmark_run_registry_v1.json"
    index_path = sweeps_root / "index.yaml"
    catalog_path = reference_root / "system_delta_catalog.yaml"
    study_path = scaling_root / "synthetic_phase2.yaml"

    _write_yaml(
        catalog_path,
        {
            "schema": "tab-foundry-system-delta-catalog-v1",
            "deltas": {
                "delta_geom_72x1": {
                    "dimension_family": "scaling",
                    "family": "scaling",
                    "description": "synthetic",
                    "upstream_delta": "anchor",
                    "expected_effect": "synthetic",
                    "adequacy_knobs": [],
                    "default_effective_surface": {
                        "data": {"surface_label": "test_data"},
                        "preprocessing": {"surface_label": "runtime_default"},
                        "training": {"surface_label": "prior_cosine_warmup", "overrides": {}},
                    },
                    "parameter_adequacy_policy": {"default_plan": []},
                },
                "delta_geom_96x2": {
                    "dimension_family": "scaling",
                    "family": "scaling",
                    "description": "synthetic",
                    "upstream_delta": "anchor",
                    "expected_effect": "synthetic",
                    "adequacy_knobs": [],
                    "default_effective_surface": {
                        "data": {"surface_label": "test_data"},
                        "preprocessing": {"surface_label": "runtime_default"},
                        "training": {"surface_label": "prior_cosine_warmup", "overrides": {}},
                    },
                    "parameter_adequacy_policy": {"default_plan": []},
                },
            },
        },
    )
    _write_yaml(
        index_path,
        {
            "schema": "tab-foundry-system-delta-sweep-index-v2",
            "sweeps": {
                "synthetic_ns": {
                    "parent_sweep_id": None,
                    "status": "draft",
                    "anchor_run_id": "anchor_run",
                    "complexity_level": "classification_md",
                    "benchmark_manifest_path": "bundle.json",
                    "control_baseline_id": "baseline",
                    "external_benchmarks": [],
                },
                "synthetic_batch": {
                    "parent_sweep_id": "synthetic_ns",
                    "status": "draft",
                    "anchor_run_id": "anchor_run",
                    "complexity_level": "classification_md",
                    "benchmark_manifest_path": "bundle.json",
                    "control_baseline_id": "baseline",
                    "external_benchmarks": [],
                },
            },
        },
    )
    sweep_template = {
        "schema": "tab-foundry-system-delta-sweep-v1",
        "status": "draft",
        "complexity_level": "classification_md",
        "anchor_run_id": "anchor_run",
        "benchmark_manifest_path": "bundle.json",
        "control_baseline_id": "baseline",
        "external_benchmarks": [],
        "training_experiment": "cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1",
        "training_config_profile": "cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_compile_eager_dynamic_v1",
        "comparison_policy": "anchor_only",
        "upstream_reference": {"name": "synthetic", "model_source": "local"},
        "anchor_surface": {"notes": [], "dimension_table": []},
        "anchor_context": {
            "run_id": "anchor_run",
            "surface_labels": {"model": "tabfoundry_sandwich"},
        },
    }
    _write_yaml(
        sweeps_root / "synthetic_ns" / "sweep.yaml",
        {
            **sweep_template,
            "sweep_id": "synthetic_ns",
            "parent_sweep_id": None,
            "surface_role": "classification_scaling_law_phase2_ns",
        },
    )
    _write_yaml(
        sweeps_root / "synthetic_batch" / "sweep.yaml",
        {
            **sweep_template,
            "sweep_id": "synthetic_batch",
            "parent_sweep_id": "synthetic_ns",
            "surface_role": "classification_scaling_law_phase2_batch",
        },
    )

    alpha_n = 0.4
    alpha_d = 0.3
    alpha_s = 0.5
    benchmark_floor = 0.3
    validation_floor = 0.25
    nc = 4.0e6
    dc = 9.0e5
    sc = 1800.0
    ns_specs = [
        ("ns_72_625", 72, 1, 625, 1_000_000, "delta_geom_72x1"),
        ("ns_72_1250", 72, 1, 1250, 1_000_000, "delta_geom_72x1"),
        ("ns_96_625", 96, 2, 625, 2_000_000, "delta_geom_96x2"),
        ("ns_96_1250", 96, 2, 1250, 2_000_000, "delta_geom_96x2"),
    ]
    ns_rows = []
    registry_runs: dict[str, Any] = {}
    for order, (run_id, d_icl, layers, steps, n_value, delta_ref) in enumerate(ns_specs, start=1):
        d_value = float(steps) * 128.0
        benchmark_loss = (
            benchmark_floor + (((nc / n_value) ** (alpha_n / alpha_d)) + (dc / d_value)) ** alpha_d
        )
        validation_loss = (
            validation_floor
            + (((nc / n_value) ** (alpha_n / alpha_s)) + (sc / float(steps))) ** alpha_s
        )
        registry_runs[run_id] = _registry_entry(
            run_id=run_id,
            d_icl=d_icl,
            layers=layers,
            canonical_n=n_value,
            benchmark_loss=benchmark_loss,
            validation_loss=validation_loss,
            final_step=steps,
            tokens_per_step=128.0,
            tokens_seen=int(d_value),
            train_flops_per_token=float(4.0 * n_value),
            root=root,
        )
        ns_rows.append(
            _queue_row(
                order=order,
                delta_ref=delta_ref,
                d_icl=d_icl,
                layers=layers,
                max_steps=steps,
                grad_accum_steps=4,
                run_id=run_id,
                benchmark_loss=benchmark_loss,
            )
        )
    _write_yaml(
        sweeps_root / "synthetic_ns" / "queue.yaml",
        {
            "schema": "tab-foundry-system-delta-sweep-queue-v1",
            "sweep_id": "synthetic_ns",
            "rows": ns_rows,
        },
    )

    batch_rows = []
    batch_specs = [
        ("batch_1", 1, 32.0),
        ("batch_4", 4, 128.0),
        ("batch_16", 16, 512.0),
    ]
    for order, (run_id, grad_accum_steps, tokens_per_step) in enumerate(batch_specs, start=1):
        validation_loss = math.sqrt(10.0 / tokens_per_step)
        benchmark_loss = validation_loss + 0.08
        registry_runs[run_id] = _registry_entry(
            run_id=run_id,
            d_icl=96,
            layers=2,
            canonical_n=2_000_000,
            benchmark_loss=benchmark_loss,
            validation_loss=validation_loss,
            final_step=1250,
            tokens_per_step=tokens_per_step,
            tokens_seen=int(tokens_per_step * 1250.0),
            train_flops_per_token=float(8.0e6),
            root=root,
        )
        batch_rows.append(
            _queue_row(
                order=order,
                delta_ref="delta_geom_96x2",
                d_icl=96,
                layers=2,
                max_steps=1250,
                grad_accum_steps=grad_accum_steps,
                run_id=run_id,
                benchmark_loss=benchmark_loss,
            )
        )
    _write_yaml(
        sweeps_root / "synthetic_batch" / "queue.yaml",
        {
            "schema": "tab-foundry-system-delta-sweep-queue-v1",
            "sweep_id": "synthetic_batch",
            "rows": batch_rows,
        },
    )
    _write_json(
        registry_path,
        {
            "schema": "tab-foundry-benchmark-runs-v2",
            "version": 2,
            "runs": registry_runs,
        },
    )
    _write_yaml(
        study_path,
        {
            "schema": "tab-foundry-scaling-study-v1",
            "study_id": "synthetic_phase2",
            "phase": 2,
            "output_root": "outputs/research_scaling/synthetic_phase2",
            "phase1_reference_sweep_id": "tf_rd_009_width_depth_medium_v1",
            "sweeps": [
                {"name": "ns_core", "sweep_id": "synthetic_ns", "family": "ns_core"},
                {
                    "name": "batch_critical",
                    "sweep_id": "synthetic_batch",
                    "family": "batch_critical",
                },
            ],
            "geometry_row_labels": ["72x1", "96x2"],
            "step_ladder": [625, 1250],
            "batch_grad_accum_ladder": [1, 4, 16],
            "canonical_loss_axes": {
                "benchmark": "benchmark_log_loss",
                "validation": "validation_loss",
            },
            "canonical_variables": {
                "N": "parameter_accounting.canonical_non_embedding_params",
                "D": "regime_budget.tokens_seen",
                "S": "tab_foundry_metrics.final_step",
                "B_eff": "regime_budget.tokens_per_step",
                "C": "compute_accounting.total_train_flops",
            },
            "slice_selection": {
                "l_n": "highest_completed_s",
                "l_d": "highest_completed_n",
                "l_c": "all_completed_runs",
                "l_nd": "full_ns_matrix",
                "l_ns": "full_ns_matrix",
                "l_cmin": "lower_envelope_from_bcrit",
                "bcrit": "batch_envelope",
            },
        },
    )
    return study_path, registry_path, index_path, catalog_path, sweeps_root


def _remove_validation_history(root: Path, registry_path: Path) -> None:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for entry in registry["runs"].values():
        history_path = root / entry["artifacts"]["history_path"]
        final_step = int(entry["tab_foundry_metrics"]["final_step"])
        _write_history(history_path, final_step=final_step, validation_loss=None)


def _write_validation_backfill_sidecars(
    root: Path,
    registry_path: Path,
    *,
    val_loss: float = 0.42,
) -> None:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for run_id, entry in registry["runs"].items():
        history_path = root / entry["artifacts"]["history_path"]
        sidecar_path = history_path.parent / VALIDATION_BACKFILL_FILENAME
        _write_json(
            sidecar_path,
            {
                "schema": VALIDATION_BACKFILL_SCHEMA,
                "version": VALIDATION_BACKFILL_VERSION,
                "study_id": "synthetic_phase2",
                "sweep_id": "synthetic",
                "row_order": 1,
                "row_label": "synthetic",
                "run_id": run_id,
                "checkpoint": {
                    "path": str(history_path.parent / "checkpoints" / "latest.pt"),
                    "source_uri": "local",
                    "global_step": int(entry["tab_foundry_metrics"]["final_step"]),
                },
                "evaluation": {
                    "split": "val",
                    "max_batches": 16,
                    "device": "cpu",
                },
                "metrics": {
                    "val_loss": float(val_loss),
                    "val_acc": 0.5,
                },
            },
        )


def _write_backfill_required_files(root: Path, registry_path: Path) -> None:
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    for entry in registry["runs"].values():
        run_dir = root / entry["artifacts"]["run_dir"]
        checkpoint_path = run_dir / "checkpoints" / "latest.pt"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint_path.write_bytes(b"placeholder checkpoint")
        _write_json(run_dir / "training_surface_record.json", {"fingerprint": "test"})


def _point(
    *,
    n_value: int,
    d_value: int,
    s_value: int,
    benchmark_loss: float,
    validation_loss: float,
    batch_value: float = 128.0,
) -> ScalingStudyRunPoint:
    return ScalingStudyRunPoint(
        family="ns_core",
        sweep_id="synthetic",
        row_order=1,
        row_label="synthetic",
        run_id=f"run_{n_value}_{d_value}_{s_value}",
        d_icl=96,
        layers=2,
        max_steps=s_value,
        grad_accum_steps=4,
        task_batch_size=16,
        strict_embedding_params=2048,
        strict_non_embedding_params=n_value,
        expanded_embedding_like_params=3072,
        expanded_non_embedding_params=n_value - 1024,
        canonical_non_embedding_params=n_value,
        benchmark_log_loss=benchmark_loss,
        validation_loss=validation_loss,
        validation_loss_source="train_history",
        validation_loss_missing_reason=None,
        steps=s_value,
        tokens_seen=d_value,
        tokens_per_step=batch_value,
        train_flops_per_token=float(4.0 * n_value),
        train_flops_per_step=float(4.0 * n_value * batch_value),
        total_train_flops=float(4.0 * n_value * d_value),
        run_dir="outputs/run/train",
        history_path="outputs/run/train/train_history.jsonl",
        telemetry_path="outputs/run/train/telemetry.json",
    )


def test_fit_loss_vs_scale_recovers_a_known_exponent() -> None:
    x_values = [1.0e6, 2.0e6, 4.0e6, 8.0e6]
    alpha = 0.35
    y_values = [0.25 + (4.0e6 / value) ** alpha for value in x_values]

    payload = fit_loss_vs_scale(
        name="L(N)",
        x_values=x_values,
        y_values=y_values,
        scale_name="Nc",
        alpha_name="alpha_n",
    )

    assert payload["parameters"]["alpha_n"] == pytest.approx(alpha, rel=0.15)


def test_fit_loss_vs_nd_and_ns_recover_known_exponents() -> None:
    points = [
        _point(
            n_value=n_value,
            d_value=s_value * 128,
            s_value=s_value,
            benchmark_loss=0.3
            + (((4.0e6 / n_value) ** (0.4 / 0.3)) + (8.0e5 / (s_value * 128))) ** 0.3,
            validation_loss=0.25 + (((4.0e6 / n_value) ** (0.4 / 0.5)) + (1500.0 / s_value)) ** 0.5,
        )
        for n_value in [1_000_000, 2_000_000, 4_000_000]
        for s_value in [625, 1250, 2500]
    ]

    nd_fit = fit_loss_vs_nd(points=points, target_key="benchmark_log_loss")
    ns_fit = fit_loss_vs_ns(points=points, target_key="validation_loss")

    assert nd_fit["parameters"]["alpha_n"] == pytest.approx(0.4, rel=0.2)
    assert nd_fit["parameters"]["alpha_d"] == pytest.approx(0.3, rel=0.2)
    assert ns_fit["parameters"]["alpha_n"] == pytest.approx(0.4, rel=0.2)
    assert ns_fit["parameters"]["alpha_s"] == pytest.approx(0.5, rel=0.2)


def test_fit_bcrit_recovers_a_known_alpha() -> None:
    points = [
        _point(
            n_value=2_000_000,
            d_value=int(batch * 1250),
            s_value=1250,
            benchmark_loss=math.sqrt(10.0 / batch) + 0.1,
            validation_loss=math.sqrt(10.0 / batch),
            batch_value=batch,
        )
        for batch in [32.0, 128.0, 512.0, 2048.0]
    ]
    for index, point in enumerate(points):
        points[index] = ScalingStudyRunPoint(**{**point.as_dict(), "family": "batch_critical"})

    fit = fit_bcrit(points)

    assert fit["parameters"]["alpha_b"] == pytest.approx(0.5, rel=0.2)


def test_fit_scaling_study_emits_artifacts_and_wandb_updates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_path, registry_path, index_path, catalog_path, sweeps_root = _study_workspace(tmp_path)
    captured_updates: list[dict[str, Any]] = []
    monkeypatch.setattr(
        scaling_fit_module,
        "posthoc_update_wandb_summary",
        lambda *, telemetry_path, payload: (
            captured_updates.append({"telemetry_path": str(telemetry_path), "payload": payload})
            or True
        ),
    )

    payload = fit_scaling_study(
        study_path=study_path,
        registry_path=registry_path,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
        out_root=tmp_path / "artifacts",
    )

    artifact_root = Path(payload["artifact_paths"]["artifact_root"])
    assert (artifact_root / "fit_summary.json").exists()
    assert (artifact_root / "summary.md").exists()
    assert (artifact_root / "wandb_summary.json").exists()
    assert (artifact_root / "plots" / "l_n.png").exists()
    assert (artifact_root / "plots" / "l_nd_surface.png").exists()
    assert (artifact_root / "plots" / "l_cmin_frontier.png").exists()
    assert payload["alphas"].keys() == {
        "alpha_n",
        "alpha_d",
        "alpha_s",
        "alpha_c",
        "alpha_cmin",
        "alpha_b",
    }
    assert "alpha_c_implied_from_l_nd" in payload["derived_kaplan_relations"]
    assert captured_updates


def test_inspect_scaling_study_reports_missing_validation_without_crashing(tmp_path: Path) -> None:
    study_path, registry_path, index_path, catalog_path, sweeps_root = _study_workspace(tmp_path)
    _remove_validation_history(tmp_path, registry_path)

    payload = inspect_scaling_study(
        study_path=study_path,
        registry_path=registry_path,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
    )

    assert payload["counts"]["total_completed_points"] == 7
    assert payload["counts"]["validation_backed_points"] == 0
    assert payload["counts"]["missing_validation_points"] == 7
    assert payload["counts"]["l_n_points"] == 2
    assert payload["counts"]["l_ns_points"] == 0
    assert (
        payload["validation_coverage"]["missing"][0]["missing_reason"]
        == "history_missing_validation_records"
    )


def test_inspect_scaling_study_uses_validation_backfill_sidecars(tmp_path: Path) -> None:
    study_path, registry_path, index_path, catalog_path, sweeps_root = _study_workspace(tmp_path)
    _remove_validation_history(tmp_path, registry_path)
    _write_validation_backfill_sidecars(tmp_path, registry_path)

    payload = inspect_scaling_study(
        study_path=study_path,
        registry_path=registry_path,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
    )

    assert payload["counts"]["total_completed_points"] == 7
    assert payload["counts"]["validation_backed_points"] == 7
    assert payload["counts"]["missing_validation_points"] == 0
    assert payload["available_points"][0]["validation_loss_source"] == "validation_backfill_v1"


def test_fit_scaling_study_requires_validation_for_strict_fits(tmp_path: Path) -> None:
    study_path, registry_path, index_path, catalog_path, sweeps_root = _study_workspace(tmp_path)
    _remove_validation_history(tmp_path, registry_path)

    with pytest.raises(
        RuntimeError,
        match="L\\(N,S\\) requires validation_loss.*runtime\\.val_batches=0.*posthoc validation backfill",
    ):
        _ = fit_scaling_study(
            study_path=study_path,
            registry_path=registry_path,
            index_path=index_path,
            catalog_path=catalog_path,
            sweeps_root=sweeps_root,
            out_root=tmp_path / "artifacts",
        )


def test_validation_backfill_dry_run_reports_ready_and_incomplete_rows(tmp_path: Path) -> None:
    study_path, registry_path, index_path, catalog_path, sweeps_root = _study_workspace(tmp_path)
    _remove_validation_history(tmp_path, registry_path)
    _write_backfill_required_files(tmp_path, registry_path)

    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    first_entry = registry["runs"]["ns_72_625"]
    (tmp_path / first_entry["artifacts"]["run_dir"] / "checkpoints" / "latest.pt").unlink()

    payload = backfill_validation_study(
        study_path=study_path,
        registry_path=registry_path,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
        preseed_gcs_root=str(tmp_path),
        dry_run=True,
    )

    assert payload["counts"]["candidate_rows"] == 7
    assert payload["counts"]["dry_run_ready"] == 6
    assert payload["counts"]["incomplete_artifacts"] == 1
    incomplete = [row for row in payload["rows"] if row["status"] == "incomplete_artifacts"]
    assert incomplete[0]["run_id"] == "ns_72_625"
    assert incomplete[0]["missing_artifacts"] == ["checkpoints/latest.pt"]


def test_validation_backfill_writes_sidecars_and_skips_existing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    study_path, registry_path, index_path, catalog_path, sweeps_root = _study_workspace(tmp_path)
    _remove_validation_history(tmp_path, registry_path)
    _write_backfill_required_files(tmp_path, registry_path)
    calls: list[Any] = []

    def _fake_evaluate_checkpoint(cfg: Any) -> EvalResult:
        calls.append(cfg)
        return EvalResult(checkpoint=Path(str(cfg.eval.checkpoint)), metrics={"loss": 0.37, "acc": 0.61})

    monkeypatch.setattr(validation_backfill_module, "evaluate_checkpoint", _fake_evaluate_checkpoint)
    monkeypatch.setattr(validation_backfill_module, "_checkpoint_global_step", lambda _path: 625)

    payload = backfill_validation_study(
        study_path=study_path,
        registry_path=registry_path,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
        preseed_gcs_root=str(tmp_path),
        cache_root=tmp_path / "cache",
        out_root=tmp_path / "validation",
        start_order=1,
        stop_after_order=1,
    )

    assert payload["counts"]["candidate_rows"] == 2
    assert payload["counts"]["validated_rows"] == 2
    assert len(calls) == 2
    first_cfg = calls[0]
    assert str(first_cfg.eval.split) == "val"
    assert int(first_cfg.eval.max_batches) == 16
    assert str(first_cfg.runtime.device) == "cpu"
    assert str(first_cfg.runtime.mixed_precision) == "no"
    assert bool(first_cfg.logging.use_wandb) is False
    first_sidecar = Path(payload["rows"][0]["sidecar_path"])
    sidecar_payload = json.loads(first_sidecar.read_text(encoding="utf-8"))
    assert sidecar_payload["schema"] == VALIDATION_BACKFILL_SCHEMA
    assert sidecar_payload["metrics"]["val_loss"] == pytest.approx(0.37)
    overlay_payload = inspect_scaling_study(
        study_path=study_path,
        registry_path=Path(payload["registry_overlay_path"]),
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
    )
    assert overlay_payload["counts"]["validation_backed_points"] == 2

    second_payload = backfill_validation_study(
        study_path=study_path,
        registry_path=registry_path,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
        preseed_gcs_root=str(tmp_path),
        cache_root=tmp_path / "cache",
        out_root=tmp_path / "validation",
        start_order=1,
        stop_after_order=1,
    )

    assert second_payload["counts"]["skipped_existing"] == 2
    assert len(calls) == 2


def test_collect_completed_scaling_points_rejects_inconsistent_tokens(tmp_path: Path) -> None:
    study_path, registry_path, index_path, catalog_path, sweeps_root = _study_workspace(tmp_path)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    registry["runs"]["ns_72_625"]["regime_budget"]["tokens_seen"] = 999
    registry_path.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    config = scaling_fit_module.load_scaling_study_config(study_path=study_path)

    with pytest.raises(RuntimeError, match="violates D = B_eff \\* S"):
        _ = collect_completed_scaling_points(
            config=config,
            registry_path=registry_path,
            index_path=index_path,
            catalog_path=catalog_path,
            sweeps_root=sweeps_root,
        )


def test_research_scaling_cli_dispatches_to_fit_and_inspect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inspect_called: dict[str, Any] = {}
    fit_called: dict[str, Any] = {}
    backfill_called: dict[str, Any] = {}
    monkeypatch.setattr(
        research_scaling_cli_module,
        "inspect_scaling_study",
        lambda **kwargs: (
            inspect_called.update(kwargs) or {"study": {"study_id": "synthetic"}, "counts": {}}
        ),
    )
    monkeypatch.setattr(
        research_scaling_cli_module,
        "fit_scaling_study",
        lambda **kwargs: (
            fit_called.update(kwargs)
            or {"study": {"study_id": "synthetic"}, "counts": {}, "fit_summary": {}}
        ),
    )
    monkeypatch.setattr(
        research_scaling_cli_module,
        "backfill_validation_study",
        lambda **kwargs: (
            backfill_called.update(kwargs)
            or {"study": {"study_id": "synthetic"}, "counts": {}, "rows": []}
        ),
    )

    inspect_result = CliRunner().invoke(
        cli_module.cli,
        [
            "research",
            "scaling",
            "inspect",
            "--study",
            "synthetic_phase2",
            "--json",
        ],
    )
    fit_result = CliRunner().invoke(
        cli_module.cli,
        [
            "research",
            "scaling",
            "fit",
            "--study",
            "synthetic_phase2",
            "--json",
        ],
    )
    backfill_result = CliRunner().invoke(
        cli_module.cli,
        [
            "research",
            "scaling",
            "backfill-validation",
            "--study",
            "synthetic_phase2",
            "--preseed-gcs-root",
            str(Path("/tmp/source")),
            "--dry-run",
            "--json",
        ],
    )

    assert inspect_result.exit_code == 0
    assert fit_result.exit_code == 0
    assert backfill_result.exit_code == 0
    assert inspect_called["study_id"] == "synthetic_phase2"
    assert fit_called["study_id"] == "synthetic_phase2"
    assert backfill_called["study_id"] == "synthetic_phase2"
    assert backfill_called["preseed_gcs_root"] == str(Path("/tmp/source"))
    assert backfill_called["dry_run"] is True
