"""Production-control corpus resolution and training for adequacy pilot."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping, cast

from omegaconf import OmegaConf

from tab_foundry.config import compose_config
from tab_foundry.data.corpus_loading import build_dagzoo_provenance_summary
from tab_foundry.data.corpus_lookup import load_corpus_record
from tab_foundry.data.corpus_materialization import (
    build_staged_corpus_manifest,
    load_staged_corpus_recipe_preview,
)
from tab_foundry.training.health import run_inspect
from tab_foundry.training.trainer import train

from .shared import (
    _MEDIUM_V4_TRAINING_SURFACE,
    _PRODUCTION_BLOCK_ID,
    _SUPPORTED_ADEQUACY_ID,
    _SUPPORTED_DEVICE,
    _TRAINING_EXPERIMENT,
    _ensure_mapping,
    _finite_float_or_none,
    _json_safe,
    _optional_mapping,
    _read_json_mapping,
    _read_last_jsonl_mapping,
    _recipe_id_from_corpus_ref,
    _repo_root,
)


def _staged_direct_manifest_record(
    *,
    requested_corpus_ref: str,
    pilot_root: Path,
    dagzoo_root: Path,
    force: bool,
    repo_root: Path | None,
) -> dict[str, Any]:
    recipe_id = _recipe_id_from_corpus_ref(requested_corpus_ref)
    candidate_manifest_paths = [
        (pilot_root.parent / "direct_training" / "manifest.parquet").resolve(),
        (pilot_root / "direct_training" / "manifest.parquet").resolve(),
    ]
    direct_manifest_path = next(
        (path for path in candidate_manifest_paths if path.exists()),
        candidate_manifest_paths[0],
    )
    if force or not direct_manifest_path.exists():
        _ = build_staged_corpus_manifest(
            recipe_id=recipe_id,
            dagzoo_root=dagzoo_root,
            out_manifest_path=direct_manifest_path,
            repo_root=repo_root,
        )

    resolved_repo_root = (repo_root or _repo_root()).expanduser().resolve()
    preview = load_staged_corpus_recipe_preview(
        recipe_id=recipe_id,
        dagzoo_root=dagzoo_root,
        repo_root=resolved_repo_root,
        stage_root=None,
        sweep_id=None,
        sweeps_root=None,
    )
    invocation_payloads = cast(list[dict[str, Any]], preview["invocations"])
    dagzoo_provenance_summary = build_dagzoo_provenance_summary(
        recipe=cast(Mapping[str, Any], preview["recipe"]),
        corpus_ref=requested_corpus_ref,
        corpus_id="staged",
        provenance={"invocations": invocation_payloads},
        surface_label=str(preview["surface_label"]),
    )
    return {
        "schema": "tab-foundry-staged-corpus-preview-v1",
        "recipe_id": str(preview["recipe_id"]),
        "corpus_id": None,
        "corpus_ref": None,
        "surface_label": str(preview["surface_label"]),
        "stage_root": str(preview["stage_root"]),
        "corpus_record_path": None,
        "manifest": {
            "manifest_path": str(direct_manifest_path),
        },
        "dagzoo_provenance": {
            "invocations": invocation_payloads,
        },
        "dagzoo_provenance_summary": dagzoo_provenance_summary,
    }


def _resolve_production_control_corpus(
    *,
    requested_corpus_ref: str,
    pilot_root: Path,
    dagzoo_root: Path,
    force: bool,
    repo_root: Path | None,
) -> dict[str, Any]:
    resolved_repo_root = (repo_root or _repo_root()).expanduser().resolve()
    try:
        corpus_record = load_corpus_record(
            requested_corpus_ref,
            repo_root=resolved_repo_root,
        )
    except RuntimeError:
        corpus_record = None
    else:
        return {
            "corpus_record": corpus_record,
            "materialization_state": "finalized",
        }

    try:
        staged_record = _staged_direct_manifest_record(
            requested_corpus_ref=requested_corpus_ref,
            pilot_root=pilot_root,
            dagzoo_root=dagzoo_root,
            force=force,
            repo_root=resolved_repo_root,
        )
    except Exception as staged_exc:
        raise RuntimeError(
            "failed to resolve the production-control corpus from a finalized record "
            "or staged direct-manifest fallback: "
            f"{type(staged_exc).__name__}: {staged_exc}"
        ) from staged_exc
    return {
        "corpus_record": staged_record,
        "materialization_state": "staged",
    }


def build_production_control_config(
    *,
    requested_corpus_ref: str,
    corpus_ref: str | None,
    manifest_path: Path | None,
    materialization_state: str,
    run_dir: Path,
    device: str,
) -> Any:
    cfg = compose_config([f"experiment={_TRAINING_EXPERIMENT}"])
    OmegaConf.update(cfg, "data.source", "manifest", merge=False, force_add=True)
    OmegaConf.update(
        cfg,
        "data.requested_corpus_ref",
        str(requested_corpus_ref),
        merge=False,
        force_add=True,
    )
    OmegaConf.update(
        cfg,
        "data.materialization_state",
        str(materialization_state),
        merge=False,
        force_add=True,
    )
    if corpus_ref is None:
        OmegaConf.update(cfg, "data.corpus_ref", None, merge=False, force_add=True)
    else:
        OmegaConf.update(cfg, "data.corpus_ref", str(corpus_ref), merge=False, force_add=True)
    if manifest_path is None:
        OmegaConf.update(cfg, "data.manifest_path", None, merge=False, force_add=True)
    else:
        OmegaConf.update(
            cfg,
            "data.manifest_path",
            str(manifest_path.expanduser().resolve()),
            merge=False,
            force_add=True,
        )
    cfg.runtime.device = str(device)
    cfg.runtime.mixed_precision = "no"
    cfg.runtime.num_workers = 0
    cfg.runtime.grad_accum_steps = 4
    cfg.runtime.grad_clip = 0.0
    cfg.runtime.max_steps = 2500
    cfg.runtime.eval_every = 25
    cfg.runtime.checkpoint_every = 25
    cfg.runtime.val_batches = 0
    cfg.runtime.seed = 1
    cfg.runtime.output_dir = str(run_dir.resolve())
    cfg.training.task_batch_size = 16
    cfg.optimizer.min_lr = 1.0e-5
    cfg.schedule.stages = [
        dict(
            cast(
                Mapping[str, Any],
                _MEDIUM_V4_TRAINING_SURFACE["schedule_stage"],
            )
        )
    ]
    cfg.logging.run_name = f"{_SUPPORTED_ADEQUACY_ID}-production-control-v4"
    cfg.logging.history_jsonl_path = str((run_dir / "train_history.jsonl").resolve())
    return cfg


def _run_inspect_excerpt(payload: Mapping[str, Any]) -> dict[str, Any]:
    health = payload.get("health")
    health_excerpt = None
    if isinstance(health, Mapping):
        health_excerpt = {
            "verdict": health.get("verdict"),
            "summary": health.get("summary"),
            "metrics": _json_safe(_optional_mapping(health.get("metrics")) or {}),
        }
    artifacts = _optional_mapping(payload.get("artifacts")) or {}
    selected_artifacts = {
        key: artifacts[key]
        for key in (
            "training_surface_record_json",
            "telemetry_json",
            "gradient_history_jsonl",
            "train_history_jsonl",
            "latest_checkpoint_pt",
            "best_checkpoint_pt",
            "checkpoints_dir",
        )
        if key in artifacts
    }
    return {
        "surface_labels": _json_safe(_optional_mapping(payload.get("surface_labels")) or {}),
        "health": health_excerpt,
        "artifacts": _json_safe(selected_artifacts),
    }


def _summarize_existing_production_control_pilot(
    *,
    requested_corpus_ref: str,
    pilot_root: Path,
) -> dict[str, Any]:
    run_root = pilot_root / _PRODUCTION_BLOCK_ID
    run_dir = (run_root / "train").expanduser().resolve()
    telemetry_payload = _read_json_mapping(
        run_dir / "telemetry.json",
        context="production control telemetry",
    )
    training_surface_payload = _read_json_mapping(
        run_dir / "training_surface_record.json",
        context="production control training surface record",
    )
    last_history = _read_last_jsonl_mapping(
        run_dir / "train_history.jsonl",
        context="production control train history",
    )
    inspect_payload = run_inspect(run_dir)

    training_surface_data = _ensure_mapping(
        training_surface_payload.get("data"),
        context="production control training surface record data",
    )
    training_surface_runtime = _ensure_mapping(
        training_surface_payload.get("runtime"),
        context="production control training surface record runtime",
    )
    training_surface_training = _ensure_mapping(
        training_surface_payload.get("training"),
        context="production control training surface record training",
    )
    telemetry_artifacts = _optional_mapping(telemetry_payload.get("artifacts")) or {}
    telemetry_wandb = _optional_mapping(telemetry_payload.get("wandb")) or {}
    manifest_payload = _optional_mapping(training_surface_data.get("manifest")) or {}
    raw_manifest_path = manifest_payload.get("manifest_path")
    manifest_path = (
        None
        if not isinstance(raw_manifest_path, str) or not raw_manifest_path.strip()
        else Path(raw_manifest_path).expanduser().resolve()
    )
    raw_corpus_ref = training_surface_data.get("corpus_ref")
    corpus_ref = (
        None
        if not isinstance(raw_corpus_ref, str) or not raw_corpus_ref.strip()
        else str(raw_corpus_ref)
    )
    materialization_state = "staged" if corpus_ref is None else "finalized"
    schedule_stages = cast(
        list[dict[str, Any]],
        _json_safe(cast(list[Any], training_surface_training.get("schedule_stages", []))),
    )

    return {
        "block_id": _PRODUCTION_BLOCK_ID,
        "status": "completed",
        "run_dir": str(run_dir),
        "config_excerpt": {
            "experiment": _TRAINING_EXPERIMENT,
            "requested_corpus_ref": requested_corpus_ref,
            "corpus_ref": corpus_ref,
            "manifest_path": (None if manifest_path is None else str(manifest_path)),
            "materialization_state": materialization_state,
            "task_batch_size": int(training_surface_training["task_batch_size"]),
            "runtime": {
                "device": _SUPPORTED_DEVICE,
                "mixed_precision": str(training_surface_runtime["mixed_precision"]),
                "num_workers": int(training_surface_runtime["num_workers"]),
                "grad_accum_steps": int(training_surface_runtime["grad_accum_steps"]),
                "grad_clip": float(training_surface_runtime["grad_clip"]),
                "max_steps": int(training_surface_runtime["max_steps"]),
                "eval_every": int(training_surface_runtime["eval_every"]),
                "checkpoint_every": int(training_surface_runtime["checkpoint_every"]),
                "val_batches": int(training_surface_runtime["val_batches"]),
                "seed": int(training_surface_runtime["seed"]),
            },
            "optimizer": {
                "name": str(training_surface_training["optimizer_name"]),
                "min_lr": float(training_surface_training["optimizer_min_lr"]),
            },
            "schedule_stages": schedule_stages,
            "logging": {
                "run_name": str(
                    telemetry_wandb.get(
                        "run_name",
                        f"{_SUPPORTED_ADEQUACY_ID}-production-control-v4",
                    )
                ),
                "use_wandb": bool(telemetry_wandb),
                "history_jsonl_path": str((run_dir / "train_history.jsonl").resolve()),
            },
            "output_dir": str(run_dir),
        },
        "metrics": {
            "best_val_loss": None,
            "best_val_step": None,
            "final_val_loss": None,
            "train_elapsed_seconds": _finite_float_or_none(last_history.get("train_elapsed_seconds")),
            "wall_elapsed_seconds": _finite_float_or_none(last_history.get("elapsed_seconds")),
        },
        "checkpoints": {
            "best_checkpoint": telemetry_artifacts.get("best_checkpoint"),
            "latest_checkpoint": telemetry_artifacts.get("latest_checkpoint"),
        },
        "run_inspect": _run_inspect_excerpt(inspect_payload),
    }


def run_production_control_pilot(
    *,
    requested_corpus_ref: str,
    corpus_ref: str | None,
    manifest_path: Path | None,
    materialization_state: str,
    pilot_root: Path,
    device: str,
    force: bool,
) -> dict[str, Any]:
    run_root = pilot_root / _PRODUCTION_BLOCK_ID
    run_dir = run_root / "train"
    if force and run_root.exists():
        shutil.rmtree(run_root)
    run_root.mkdir(parents=True, exist_ok=True)

    cfg = build_production_control_config(
        requested_corpus_ref=requested_corpus_ref,
        corpus_ref=corpus_ref,
        manifest_path=manifest_path,
        materialization_state=materialization_state,
        run_dir=run_dir,
        device=device,
    )
    config_excerpt = {
        "experiment": _TRAINING_EXPERIMENT,
        "requested_corpus_ref": requested_corpus_ref,
        "corpus_ref": corpus_ref,
        "manifest_path": (None if manifest_path is None else str(manifest_path.resolve())),
        "materialization_state": materialization_state,
        "task_batch_size": int(cfg.training.task_batch_size),
        "runtime": {
            "device": str(cfg.runtime.device),
            "mixed_precision": str(cfg.runtime.mixed_precision),
            "num_workers": int(cfg.runtime.num_workers),
            "grad_accum_steps": int(cfg.runtime.grad_accum_steps),
            "grad_clip": float(cfg.runtime.grad_clip),
            "max_steps": int(cfg.runtime.max_steps),
            "eval_every": int(cfg.runtime.eval_every),
            "checkpoint_every": int(cfg.runtime.checkpoint_every),
            "val_batches": int(cfg.runtime.val_batches),
            "seed": int(cfg.runtime.seed),
        },
        "optimizer": {
            "name": str(cfg.optimizer.name),
            "min_lr": float(cfg.optimizer.min_lr),
        },
        "schedule_stages": cast(
            list[dict[str, Any]],
            OmegaConf.to_container(cfg.schedule.stages, resolve=True),
        ),
        "logging": {
            "run_name": str(cfg.logging.run_name),
            "use_wandb": bool(cfg.logging.use_wandb),
            "history_jsonl_path": str(cfg.logging.history_jsonl_path),
        },
        "output_dir": str(run_dir.resolve()),
    }

    try:
        result = train(cfg)
    except Exception as exc:
        inspect_payload = None
        if run_dir.exists():
            try:
                inspect_payload = run_inspect(run_dir)
            except Exception:
                inspect_payload = None
        if inspect_payload is not None:
            from .shared import _write_json

            _write_json(run_root / "run_inspect.json", cast(Mapping[str, Any], inspect_payload))
        return {
            "block_id": _PRODUCTION_BLOCK_ID,
            "status": "error",
            "error": {
                "type": type(exc).__name__,
                "message": str(exc),
            },
            "run_dir": str(run_dir.resolve()),
            "config_excerpt": config_excerpt,
            "run_inspect": (
                None
                if inspect_payload is None
                else _run_inspect_excerpt(cast(Mapping[str, Any], inspect_payload))
            ),
        }

    inspect_payload = run_inspect(result.output_dir)
    from .shared import _write_json

    _write_json(run_root / "run_inspect.json", cast(Mapping[str, Any], inspect_payload))
    return {
        "block_id": _PRODUCTION_BLOCK_ID,
        "status": "completed",
        "run_dir": str(result.output_dir.resolve()),
        "config_excerpt": config_excerpt,
        "metrics": {
            "best_val_loss": _finite_float_or_none(result.metrics.get("best_val_loss")),
            "best_val_step": _finite_float_or_none(result.metrics.get("best_val_step")),
            "final_val_loss": _finite_float_or_none(result.metrics.get("final_val_loss")),
            "train_elapsed_seconds": _finite_float_or_none(result.metrics.get("train_elapsed_seconds")),
            "wall_elapsed_seconds": _finite_float_or_none(result.metrics.get("wall_elapsed_seconds")),
        },
        "checkpoints": {
            "best_checkpoint": (
                None
                if result.best_checkpoint is None
                else str(result.best_checkpoint.resolve())
            ),
            "latest_checkpoint": (
                None
                if result.latest_checkpoint is None
                else str(result.latest_checkpoint.resolve())
            ),
        },
        "run_inspect": _run_inspect_excerpt(inspect_payload),
    }


__all__ = [
    "build_production_control_config",
    "run_production_control_pilot",
]
