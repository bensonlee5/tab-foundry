"""Shared surface resolution helpers for system-delta sweep inspection and graphing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable, Mapping, cast

from omegaconf import OmegaConf
import torch

from tab_foundry.benchmark_registry import (
    load_benchmark_run_registry,
    resolve_registry_path_value,
)
from tab_foundry.model.spec import (
    ModelBuildSpec,
    checkpoint_model_build_spec_from_mappings,
    model_build_spec_from_mappings,
)
from tab_foundry.research.lane_contract import resolve_training_experiment
from tab_foundry.training.surface import build_training_surface_record

from .configuration import compose_cfg
from .materialize import load_system_delta_queue_for_inspection, ordered_rows
from .paths_io import default_registry_path, repo_root


def _require_non_empty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{context} must be a non-empty string")
    return str(value)


def _require_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _load_json_mapping(path: Path, *, context: str) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} must decode to a JSON object: {path.expanduser().resolve()}")
    return cast(dict[str, Any], payload)


def queue_row_run_dir(queue_row: Mapping[str, Any]) -> Path:
    delta_id = _require_non_empty_string(
        queue_row.get("delta_id", queue_row.get("delta_ref")),
        context="queue row delta_id",
    )
    return repo_root() / "outputs" / ".graph_spec_resolution" / delta_id / "train"


def resolve_queue_row_cfg_mapping(
    queue_row: Mapping[str, Any],
    *,
    run_dir: Path,
    training_experiment: str,
    sweep_id: str | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    cfg = compose_cfg(
        row=queue_row,
        run_dir=run_dir,
        device="cpu",
        training_experiment=training_experiment,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )
    payload = OmegaConf.to_container(cfg, resolve=True)
    if not isinstance(payload, dict):
        raise RuntimeError("resolved config must be a mapping")
    return {str(key): value for key, value in payload.items()}


def inspection_raw_cfg_mapping(
    *,
    row: Mapping[str, Any],
    training_experiment: str,
    sweep_id: str | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    target_sweep_id = sweep_id if sweep_id is not None else "inspection"
    return resolve_queue_row_cfg_mapping(
        row,
        run_dir=repo_root()
        / "outputs"
        / ".inspection"
        / "research"
        / target_sweep_id
        / "row"
        / "raw_cfg"
        / "train",
        training_experiment=training_experiment,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )


def resolve_queue_row_model_spec(
    queue_row: Mapping[str, Any],
    *,
    training_experiment: str,
) -> ModelBuildSpec:
    raw_cfg = resolve_queue_row_cfg_mapping(
        queue_row,
        run_dir=queue_row_run_dir(queue_row),
        training_experiment=training_experiment,
    )
    task = str(raw_cfg.get("task", "classification")).strip().lower()
    raw_model_cfg = raw_cfg.get("model")
    if not isinstance(raw_model_cfg, Mapping):
        raise RuntimeError("resolved queue-row cfg.model must be a mapping")
    normalized_model_cfg = {str(key): value for key, value in raw_model_cfg.items()}
    return model_build_spec_from_mappings(task=task, primary=normalized_model_cfg)


def training_surface_record_model_spec(training_surface_record_path: Path) -> ModelBuildSpec:
    payload = _load_json_mapping(
        training_surface_record_path,
        context="training surface record",
    )
    model_payload = _require_mapping(payload.get("model"), context="training surface record model")
    build_spec_payload = _require_mapping(
        model_payload.get("build_spec"),
        context="training surface record model.build_spec",
    )
    task = str(build_spec_payload.get("task", "classification")).strip().lower()
    return model_build_spec_from_mappings(task=task, primary=build_spec_payload)


def _checkpoint_model_spec_from_path(checkpoint_path: Path) -> ModelBuildSpec:
    payload = torch.load(
        checkpoint_path.expanduser().resolve(),
        map_location="cpu",
        weights_only=False,
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint payload must be a mapping: {checkpoint_path.expanduser().resolve()}")
    raw_cfg = payload.get("config")
    checkpoint_cfg = raw_cfg if isinstance(raw_cfg, Mapping) else {}
    task = str(checkpoint_cfg.get("task", "classification")).strip().lower()
    raw_model_cfg = checkpoint_cfg.get("model")
    model_cfg = raw_model_cfg if isinstance(raw_model_cfg, Mapping) else {}
    raw_state_dict = payload.get("model")
    state_dict = raw_state_dict if isinstance(raw_state_dict, Mapping) else None
    return checkpoint_model_build_spec_from_mappings(
        task=task,
        primary={str(key): value for key, value in model_cfg.items()},
        state_dict=cast(dict[str, Any] | None, state_dict),
    )


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    if isinstance(value, str):
        stripped = value.strip()
        if stripped:
            try:
                return int(stripped)
            except ValueError:
                return None
    return None


def resolve_anchor_originating_queue_row(
    *,
    queue: Mapping[str, Any],
    registry_path: Path | None = None,
    index_path: Path | None = None,
    sweeps_root: Path | None = None,
    load_registry: Callable[[Path], Mapping[str, Any]] = load_benchmark_run_registry,
) -> tuple[dict[str, Any], dict[str, Any]] | None:
    anchor_run_id = _require_non_empty_string(queue.get("anchor_run_id"), context="anchor_run_id")
    registry = load_registry(registry_path or default_registry_path())
    runs = _require_mapping(registry.get("runs"), context="benchmark registry runs")
    raw_run = runs.get(anchor_run_id)
    if not isinstance(raw_run, Mapping):
        return None
    run = cast(Mapping[str, Any], raw_run)
    sweep_payload = run.get("sweep")
    if not isinstance(sweep_payload, Mapping):
        return None
    raw_sweep_id = sweep_payload.get("sweep_id")
    if not isinstance(raw_sweep_id, str) or not raw_sweep_id.strip():
        return None

    source_queue = load_system_delta_queue_for_inspection(
        sweep_id=str(raw_sweep_id),
        index_path=index_path,
        sweeps_root=sweeps_root,
    )
    source_training_experiment = resolve_training_experiment(source_queue)
    queue_order = _optional_int(sweep_payload.get("queue_order"))
    delta_id = sweep_payload.get("delta_id")
    normalized_delta_id = str(delta_id).strip() if isinstance(delta_id, str) and delta_id.strip() else None

    for row in ordered_rows(source_queue):
        row_run_id = row.get("run_id")
        if isinstance(row_run_id, str) and row_run_id == anchor_run_id:
            return row, {
                "source": "originating_sweep_row",
                "run_id": anchor_run_id,
                "sweep_id": str(raw_sweep_id),
                "order": int(row["order"]),
                "delta_id": str(row["delta_id"]),
                "training_experiment": source_training_experiment,
            }

    if queue_order is not None:
        for row in ordered_rows(source_queue):
            if int(row["order"]) != queue_order:
                continue
            if normalized_delta_id is not None and str(row["delta_id"]) != normalized_delta_id:
                continue
            return row, {
                "source": "originating_sweep_row",
                "run_id": anchor_run_id,
                "sweep_id": str(raw_sweep_id),
                "order": int(row["order"]),
                "delta_id": str(row["delta_id"]),
                "training_experiment": source_training_experiment,
            }

    if normalized_delta_id is not None:
        for row in ordered_rows(source_queue):
            if str(row["delta_id"]) != normalized_delta_id:
                continue
            return row, {
                "source": "originating_sweep_row",
                "run_id": anchor_run_id,
                "sweep_id": str(raw_sweep_id),
                "order": int(row["order"]),
                "delta_id": str(row["delta_id"]),
                "training_experiment": source_training_experiment,
            }

    return None


def resolve_anchor_model_spec(
    *,
    queue: Mapping[str, Any],
    registry_path: Path | None = None,
    index_path: Path | None = None,
    sweeps_root: Path | None = None,
    load_registry: Callable[[Path], Mapping[str, Any]] = load_benchmark_run_registry,
    resolve_registry_path: Callable[[str], Path] = resolve_registry_path_value,
) -> tuple[ModelBuildSpec, dict[str, Any]]:
    anchor_run_id = _require_non_empty_string(queue.get("anchor_run_id"), context="anchor_run_id")
    training_experiment = resolve_training_experiment(queue)
    for row in ordered_rows(queue):
        row_run_id = row.get("run_id")
        if isinstance(row_run_id, str) and row_run_id == anchor_run_id:
            return resolve_queue_row_model_spec(
                row,
                training_experiment=training_experiment,
            ), {
                "source": "queue_row",
                "run_id": anchor_run_id,
                "order": int(row["order"]),
                "delta_id": str(row["delta_id"]),
            }

    registry = load_registry(registry_path or default_registry_path())
    runs = _require_mapping(registry.get("runs"), context="benchmark registry runs")
    raw_run = runs.get(anchor_run_id)
    if not isinstance(raw_run, Mapping):
        raise RuntimeError(f"anchor_run_id {anchor_run_id!r} is missing from the benchmark registry")
    run = cast(Mapping[str, Any], raw_run)
    artifacts = _require_mapping(
        run.get("artifacts"),
        context=f"benchmark registry run {anchor_run_id}.artifacts",
    )

    raw_training_surface_path = artifacts.get("training_surface_record_path")
    if isinstance(raw_training_surface_path, str) and raw_training_surface_path.strip():
        training_surface_path = resolve_registry_path(raw_training_surface_path)
        if training_surface_path.exists():
            return training_surface_record_model_spec(training_surface_path), {
                "source": "training_surface_record",
                "run_id": anchor_run_id,
                "training_surface_record_path": str(training_surface_path),
            }

    raw_checkpoint_path = artifacts.get("best_checkpoint_path")
    if isinstance(raw_checkpoint_path, str) and raw_checkpoint_path.strip():
        checkpoint_path = resolve_registry_path(raw_checkpoint_path)
        if checkpoint_path.exists():
            return _checkpoint_model_spec_from_path(checkpoint_path), {
                "source": "checkpoint",
                "run_id": anchor_run_id,
                "checkpoint_path": str(checkpoint_path),
            }

    originating_row = resolve_anchor_originating_queue_row(
        queue=queue,
        registry_path=registry_path,
        index_path=index_path,
        sweeps_root=sweeps_root,
        load_registry=load_registry,
    )
    if originating_row is not None:
        row, metadata = originating_row
        return resolve_queue_row_model_spec(
            row,
            training_experiment=str(metadata["training_experiment"]),
        ), metadata

    raise RuntimeError(
        "unable to resolve anchor model spec for "
        f"{anchor_run_id!r}: no matching completed sweep row, readable "
        "`training_surface_record.json`, readable best checkpoint config, or "
        "originating sweep row"
    )


def build_lightweight_training_surface_record(
    *,
    raw_cfg: Mapping[str, Any],
    run_dir: Path,
    state_dict: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    return build_training_surface_record(
        raw_cfg=raw_cfg,
        run_dir=run_dir,
        state_dict=state_dict,
        include_manifest_characteristics=False,
        allow_unresolved_corpus_ref=True,
    )


def inspection_spec_and_record(
    *,
    row: Mapping[str, Any],
    run_dir: Path,
    training_experiment: str,
    sweep_id: str,
    sweeps_root: Path | None = None,
) -> tuple[ModelBuildSpec, dict[str, Any]]:
    raw_cfg = resolve_queue_row_cfg_mapping(
        row,
        run_dir=run_dir,
        training_experiment=training_experiment,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )
    task = str(raw_cfg.get("task", "classification")).strip().lower()
    raw_model_cfg = raw_cfg.get("model")
    if not isinstance(raw_model_cfg, Mapping):
        raise RuntimeError("inspection fallback requires cfg.model to resolve to a mapping")
    spec = model_build_spec_from_mappings(
        task=task,
        primary={str(key): value for key, value in raw_model_cfg.items()},
    )
    training_surface_record = build_lightweight_training_surface_record(
        raw_cfg=raw_cfg,
        run_dir=run_dir,
    )
    return spec, training_surface_record
