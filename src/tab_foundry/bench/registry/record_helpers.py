"""Pure record-derivation helpers for the benchmark run registry."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, cast

from tab_foundry.bench.nanotabpfn import (
    resolve_tab_foundry_best_checkpoint,
    resolve_tab_foundry_run_artifact_paths,
)
from tab_foundry.model.architectures.tabfoundry_staged.resolved import resolve_staged_surface
from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.model.spec import (
    checkpoint_model_build_spec_from_mappings,
    ModelBuildSpec,
)
from tab_foundry.training.instability import grad_norm_summary_from_values
from tab_foundry.training.schedule import build_stage_configs, warmup_steps_for_stage
from tab_foundry.training.surface import write_training_surface_record


def _ensure_non_empty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{context} must be a non-empty string")
    return str(value)


def _history_variance(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / float(len(values))
    return sum((value - mean) ** 2 for value in values) / float(len(values))


def _finite_history_values(history: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in history:
        raw = record.get(key)
        if raw is None:
            continue
        value = float(raw)
        if math.isfinite(value):
            values.append(value)
    return values


def _training_diagnostics_from_history(
    history: list[dict[str, Any]],
    *,
    raw_cfg: dict[str, Any],
) -> dict[str, float | None]:
    val_records: list[tuple[float, float]] = []
    for record in history:
        raw_val_loss = record.get("val_loss")
        if raw_val_loss is None:
            continue
        value = float(raw_val_loss)
        if math.isfinite(value):
            val_records.append((float(record["step"]), value))

    best_val_loss: float | None = None
    best_val_step: float | None = None
    final_val_loss: float | None = None
    if val_records:
        best_val_step, best_val_loss = min(val_records, key=lambda item: (item[1], item[0]))
        final_val_loss = float(val_records[-1][1])

    grad_norms = _finite_history_values(history, "grad_norm")
    raw_schedule = raw_cfg.get("schedule")
    warmup_steps = 0
    if isinstance(raw_schedule, dict):
        raw_stages = raw_schedule.get("stages")
        if isinstance(raw_stages, list):
            normalized_stages = [
                {str(key): value for key, value in stage.items()}
                for stage in raw_stages
                if isinstance(stage, dict)
            ]
            if normalized_stages:
                stage_configs = build_stage_configs(normalized_stages)
                if stage_configs:
                    warmup_steps = warmup_steps_for_stage(stage_configs[0])
    post_warmup_losses = [
        float(record["train_loss"])
        for record in history
        if int(record["step"]) > warmup_steps
        and record.get("train_loss") is not None
        and math.isfinite(float(record["train_loss"]))
    ]
    last_record = history[-1]
    train_elapsed = float(last_record.get("train_elapsed_seconds", 0.0))
    wall_elapsed = float(last_record.get("elapsed_seconds", train_elapsed))
    return {
        "best_val_loss": None if best_val_loss is None else float(best_val_loss),
        "final_val_loss": None if final_val_loss is None else float(final_val_loss),
        "best_val_step": None if best_val_step is None else float(best_val_step),
        "post_warmup_train_loss_var": _history_variance(post_warmup_losses),
        **grad_norm_summary_from_values(grad_norms),
        "train_elapsed_seconds": train_elapsed if math.isfinite(train_elapsed) else None,
        "wall_elapsed_seconds": wall_elapsed if math.isfinite(wall_elapsed) else None,
    }


def _checkpoint_model_spec_from_cfg(
    raw_cfg: dict[str, Any],
    *,
    state_dict: dict[str, Any] | None,
) -> ModelBuildSpec:
    task = _ensure_non_empty_string(raw_cfg.get("task"), context="checkpoint config.task")
    raw_model_cfg = raw_cfg.get("model")
    if not isinstance(raw_model_cfg, dict):
        raise RuntimeError("checkpoint config must include a model mapping")
    model_cfg = {str(key): value for key, value in raw_model_cfg.items()}
    arch = model_cfg.get("arch")
    if not isinstance(arch, str) or not arch.strip():
        raise RuntimeError(
            "checkpoint config must include explicit model.arch metadata for benchmark registration; "
            "legacy checkpoints without persisted model.arch cannot be registered"
        )
    return checkpoint_model_build_spec_from_mappings(
        task=task,
        primary=model_cfg,
        state_dict=state_dict,
    )


def _count_parameters_from_cfg(
    raw_cfg: dict[str, Any],
    *,
    state_dict: dict[str, Any] | None,
) -> dict[str, int]:
    model_spec = _checkpoint_model_spec_from_cfg(raw_cfg, state_dict=state_dict)
    model = build_model_from_spec(model_spec)
    total_params = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable_params = sum(
        int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
    }


def _model_payload_from_cfg(
    raw_cfg: dict[str, Any],
    *,
    state_dict: dict[str, Any] | None,
    summary_tab_foundry: Mapping[str, Any],
) -> dict[str, Any]:
    model_spec = _checkpoint_model_spec_from_cfg(raw_cfg, state_dict=state_dict)
    benchmark_profile_raw = summary_tab_foundry.get("benchmark_profile")
    payload: dict[str, Any] = {
        "arch": str(model_spec.arch),
        "stage": None if model_spec.stage is None else str(model_spec.stage),
        "stage_label": None if model_spec.stage_label is None else str(model_spec.stage_label),
        "benchmark_profile": None if benchmark_profile_raw is None else str(benchmark_profile_raw),
        "d_icl": int(model_spec.d_icl),
        "tficl_n_heads": int(model_spec.tficl_n_heads),
        "tficl_n_layers": int(model_spec.tficl_n_layers),
        "head_hidden_dim": int(model_spec.head_hidden_dim),
        "input_normalization": str(model_spec.input_normalization),
        "many_class_base": int(model_spec.many_class_base),
    }
    if model_spec.arch == "tabfoundry_staged":
        surface = resolve_staged_surface(model_spec)
        if payload["stage_label"] is None:
            payload["stage_label"] = str(surface.stage_label)
        if payload["benchmark_profile"] is None:
            payload["benchmark_profile"] = str(surface.benchmark_profile)
        payload["module_selection"] = surface.module_selection()
        payload["module_hyperparameters"] = surface.component_hyperparameters()
    return payload


def _training_surface_record(
    *,
    run_dir: Path,
    raw_cfg: dict[str, Any],
    raw_state_dict: dict[str, Any] | None,
    benchmark_run_record_path: Path | None,
) -> tuple[dict[str, Any] | None, Path | None]:
    run_record_path = run_dir.expanduser().resolve() / "training_surface_record.json"
    if run_record_path.exists():
        with run_record_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise RuntimeError(f"training_surface_record.json must be a JSON object: {run_record_path}")
        return cast(dict[str, Any], payload), run_record_path
    if benchmark_run_record_path is None:
        return None, None
    derived_path = benchmark_run_record_path.parent / "training_surface_record.json"
    payload = write_training_surface_record(
        derived_path,
        raw_cfg=raw_cfg,
        run_dir=run_dir,
        state_dict=raw_state_dict,
    )
    return payload, derived_path


def _coerce_integral_step(value: Any) -> int | None:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return None
    value_f = float(value)
    if not math.isfinite(value_f) or not value_f.is_integer():
        return None
    return int(value_f)


def _resolve_record_checkpoint_path(
    run_dir: Path,
    *,
    summary_tab_foundry: Mapping[str, Any],
) -> Path:
    try:
        return resolve_tab_foundry_best_checkpoint(run_dir)
    except RuntimeError as best_exc:
        _history_path, checkpoint_dir = resolve_tab_foundry_run_artifact_paths(run_dir)
        candidate_steps: list[int] = []
        for key in ("best_step", "final_step"):
            step = _coerce_integral_step(summary_tab_foundry.get(key))
            if step is not None and step > 0 and step not in candidate_steps:
                candidate_steps.append(step)
        for step in candidate_steps:
            step_path = checkpoint_dir / f"step_{step:06d}.pt"
            if step_path.exists():
                return step_path.resolve()
        latest_path = checkpoint_dir / "latest.pt"
        if latest_path.exists():
            return latest_path.resolve()
        step_paths = sorted(checkpoint_dir.glob("step_*.pt"))
        if step_paths:
            return step_paths[-1].resolve()
        raise RuntimeError(
            "missing checkpoint artifact suitable for benchmark run record under "
            f"{run_dir.expanduser().resolve()}; best.pt lookup failed with: {best_exc}"
        ) from best_exc
