"""Prior-dump training for exact-parity tabfoundry classification models."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from tab_foundry.bench.nanotabpfn import resolve_device
from tab_foundry.bench.prior_dump import (
    PriorDumpBatchMissingness,
    PriorDumpNonFinitePolicy,
    PriorDumpTaskBatchReader,
)
from tab_foundry.config import compose_config
from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.model.architectures.tabfoundry_staged.resolved import (
    ResolvedStageSurface,
    resolve_staged_surface,
)
from tab_foundry.model.spec import ModelBuildSpec, model_build_spec_from_mappings
from tab_foundry.training.artifacts import (
    append_jsonl_record,
    append_history_record,
    assert_clean_training_output,
    gradient_history_record,
    history_path_from_cfg,
    history_record,
    save_checkpoint,
)
from tab_foundry.training.instability import (
    build_training_telemetry,
    gradient_history_path,
    module_grad_norms,
    normalize_grad_norm_value,
    telemetry_path,
    total_grad_norm,
    train_loss_delta,
    update_loss_ema,
    write_training_telemetry,
)
from tab_foundry.training.losses import classification_loss
from tab_foundry.training.optimizer import build_optimizer
from tab_foundry.training.surface import write_training_surface_record
from tab_foundry.training.schedule import build_stage_configs, stage_base_lr, StageConfig
from tab_foundry.training.trainer import _set_optimizer_base_lr, _set_optimizer_training_mode
from tab_foundry.training.wandb import (
    finish_wandb_run,
    init_wandb_run,
    log_wandb_metrics,
    update_wandb_summary,
)
from tab_foundry.types import TrainResult


DEFAULT_PRIOR_DUMP_PATH = Path("~/dev/nanoTabPFN/300k_150x5_2.h5")
DEFAULT_BATCH_SIZE = 32
DEFAULT_EXPERIMENT = "cls_benchmark_linear_simple_prior"
_PRIOR_STAGE_NAME = "prior_dump"
_DEFAULT_STAGED_PRIOR_WANDB_RUN_NAME = "cls-benchmark-staged-prior"
_SUPPORTED_PRIOR_DUMP_LR_SCALE_RULES = ("none", "sqrt", "linear")


@dataclass(slots=True, frozen=True)
class _PriorDumpBatchConfig:
    batch_size: int
    lr_scale_rule: str
    reference_batch_size: int
    effective_lr_scale_factor: float


def _resolve_positive_int(value: object, *, name: str) -> int:
    if not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} must be int-compatible, got {value!r}")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be >= 1, got {resolved}")
    return resolved


def _resolve_runtime_bool(value: object, *, name: str) -> bool:
    if value is None:
        return False
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "on"}:
            return True
        if normalized in {"0", "false", "no", "off", ""}:
            return False
    raise ValueError(f"{name} must be boolean-compatible, got {value!r}")


def _resolve_mapping(value: object, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    resolved = OmegaConf.to_container(value, resolve=True)
    if resolved is None:
        return {}
    if not isinstance(resolved, dict):
        raise ValueError(f"{name} must resolve to a mapping, got {resolved!r}")
    return {str(key): item for key, item in resolved.items()}



def _resolve_prior_missingness_config(cfg: DictConfig) -> dict[str, Any] | None:
    training_cfg = getattr(cfg, "training", None)
    overrides = _resolve_mapping(
        None if training_cfg is None else getattr(training_cfg, "overrides", None),
        name="training.overrides",
    )
    raw = overrides.get("prior_missingness")
    if raw is None:
        return None
    if not isinstance(raw, dict):
        raise ValueError(
            "training.overrides.prior_missingness must resolve to a mapping"
        )
    enabled = _resolve_runtime_bool(
        raw.get("enabled", False),
        name="training.overrides.prior_missingness.enabled",
    )
    min_rate = float(raw.get("min_rate", 0.0))
    max_rate = float(raw.get("max_rate", min_rate))
    if not 0.0 <= min_rate <= 1.0:
        raise ValueError(
            "training.overrides.prior_missingness.min_rate must be in [0, 1], "
            f"got {min_rate}"
        )
    if not 0.0 <= max_rate <= 1.0:
        raise ValueError(
            "training.overrides.prior_missingness.max_rate must be in [0, 1], "
            f"got {max_rate}"
        )
    if min_rate > max_rate:
        raise ValueError(
            "training.overrides.prior_missingness.min_rate must be <= max_rate, "
            f"got min_rate={min_rate}, max_rate={max_rate}"
        )
    if not enabled:
        return None
    return {
        "enabled": True,
        "min_rate": min_rate,
        "max_rate": max_rate,
    }


def _resolve_prior_dump_non_finite_policy(cfg: DictConfig) -> PriorDumpNonFinitePolicy:
    training_cfg = getattr(cfg, "training", None)
    raw_value = (
        "error"
        if training_cfg is None
        else getattr(training_cfg, "prior_dump_non_finite_policy", "error")
    )
    normalized = str(raw_value).strip().lower()
    if normalized not in {"error", "skip"}:
        raise ValueError(
            "training.prior_dump_non_finite_policy must be one of {'error', 'skip'}, "
            f"got {raw_value!r}"
        )
    return cast(PriorDumpNonFinitePolicy, normalized)


def _queue_aware_run_name_from_output_dir(output_dir: Path) -> str | None:
    candidate = output_dir.parent.name if output_dir.name == "train" else output_dir.name
    if not candidate.startswith("sd_"):
        return None
    stem, separator, version = candidate.rpartition("_v")
    if not separator or not stem or not version.isdigit():
        return None
    return candidate


def _resolve_prior_wandb_run_name(cfg: DictConfig) -> str:
    output_dir = Path(str(cfg.runtime.output_dir)).expanduser().resolve()
    queue_aware_run_name = _queue_aware_run_name_from_output_dir(output_dir)
    if queue_aware_run_name is not None:
        return queue_aware_run_name

    raw_run_name = str(getattr(cfg.logging, "run_name", "") or "").strip()
    if raw_run_name and raw_run_name != _DEFAULT_STAGED_PRIOR_WANDB_RUN_NAME:
        return raw_run_name

    candidate = output_dir.parent.name if output_dir.name == "train" else output_dir.name
    if candidate.startswith("sd_stability_followup_dpnb_") and candidate.endswith("_v1"):
        parts = candidate.split("_", 5)
        if len(parts) == 6:
            suffix = parts[-1].removesuffix("_v1")
            return f"dpnb_{suffix}"
    if candidate.startswith("sd_stability_followup_") and candidate.endswith("_v1"):
        parts = candidate.split("_", 4)
        if len(parts) == 5:
            return str(parts[-1].removesuffix("_v1"))
    if candidate:
        return str(candidate)

    stage_label = str(getattr(cfg.model, "stage_label", "") or "").strip()
    if stage_label:
        return stage_label
    return _DEFAULT_STAGED_PRIOR_WANDB_RUN_NAME


def _resolve_lr(cfg: DictConfig) -> float:
    raw_lr = getattr(cfg.optimizer, "min_lr", None)
    if raw_lr is None:
        raise ValueError("exact-parity prior training requires optimizer.min_lr")
    lr = float(raw_lr)
    if lr <= 0:
        raise ValueError(f"optimizer.min_lr must be > 0, got {lr}")
    return lr


def _resolve_prior_dump_batch_size(
    cfg: DictConfig,
    *,
    override: int | None = None,
) -> int:
    if override is not None:
        return _resolve_positive_int(override, name="batch_size")
    training_cfg = getattr(cfg, "training", None)
    raw_value = (
        DEFAULT_BATCH_SIZE
        if training_cfg is None
        else getattr(training_cfg, "prior_dump_batch_size", DEFAULT_BATCH_SIZE)
    )
    return _resolve_positive_int(raw_value, name="training.prior_dump_batch_size")


def _resolve_prior_dump_lr_scale_rule(cfg: DictConfig) -> str:
    training_cfg = getattr(cfg, "training", None)
    raw_value = (
        "none"
        if training_cfg is None
        else getattr(training_cfg, "prior_dump_lr_scale_rule", "none")
    )
    normalized = str(raw_value).strip().lower()
    if normalized not in _SUPPORTED_PRIOR_DUMP_LR_SCALE_RULES:
        raise ValueError(
            "training.prior_dump_lr_scale_rule must be one of "
            f"{_SUPPORTED_PRIOR_DUMP_LR_SCALE_RULES}, got {raw_value!r}"
        )
    return normalized


def _resolve_prior_dump_batch_reference_size(cfg: DictConfig) -> int:
    training_cfg = getattr(cfg, "training", None)
    raw_value = (
        DEFAULT_BATCH_SIZE
        if training_cfg is None
        else getattr(training_cfg, "prior_dump_batch_reference_size", DEFAULT_BATCH_SIZE)
    )
    return _resolve_positive_int(raw_value, name="training.prior_dump_batch_reference_size")


def _resolve_prior_dump_batch_config(
    cfg: DictConfig,
    *,
    batch_size_override: int | None = None,
) -> _PriorDumpBatchConfig:
    batch_size = _resolve_prior_dump_batch_size(cfg, override=batch_size_override)
    lr_scale_rule = _resolve_prior_dump_lr_scale_rule(cfg)
    reference_batch_size = _resolve_prior_dump_batch_reference_size(cfg)
    ratio = float(batch_size) / float(reference_batch_size)
    if lr_scale_rule == "none":
        factor = 1.0
    elif lr_scale_rule == "sqrt":
        factor = math.sqrt(ratio)
    else:
        factor = ratio
    return _PriorDumpBatchConfig(
        batch_size=batch_size,
        lr_scale_rule=lr_scale_rule,
        reference_batch_size=reference_batch_size,
        effective_lr_scale_factor=float(factor),
    )


def _resolve_prior_schedule(
    cfg: DictConfig,
    *,
    max_steps: int,
    lr_min: float,
    lr_scale_factor: float = 1.0,
) -> StageConfig | None:
    training_cfg = getattr(cfg, "training", None)
    apply_schedule = bool(getattr(training_cfg, "apply_schedule", False))
    if not apply_schedule:
        return None

    schedule_cfg = getattr(cfg, "schedule", None)
    raw_stages = getattr(schedule_cfg, "stages", None)
    if raw_stages is None:
        raise ValueError("exact-parity prior training requires schedule.stages when training.apply_schedule=true")
    resolved = OmegaConf.to_container(raw_stages, resolve=True)
    if not isinstance(resolved, list):
        raise ValueError("exact-parity prior training requires schedule.stages to resolve to a list")
    stages = build_stage_configs(cast(list[dict[str, object]], resolved))
    if len(stages) != 1:
        raise ValueError("exact-parity prior training currently supports exactly one schedule stage")
    stage = stages[0]
    if lr_scale_factor <= 0.0:
        raise ValueError(f"lr_scale_factor must be > 0, got {lr_scale_factor}")
    stage = StageConfig(
        name=str(stage.name),
        steps=int(stage.steps),
        lr_max=float(stage.lr_max) * float(lr_scale_factor),
        lr_schedule=str(stage.lr_schedule),
        warmup_ratio=float(stage.warmup_ratio),
    )
    if int(stage.steps) != int(max_steps):
        raise ValueError(
            "exact-parity prior training requires schedule.stages[0].steps to equal runtime.max_steps; "
            f"got steps={stage.steps}, max_steps={max_steps}"
        )
    if float(lr_min) > float(stage.lr_max):
        raise ValueError(
            "exact-parity prior training requires optimizer.min_lr <= schedule.stages[0].lr_max; "
            f"got min_lr={lr_min}, lr_max={stage.lr_max}"
        )
    return stage


def _optimizer_kwargs(cfg: DictConfig) -> dict[str, object]:
    kwargs: dict[str, object] = {}
    raw_betas = getattr(cfg.optimizer, "betas", None)
    if raw_betas is not None:
        kwargs["betas"] = tuple(float(value) for value in raw_betas)
    return kwargs


def _model_spec_from_cfg(cfg: DictConfig):
    raw_model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(raw_model_cfg, dict):
        raise RuntimeError("cfg.model must resolve to a mapping")
    primary_cfg = {str(key): value for key, value in raw_model_cfg.items()}
    return model_build_spec_from_mappings(
        task=str(cfg.task),
        primary=primary_cfg,
    )


def _validate_prior_training_model_spec(
    spec: ModelBuildSpec,
) -> ResolvedStageSurface | None:
    if spec.arch not in {"tabfoundry_simple", "tabfoundry_staged"}:
        raise ValueError(
            "prior-dump training requires model.arch in {'tabfoundry_simple', 'tabfoundry_staged'}, "
            f"got {spec.arch!r}"
        )
    if spec.arch != "tabfoundry_staged":
        return None

    surface = resolve_staged_surface(spec)
    if surface.head == "many_class":
        raise ValueError(
            "prior-dump training requires a staged recipe with forward_batched() tensor logits, "
            f"got model.stage={surface.stage!r}"
        )
    return surface


def _resolve_prior_training_device_name(
    cfg: DictConfig,
    *,
    spec: ModelBuildSpec,
    staged_surface: ResolvedStageSurface | None,
) -> str:
    requested_device = str(getattr(cfg.runtime, "device", "auto") or "auto").strip()
    resolved_device = resolve_device(requested_device)
    if (
        resolved_device != "mps"
        or spec.arch != "tabfoundry_staged"
        or staged_surface is None
        or staged_surface.row_pool != "row_cls"
    ):
        return resolved_device

    row_pool_layers = int(staged_surface.row_pool_config.n_layers or 0)
    if row_pool_layers <= 1:
        return resolved_device

    print(
        "Warning: exact prior-dump training requested "
        f"runtime.device={requested_device!r} resolved to 'mps', but staged "
        f"row_pool='row_cls' with tfrow_n_layers={row_pool_layers} is unstable on MPS; "
        "falling back to CPU for this run.",
        file=sys.stderr,
        flush=True,
    )
    cfg.runtime.device = "cpu"
    return "cpu"


def _stack_prior_step(
    prior_step,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if prior_step.x_batch is not None and prior_step.y_batch is not None:
        x_batch = prior_step.x_batch.to(device=device, dtype=torch.float32)
        y_batch = prior_step.y_batch.to(device=device, dtype=torch.float32)
        return (
            x_batch,
            y_batch[:, : prior_step.train_test_split_index],
            y_batch,
        )

    tasks = prior_step.tasks
    if len(tasks) <= 0:
        raise RuntimeError(f"prior dump step {prior_step.step_index} produced no tasks")
    first_x_all = torch.cat([tasks[0].x_train, tasks[0].x_test], dim=0)
    row_count = int(first_x_all.shape[0])
    feature_count = int(first_x_all.shape[1])
    for task in tasks[1:]:
        x_all = torch.cat([task.x_train, task.x_test], dim=0)
        if tuple(x_all.shape) != (row_count, feature_count):
            raise RuntimeError(
                "exact nanoTabPFN parity requires rectangular prior batches, but got "
                f"mixed x_all shapes {(row_count, feature_count)} and {tuple(x_all.shape)}"
            )
        y_all = torch.cat([task.y_train, task.y_test], dim=0)
        if tuple(y_all.shape) != (row_count,):
            raise RuntimeError(
                "exact nanoTabPFN parity requires matching label lengths across the batch, but got "
                f"{row_count} and {tuple(y_all.shape)}"
            )
    x_batch = torch.stack(
        [torch.cat([task.x_train, task.x_test], dim=0) for task in tasks],
        dim=0,
    ).to(device=device, dtype=torch.float32)
    y_train_batch = torch.stack([task.y_train for task in tasks], dim=0).to(
        device=device,
        dtype=torch.float32,
    )
    y_all_batch = torch.stack(
        [torch.cat([task.y_train, task.y_test], dim=0) for task in tasks],
        dim=0,
    ).to(device=device, dtype=torch.float32)
    return x_batch, y_train_batch, y_all_batch


def _save_eval_mode_checkpoint(
    prepared_opts: list[tuple[str, torch.optim.Optimizer]],
    *,
    path: Path,
    model: torch.nn.Module,
    global_step: int,
    cfg: DictConfig,
    restore_training: bool,
) -> None:
    _set_optimizer_training_mode(prepared_opts, training=False)
    save_checkpoint(
        path,
        model_state=model.state_dict(),
        global_step=global_step,
        cfg=cfg,
    )
    if restore_training:
        _set_optimizer_training_mode(prepared_opts, training=True)


def _initial_missingness_summary(
    prior_dump_path: Path,
    *,
    prior_missingness_config: dict[str, Any] | None,
    prior_dump_non_finite_policy: str,
) -> dict[str, Any]:
    synthetic_enabled = prior_missingness_config is not None
    return {
        "prior_dump": {
            "path": str(prior_dump_path.expanduser().resolve()),
            "allow_missing_values": False,
            "non_finite_policy": str(prior_dump_non_finite_policy),
            "batches_seen": 0,
            "affected_batch_count": 0,
            "affected_dataset_indices": [],
            "non_finite_feature_count": 0,
            "non_finite_label_count": 0,
            "last_batch": None,
            "skipped_batch_count": 0,
            "skipped_dataset_indices": [],
            "skipped_non_finite_feature_count": 0,
            "skipped_non_finite_label_count": 0,
            "last_skipped_batch": None,
        },
        "synthetic_prior": {
            "enabled": synthetic_enabled,
            "min_rate": None if prior_missingness_config is None else float(prior_missingness_config["min_rate"]),
            "max_rate": None if prior_missingness_config is None else float(prior_missingness_config["max_rate"]),
            "batches_seen": 0,
            "affected_batch_count": 0,
            "affected_dataset_indices": [],
            "masked_feature_count": 0,
            "last_batch": None,
        },
    }


def _accumulate_missingness(
    summary: dict[str, Any],
    *,
    batch_missingness: PriorDumpBatchMissingness,
    skipped: bool = False,
) -> None:
    prior_dump = cast(dict[str, Any], summary["prior_dump"])
    prior_dump["batches_seen"] = int(prior_dump["batches_seen"]) + 1
    prior_dump["non_finite_feature_count"] = int(prior_dump["non_finite_feature_count"]) + int(
        batch_missingness.non_finite_feature_count
    )
    prior_dump["non_finite_label_count"] = int(prior_dump["non_finite_label_count"]) + int(
        batch_missingness.non_finite_label_count
    )
    if batch_missingness.affected_batch_count > 0:
        prior_dump["affected_batch_count"] = int(prior_dump["affected_batch_count"]) + 1
        known_dataset_indices = {
            int(dataset_index) for dataset_index in prior_dump["affected_dataset_indices"]
        }
        known_dataset_indices.update(int(index) for index in batch_missingness.affected_dataset_indices)
        prior_dump["affected_dataset_indices"] = sorted(known_dataset_indices)
        prior_dump["last_batch"] = batch_missingness.to_dict()
    if skipped:
        prior_dump["skipped_batch_count"] = int(prior_dump["skipped_batch_count"]) + 1
        prior_dump["skipped_non_finite_feature_count"] = int(
            prior_dump["skipped_non_finite_feature_count"]
        ) + int(batch_missingness.non_finite_feature_count)
        prior_dump["skipped_non_finite_label_count"] = int(
            prior_dump["skipped_non_finite_label_count"]
        ) + int(batch_missingness.non_finite_label_count)
        skipped_dataset_indices = {
            int(dataset_index) for dataset_index in prior_dump["skipped_dataset_indices"]
        }
        skipped_dataset_indices.update(int(index) for index in batch_missingness.affected_dataset_indices)
        prior_dump["skipped_dataset_indices"] = sorted(skipped_dataset_indices)
        prior_dump["last_skipped_batch"] = batch_missingness.to_dict()


def _accumulate_synthetic_missingness(
    summary: dict[str, Any],
    *,
    batch_missingness: dict[str, Any] | None,
) -> None:
    synthetic_prior = cast(dict[str, Any], summary["synthetic_prior"])
    synthetic_prior["batches_seen"] = int(synthetic_prior["batches_seen"]) + 1
    if batch_missingness is None:
        return
    masked_feature_count = int(batch_missingness["masked_feature_count"])
    synthetic_prior["masked_feature_count"] = int(synthetic_prior["masked_feature_count"]) + masked_feature_count
    if masked_feature_count <= 0:
        return
    synthetic_prior["affected_batch_count"] = int(synthetic_prior["affected_batch_count"]) + 1
    known_dataset_indices = {
        int(dataset_index) for dataset_index in synthetic_prior["affected_dataset_indices"]
    }
    known_dataset_indices.update(int(index) for index in batch_missingness["affected_dataset_indices"])
    synthetic_prior["affected_dataset_indices"] = sorted(known_dataset_indices)
    synthetic_prior["last_batch"] = batch_missingness



def _apply_prior_missingness(
    x_batch: torch.Tensor,
    *,
    prior_step: Any,
    generator: torch.Generator | None,
    prior_missingness_config: dict[str, Any] | None,
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    if prior_missingness_config is None:
        return x_batch, None

    masked = x_batch.clone()
    min_rate = float(prior_missingness_config["min_rate"])
    max_rate = float(prior_missingness_config["max_rate"])
    affected_dataset_indices: list[int] = []
    affected_datasets: list[dict[str, int | float]] = []
    masked_feature_count = 0
    for local_index, task in enumerate(prior_step.tasks):
        num_rows = int(task.x_train.shape[0] + task.x_test.shape[0])
        num_features = int(task.x_train.shape[1])
        if num_rows <= 0 or num_features <= 0:
            continue
        if max_rate > min_rate:
            assert generator is not None
            rate = min_rate + ((max_rate - min_rate) * float(torch.rand((), generator=generator).item()))
        else:
            rate = min_rate
        if rate <= 0.0:
            continue
        mask = torch.rand((num_rows, num_features), generator=generator, device="cpu") < rate
        dataset_masked_feature_count = int(mask.sum().item())
        if dataset_masked_feature_count <= 0:
            continue
        masked_feature_count += dataset_masked_feature_count
        dataset_index = int(prior_step.dataset_indices[local_index])
        affected_dataset_indices.append(dataset_index)
        affected_datasets.append(
            {
                "dataset_index": dataset_index,
                "masked_feature_count": dataset_masked_feature_count,
                "applied_rate": float(rate),
            }
        )
        masked[local_index, :num_rows, :num_features][mask.to(device=masked.device)] = float("nan")
    return masked, {
        "step_index": int(prior_step.step_index),
        "dataset_indices": [int(index) for index in prior_step.dataset_indices],
        "masked_feature_count": int(masked_feature_count),
        "affected_batch_count": 1 if masked_feature_count > 0 else 0,
        "affected_dataset_count": int(len(affected_dataset_indices)),
        "affected_dataset_indices": sorted(affected_dataset_indices),
        "affected_datasets": affected_datasets,
    }


def _prior_wandb_summary_payload(
    *,
    output_dir: Path,
    global_step: int,
    telemetry_payload: dict[str, Any],
) -> dict[str, Any]:
    raw_gradient_summary = telemetry_payload.get("gradient_summary")
    gradient_global = (
        raw_gradient_summary.get("global")
        if isinstance(raw_gradient_summary, dict)
        else None
    )
    raw_missingness = telemetry_payload.get("missingness")
    prior_dump_missingness = (
        raw_missingness.get("prior_dump")
        if isinstance(raw_missingness, dict)
        else None
    )
    synthetic_prior_missingness = (
        raw_missingness.get("synthetic_prior")
        if isinstance(raw_missingness, dict)
        else None
    )
    affected_indices = (
        prior_dump_missingness.get("affected_dataset_indices")
        if isinstance(prior_dump_missingness, dict)
        else None
    )
    raw_artifacts = telemetry_payload.get("artifacts")
    latest_checkpoint = (
        raw_artifacts.get("latest_checkpoint")
        if isinstance(raw_artifacts, dict)
        else None
    )
    summary: dict[str, Any] = {
        "run": {
            "output_dir": str(output_dir),
            "global_step": int(global_step),
        },
        "telemetry": {
            "success": telemetry_payload.get("success"),
            "checkpoint_snapshot_count": len(telemetry_payload.get("checkpoint_snapshots", [])),
        },
        "artifacts": {
            "latest_checkpoint": latest_checkpoint,
        },
        "loss_summary": telemetry_payload.get("loss_summary"),
        "gradient_summary": {
            "global": gradient_global,
        },
        "diagnostics": telemetry_payload.get("diagnostics"),
    }
    if isinstance(prior_dump_missingness, dict) or isinstance(synthetic_prior_missingness, dict):
        summary["missingness"] = {}
        if isinstance(prior_dump_missingness, dict):
            summary["missingness"]["prior_dump"] = {
                "batches_seen": prior_dump_missingness.get("batches_seen"),
                "non_finite_policy": prior_dump_missingness.get("non_finite_policy"),
                "affected_batch_count": prior_dump_missingness.get("affected_batch_count"),
                "affected_dataset_count": len(affected_indices)
                if isinstance(affected_indices, list)
                else None,
                "non_finite_feature_count": prior_dump_missingness.get("non_finite_feature_count"),
                "non_finite_label_count": prior_dump_missingness.get("non_finite_label_count"),
                "skipped_batch_count": prior_dump_missingness.get("skipped_batch_count"),
                "skipped_dataset_count": len(prior_dump_missingness.get("skipped_dataset_indices", []))
                if isinstance(prior_dump_missingness.get("skipped_dataset_indices"), list)
                else None,
            }
        if isinstance(synthetic_prior_missingness, dict):
            summary["missingness"]["synthetic_prior"] = {
                "enabled": synthetic_prior_missingness.get("enabled"),
                "batches_seen": synthetic_prior_missingness.get("batches_seen"),
                "affected_batch_count": synthetic_prior_missingness.get("affected_batch_count"),
                "masked_feature_count": synthetic_prior_missingness.get("masked_feature_count"),
            }
    raw_error = telemetry_payload.get("error")
    if isinstance(raw_error, dict):
        summary["error"] = {
            "type": raw_error.get("type"),
            "message": raw_error.get("message"),
        }
    return summary


def _merge_activation_norms(
    weighted_sums: dict[str, float],
    weight_totals: dict[str, float],
) -> dict[str, float] | None:
    if not weighted_sums:
        return None
    merged: dict[str, float] = {}
    for name, value in weighted_sums.items():
        weight = float(weight_totals.get(name, 0.0))
        if weight <= 0.0:
            continue
        merged[name] = float(value / weight)
    return merged or None


def _run_prior_step_with_microbatch_retry(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    x_batch: torch.Tensor,
    y_train_batch: torch.Tensor,
    y_all_batch: torch.Tensor,
    train_test_split_index: int,
    trace_activations: bool,
    flush_activation_trace: object,
) -> tuple[float, float, dict[str, float] | None, int, int]:
    forward_batched = getattr(model, "forward_batched", None)
    if not callable(forward_batched):
        raise RuntimeError("prior-dump training requires a model with forward_batched()")

    total_batch_size = int(x_batch.shape[0])
    if total_batch_size <= 0:
        raise RuntimeError("prior-dump training requires batch_size >= 1")

    def _attempt(microbatch_size: int) -> tuple[float, float, dict[str, float] | None, int]:
        weighted_loss = 0.0
        weighted_acc = 0.0
        activation_weighted_sums: dict[str, float] = {}
        activation_weight_totals: dict[str, float] = {}
        microbatch_count = 0
        for start in range(0, total_batch_size, microbatch_size):
            stop = min(start + microbatch_size, total_batch_size)
            microbatch_count += 1
            weight = float(stop - start) / float(total_batch_size)
            logits = forward_batched(
                x_all=x_batch[start:stop],
                y_train=y_train_batch[start:stop],
                train_test_split_index=train_test_split_index,
            )
            if not isinstance(logits, torch.Tensor):
                raise RuntimeError("prior-dump training requires tensor logits")
            activation_norms = (
                flush_activation_trace()
                if trace_activations and callable(flush_activation_trace)
                else None
            )
            targets = y_all_batch[start:stop, train_test_split_index:].reshape(-1).to(torch.int64)
            loss = classification_loss(
                logits.reshape(-1, int(logits.shape[-1])),
                targets,
            )
            (loss * weight).backward()
            weighted_loss += float(loss.detach().item()) * weight
            weighted_acc += (
                float(
                    (
                        logits.argmax(dim=-1)
                        == y_all_batch[start:stop, train_test_split_index:].to(torch.int64)
                    )
                    .float()
                    .mean()
                    .item()
                )
                * weight
            )
            if activation_norms is not None:
                for activation_name, activation_value in activation_norms.items():
                    activation_weighted_sums[activation_name] = (
                        activation_weighted_sums.get(activation_name, 0.0)
                        + (float(activation_value) * weight)
                    )
                    activation_weight_totals[activation_name] = (
                        activation_weight_totals.get(activation_name, 0.0) + weight
                    )
        return (
            weighted_loss,
            weighted_acc,
            _merge_activation_norms(activation_weighted_sums, activation_weight_totals),
            microbatch_count,
        )

    microbatch_size = total_batch_size
    while True:
        optimizer.zero_grad(set_to_none=True)
        try:
            loss_value, acc_value, activation_norms, microbatch_count = _attempt(microbatch_size)
            return (
                loss_value,
                acc_value,
                activation_norms,
                microbatch_size,
                microbatch_count,
            )
        except torch.OutOfMemoryError:
            if callable(flush_activation_trace):
                _ = flush_activation_trace()
            optimizer.zero_grad(set_to_none=True)
            if x_batch.device.type == "cuda":
                torch.cuda.empty_cache()
            if microbatch_size <= 1:
                raise
            next_microbatch_size = max(1, microbatch_size // 2)
            if next_microbatch_size == microbatch_size:
                next_microbatch_size = microbatch_size - 1
            print(
                "Warning: prior-dump step hit OOM; "
                f"retrying with microbatch_size={next_microbatch_size} "
                f"(effective_batch_size={total_batch_size})",
                file=sys.stderr,
                flush=True,
            )
            microbatch_size = next_microbatch_size


def train_tabfoundry_simple_prior(
    cfg: DictConfig,
    *,
    prior_dump_path: Path = DEFAULT_PRIOR_DUMP_PATH,
    batch_size: int | None = None,
) -> TrainResult:
    """Train an exact-parity staged/simple classifier on the nanoTabPFN prior dump."""

    if str(cfg.task).strip().lower() != "classification":
        raise ValueError(f"prior-dump training requires task='classification', got {cfg.task!r}")

    spec = _model_spec_from_cfg(cfg)
    staged_surface = _validate_prior_training_model_spec(spec)
    if str(cfg.runtime.mixed_precision).strip().lower() != "no":
        raise ValueError(
            "exact-parity prior-dump training requires runtime.mixed_precision='no', "
            f"got {cfg.runtime.mixed_precision!r}"
        )

    seed = int(cfg.runtime.seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    output_dir = Path(str(cfg.runtime.output_dir)).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_path_from_cfg(cfg)
    assert_clean_training_output(output_dir, history_path=history_path)
    gradient_path = gradient_history_path(output_dir)
    telemetry_output_path = telemetry_path(output_dir)

    max_steps = _resolve_positive_int(getattr(cfg.runtime, "max_steps", None), name="runtime.max_steps")
    eval_every = _resolve_positive_int(getattr(cfg.runtime, "eval_every", None), name="runtime.eval_every")
    checkpoint_every = _resolve_positive_int(
        getattr(cfg.runtime, "checkpoint_every", None),
        name="runtime.checkpoint_every",
    )
    grad_clip = float(cfg.runtime.grad_clip)
    trace_activations = _resolve_runtime_bool(
        getattr(cfg.runtime, "trace_activations", False),
        name="runtime.trace_activations",
    )
    if grad_clip <= 0:
        raise ValueError(f"runtime.grad_clip must be > 0 for prior-dump training, got {grad_clip}")

    prior_batch_config = _resolve_prior_dump_batch_config(cfg, batch_size_override=batch_size)
    lr_min = _resolve_lr(cfg) * prior_batch_config.effective_lr_scale_factor
    device = torch.device(
        _resolve_prior_training_device_name(
            cfg,
            spec=spec,
            staged_surface=staged_surface,
        )
    )
    model = build_model_from_spec(spec)
    enable_activation_trace = getattr(model, "enable_activation_trace", None)
    flush_activation_trace = getattr(model, "flush_activation_trace", None)
    if trace_activations and callable(enable_activation_trace):
        enable_activation_trace()
    model.to(device)
    model.train()
    cfg.logging.run_name = _resolve_prior_wandb_run_name(cfg)
    raw_cfg = cast(dict[str, object], OmegaConf.to_container(cfg, resolve=True))
    raw_training_cfg = raw_cfg.get("training")
    if not isinstance(raw_training_cfg, dict):
        raw_training_cfg = {}
        raw_cfg["training"] = raw_training_cfg
    raw_training_cfg["prior_dump_batch_size"] = int(prior_batch_config.batch_size)
    raw_training_cfg["prior_dump_lr_scale_rule"] = str(prior_batch_config.lr_scale_rule)
    raw_training_cfg["prior_dump_batch_reference_size"] = int(
        prior_batch_config.reference_batch_size
    )
    raw_training_cfg["effective_lr_scale_factor"] = float(
        prior_batch_config.effective_lr_scale_factor
    )
    raw_optimizer_cfg = raw_cfg.get("optimizer")
    if isinstance(raw_optimizer_cfg, dict) and raw_optimizer_cfg.get("min_lr") is not None:
        raw_optimizer_cfg["min_lr"] = float(lr_min)
    raw_schedule_cfg = raw_cfg.get("schedule")
    if isinstance(raw_schedule_cfg, dict):
        raw_stages = raw_schedule_cfg.get("stages")
        if isinstance(raw_stages, list):
            for stage in raw_stages:
                if isinstance(stage, dict) and stage.get("lr_max") is not None:
                    stage["lr_max"] = float(stage["lr_max"]) * float(
                        prior_batch_config.effective_lr_scale_factor
                    )
    training_surface_path = output_dir / "training_surface_record.json"
    training_surface_payload = write_training_surface_record(
        training_surface_path,
        raw_cfg=raw_cfg,
        run_dir=output_dir,
    )
    run = init_wandb_run(
        cfg,
        enabled=bool(getattr(cfg.logging, "use_wandb", False)),
    )
    artifacts: dict[str, Any] = {
        "train_history_jsonl": None if history_path is None else str(history_path),
        "gradient_history_jsonl": str(gradient_path),
        "telemetry_json": str(telemetry_output_path),
        "training_surface_record_json": str(training_surface_path.resolve()),
        "checkpoints_dir": str((output_dir / "checkpoints").resolve()),
        "latest_checkpoint": None,
    }
    history_records: list[dict[str, Any]] = []
    gradient_records: list[dict[str, Any]] = []
    checkpoint_snapshots: list[dict[str, Any]] = []
    prior_missingness_config = _resolve_prior_missingness_config(cfg)
    prior_dump_non_finite_policy = _resolve_prior_dump_non_finite_policy(cfg)
    missingness_summary = _initial_missingness_summary(
        prior_dump_path,
        prior_missingness_config=prior_missingness_config,
        prior_dump_non_finite_policy=prior_dump_non_finite_policy,
    )

    prior_stage = _resolve_prior_schedule(
        cfg,
        max_steps=max_steps,
        lr_min=lr_min,
        lr_scale_factor=prior_batch_config.effective_lr_scale_factor,
    )
    initial_lr = (
        float(stage_base_lr(prior_stage, step=1, lr_min=lr_min))
        if prior_stage is not None
        else float(lr_min)
    )

    optimizer_selection = build_optimizer(
        model,
        name=str(cfg.optimizer.name),
        lr=initial_lr,
        weight_decay=float(cfg.optimizer.weight_decay),
        extra_kwargs=_optimizer_kwargs(cfg),
        require_requested=bool(cfg.optimizer.require_requested),
        muon_per_parameter_lr=bool(getattr(cfg.optimizer, "muon_per_parameter_lr", True)),
        muon_lr_scale_base=float(getattr(cfg.optimizer, "muon_lr_scale_base", 0.2)),
        muon_partition_non2d=bool(getattr(cfg.optimizer, "muon_partition_non2d", True)),
    )
    if optimizer_selection.resolved_name not in {"schedulefree_adamw", "adamw"}:
        raise RuntimeError(
            "exact-parity prior-dump training requires optimizer 'schedulefree_adamw' or 'adamw', "
            f"resolved {optimizer_selection.resolved_name!r}"
        )
    if len(optimizer_selection.optimizers) != 1:
        raise RuntimeError(
            "exact-parity prior-dump training expects exactly one optimizer instance, "
            f"got {len(optimizer_selection.optimizers)}"
        )
    prepared_opts = list(optimizer_selection.optimizers)
    optimizer = prepared_opts[0][1]
    _set_optimizer_training_mode(prepared_opts, training=True)
    lr_scales = [1.0 for _ in optimizer.param_groups]
    _set_optimizer_base_lr(
        optimizer,
        base_lr=initial_lr,
        scales=lr_scales,
    )

    history_step_loss = 0.0
    history_step_acc = 0.0
    global_step = 0
    latest_checkpoint: Path | None = None
    final_grad_norm = 0.0
    grad_norm_sum = 0.0
    grad_norm_count = 0
    max_grad_norm = 0.0
    clipped_step_count = 0
    train_start = time.perf_counter()
    train_elapsed_seconds = 0.0
    previous_train_loss: float | None = None
    loss_ema: float | None = None
    prior_missingness_generator = None
    if prior_missingness_config is not None:
        prior_missingness_generator = torch.Generator(device="cpu")
        prior_missingness_generator.manual_seed(seed)

    def _record_non_finite_batch(batch_missingness: PriorDumpBatchMissingness) -> None:
        _accumulate_missingness(
            missingness_summary,
            batch_missingness=batch_missingness,
            skipped=prior_dump_non_finite_policy == "skip",
        )

    reader = PriorDumpTaskBatchReader(
        prior_dump_path,
        num_steps=max_steps,
        batch_size=prior_batch_config.batch_size,
        non_finite_policy=prior_dump_non_finite_policy,
        on_non_finite_batch=_record_non_finite_batch,
    )

    try:
        for prior_step in reader:
            if prior_step.missingness is not None:
                _accumulate_missingness(
                    missingness_summary,
                    batch_missingness=prior_step.missingness,
                )
            optimizer.zero_grad(set_to_none=True)
            step_train_start = time.perf_counter()
            if prior_stage is not None:
                current_base_lr = float(
                    stage_base_lr(prior_stage, step=int(prior_step.step_index), lr_min=lr_min)
                )
                _set_optimizer_base_lr(
                    optimizer,
                    base_lr=current_base_lr,
                    scales=lr_scales,
                )
            x_batch, y_train_batch, y_all_batch = _stack_prior_step(prior_step, device=device)
            x_batch, synthetic_missingness = _apply_prior_missingness(
                x_batch,
                prior_step=prior_step,
                generator=prior_missingness_generator,
                prior_missingness_config=prior_missingness_config,
            )
            _accumulate_synthetic_missingness(
                missingness_summary,
                batch_missingness=synthetic_missingness,
            )
            (
                history_step_loss,
                history_step_acc,
                activation_norms,
                microbatch_size_used,
                microbatch_count,
            ) = _run_prior_step_with_microbatch_retry(
                model=model,
                optimizer=optimizer,
                x_batch=x_batch,
                y_train_batch=y_train_batch,
                y_all_batch=y_all_batch,
                train_test_split_index=prior_step.train_test_split_index,
                trace_activations=trace_activations,
                flush_activation_trace=flush_activation_trace,
            )

            pre_clip_module_grad_norms = module_grad_norms(model)
            local_grad_norm = total_grad_norm(model.parameters())
            clipped = torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            local_grad_norm = normalize_grad_norm_value(clipped, fallback=local_grad_norm)
            grad_clip_triggered = bool(local_grad_norm > grad_clip)
            if grad_clip_triggered:
                clipped_step_count += 1

            optimizer.step()
            step_train_duration = time.perf_counter() - step_train_start
            train_elapsed_seconds += step_train_duration
            global_step = int(prior_step.step_index)

            final_grad_norm = float(local_grad_norm)
            grad_norm_sum += final_grad_norm
            grad_norm_count += 1
            max_grad_norm = max(max_grad_norm, final_grad_norm)

            loss_delta_value = train_loss_delta(
                history_step_loss,
                previous_train_loss=previous_train_loss,
            )
            loss_ema = update_loss_ema(history_step_loss, previous_ema=loss_ema)
            previous_train_loss = history_step_loss
            elapsed_seconds = time.perf_counter() - train_start
            prior_dump_missingness = cast(dict[str, Any], missingness_summary["prior_dump"])
            synthetic_prior_missingness = cast(dict[str, Any], missingness_summary["synthetic_prior"])
            train_log: dict[str, Any] = {
                "train/loss": float(history_step_loss),
                "train/acc": float(history_step_acc),
                "train/lr": float(optimizer.param_groups[0]["lr"]),
                "train/grad_norm": float(final_grad_norm),
                "train/loss_delta": loss_delta_value,
                "train/loss_ema": loss_ema,
                "train/elapsed_seconds": float(elapsed_seconds),
                "train/train_elapsed_seconds": float(train_elapsed_seconds),
                "train/grad_clip_threshold": float(grad_clip),
                "train/grad_clip_triggered": grad_clip_triggered,
                "train/grad_clip_count_so_far": int(clipped_step_count),
                "train/grad_clip_fraction_so_far": float(clipped_step_count / global_step),
                "train/prior_dump_skipped_batch_count": int(prior_dump_missingness["skipped_batch_count"]),
                "train/prior_dump_non_finite_feature_count": int(prior_dump_missingness["non_finite_feature_count"]),
                "train/prior_dump_non_finite_label_count": int(prior_dump_missingness["non_finite_label_count"]),
                "train/synthetic_prior_masked_feature_count": int(synthetic_prior_missingness["masked_feature_count"]),
                "train/prior_dump_microbatch_size": int(microbatch_size_used),
                "train/prior_dump_microbatch_count": int(microbatch_count),
                "train/stage": _PRIOR_STAGE_NAME,
            }
            for module_name, module_value in pre_clip_module_grad_norms.items():
                train_log[f"train/module_grad_norm/{module_name}"] = float(module_value)
            feature_grad = pre_clip_module_grad_norms.get("feature_encoder")
            head_grad = pre_clip_module_grad_norms.get("direct_head")
            if feature_grad is not None and float(feature_grad) > 0.0 and head_grad is not None:
                train_log["train/module_balance/direct_head_to_feature_encoder"] = float(head_grad) / float(feature_grad)
            if head_grad is not None and float(head_grad) > 0.0 and feature_grad is not None:
                train_log["train/module_balance/feature_encoder_to_direct_head"] = float(feature_grad) / float(head_grad)
            if activation_norms is not None:
                for activation_name, activation_value in activation_norms.items():
                    train_log[f"train/activation_norm/{activation_name}"] = float(activation_value)
            log_wandb_metrics(run, train_log, step=global_step)
            history_payload = history_record(
                global_step=global_step,
                stage_name=_PRIOR_STAGE_NAME,
                train_loss=history_step_loss,
                train_metrics={"acc": history_step_acc},
                lr=float(optimizer.param_groups[0]["lr"]),
                grad_norm=final_grad_norm,
                elapsed_seconds=elapsed_seconds,
                train_elapsed_seconds=train_elapsed_seconds,
                val_metrics=None,
                train_loss_delta=loss_delta_value,
                train_loss_ema=loss_ema,
                grad_clip_threshold=float(grad_clip),
                grad_clip_triggered=grad_clip_triggered,
            )
            history_records.append(history_payload)
            if history_path is not None:
                append_history_record(history_path, history_payload)

            gradient_payload = gradient_history_record(
                global_step=global_step,
                stage_name=_PRIOR_STAGE_NAME,
                train_loss=history_step_loss,
                train_acc=history_step_acc,
                lr=float(optimizer.param_groups[0]["lr"]),
                global_grad_norm=final_grad_norm,
                module_grad_norms=pre_clip_module_grad_norms,
                activation_norms=activation_norms,
                elapsed_seconds=elapsed_seconds,
                train_elapsed_seconds=train_elapsed_seconds,
                grad_clip_threshold=float(grad_clip),
                grad_clip_triggered=grad_clip_triggered,
            )
            gradient_records.append(gradient_payload)
            append_jsonl_record(gradient_path, gradient_payload)

            if global_step % checkpoint_every == 0:
                checkpoint_path = output_dir / "checkpoints" / f"step_{global_step:06d}.pt"
                _save_eval_mode_checkpoint(
                    prepared_opts,
                    path=checkpoint_path,
                    model=model,
                    global_step=global_step,
                    cfg=cfg,
                    restore_training=True,
                )
                checkpoint_snapshots.append(
                    {
                        "step": int(global_step),
                        "path": str(checkpoint_path.resolve()),
                        "elapsed_seconds": float(train_elapsed_seconds),
                        "train_elapsed_seconds": float(train_elapsed_seconds),
                    }
                )
            if global_step % eval_every == 0:
                print(
                    f"time {train_elapsed_seconds:7.1f}s | "
                    f"step {global_step:4d} | "
                    f"loss {history_step_loss:7.4f} | "
                    f"acc {history_step_acc:7.4f}"
                )

        latest_checkpoint = output_dir / "checkpoints" / "latest.pt"
        _save_eval_mode_checkpoint(
            prepared_opts,
            path=latest_checkpoint,
            model=model,
            global_step=global_step,
            cfg=cfg,
            restore_training=False,
        )
        artifacts["latest_checkpoint"] = str(latest_checkpoint.resolve())
        wall_elapsed_seconds = time.perf_counter() - train_start
        telemetry_payload = build_training_telemetry(
            run_dir=output_dir,
            success=True,
            artifacts=artifacts,
            checkpoint_snapshots=checkpoint_snapshots,
            history_records=history_records,
            gradient_records=gradient_records,
            missingness=missingness_summary,
            training_surface_record=training_surface_payload,
        )
        write_training_telemetry(telemetry_output_path, telemetry_payload)
        update_wandb_summary(
            run,
            _prior_wandb_summary_payload(
                output_dir=output_dir,
                global_step=global_step,
                telemetry_payload=telemetry_payload,
            ),
        )
        return TrainResult(
            output_dir=output_dir,
            best_checkpoint=None,
            latest_checkpoint=latest_checkpoint,
            global_step=global_step,
            metrics={
                "final_train_loss": float(history_step_loss),
                "final_train_acc": float(history_step_acc),
                "final_grad_norm": float(final_grad_norm),
                "mean_grad_norm": float(grad_norm_sum / grad_norm_count) if grad_norm_count > 0 else 0.0,
                "max_grad_norm": float(max_grad_norm),
                "train_elapsed_seconds": float(train_elapsed_seconds),
                "wall_elapsed_seconds": float(wall_elapsed_seconds),
            },
        )
    except Exception as exc:
        telemetry_payload = build_training_telemetry(
            run_dir=output_dir,
            success=False,
            artifacts=artifacts,
            checkpoint_snapshots=checkpoint_snapshots,
            history_records=history_records,
            gradient_records=gradient_records,
            missingness=missingness_summary,
            training_surface_record=training_surface_payload,
            error=exc,
        )
        write_training_telemetry(telemetry_output_path, telemetry_payload)
        update_wandb_summary(
            run,
            _prior_wandb_summary_payload(
                output_dir=output_dir,
                global_step=global_step,
                telemetry_payload=telemetry_payload,
            ),
        )
        raise
    finally:
        finish_wandb_run(run)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train an exact-parity tabfoundry classifier on the nanoTabPFN prior dump"
    )
    parser.add_argument(
        "--prior-dump",
        default=str(DEFAULT_PRIOR_DUMP_PATH),
        help="Path to the nanoTabPFN prior dump (.h5)",
    )
    parser.add_argument("overrides", nargs="*", help="Optional Hydra override strings")
    return parser


def _compose_prior_cfg(overrides: Sequence[str]) -> DictConfig:
    resolved_overrides = list(overrides)
    if not any(str(override).startswith("experiment=") for override in resolved_overrides):
        resolved_overrides.insert(0, f"experiment={DEFAULT_EXPERIMENT}")
    return compose_config(resolved_overrides)


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    cfg = _compose_prior_cfg(args.overrides)
    result = train_tabfoundry_simple_prior(
        cfg,
        prior_dump_path=Path(str(args.prior_dump)),
    )
    print(
        "Training complete:",
        f"output_dir={result.output_dir}",
        f"latest={result.latest_checkpoint}",
        f"step={result.global_step}",
        f"metrics={result.metrics}",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
