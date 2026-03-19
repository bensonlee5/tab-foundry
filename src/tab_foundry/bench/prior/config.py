"""Config and model-spec helpers for prior-dump training."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from tab_foundry.bench.prior_dump import PriorDumpNonFinitePolicy
from tab_foundry.model.architectures.tabfoundry_staged.resolved import (
    ResolvedStageSurface,
    resolve_staged_surface,
)
from tab_foundry.model.spec import ModelBuildSpec, model_build_spec_from_mappings
from tab_foundry.training.schedule import StageConfig, build_stage_configs


DEFAULT_BATCH_SIZE = 32
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
