"""Shared prior-backend settings models and resolution helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

from omegaconf import OmegaConf
from pydantic import BaseModel, ConfigDict, Field, ValidationError, ValidationInfo, field_validator, model_validator

from tab_foundry.training.prior_dump import PriorDumpNonFinitePolicy


DEFAULT_BATCH_SIZE = 32
SUPPORTED_LR_SCALE_RULES = ("none", "sqrt", "linear")


def _resolve_config_mapping(value: object, *, name: str) -> dict[str, Any]:
    if value is None:
        return {}
    resolved = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    if resolved is None:
        return {}
    if not isinstance(resolved, Mapping):
        raise ValueError(f"{name} must resolve to a mapping, got {resolved!r}")
    return {str(key): item for key, item in resolved.items()}


def _coerce_positive_int(value: Any, *, name: str) -> int:
    if not isinstance(value, (int, float, str)):
        raise ValueError(f"{name} must be int-compatible, got {value!r}")
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be >= 1, got {resolved}")
    return resolved


def _coerce_runtime_bool(value: Any, *, name: str) -> bool:
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


def _coerce_probability(value: Any, *, name: str) -> float:
    resolved = float(value)
    if not 0.0 <= resolved <= 1.0:
        raise ValueError(f"{name} must be in [0, 1], got {resolved}")
    return resolved


class PriorBackendSurfaceConfig(BaseModel):
    """Backend-scoped prior settings used by exact-prior training and inspection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    non_finite_policy: Literal["error", "skip"] = "error"
    batch_size: int | None = Field(default=None, gt=0)
    lr_scale_rule: Literal["none", "sqrt", "linear"] | None = None
    batch_reference_size: int | None = Field(default=None, gt=0)
    effective_lr_scale_factor: float | None = Field(default=None, gt=0.0)

    @field_validator("batch_size", "batch_reference_size", mode="before")
    @classmethod
    def _validate_optional_positive_ints(cls, value: Any, info: ValidationInfo) -> int | None:
        if value is None:
            return None
        assert info.field_name is not None
        return _coerce_positive_int(value, name=str(info.field_name))

    @field_validator("effective_lr_scale_factor", mode="before")
    @classmethod
    def _validate_optional_positive_float(cls, value: Any) -> float | None:
        if value is None:
            return None
        resolved = float(value)
        if resolved <= 0.0:
            raise ValueError(f"effective_lr_scale_factor must be > 0, got {resolved}")
        return resolved

    def to_dict(self) -> dict[str, Any]:
        return {
            "non_finite_policy": str(self.non_finite_policy),
            "batch_size": None if self.batch_size is None else int(self.batch_size),
            "lr_scale_rule": None if self.lr_scale_rule is None else str(self.lr_scale_rule),
            "batch_reference_size": (
                None if self.batch_reference_size is None else int(self.batch_reference_size)
            ),
            "effective_lr_scale_factor": (
                None
                if self.effective_lr_scale_factor is None
                else float(self.effective_lr_scale_factor)
            ),
        }


class PriorRuntimeConfig(BaseModel):
    """Typed runtime fields required by exact-prior training."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    max_steps: int
    eval_every: int
    checkpoint_every: int
    trace_activations: bool = False

    @field_validator("max_steps", "eval_every", "checkpoint_every", mode="before")
    @classmethod
    def _validate_positive_ints(cls, value: Any, info: ValidationInfo) -> int:
        assert info.field_name is not None
        return _coerce_positive_int(value, name=f"runtime.{info.field_name}")

    @field_validator("trace_activations", mode="before")
    @classmethod
    def _validate_trace_activations(cls, value: Any) -> bool:
        return _coerce_runtime_bool(value, name="runtime.trace_activations")

    @classmethod
    def from_runtime_cfg(cls, runtime_cfg: object) -> PriorRuntimeConfig:
        payload = _resolve_config_mapping(runtime_cfg, name="runtime")
        try:
            return cls.model_validate(
                {
                    "max_steps": payload.get("max_steps"),
                    "eval_every": payload.get("eval_every"),
                    "checkpoint_every": payload.get("checkpoint_every"),
                    "trace_activations": payload.get("trace_activations", False),
                }
            )
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc


class PriorMissingnessConfig(BaseModel):
    """Typed synthetic-missingness settings for exact-prior training."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    enabled: bool = False
    min_rate: float = 0.0
    max_rate: float = 0.0

    @field_validator("enabled", mode="before")
    @classmethod
    def _validate_enabled(cls, value: Any) -> bool:
        return _coerce_runtime_bool(value, name="training.overrides.prior_missingness.enabled")

    @field_validator("min_rate", "max_rate", mode="before")
    @classmethod
    def _validate_rates(cls, value: Any, info: ValidationInfo) -> float:
        assert info.field_name is not None
        return _coerce_probability(
            value,
            name=f"training.overrides.prior_missingness.{info.field_name}",
        )

    @model_validator(mode="after")
    def _validate_rate_order(self) -> PriorMissingnessConfig:
        if self.min_rate > self.max_rate:
            raise ValueError(
                "training.overrides.prior_missingness.min_rate must be <= max_rate, "
                f"got min_rate={self.min_rate}, max_rate={self.max_rate}"
            )
        return self

    def to_runtime_dict(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "min_rate": float(self.min_rate),
            "max_rate": float(self.max_rate),
        }

    @classmethod
    def from_training_overrides(cls, overrides_cfg: object) -> PriorMissingnessConfig | None:
        overrides = _resolve_config_mapping(overrides_cfg, name="training.overrides")
        raw = overrides.get("prior_missingness")
        if raw is None:
            return None
        payload = _resolve_config_mapping(
            raw,
            name="training.overrides.prior_missingness",
        )
        try:
            resolved = cls.model_validate(
                {
                    "enabled": payload.get("enabled", False),
                    "min_rate": payload.get("min_rate", 0.0),
                    "max_rate": payload.get("max_rate", payload.get("min_rate", 0.0)),
                }
            )
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc
        return None if not resolved.enabled else resolved


def _pick(
    *,
    training_cfg: Mapping[str, Any] | None,
    legacy_prior_cfg: Mapping[str, Any] | None,
    flat_key: str,
    scoped_key: str,
) -> Any:
    if training_cfg is not None and training_cfg.get(flat_key) is not None:
        return training_cfg[flat_key]
    if legacy_prior_cfg is not None and legacy_prior_cfg.get(scoped_key) is not None:
        return legacy_prior_cfg[scoped_key]
    return None


def resolve_prior_backend_surface_config(
    *,
    training_cfg: Mapping[str, Any] | object | None = None,
    legacy_prior_cfg: Mapping[str, Any] | object | None = None,
) -> PriorBackendSurfaceConfig:
    resolved_training_cfg = _resolve_config_mapping(training_cfg, name="training")
    resolved_legacy_prior_cfg = _resolve_config_mapping(legacy_prior_cfg, name="legacy_prior")
    payload: dict[str, Any] = {}
    non_finite_policy = _pick(
        training_cfg=resolved_training_cfg,
        legacy_prior_cfg=resolved_legacy_prior_cfg,
        flat_key="prior_dump_non_finite_policy",
        scoped_key="non_finite_policy",
    )
    if non_finite_policy is not None:
        payload["non_finite_policy"] = non_finite_policy
    batch_size = _pick(
        training_cfg=resolved_training_cfg,
        legacy_prior_cfg=resolved_legacy_prior_cfg,
        flat_key="prior_dump_batch_size",
        scoped_key="batch_size",
    )
    if batch_size is not None:
        payload["batch_size"] = batch_size
    lr_scale_rule = _pick(
        training_cfg=resolved_training_cfg,
        legacy_prior_cfg=resolved_legacy_prior_cfg,
        flat_key="prior_dump_lr_scale_rule",
        scoped_key="lr_scale_rule",
    )
    if lr_scale_rule is not None:
        payload["lr_scale_rule"] = lr_scale_rule
    batch_reference_size = _pick(
        training_cfg=resolved_training_cfg,
        legacy_prior_cfg=resolved_legacy_prior_cfg,
        flat_key="prior_dump_batch_reference_size",
        scoped_key="batch_reference_size",
    )
    if batch_reference_size is not None:
        payload["batch_reference_size"] = batch_reference_size
    effective_lr_scale_factor = _pick(
        training_cfg=resolved_training_cfg,
        legacy_prior_cfg=resolved_legacy_prior_cfg,
        flat_key="effective_lr_scale_factor",
        scoped_key="effective_lr_scale_factor",
    )
    if effective_lr_scale_factor is not None:
        payload["effective_lr_scale_factor"] = effective_lr_scale_factor
    return PriorBackendSurfaceConfig.model_validate(payload)


def resolve_prior_dump_non_finite_policy(
    *,
    training_cfg: Mapping[str, Any] | None = None,
    legacy_prior_cfg: Mapping[str, Any] | None = None,
) -> PriorDumpNonFinitePolicy:
    resolved = resolve_prior_backend_surface_config(
        training_cfg=training_cfg,
        legacy_prior_cfg=legacy_prior_cfg,
    )
    return cast(PriorDumpNonFinitePolicy, resolved.non_finite_policy)
