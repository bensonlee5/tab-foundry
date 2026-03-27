"""Shared prior-backend settings models and resolution helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field

from tab_foundry.training.prior_dump import PriorDumpNonFinitePolicy


DEFAULT_BATCH_SIZE = 32
SUPPORTED_LR_SCALE_RULES = ("none", "sqrt", "linear")


class PriorBackendSurfaceConfig(BaseModel):
    """Backend-scoped prior settings used by exact-prior training and inspection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    non_finite_policy: Literal["error", "skip"] = "error"
    batch_size: int | None = Field(default=None, gt=0)
    lr_scale_rule: Literal["none", "sqrt", "linear"] | None = None
    batch_reference_size: int | None = Field(default=None, gt=0)
    effective_lr_scale_factor: float | None = Field(default=None, gt=0.0)

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
    training_cfg: Mapping[str, Any] | None = None,
    legacy_prior_cfg: Mapping[str, Any] | None = None,
) -> PriorBackendSurfaceConfig:
    payload: dict[str, Any] = {}
    non_finite_policy = _pick(
        training_cfg=training_cfg,
        legacy_prior_cfg=legacy_prior_cfg,
        flat_key="prior_dump_non_finite_policy",
        scoped_key="non_finite_policy",
    )
    if non_finite_policy is not None:
        payload["non_finite_policy"] = non_finite_policy
    batch_size = _pick(
        training_cfg=training_cfg,
        legacy_prior_cfg=legacy_prior_cfg,
        flat_key="prior_dump_batch_size",
        scoped_key="batch_size",
    )
    if batch_size is not None:
        payload["batch_size"] = batch_size
    lr_scale_rule = _pick(
        training_cfg=training_cfg,
        legacy_prior_cfg=legacy_prior_cfg,
        flat_key="prior_dump_lr_scale_rule",
        scoped_key="lr_scale_rule",
    )
    if lr_scale_rule is not None:
        payload["lr_scale_rule"] = lr_scale_rule
    batch_reference_size = _pick(
        training_cfg=training_cfg,
        legacy_prior_cfg=legacy_prior_cfg,
        flat_key="prior_dump_batch_reference_size",
        scoped_key="batch_reference_size",
    )
    if batch_reference_size is not None:
        payload["batch_reference_size"] = batch_reference_size
    effective_lr_scale_factor = _pick(
        training_cfg=training_cfg,
        legacy_prior_cfg=legacy_prior_cfg,
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
