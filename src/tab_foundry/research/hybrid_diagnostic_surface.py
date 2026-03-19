"""Shared helpers for the locked hybrid-diagnostic surface."""

from __future__ import annotations

from typing import Any, cast

from omegaconf import DictConfig, OmegaConf

from tab_foundry.config import compose_config
from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.model.spec import model_build_spec_from_mappings
from tab_foundry.research.lane_contract import HYBRID_DIAGNOSTIC_SURFACE


def hybrid_diagnostic_surface_cfg() -> DictConfig:
    """Resolve the current locked hybrid-diagnostic experiment config."""

    return compose_config([f"experiment={HYBRID_DIAGNOSTIC_SURFACE}"])


def hybrid_diagnostic_surface_payload() -> dict[str, Any]:
    """Return the locked hybrid-diagnostic surface payload."""

    cfg = hybrid_diagnostic_surface_cfg()
    module_overrides = cast(
        dict[str, Any],
        OmegaConf.to_container(cfg.model.module_overrides, resolve=True),
    )
    stage = cfg.schedule.stages[0]
    return {
        "task": str(cfg.task),
        "model": {
            "arch": str(cfg.model.arch),
            "stage": str(cfg.model.stage),
            "stage_label": str(cfg.model.stage_label),
            "input_normalization": str(cfg.model.input_normalization),
            "module_overrides": module_overrides,
            "tfrow_n_heads": int(cfg.model.tfrow_n_heads),
            "tfrow_n_layers": int(cfg.model.tfrow_n_layers),
            "tfrow_cls_tokens": int(cfg.model.tfrow_cls_tokens),
            "tfrow_norm": str(cfg.model.tfrow_norm),
            "d_icl": int(cfg.model.d_icl),
            "tficl_n_heads": int(cfg.model.tficl_n_heads),
            "tficl_n_layers": int(cfg.model.tficl_n_layers),
            "head_hidden_dim": int(cfg.model.head_hidden_dim),
            "many_class_base": int(cfg.model.many_class_base),
        },
        "training": {
            "surface_label": str(cfg.training.surface_label),
            "prior_dump_non_finite_policy": str(cfg.training.prior_dump_non_finite_policy),
            "prior_dump_batch_size": int(cfg.training.prior_dump_batch_size),
            "prior_dump_lr_scale_rule": str(cfg.training.prior_dump_lr_scale_rule),
            "prior_dump_batch_reference_size": int(cfg.training.prior_dump_batch_reference_size),
            "optimizer_min_lr": float(cfg.optimizer.min_lr),
            "runtime": {
                "grad_clip": float(cfg.runtime.grad_clip),
                "max_steps": int(cfg.runtime.max_steps),
                "trace_activations": bool(cfg.runtime.trace_activations),
            },
            "schedule_stage": {
                "name": str(stage["name"]),
                "steps": int(stage["steps"]),
                "lr_max": float(stage["lr_max"]),
                "lr_schedule": str(stage["lr_schedule"]),
                "warmup_ratio": float(stage["warmup_ratio"]),
            },
        },
        "logging": {
            "run_name": str(cfg.logging.run_name),
        },
    }


def build_hybrid_diagnostic_surface_model():
    """Instantiate the locked hybrid-diagnostic model from the canonical experiment config."""

    cfg = hybrid_diagnostic_surface_cfg()
    raw_model_cfg = OmegaConf.to_container(cfg.model, resolve=True)
    if not isinstance(raw_model_cfg, dict):
        raise RuntimeError("cfg.model must resolve to a mapping")
    spec = model_build_spec_from_mappings(
        task=str(cfg.task),
        primary={str(key): value for key, value in raw_model_cfg.items()},
    )
    return build_model_from_spec(spec)
