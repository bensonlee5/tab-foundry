"""Shared config inspection helpers used by developer-facing CLI surfaces."""

from __future__ import annotations

from typing import Any, Mapping, Sequence, cast

from omegaconf import OmegaConf

from tab_foundry.config import compose_config, config_dir
from tab_foundry.data.surface import resolve_data_surface
from tab_foundry.model.inspection import model_surface_payload, parameter_counts_from_model_spec
from tab_foundry.model.spec import model_build_spec_from_mappings
from tab_foundry.preprocessing import resolve_preprocessing_surface
from tab_foundry.training.prior.settings import resolve_prior_backend_surface_config
from tab_foundry.training.surface import resolve_training_backend_from_data_cfg

_MODEL_ARCH_PREFIXES = ("model.arch=", "+model.arch=", "++model.arch=")
_STAGED_COMPAT_PREFIXES = (
    "model.stage=",
    "+model.stage=",
    "++model.stage=",
    "model.stage_label=",
    "+model.stage_label=",
    "++model.stage_label=",
    "model.module_overrides",
    "+model.module_overrides",
    "++model.module_overrides",
)


def mapping_from_node(value: Any, *, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    payload = OmegaConf.to_container(value, resolve=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} must resolve to a mapping")
    return {str(key): item for key, item in payload.items()}


def _project_default_experiment() -> str | None:
    root_payload = OmegaConf.to_container(OmegaConf.load(config_dir() / "config.yaml"), resolve=True)
    if not isinstance(root_payload, dict):
        return None
    raw_defaults = root_payload.get("defaults")
    if not isinstance(raw_defaults, list):
        return None
    for entry in raw_defaults:
        if not isinstance(entry, Mapping):
            continue
        raw_experiment = entry.get("experiment")
        if isinstance(raw_experiment, str) and raw_experiment.strip():
            return str(raw_experiment).strip()
    return None


def resolve_experiment_name(overrides: Sequence[str]) -> str | None:
    for override in reversed(list(overrides)):
        token = str(override).strip()
        if not token.startswith("experiment="):
            continue
        value = token.split("=", 1)[1].strip()
        return value or None
    return _project_default_experiment()


def normalized_dev_overrides(overrides: Sequence[str]) -> list[str]:
    normalized = [str(override) for override in overrides]
    if any(token.strip().startswith(_MODEL_ARCH_PREFIXES) for token in normalized):
        return normalized
    if any(token.strip().startswith(_STAGED_COMPAT_PREFIXES) for token in normalized):
        return [*normalized, "model.arch=tabfoundry_staged"]
    return normalized


def _training_surface_payload(
    training_cfg: Mapping[str, Any],
    *,
    legacy_prior_cfg: Mapping[str, Any],
    backend: str | None,
    optimizer_cfg: Mapping[str, Any],
    schedule_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    raw_stages = schedule_cfg.get("stages")
    rendered_stages: list[dict[str, Any]] = []
    if isinstance(raw_stages, list):
        for item in raw_stages:
            if not isinstance(item, Mapping):
                continue
            rendered_stages.append(
                {
                    "name": item.get("name"),
                    "steps": None if item.get("steps") is None else int(item["steps"]),
                    "lr_max": None if item.get("lr_max") is None else float(item["lr_max"]),
                    "warmup_ratio": (
                        None if item.get("warmup_ratio") is None else float(item["warmup_ratio"])
                    ),
                    "lr_schedule": None if item.get("lr_schedule") is None else str(item["lr_schedule"]),
                }
            )
    payload = {
        "surface_label": str(training_cfg.get("surface_label", "training_default")),
        "apply_schedule": bool(training_cfg.get("apply_schedule", False)),
        "task_batch_size": int(training_cfg.get("task_batch_size", 1)),
        "overrides": dict(cast(dict[str, Any], training_cfg.get("overrides", {}))),
        "optimizer_name": None if optimizer_cfg.get("name") is None else str(optimizer_cfg["name"]),
        "optimizer_min_lr": None
        if optimizer_cfg.get("min_lr") is None
        else float(optimizer_cfg["min_lr"]),
        "schedule_stages": rendered_stages,
    }
    if backend is not None:
        payload["backend"] = backend
    if backend == "legacy_prior":
        payload["legacy_prior"] = resolve_prior_backend_surface_config(
            training_cfg=training_cfg,
            legacy_prior_cfg=legacy_prior_cfg,
        ).to_dict()
    return payload


def _inspection_training_backend(data_cfg: Mapping[str, Any]) -> str | None:
    try:
        return resolve_training_backend_from_data_cfg(
            data_cfg,
            allow_unresolved_corpus_ref=True,
        )
    except RuntimeError:
        return None


def resolve_config_payload(overrides: Sequence[str]) -> dict[str, Any]:
    normalized_overrides = normalized_dev_overrides(overrides)
    cfg = compose_config(normalized_overrides)
    task = str(getattr(cfg, "task", "classification")).strip().lower()
    model_cfg = mapping_from_node(getattr(cfg, "model", None), context="cfg.model")
    spec = model_build_spec_from_mappings(task=task, primary=model_cfg)
    data_cfg = mapping_from_node(getattr(cfg, "data", None), context="cfg.data")
    data_surface = resolve_data_surface(
        data_cfg,
        allow_unresolved_corpus_ref=True,
    )
    preprocessing_surface = resolve_preprocessing_surface(
        mapping_from_node(getattr(cfg, "preprocessing", None), context="cfg.preprocessing")
    )
    legacy_prior_cfg = mapping_from_node(getattr(cfg, "legacy_prior", None), context="cfg.legacy_prior")
    backend = _inspection_training_backend(data_cfg)
    training_payload = _training_surface_payload(
        mapping_from_node(getattr(cfg, "training", None), context="cfg.training"),
        legacy_prior_cfg=legacy_prior_cfg,
        backend=backend,
        optimizer_cfg=mapping_from_node(getattr(cfg, "optimizer", None), context="cfg.optimizer"),
        schedule_cfg=mapping_from_node(getattr(cfg, "schedule", None), context="cfg.schedule"),
    )
    runtime_cfg = mapping_from_node(getattr(cfg, "runtime", None), context="cfg.runtime")
    return {
        "experiment": resolve_experiment_name(overrides),
        "task": task,
        "model": {
            **model_surface_payload(spec),
            "parameter_counts": parameter_counts_from_model_spec(spec),
        },
        "data": data_surface.to_dict(),
        "preprocessing": preprocessing_surface.to_dict(),
        "training": training_payload,
        "runtime": {
            "device": runtime_cfg.get("device"),
            "output_dir": runtime_cfg.get("output_dir"),
            "seed": runtime_cfg.get("seed"),
        },
    }


__all__ = [
    "mapping_from_node",
    "normalized_dev_overrides",
    "resolve_config_payload",
    "resolve_experiment_name",
]
