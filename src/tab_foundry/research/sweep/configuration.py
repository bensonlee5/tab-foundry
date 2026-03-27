"""Configuration composition helpers for system-delta sweep execution."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Mapping, cast

from omegaconf import DictConfig, OmegaConf

from tab_foundry.config import compose_config
from tab_foundry.training.prior.settings import resolve_prior_backend_surface_config
from tab_foundry.training.surface import resolve_training_backend_from_data_cfg


def row_id_for_order(sweep_id: str, order: int, delta_ref: str, existing_run_id: str | None) -> str:
    base = f"sd_{sweep_id}_{order:02d}_{delta_ref}"
    if existing_run_id is None:
        return f"{base}_v1"
    match = re.fullmatch(rf"{re.escape(base)}_v(\d+)", existing_run_id)
    if match is None:
        return f"{base}_v1"
    return f"{base}_v{int(match.group(1)) + 1}"


def apply_mapping(cfg: DictConfig, prefix: str, payload: Mapping[str, Any]) -> None:
    if (
        prefix == "model"
        and payload.get("arch") is None
        and any(payload.get(key) is not None for key in ("stage", "stage_label", "module_overrides"))
    ):
        OmegaConf.update(cfg, "model.arch", "tabfoundry_staged", merge=False)
    for key, value in payload.items():
        if prefix == "data" and key == "corpus_ref":
            OmegaConf.update(cfg, f"{prefix}.surface_overrides.{key}", value, merge=True)
            continue
        merge = not (
            prefix == "model"
            and key == "module_overrides"
            and isinstance(value, Mapping)
        )
        OmegaConf.update(cfg, f"{prefix}.{key}", value, merge=merge)


def _queue_aware_run_name(*, run_dir: Path) -> str:
    return str(run_dir.parent.name if run_dir.name == "train" else run_dir.name)


def _corpus_lookup_context(
    *,
    sweep_id: str | None,
    sweeps_root: Path | None,
) -> dict[str, str]:
    if sweep_id is None:
        return {}
    payload = {
        "corpus_lookup_sweep_id": str(sweep_id),
    }
    if sweeps_root is not None:
        payload["corpus_lookup_sweeps_root"] = str(sweeps_root.expanduser().resolve())
    return payload


def _apply_corpus_lookup_context(
    cfg: DictConfig,
    *,
    sweep_id: str | None,
    sweeps_root: Path | None,
) -> None:
    if sweep_id is None:
        return
    raw_corpus_ref = OmegaConf.select(cfg, "data.surface_overrides.corpus_ref")
    if raw_corpus_ref is None:
        return
    for key, value in _corpus_lookup_context(
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    ).items():
        OmegaConf.update(
            cfg,
            f"data.surface_overrides.{key}",
            value,
            merge=True,
            force_add=True,
        )


def _has_explicit_prior_backend_settings(
    *,
    training_payload: Mapping[str, Any],
    legacy_prior_payload: Mapping[str, Any] | None,
) -> bool:
    if legacy_prior_payload is not None:
        return True
    return any(
        training_payload.get(key) is not None
        for key in (
            "prior_dump_non_finite_policy",
            "prior_dump_batch_size",
            "prior_dump_lr_scale_rule",
            "prior_dump_batch_reference_size",
            "effective_lr_scale_factor",
        )
    )


def compose_cfg(
    *,
    row: Mapping[str, Any],
    run_dir: Path,
    device: str,
    training_experiment: str = "cls_benchmark_staged_corpus",
    sweep_id: str | None = None,
    sweeps_root: Path | None = None,
) -> DictConfig:
    cfg = compose_config([f"experiment={training_experiment}"])
    cfg.runtime.output_dir = str(run_dir.resolve())
    cfg.runtime.device = str(device)
    cfg.logging.run_name = _queue_aware_run_name(run_dir=run_dir)
    if sweep_id is not None:
        cfg.logging.group = str(sweep_id)
    apply_mapping(cfg, "model", cast(Mapping[str, Any], row.get("model", {})))
    data_payload = cast(Mapping[str, Any], row.get("data", {}))
    apply_mapping(cfg, "data", data_payload)
    if "corpus_ref" in data_payload and "allow_missing_values" not in data_payload:
        OmegaConf.update(cfg, "data.allow_missing_values", None, merge=False)
    _apply_corpus_lookup_context(cfg, sweep_id=sweep_id, sweeps_root=sweeps_root)
    apply_mapping(cfg, "preprocessing", cast(Mapping[str, Any], row.get("preprocessing", {})))

    training_payload = cast(Mapping[str, Any], row.get("training", {}))
    for key in (
        "surface_label",
        "task_batch_size",
    ):
        if key in training_payload:
            OmegaConf.update(cfg, f"training.{key}", training_payload[key], merge=True)
    legacy_prior_payload = (
        legacy
        if isinstance((legacy := training_payload.get("legacy_prior")), Mapping)
        else None
    )
    normalized_prior_backend = resolve_prior_backend_surface_config(
        training_cfg=training_payload,
        legacy_prior_cfg=legacy_prior_payload,
    )
    if _has_explicit_prior_backend_settings(
        training_payload=training_payload,
        legacy_prior_payload=legacy_prior_payload,
    ):
        for source_key, value in normalized_prior_backend.to_dict().items():
            if value is None or source_key == "effective_lr_scale_factor":
                continue
            OmegaConf.update(
                cfg,
                f"legacy_prior.{source_key}",
                value,
                merge=True,
                force_add=True,
            )

    overrides = cast(Mapping[str, Any], training_payload.get("overrides", {}))
    if "apply_schedule" in overrides:
        OmegaConf.update(cfg, "training.apply_schedule", overrides["apply_schedule"], merge=True)
    for key in ("optimizer", "runtime", "schedule"):
        override_payload = overrides.get(key)
        if isinstance(override_payload, dict):
            apply_mapping(cfg, key, cast(Mapping[str, Any], override_payload))
    return cfg


def _cfg_data_mapping(cfg: Any) -> Mapping[str, Any] | None:
    raw_data_cfg = getattr(cfg, "data", None)
    if raw_data_cfg is None and isinstance(cfg, Mapping):
        raw_data_cfg = cfg.get("data")
    if raw_data_cfg is None:
        return None
    if isinstance(raw_data_cfg, DictConfig):
        raw_data_cfg = OmegaConf.to_container(raw_data_cfg, resolve=True)
    if not isinstance(raw_data_cfg, Mapping):
        raise RuntimeError("cfg.data must be a mapping when present")
    return cast(Mapping[str, Any], raw_data_cfg)


def resolve_training_backend(
    cfg: Any,
    *,
    allow_unresolved_corpus_ref: bool = False,
) -> str:
    return resolve_training_backend_from_data_cfg(
        _cfg_data_mapping(cfg),
        allow_unresolved_corpus_ref=allow_unresolved_corpus_ref,
    )
