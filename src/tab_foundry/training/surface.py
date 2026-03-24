"""Training-surface resolution and artifact helpers."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

from tab_foundry.data.inspection import manifest_characteristics
from tab_foundry.data.surface import resolve_data_surface
from tab_foundry.model.architectures.tabfoundry_staged.resolved import resolve_staged_surface
from tab_foundry.model.spec import (
    checkpoint_model_build_spec_from_mappings,
    model_build_spec_from_mappings,
)
from tab_foundry.preprocessing import resolve_preprocessing_surface
from tab_foundry.timestamps import utc_now as _shared_utc_now


TRAINING_SURFACE_SCHEMA = "tab-foundry-training-surface-v1"
TRAINING_BACKEND_MANIFEST = "manifest"
TRAINING_BACKEND_PRIOR_DUMP = "prior_dump"
_VALID_TRAINING_BACKENDS = {
    TRAINING_BACKEND_MANIFEST,
    TRAINING_BACKEND_PRIOR_DUMP,
}


def _utc_now() -> str:
    return _shared_utc_now()


def _normalize_training_backend(value: Any) -> str | None:
    if value is None:
        return None
    backend = str(value).strip().lower()
    if not backend:
        return None
    if backend not in _VALID_TRAINING_BACKENDS:
        raise ValueError(
            f"training backend must be one of {sorted(_VALID_TRAINING_BACKENDS)}, got {value!r}"
        )
    return backend


def resolve_training_backend_from_data_cfg(data_cfg: Mapping[str, Any] | None) -> str:
    """Resolve the training backend from one data surface mapping."""

    if data_cfg is None:
        return TRAINING_BACKEND_PRIOR_DUMP
    source = str(resolve_data_surface(data_cfg).source).strip().lower()
    if source not in _VALID_TRAINING_BACKENDS:
        raise RuntimeError(
            f"unsupported training backend source {source!r}; expected one of {sorted(_VALID_TRAINING_BACKENDS)}"
        )
    return source


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def build_training_surface_record(
    *,
    raw_cfg: Mapping[str, Any],
    run_dir: Path,
    state_dict: Mapping[str, Any] | None = None,
    include_manifest_characteristics: bool = True,
    backend: str | None = None,
    allow_unresolved_corpus_ref: bool = False,
) -> dict[str, Any]:
    """Build one machine-readable training-surface record."""

    task = str(raw_cfg.get("task", "classification")).strip().lower()
    raw_model_cfg = raw_cfg.get("model")
    raw_data_cfg = raw_cfg.get("data")
    raw_preprocessing_cfg = raw_cfg.get("preprocessing")
    raw_training_cfg = raw_cfg.get("training")
    raw_optimizer_cfg = raw_cfg.get("optimizer")
    raw_schedule_cfg = raw_cfg.get("schedule")
    if not isinstance(raw_model_cfg, Mapping):
        raise RuntimeError("training surface record requires cfg.model to be a mapping")

    model_cfg = {str(key): value for key, value in raw_model_cfg.items()}
    if not isinstance(raw_data_cfg, Mapping):
        data_cfg = {
            "source": "prior_dump",
            "surface_label": "prior_dump",
            "surface_overrides": {},
        }
    else:
        data_cfg = {str(key): value for key, value in raw_data_cfg.items()}
    preprocessing_cfg = (
        None
        if not isinstance(raw_preprocessing_cfg, Mapping)
        else {str(key): value for key, value in raw_preprocessing_cfg.items()}
    )
    training_cfg = (
        None
        if not isinstance(raw_training_cfg, Mapping)
        else {str(key): value for key, value in raw_training_cfg.items()}
    )
    optimizer_cfg = (
        None
        if not isinstance(raw_optimizer_cfg, Mapping)
        else {str(key): value for key, value in raw_optimizer_cfg.items()}
    )
    schedule_cfg = (
        None
        if not isinstance(raw_schedule_cfg, Mapping)
        else {str(key): value for key, value in raw_schedule_cfg.items()}
    )
    if state_dict is None:
        model_spec = model_build_spec_from_mappings(task=task, primary=model_cfg)
    else:
        model_spec = checkpoint_model_build_spec_from_mappings(
            task=task,
            primary=model_cfg,
            state_dict=state_dict,
        )
    data_surface = resolve_data_surface(
        data_cfg,
        allow_unresolved_corpus_ref=allow_unresolved_corpus_ref,
    )
    preprocessing_surface = resolve_preprocessing_surface(preprocessing_cfg)
    resolved_backend = _normalize_training_backend(backend)
    if resolved_backend is None:
        resolved_backend = str(data_surface.source).strip().lower()
        if resolved_backend not in _VALID_TRAINING_BACKENDS:
            raise RuntimeError(
                "unsupported training backend source "
                f"{resolved_backend!r}; expected one of {sorted(_VALID_TRAINING_BACKENDS)}"
            )
    manifest_payload: dict[str, Any] | None = None
    if data_surface.manifest_path is not None:
        manifest_payload = {
            "manifest_path": str(data_surface.manifest_path),
        }
        if data_surface.manifest_path.exists():
            manifest_payload["manifest_sha256"] = _sha256_path(data_surface.manifest_path)
            if include_manifest_characteristics:
                try:
                    manifest_payload["characteristics"] = manifest_characteristics(data_surface.manifest_path)
                except Exception as exc:  # pragma: no cover - defensive compatibility fallback
                    manifest_payload["characteristics"] = None
                    manifest_payload["characteristics_error"] = str(exc)

    model_payload: dict[str, Any] = {
        "arch": str(model_spec.arch),
        "stage": None if model_spec.stage is None else str(model_spec.stage),
        "stage_label": None if model_spec.stage_label is None else str(model_spec.stage_label),
        "input_normalization": str(model_spec.input_normalization),
        "feature_group_size": int(model_spec.feature_group_size),
        "many_class_base": int(model_spec.many_class_base),
        "build_spec": model_spec.to_dict(),
    }
    model_label = str(model_spec.arch)
    if model_spec.arch == "tabfoundry_staged":
        surface = resolve_staged_surface(model_spec)
        model_payload["benchmark_profile"] = str(surface.benchmark_profile)
        model_payload["module_selection"] = surface.module_selection()
        model_payload["module_hyperparameters"] = surface.component_hyperparameters()
        model_label = str(surface.stage_label)

    data_label = str(data_surface.surface_label)
    preprocessing_label = str(preprocessing_surface.surface_label)
    labels: dict[str, Any] = {
        "model": model_label,
        "data": data_label,
        "preprocessing": preprocessing_label,
    }
    payload = {
        "schema": TRAINING_SURFACE_SCHEMA,
        "generated_at_utc": _utc_now(),
        "run_dir": str(run_dir.expanduser().resolve()),
        "labels": labels,
        "model": model_payload,
        "data": {
            "surface_label": data_label,
            "source": str(data_surface.source),
            "filter_policy": data_surface.filter_policy,
            "allow_missing_values": bool(data_surface.allow_missing_values),
            "corpus_ref": data_surface.corpus_ref,
            "recipe_id": data_surface.recipe_id,
            "corpus_id": data_surface.corpus_id,
            "corpus_record_path": (
                None
                if data_surface.corpus_record_path is None
                else str(data_surface.corpus_record_path)
            ),
            "manifest": manifest_payload,
            "dagzoo_provenance": data_surface.dagzoo_provenance,
            "train_row_cap": data_surface.train_row_cap,
            "test_row_cap": data_surface.test_row_cap,
            "overrides": data_surface.overrides,
        },
        "preprocessing": {
            "surface_label": preprocessing_label,
            "impute_missing": bool(preprocessing_surface.impute_missing),
            "all_nan_fill": float(preprocessing_surface.all_nan_fill),
            "label_mapping": str(preprocessing_surface.label_mapping),
            "unseen_test_label_policy": str(preprocessing_surface.unseen_test_label_policy),
            "feature_order_policy": str(preprocessing_surface.feature_order_policy),
            "dtype_policy": dict(preprocessing_surface.dtype_policy),
            "overrides": preprocessing_surface.overrides,
        },
    }
    if training_cfg is not None or resolved_backend is not None:
        training_payload: dict[str, Any] = {}
        if training_cfg is not None:
            training_label = str(training_cfg.get("surface_label", "training_default"))
            labels["training"] = training_label
            training_payload = {
                "surface_label": training_label,
                "apply_schedule": bool(training_cfg.get("apply_schedule", False)),
                "task_batch_size": int(training_cfg.get("task_batch_size", 1)),
                "prior_dump_non_finite_policy": str(
                    training_cfg.get("prior_dump_non_finite_policy", "error")
                ),
                "prior_dump_batch_size": None
                if training_cfg.get("prior_dump_batch_size") is None
                else int(training_cfg["prior_dump_batch_size"]),
                "prior_dump_lr_scale_rule": None
                if training_cfg.get("prior_dump_lr_scale_rule") is None
                else str(training_cfg["prior_dump_lr_scale_rule"]),
                "prior_dump_batch_reference_size": None
                if training_cfg.get("prior_dump_batch_reference_size") is None
                else int(training_cfg["prior_dump_batch_reference_size"]),
                "effective_lr_scale_factor": None
                if training_cfg.get("effective_lr_scale_factor") is None
                else float(training_cfg["effective_lr_scale_factor"]),
                "optimizer_name": None
                if optimizer_cfg is None or optimizer_cfg.get("name") is None
                else str(optimizer_cfg["name"]),
                "optimizer_min_lr": None
                if optimizer_cfg is None or optimizer_cfg.get("min_lr") is None
                else float(optimizer_cfg["min_lr"]),
                "schedule_stages": None
                if schedule_cfg is None
                else schedule_cfg.get("stages"),
                "overrides": training_cfg.get("overrides", {}),
            }
        else:
            training_payload = {
                "task_batch_size": 1,
            }
        if resolved_backend is not None:
            training_payload["backend"] = resolved_backend
        payload["training"] = training_payload
    return payload


def write_training_surface_record(
    path: Path,
    *,
    raw_cfg: Mapping[str, Any],
    run_dir: Path,
    state_dict: Mapping[str, Any] | None = None,
    backend: str | None = None,
) -> dict[str, Any]:
    """Write one training-surface record and return the payload."""

    record = build_training_surface_record(
        raw_cfg=raw_cfg,
        run_dir=run_dir,
        state_dict=state_dict,
        backend=backend,
    )
    resolved_path = path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record
