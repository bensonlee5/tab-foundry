"""Training-surface resolution and artifact helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from omegaconf import OmegaConf

from tab_foundry.data.manifest_characteristics import compute_manifest_characteristics
from tab_foundry.data.surface import DataSurfaceConfig, resolve_data_surface
from tab_foundry.hashing import sha256_path
from tab_foundry.model.architectures.tabfoundry_staged.resolved import resolve_staged_surface
from tab_foundry.model.spec import (
    SANDWICH_MODEL_ARCH,
    checkpoint_model_build_spec_from_mappings,
    model_build_spec_from_mappings,
)
from tab_foundry.preprocessing import resolve_preprocessing_surface
from tab_foundry.repo_paths import normalize_repo_relative_path
from tab_foundry.training.prior.settings import resolve_prior_backend_surface_config
from tab_foundry.timestamps import utc_now as _shared_utc_now

from .trainer_runtime_config import (
    _resolve_activation_checkpointing,
    _resolve_compile_backend,
    _resolve_compile_dynamic,
    _resolve_compile_mode,
    _resolve_compile_model,
    _resolve_trace_activations,
)


TRAINING_SURFACE_SCHEMA = "tab-foundry-training-surface-v1"
TRAINING_BACKEND_MANIFEST = "manifest"
TRAINING_BACKEND_LEGACY_PRIOR = "legacy_prior"
_TRAINING_BACKEND_ALIASES = {
    TRAINING_BACKEND_MANIFEST: TRAINING_BACKEND_MANIFEST,
    TRAINING_BACKEND_LEGACY_PRIOR: TRAINING_BACKEND_LEGACY_PRIOR,
}
_VALID_TRAINING_BACKENDS = {
    TRAINING_BACKEND_MANIFEST,
    TRAINING_BACKEND_LEGACY_PRIOR,
}
_EXCLUDED_RUNTIME_SURFACE_KEYS = frozenset({"device", "output_dir"})


def _utc_now() -> str:
    return _shared_utc_now()


def _normalized_surface_path(value: Any) -> str | None:
    if value is None:
        return None
    normalized = str(value).strip()
    if not normalized:
        return None
    return normalize_repo_relative_path(Path(normalized))


def _normalize_runtime_surface_cfg(raw_runtime_cfg: Mapping[str, Any]) -> dict[str, Any]:
    runtime_cfg = {
        str(key): value
        for key, value in raw_runtime_cfg.items()
        if str(key) not in _EXCLUDED_RUNTIME_SURFACE_KEYS
    }
    resolved_runtime_cfg = OmegaConf.create(runtime_cfg)
    runtime_cfg["compile_model"] = _resolve_compile_model(resolved_runtime_cfg)
    runtime_cfg["compile_dynamic"] = _resolve_compile_dynamic(resolved_runtime_cfg)
    runtime_cfg["compile_backend"] = _resolve_compile_backend(resolved_runtime_cfg)
    runtime_cfg["compile_mode"] = _resolve_compile_mode(resolved_runtime_cfg)
    runtime_cfg["trace_activations"] = _resolve_trace_activations(resolved_runtime_cfg)
    runtime_cfg["activation_checkpointing"] = _resolve_activation_checkpointing(
        resolved_runtime_cfg
    )
    return runtime_cfg


def normalize_training_backend(value: Any) -> str | None:
    if value is None:
        return None
    backend = str(value).strip().lower()
    if not backend:
        return None
    normalized = _TRAINING_BACKEND_ALIASES.get(backend)
    if normalized is None:
        raise ValueError(
            "training backend must be one of "
            f"{sorted(_VALID_TRAINING_BACKENDS)}, got {value!r}"
        )
    return normalized


def resolve_training_backend_from_data_surface(data_surface: DataSurfaceConfig) -> str:
    source = str(data_surface.source).strip().lower()
    try:
        normalized = normalize_training_backend(source)
    except ValueError as exc:
        raise RuntimeError(
            f"unsupported training backend source {source!r}; expected one of {sorted(_VALID_TRAINING_BACKENDS)}"
        ) from exc
    if normalized is None:
        raise RuntimeError(
            f"unsupported training backend source {source!r}; expected one of {sorted(_VALID_TRAINING_BACKENDS)}"
        )
    return normalized


def resolve_training_backend_from_data_cfg(
    data_cfg: Mapping[str, Any] | None,
    *,
    allow_unresolved_corpus_ref: bool = False,
) -> str:
    """Resolve the training backend from one data surface mapping."""

    if data_cfg is None:
        return TRAINING_BACKEND_LEGACY_PRIOR
    data_surface = resolve_data_surface(
        data_cfg,
        allow_unresolved_corpus_ref=allow_unresolved_corpus_ref,
    )
    return resolve_training_backend_from_data_surface(data_surface)


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
    raw_legacy_prior_cfg = raw_cfg.get("legacy_prior")
    raw_optimizer_cfg = raw_cfg.get("optimizer")
    raw_schedule_cfg = raw_cfg.get("schedule")
    raw_runtime_cfg = raw_cfg.get("runtime")
    if not isinstance(raw_model_cfg, Mapping):
        raise RuntimeError("training surface record requires cfg.model to be a mapping")

    model_cfg = {str(key): value for key, value in raw_model_cfg.items()}
    if not isinstance(raw_data_cfg, Mapping):
        data_cfg = {
            "source": TRAINING_BACKEND_LEGACY_PRIOR,
            "surface_label": TRAINING_BACKEND_LEGACY_PRIOR,
            "surface_overrides": {},
        }
    else:
        data_cfg = {str(key): value for key, value in raw_data_cfg.items()}
    raw_data_surface_overrides = data_cfg.get("surface_overrides")
    data_surface_overrides = (
        {}
        if not isinstance(raw_data_surface_overrides, Mapping)
        else {str(key): value for key, value in raw_data_surface_overrides.items()}
    )
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
    legacy_prior_cfg = (
        None
        if not isinstance(raw_legacy_prior_cfg, Mapping)
        else {str(key): value for key, value in raw_legacy_prior_cfg.items()}
    )
    schedule_cfg = (
        None
        if not isinstance(raw_schedule_cfg, Mapping)
        else {str(key): value for key, value in raw_schedule_cfg.items()}
    )
    runtime_cfg = (
        None
        if not isinstance(raw_runtime_cfg, Mapping)
        else _normalize_runtime_surface_cfg(raw_runtime_cfg)
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
    resolved_backend = normalize_training_backend(backend)
    if resolved_backend is None:
        resolved_backend = resolve_training_backend_from_data_surface(data_surface)
    prior_backend_config = resolve_prior_backend_surface_config(
        training_cfg=training_cfg,
        legacy_prior_cfg=legacy_prior_cfg,
    )
    manifest_payload: dict[str, Any] | None = None
    if data_surface.manifest_path is not None:
        manifest_payload = {
            "manifest_path": normalize_repo_relative_path(data_surface.manifest_path),
        }
        if data_surface.manifest_path.exists():
            manifest_payload["manifest_sha256"] = sha256_path(data_surface.manifest_path)
            if include_manifest_characteristics:
                try:
                    manifest_payload["characteristics"] = compute_manifest_characteristics(
                        data_surface.manifest_path
                    )
                except Exception as exc:  # pragma: no cover - defensive compatibility fallback
                    manifest_payload["characteristics"] = None
                    manifest_payload["characteristics_error"] = str(exc)

    raw_requested_corpus_ref = data_surface_overrides.get("requested_corpus_ref")
    if raw_requested_corpus_ref is None:
        raw_requested_corpus_ref = data_cfg.get("requested_corpus_ref")
    if raw_requested_corpus_ref is None:
        raw_requested_corpus_ref = data_cfg.get("corpus_ref")
    requested_corpus_ref = (
        None
        if raw_requested_corpus_ref is None
        else str(raw_requested_corpus_ref).strip() or None
    )
    raw_materialization_state = data_surface_overrides.get("materialization_state")
    if raw_materialization_state is None:
        raw_materialization_state = data_cfg.get("materialization_state")
    if raw_materialization_state is None:
        if data_surface.corpus_record_path is not None:
            materialization_state = "finalized"
        elif data_surface.manifest_path is not None:
            materialization_state = "direct_manifest"
        else:
            materialization_state = None
    else:
        materialization_state = str(raw_materialization_state).strip().lower() or None

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
    elif model_spec.arch == SANDWICH_MODEL_ARCH:
        model_payload["architecture"] = {
            "initial_input_tokens": "full_cell_plus_row_col_summary_stream",
            "initial_input_token_count": "R_times_C_plus_K_times_(R_plus_C)",
            "repeated_input_tokens": "row_col_summary_stream",
            "repeated_input_token_count": "K_times_(R_plus_C)",
            "summary_tokens_per_axis": int(model_spec.sandwich_summary_tokens_per_axis),
            "pre_perceiver_cell_mixer": "row_feature_self_attention_then_column_row_isab",
            "pre_row_attention_layers": int(model_spec.sandwich_pre_row_attention_layers),
            "pre_column_attention_layers": int(model_spec.sandwich_pre_column_attention_layers),
            "pre_column_inducing_tokens": int(model_spec.sandwich_pre_column_inducing_tokens),
            "label_injection": "fused_into_row_summaries_and_feature_cells",
            "summary_builder": "summary_query_attention",
            "position_encoding": "shared_fourier_row_col",
            "feature_type_encoding": str(model_spec.feature_type_conditioning),
            "floating_likelihood": str(model_spec.floating_likelihood),
            "integer_likelihood": str(model_spec.integer_likelihood),
            "sandwich_activation": str(model_spec.sandwich_activation),
            "sandwich_block_norm": str(model_spec.sandwich_block_norm),
            "latent_core": "stage0_full_cell_plus_summary_then_summary_repeated_cross_self_stages",
            "layer_semantics": "stage0_hybrid_then_summary_repeated_stages",
            "readout": "latent_then_full_cell_cross_attention_then_latent_conditioned_query_pool",
            "latents": int(model_spec.sandwich_latents),
            "layers": int(model_spec.sandwich_layers),
            "heads": int(model_spec.sandwich_heads),
            "ff_expansion": int(model_spec.sandwich_ff_expansion),
            "self_attention_per_cross": int(model_spec.sandwich_self_attention_per_cross),
        }

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
        "run_dir": normalize_repo_relative_path(run_dir),
        "labels": labels,
        "model": model_payload,
        "data": {
            "surface_label": data_label,
            "source": str(data_surface.source),
            "filter_policy": data_surface.filter_policy,
            "allow_missing_values": bool(data_surface.allow_missing_values),
            "corpus_ref": data_surface.corpus_ref,
            "requested_corpus_ref": requested_corpus_ref,
            "materialization_state": materialization_state,
            "recipe_id": data_surface.recipe_id,
            "corpus_id": data_surface.corpus_id,
            "corpus_record_path": (
                None
                if data_surface.corpus_record_path is None
                else normalize_repo_relative_path(data_surface.corpus_record_path)
            ),
            "manifest": manifest_payload,
            "dagzoo_provenance": data_surface.dagzoo_provenance,
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
                "loss_surface": str(training_cfg.get("loss_surface", "classification")),
                "apply_schedule": bool(training_cfg.get("apply_schedule", False)),
                "task_batch_size": int(training_cfg.get("task_batch_size", 1)),
                "initial_checkpoint_path": _normalized_surface_path(
                    training_cfg.get("initial_checkpoint_path")
                ),
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
        if resolved_backend == TRAINING_BACKEND_LEGACY_PRIOR:
            training_payload["legacy_prior"] = prior_backend_config.to_dict()
        if resolved_backend is not None:
            training_payload["backend"] = resolved_backend
        payload["training"] = training_payload
    if runtime_cfg:
        payload["runtime"] = runtime_cfg
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
