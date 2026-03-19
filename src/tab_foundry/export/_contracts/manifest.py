"""Manifest validation for export bundles."""

from __future__ import annotations

from typing import Any

from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.model.spec import SUPPORTED_MODEL_ARCHES, model_build_spec_from_mappings

from .common import (
    _as_bool,
    _as_float,
    _as_int,
    _as_str,
    _require_keys,
    _validate_created_at_utc,
    _validate_hex_digest,
    _validate_input_normalization,
    _validate_many_class_train_mode,
    _validate_payload_model,
)
from .inference import validate_inference_config_dict
from .models import (
    SCHEMA_VERSION_V3,
    SUPPORTED_SCHEMA_VERSIONS,
    SUPPORTED_TASKS,
    _ManifestModelPayloadV2,
    _ManifestModelPayloadV3,
    ExportFiles,
    ExportManifest,
    ExportModelSpec,
    ExportPreprocessorState,
    ExportWeights,
    ProducerInfo,
)
from .preprocessor import validate_preprocessor_state_dict


def _manifest_model_primary_dict(model_raw: dict[str, Any]) -> dict[str, Any]:
    primary: dict[str, Any] = {
        "arch": _as_str(model_raw["arch"], context="manifest.model.arch"),
        "d_col": _as_int(model_raw["d_col"], context="manifest.model.d_col"),
        "d_icl": _as_int(model_raw["d_icl"], context="manifest.model.d_icl"),
        "feature_group_size": _as_int(
            model_raw["feature_group_size"],
            context="manifest.model.feature_group_size",
        ),
        "many_class_train_mode": _validate_many_class_train_mode(
            model_raw["many_class_train_mode"],
            context="manifest.model.many_class_train_mode",
        ),
        "max_mixed_radix_digits": _as_int(
            model_raw["max_mixed_radix_digits"],
            context="manifest.model.max_mixed_radix_digits",
        ),
    }
    if "stage" in model_raw:
        primary["stage"] = model_raw["stage"]
    if "stage_label" in model_raw:
        primary["stage_label"] = _as_str(
            model_raw["stage_label"],
            context="manifest.model.stage_label",
        )
    if "module_overrides" in model_raw:
        module_overrides = model_raw["module_overrides"]
        if not isinstance(module_overrides, dict):
            raise ValueError("manifest.model.module_overrides must be object")
        primary["module_overrides"] = module_overrides
    optional_fields: tuple[tuple[str, str], ...] = (
        ("input_normalization", "manifest.model.input_normalization"),
        ("norm_type", "manifest.model.norm_type"),
        ("tfcol_n_heads", "manifest.model.tfcol_n_heads"),
        ("tfcol_n_layers", "manifest.model.tfcol_n_layers"),
        ("tfcol_n_inducing", "manifest.model.tfcol_n_inducing"),
        ("tfrow_n_heads", "manifest.model.tfrow_n_heads"),
        ("tfrow_n_layers", "manifest.model.tfrow_n_layers"),
        ("tfrow_cls_tokens", "manifest.model.tfrow_cls_tokens"),
        ("tfrow_norm", "manifest.model.tfrow_norm"),
        ("tficl_n_heads", "manifest.model.tficl_n_heads"),
        ("tficl_n_layers", "manifest.model.tficl_n_layers"),
        ("tficl_ff_expansion", "manifest.model.tficl_ff_expansion"),
        ("many_class_base", "manifest.model.many_class_base"),
        ("head_hidden_dim", "manifest.model.head_hidden_dim"),
        ("staged_dropout", "manifest.model.staged_dropout"),
        ("pre_encoder_clip", "manifest.model.pre_encoder_clip"),
    )
    for field_name, context in optional_fields:
        if field_name not in model_raw:
            continue
        if field_name == "input_normalization":
            primary[field_name] = _validate_input_normalization(model_raw[field_name], context=context)
        elif field_name in {"norm_type", "tfrow_norm"}:
            primary[field_name] = _as_str(model_raw[field_name], context=context)
        elif field_name in {"staged_dropout", "pre_encoder_clip"}:
            primary[field_name] = _as_float(model_raw[field_name], context=context)
        else:
            primary[field_name] = _as_int(model_raw[field_name], context=context)
    if "use_digit_position_embed" in model_raw:
        primary["use_digit_position_embed"] = _as_bool(
            model_raw["use_digit_position_embed"],
            context="manifest.model.use_digit_position_embed",
        )
    return primary


def _validate_producer_info(payload: Any) -> ProducerInfo:
    if not isinstance(payload, dict):
        raise ValueError("manifest.producer must be object")
    _require_keys(
        payload,
        keys={"name", "version", "git_sha"},
        context="manifest.producer",
    )
    git_sha_raw = payload["git_sha"]
    if git_sha_raw is not None and not isinstance(git_sha_raw, str):
        raise ValueError("manifest.producer.git_sha must be string or null")
    return ProducerInfo(
        name=_as_str(payload["name"], context="manifest.producer.name"),
        version=_as_str(payload["version"], context="manifest.producer.version"),
        git_sha=git_sha_raw,
    )


def _validate_model_spec(
    payload: Any,
    *,
    task: str,
    schema_version: str,
) -> ExportModelSpec:
    if not isinstance(payload, dict):
        raise ValueError("manifest.model must be object")
    required_model_keys = {
        "arch",
        "d_col",
        "d_icl",
        "feature_group_size",
        "many_class_train_mode",
        "max_mixed_radix_digits",
    }
    optional_model_keys = {
        "stage",
        "norm_type",
        "tfcol_n_heads",
        "tfcol_n_layers",
        "tfcol_n_inducing",
        "tfrow_n_heads",
        "tfrow_n_layers",
        "tfrow_cls_tokens",
        "tfrow_norm",
        "tficl_n_heads",
        "tficl_n_layers",
        "tficl_ff_expansion",
        "many_class_base",
        "head_hidden_dim",
        "use_digit_position_embed",
    }
    if schema_version == SCHEMA_VERSION_V3:
        required_model_keys.add("input_normalization")
        optional_model_keys.update(
            {
                "stage_label",
                "module_overrides",
                "staged_dropout",
                "pre_encoder_clip",
            }
        )
    else:
        optional_model_keys.add("input_normalization")
    _require_keys(
        payload,
        keys=required_model_keys,
        context="manifest.model",
        optional_keys=optional_model_keys,
    )
    payload_model = (
        _ManifestModelPayloadV3 if schema_version == SCHEMA_VERSION_V3 else _ManifestModelPayloadV2
    )
    validated_payload = _validate_payload_model(
        payload_model,
        payload,
        context="manifest.model",
    )
    payload_dict = validated_payload.model_dump(exclude_none=True)
    arch = _as_str(payload_dict["arch"], context="manifest.model.arch")
    if arch not in SUPPORTED_MODEL_ARCHES:
        raise ValueError(f"Unsupported model arch: {arch!r}")
    model_spec = model_build_spec_from_mappings(
        task=task,
        primary=_manifest_model_primary_dict(payload_dict),
    )
    try:
        _ = build_model_from_spec(model_spec)
    except (RuntimeError, ValueError) as exc:
        raise ValueError(str(exc)) from exc
    return ExportModelSpec.from_build_spec(model_spec, arch=arch)


def validate_manifest_dict(payload: dict[str, Any]) -> ExportManifest:
    schema_version = _as_str(payload.get("schema_version"), context="manifest.schema_version")
    if schema_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported schema version: {schema_version!r}")

    common_keys = {"schema_version", "producer", "task", "model", "created_at_utc"}
    if schema_version == SCHEMA_VERSION_V3:
        if "manifest_sha256" not in payload:
            raise ValueError(
                "manifest.manifest_sha256 is required for tab-foundry-export-v3 bundles; "
                "older v3 bundles must be regenerated"
            )
        _require_keys(
            payload,
            keys=common_keys | {"manifest_sha256", "inference", "preprocessor", "weights"},
            context="manifest",
        )
    else:
        _require_keys(
            payload,
            keys=common_keys | {"files", "checksums"},
            context="manifest",
        )

    producer = _validate_producer_info(payload["producer"])
    task = _as_str(payload["task"], context="manifest.task")
    if task not in SUPPORTED_TASKS:
        raise ValueError(f"Unsupported manifest task: {task!r}")
    model = _validate_model_spec(payload["model"], task=task, schema_version=schema_version)
    created_at_utc = _validate_created_at_utc(payload["created_at_utc"])
    manifest_sha256: str | None = None

    if schema_version == SCHEMA_VERSION_V3:
        manifest_sha256 = _validate_hex_digest(
            payload["manifest_sha256"],
            context="manifest.manifest_sha256",
        )
        inference_raw = payload["inference"]
        if not isinstance(inference_raw, dict):
            raise ValueError("manifest.inference must be object")
        inference = validate_inference_config_dict(inference_raw)
        if inference.task != task:
            raise ValueError("manifest.task and manifest.inference.task mismatch")
        if inference.model_arch != model.arch:
            raise ValueError("manifest.model.arch and manifest.inference.model_arch mismatch")
        if inference.model_stage != model.stage:
            raise ValueError("manifest.model.stage and manifest.inference.model_stage mismatch")
        if inference.feature_group_size != model.feature_group_size:
            raise ValueError("feature_group_size mismatch between manifest.model and manifest.inference")
        preprocessor_raw = payload["preprocessor"]
        if not isinstance(preprocessor_raw, dict):
            raise ValueError("manifest.preprocessor must be object")
        preprocessor = validate_preprocessor_state_dict(
            preprocessor_raw,
            schema_version=schema_version,
            task=task,
        )
        if not isinstance(preprocessor, ExportPreprocessorState):
            raise RuntimeError("v3 manifest preprocessor must validate to export preprocessor state")
        weights_raw = payload["weights"]
        if not isinstance(weights_raw, dict):
            raise ValueError("manifest.weights must be object")
        _require_keys(
            weights_raw,
            keys={"file", "sha256"},
            context="manifest.weights",
        )
        weights = ExportWeights(
            file=_as_str(weights_raw["file"], context="manifest.weights.file"),
            sha256=_validate_hex_digest(weights_raw["sha256"], context="manifest.weights.sha256"),
        )
        return ExportManifest(
            schema_version=schema_version,
            producer=producer,
            task=task,
            model=model,
            created_at_utc=created_at_utc,
            manifest_sha256=manifest_sha256,
            inference=inference,
            preprocessor=preprocessor,
            weights=weights,
        )

    files_raw = payload["files"]
    if not isinstance(files_raw, dict):
        raise ValueError("manifest.files must be object")
    _require_keys(
        files_raw,
        keys={"weights", "inference_config", "preprocessor_state"},
        context="manifest.files",
    )
    files = ExportFiles(
        weights=_as_str(files_raw["weights"], context="manifest.files.weights"),
        inference_config=_as_str(files_raw["inference_config"], context="manifest.files.inference_config"),
        preprocessor_state=_as_str(
            files_raw["preprocessor_state"],
            context="manifest.files.preprocessor_state",
        ),
    )

    checksums_raw = payload["checksums"]
    if not isinstance(checksums_raw, dict):
        raise ValueError("manifest.checksums must be object")
    _require_keys(
        checksums_raw,
        keys={"weights", "inference_config", "preprocessor_state"},
        context="manifest.checksums",
    )
    checksums = {
        key: _validate_hex_digest(value, context=f"manifest.checksums.{key}")
        for key, value in checksums_raw.items()
    }
    return ExportManifest(
        schema_version=schema_version,
        producer=producer,
        task=task,
        model=model,
        created_at_utc=created_at_utc,
        files=files,
        checksums=checksums,
    )
