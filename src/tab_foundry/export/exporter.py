"""Export checkpoints into versioned inference bundles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import subprocess
from typing import Any

from safetensors.torch import save_file
import torch

from tab_foundry.model.missingness import validate_missingness_runtime_policy
from tab_foundry.model.spec import ModelBuildSpec, checkpoint_model_build_spec_from_mappings
from tab_foundry.preprocessing import resolve_preprocessing_surface

from .checksums import sha256_file
from .contracts import (
    compute_v3_manifest_sha256,
    ExportClassificationLabelPolicy,
    ExportFiles,
    ExportManifest,
    ExportMissingValuePolicy,
    ExportModelSpec,
    ExportPreprocessorState,
    ExportWeights,
    InferenceConfig,
    LegacyPreprocessorState,
    ProducerInfo,
    SCHEMA_VERSION_V3,
    SUPPORTED_SCHEMA_VERSIONS,
    ValidatedBundle,
    read_json_dict,
    validate_inference_config_dict,
    validate_manifest_dict,
    validate_preprocessor_state_dict,
)


DEFAULT_GROUP_SHIFTS = [0, 1, 3]
DEFAULT_MANY_CLASS_THRESHOLD = 10


@dataclass(slots=True)
class ExportResult:
    bundle_dir: Path
    manifest_path: Path
    schema_version: str


def _git_sha() -> str | None:
    try:
        output = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except Exception:
        return None
    value = output.strip()
    return value or None


def _producer_info() -> ProducerInfo:
    try:
        version = importlib.metadata.version("tab-foundry")
    except Exception:
        version = "0.0.0"
    return ProducerInfo(
        name="tab-foundry",
        version=version,
        git_sha=_git_sha(),
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _require_mapping(payload: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} must be an object")
    return payload


def _checkpoint_model_spec(
    cfg: dict[str, Any],
    *,
    task: str,
    state_dict: dict[str, torch.Tensor] | None = None,
) -> ModelBuildSpec:
    model_cfg = cfg.get("model")
    model_cfg = model_cfg if isinstance(model_cfg, dict) else {}
    return checkpoint_model_build_spec_from_mappings(
        task=task,
        primary=model_cfg,
        state_dict=state_dict,
    )


def _inference_config(task: str, model_spec: ExportModelSpec) -> InferenceConfig:
    quantile_levels: list[float] | None = None
    if task == "regression":
        levels = torch.arange(1, 1000, dtype=torch.float32) / 1000.0
        quantile_levels = [float(v) for v in levels.tolist()]
    return InferenceConfig(
        task=task,
        model_arch=str(model_spec.arch),
        model_stage=None if model_spec.stage is None else str(model_spec.stage),
        missingness_mode=str(model_spec.missingness_mode),
        group_shifts=list(DEFAULT_GROUP_SHIFTS),
        feature_group_size=int(model_spec.feature_group_size),
        many_class_threshold=DEFAULT_MANY_CLASS_THRESHOLD,
        many_class_inference_mode="full_probs",
        quantile_levels=quantile_levels,
    )


def _preprocessor_state_v2(*, impute_missing: bool) -> LegacyPreprocessorState:
    return LegacyPreprocessorState(
        feature_order_policy="lexicographic_f_columns",
        missing_value_policy={"strategy": "train_mean", "all_nan_fill": 0.0},
        classification_label_policy={
            "mapping": "train_only_remap",
            "unseen_test_label": "filter",
        },
        dtype_policy={
            "features": "float32",
            "classification_labels": "int64",
            "regression_targets": "float32",
        },
        impute_missing=bool(impute_missing),
    )


def _preprocessor_state_v3(task: str, *, impute_missing: bool) -> ExportPreprocessorState:
    classification_label_policy: ExportClassificationLabelPolicy | None = None
    if task == "classification":
        classification_label_policy = ExportClassificationLabelPolicy(
            mapping="train_only_remap",
            unseen_test_label="filter",
        )
    return ExportPreprocessorState(
        feature_order_policy="positional_feature_ids",
        missing_value_policy=ExportMissingValuePolicy(
            strategy="train_mean",
            all_nan_fill=0.0,
        ),
        classification_label_policy=classification_label_policy,
        dtype_policy={
            "features": "float32",
            "classification_labels": "int64",
            "regression_targets": "float32",
        },
        impute_missing=bool(impute_missing),
    )


def _normalize_state_dict(state_dict: Any) -> dict[str, torch.Tensor]:
    if not isinstance(state_dict, dict):
        raise RuntimeError("checkpoint['model'] must be a state_dict object")
    out: dict[str, torch.Tensor] = {}
    for key, value in state_dict.items():
        if not isinstance(key, str):
            raise RuntimeError("state_dict keys must be strings")
        if not isinstance(value, torch.Tensor):
            raise RuntimeError(f"state_dict[{key!r}] must be a tensor")
        out[key] = value.detach().cpu().contiguous()
    return out


def export_checkpoint(
    checkpoint_path: Path,
    out_dir: Path,
    *,
    artifact_version: str = SCHEMA_VERSION_V3,
) -> ExportResult:
    """Export one training checkpoint as an inference bundle."""

    checkpoint = checkpoint_path.expanduser().resolve()
    if artifact_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise ValueError(f"Unsupported artifact_version: {artifact_version!r}")

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    payload = _require_mapping(payload, context="checkpoint payload")
    if "model" not in payload or "config" not in payload:
        raise RuntimeError("checkpoint payload must contain 'model' and 'config'")

    cfg = _require_mapping(payload["config"], context="checkpoint payload.config")
    task = str(cfg.get("task", "")).strip().lower()
    if task not in ("classification", "regression"):
        raise RuntimeError(f"Unsupported checkpoint task value: {task!r}")

    state_dict = _normalize_state_dict(payload["model"])
    model_build_spec = _checkpoint_model_spec(cfg, task=task, state_dict=state_dict)
    model_spec = ExportModelSpec.from_build_spec(model_build_spec)
    preprocessing_cfg = cfg.get("preprocessing")
    preprocessing_surface = resolve_preprocessing_surface(
        preprocessing_cfg if isinstance(preprocessing_cfg, dict) else None
    )
    validate_missingness_runtime_policy(
        missingness_mode=model_build_spec.missingness_mode,
        impute_missing=preprocessing_surface.impute_missing,
        context="export_checkpoint",
    )

    bundle_dir = out_dir.expanduser().resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)

    weights_name = "weights.safetensors"
    manifest_name = "manifest.json"
    weights_path = bundle_dir / weights_name
    manifest_path = bundle_dir / manifest_name

    save_file(state_dict, str(weights_path))

    if artifact_version == SCHEMA_VERSION_V3:
        manifest = ExportManifest(
            schema_version=artifact_version,
            producer=_producer_info(),
            task=task,
            model=model_spec,
            created_at_utc=datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            inference=_inference_config(task, model_spec),
            preprocessor=_preprocessor_state_v3(
                task,
                impute_missing=preprocessing_surface.impute_missing,
            ),
            weights=ExportWeights(
                file=weights_name,
                sha256=sha256_file(weights_path),
            ),
        )
        manifest.manifest_sha256 = compute_v3_manifest_sha256(manifest.to_dict())
    else:
        inference_name = "inference_config.json"
        preproc_name = "preprocessor_state.json"
        inference_path = bundle_dir / inference_name
        preproc_path = bundle_dir / preproc_name
        _write_json(inference_path, _inference_config(task, model_spec).to_dict())
        _write_json(
            preproc_path,
            _preprocessor_state_v2(
                impute_missing=preprocessing_surface.impute_missing,
            ).to_dict(),
        )
        manifest = ExportManifest(
            schema_version=artifact_version,
            producer=_producer_info(),
            task=task,
            model=model_spec,
            created_at_utc=datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
            files=ExportFiles(
                weights=weights_name,
                inference_config=inference_name,
                preprocessor_state=preproc_name,
            ),
            checksums={
                "weights": sha256_file(weights_path),
                "inference_config": sha256_file(inference_path),
                "preprocessor_state": sha256_file(preproc_path),
            },
        )

    _write_json(manifest_path, manifest.to_dict())
    _ = validate_export_bundle(bundle_dir)

    return ExportResult(
        bundle_dir=bundle_dir,
        manifest_path=manifest_path,
        schema_version=artifact_version,
    )


def validate_export_bundle(bundle_dir: Path) -> ValidatedBundle:
    """Validate schema and checksums for an export bundle."""

    root = bundle_dir.expanduser().resolve()
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"manifest file not found: {manifest_path}")

    manifest_dict = read_json_dict(manifest_path)
    manifest: ExportManifest = validate_manifest_dict(manifest_dict)

    if manifest.schema_version == SCHEMA_VERSION_V3:
        if manifest.inference is None or manifest.preprocessor is None or manifest.weights is None:
            raise RuntimeError("v3 bundle validation requires embedded inference, preprocessor, and weights")
        if manifest.manifest_sha256 is None:
            raise RuntimeError("v3 bundle validation requires manifest.manifest_sha256")
        expected_manifest_sha256 = compute_v3_manifest_sha256(manifest_dict)
        if expected_manifest_sha256 != manifest.manifest_sha256:
            raise ValueError(
                "manifest.manifest_sha256 mismatch: bundle metadata was modified after export; "
                "regenerate the bundle"
            )
        weights_path = root / manifest.weights.file
        if not weights_path.exists():
            raise ValueError(f"bundle file not found: {weights_path}")
        current_checksum = sha256_file(weights_path)
        if current_checksum != manifest.weights.sha256:
            raise ValueError(
                "checksum mismatch for weights: "
                f"expected={manifest.weights.sha256}, actual={current_checksum}"
            )
        if manifest.inference.task != manifest.task:
            raise ValueError("manifest.task and manifest.inference.task mismatch")
        if manifest.inference.feature_group_size != manifest.model.feature_group_size:
            raise ValueError("feature_group_size mismatch between manifest.model and manifest.inference")
        return ValidatedBundle(
            manifest=manifest,
            inference_config=manifest.inference,
            preprocessor_state=manifest.preprocessor,
        )

    if manifest.files is None or manifest.checksums is None:
        raise RuntimeError("v2 bundle validation requires files and checksums")
    weights_path = root / manifest.files.weights
    inference_path = root / manifest.files.inference_config
    preproc_path = root / manifest.files.preprocessor_state
    for path in (weights_path, inference_path, preproc_path):
        if not path.exists():
            raise ValueError(f"bundle file not found: {path}")

    inference_dict = read_json_dict(inference_path)
    preproc_dict = read_json_dict(preproc_path)
    inference = validate_inference_config_dict(
        inference_dict,
        schema_version=manifest.schema_version,
    )
    preprocessor_state = validate_preprocessor_state_dict(
        preproc_dict,
        schema_version=manifest.schema_version,
        task=manifest.task,
    )

    if inference.task != manifest.task:
        raise ValueError("manifest.task and inference_config.task mismatch")
    if inference.model_arch != manifest.model.arch:
        raise ValueError("manifest.model.arch and inference_config.model_arch mismatch")
    if inference.model_stage != manifest.model.stage:
        raise ValueError("manifest.model.stage and inference_config.model_stage mismatch")
    if inference.missingness_mode != manifest.model.missingness_mode:
        raise ValueError("missingness_mode mismatch between manifest and inference_config")
    if inference.feature_group_size != manifest.model.feature_group_size:
        raise ValueError("feature_group_size mismatch between manifest and inference_config")
    validate_missingness_runtime_policy(
        missingness_mode=manifest.model.missingness_mode,
        impute_missing=preprocessor_state.impute_missing,
        context="validate_export_bundle",
    )

    current_checksums = {
        "weights": sha256_file(weights_path),
        "inference_config": sha256_file(inference_path),
        "preprocessor_state": sha256_file(preproc_path),
    }
    for key, current in current_checksums.items():
        expected = manifest.checksums[key]
        if current != expected:
            raise ValueError(f"checksum mismatch for {key}: expected={expected}, actual={current}")

    return ValidatedBundle(
        manifest=manifest,
        inference_config=inference,
        preprocessor_state=preprocessor_state,
    )
