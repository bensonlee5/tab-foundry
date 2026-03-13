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

from tab_foundry.model.spec import (
    ModelBuildSpec,
    checkpoint_model_build_spec_from_mappings,
)

from .checksums import sha256_file
from .contracts import (
    ExportModelSpec,
    SCHEMA_VERSION_V2,
    ExportManifest,
    InferenceConfig,
    PreprocessorState,
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


def _producer_info() -> dict[str, str | None]:
    try:
        version = importlib.metadata.version("tab-foundry")
    except Exception:
        version = "0.0.0"
    return {
        "name": "tab-foundry",
        "version": version,
        "git_sha": _git_sha(),
    }


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


def _inference_config(task: str, model_spec: ExportModelSpec) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "task": task,
        "model_arch": "tabfoundry",
        "group_shifts": list(DEFAULT_GROUP_SHIFTS),
        "feature_group_size": int(model_spec.feature_group_size),
        "many_class_threshold": DEFAULT_MANY_CLASS_THRESHOLD,
        "many_class_inference_mode": "full_probs",
    }
    if task == "regression":
        levels = torch.arange(1, 1000, dtype=torch.float32) / 1000.0
        payload["quantile_levels"] = [float(v) for v in levels.tolist()]
    return payload


def _preprocessor_state() -> dict[str, Any]:
    return {
        "feature_order_policy": "lexicographic_f_columns",
        "missing_value_policy": {"strategy": "train_mean", "all_nan_fill": 0.0},
        "classification_label_policy": {
            "mapping": "train_only_remap",
            "unseen_test_label": "filter",
        },
        "dtype_policy": {
            "features": "float32",
            "classification_labels": "int64",
            "regression_targets": "float32",
        },
    }


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
    artifact_version: str = SCHEMA_VERSION_V2,
) -> ExportResult:
    """Export one training checkpoint as an inference bundle."""

    checkpoint = checkpoint_path.expanduser().resolve()
    if artifact_version != SCHEMA_VERSION_V2:
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

    bundle_dir = out_dir.expanduser().resolve()
    bundle_dir.mkdir(parents=True, exist_ok=True)

    weights_name = "weights.safetensors"
    inference_name = "inference_config.json"
    preproc_name = "preprocessor_state.json"
    manifest_name = "manifest.json"

    weights_path = bundle_dir / weights_name
    inference_path = bundle_dir / inference_name
    preproc_path = bundle_dir / preproc_name
    manifest_path = bundle_dir / manifest_name

    save_file(state_dict, str(weights_path))

    inference_payload = _inference_config(task, model_spec)
    preproc_payload = _preprocessor_state()
    _write_json(inference_path, inference_payload)
    _write_json(preproc_path, preproc_payload)

    manifest_payload: dict[str, Any] = {
        "schema_version": artifact_version,
        "producer": _producer_info(),
        "task": task,
        "model": model_spec.to_dict(),
        "files": {
            "weights": weights_name,
            "inference_config": inference_name,
            "preprocessor_state": preproc_name,
        },
        "checksums": {
            "weights": sha256_file(weights_path),
            "inference_config": sha256_file(inference_path),
            "preprocessor_state": sha256_file(preproc_path),
        },
        "created_at_utc": datetime.now(tz=timezone.utc).isoformat().replace("+00:00", "Z"),
    }
    _write_json(manifest_path, manifest_payload)

    # Fail fast if any emitted JSON/schema/checksum contract is invalid.
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

    weights_path = root / manifest.files.weights
    inference_path = root / manifest.files.inference_config
    preproc_path = root / manifest.files.preprocessor_state
    for path in (weights_path, inference_path, preproc_path):
        if not path.exists():
            raise ValueError(f"bundle file not found: {path}")

    inference_dict = read_json_dict(inference_path)
    preproc_dict = read_json_dict(preproc_path)
    inference: InferenceConfig = validate_inference_config_dict(inference_dict)
    preprocessor_state: PreprocessorState = validate_preprocessor_state_dict(preproc_dict)

    if inference.task != manifest.task:
        raise ValueError("manifest.task and inference_config.task mismatch")
    if inference.feature_group_size != manifest.model.feature_group_size:
        raise ValueError("feature_group_size mismatch between manifest and inference_config")

    current_checksums = {
        "weights": sha256_file(weights_path),
        "inference_config": sha256_file(inference_path),
        "preprocessor_state": sha256_file(preproc_path),
    }
    for key, current in current_checksums.items():
        expected = manifest.checksums[key]
        if current != expected:
            raise ValueError(
                f"checksum mismatch for {key}: expected={expected}, actual={current}"
            )

    return ValidatedBundle(
        manifest=manifest,
        inference_config=inference,
        preprocessor_state=preprocessor_state,
    )
