"""Shared helpers for the adequacy pilot stack."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tab_foundry.repo_paths import repo_root as shared_repo_root

_SUPPORTED_ADEQUACY_ID = "tf_rd_010_synthetic_adequacy_v3"
_SUPPORTED_DEVICE = "cpu"
_LATENT_TARGET_DERIVATION = "tabiclv2_latent_node"
_CANARY_BLOCK_ID = "latent_target_canary_curated_v3"
_PRODUCTION_BLOCK_ID = "production_control_curated_v5"
_TRAINING_EXPERIMENT = "cls_benchmark_sandwich_classification_evolution_v1"
_CANARY_PREDICTORS = frozenset({"chance", "logistic_regression"})
_SUMMARY_JSON_NAME = "summary.json"
_SUMMARY_MARKDOWN_NAME = "summary.md"
_MAX_REPORTED_TASK_ERRORS = 12
_ABSOLUTE_CANARY_IMPROVEMENT_THRESHOLD = 0.05
_CONTRACT_CHECK_MODES = frozenset({"fast", "full"})

_MEDIUM_V4_TRAINING_SURFACE = {
    "experiment": _TRAINING_EXPERIMENT,
    "task_batch_size": 16,
    "grad_accum_steps": 4,
    "grad_clip": 0.0,
    "max_steps": 2500,
    "optimizer_min_lr": 1.0e-5,
    "runtime": {
        "device": "cpu",
        "mixed_precision": "no",
        "num_workers": 0,
        "eval_every": 25,
        "checkpoint_every": 25,
        "val_batches": 0,
        "seed": 1,
    },
    "schedule_stage": {
        "name": "prior_dump",
        "steps": 2500,
        "lr_max": 1.0e-3,
        "lr_schedule": "linear",
        "warmup_ratio": 0.10,
    },
}


def _repo_root() -> Path:
    return shared_repo_root()


def _ensure_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _optional_mapping(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {str(key): item for key, item in value.items()}


def _finite_float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    if not math.isfinite(numeric):
        return None
    return numeric


def _int_or_none(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_safe(item) for item in value]
    if isinstance(value, list):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    return value


def _drop_none_values(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in payload.items()
        if value is not None
    }


def _recipe_id_from_corpus_ref(corpus_ref: str) -> str:
    recipe_id, _separator, _corpus_id = str(corpus_ref).partition("/")
    resolved_recipe_id = recipe_id.strip()
    if not resolved_recipe_id:
        raise RuntimeError(f"corpus_ref must include a recipe id, got {corpus_ref!r}")
    return resolved_recipe_id


def _normalize_contract_check_mode(mode: str) -> str:
    normalized = str(mode).strip().lower()
    if normalized not in _CONTRACT_CHECK_MODES:
        expected = ", ".join(sorted(_CONTRACT_CHECK_MODES))
        raise ValueError(f"contract_check must be one of {expected}, got {mode!r}")
    return normalized


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(payload), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _read_json_mapping(path: Path, *, context: str) -> dict[str, Any]:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.exists():
        raise RuntimeError(f"{context} is missing: {resolved_path}")
    try:
        payload = json.loads(resolved_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{context} is not valid JSON: {resolved_path}: {exc}") from exc
    return _ensure_mapping(payload, context=context)


def _read_last_jsonl_mapping(path: Path, *, context: str) -> dict[str, Any]:
    resolved_path = path.expanduser().resolve()
    if not resolved_path.exists():
        raise RuntimeError(f"{context} is missing: {resolved_path}")
    last_line: str | None = None
    with resolved_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            stripped = raw_line.strip()
            if stripped:
                last_line = stripped
    if last_line is None:
        raise RuntimeError(f"{context} has no JSON records: {resolved_path}")
    try:
        payload = json.loads(last_line)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{context} contains invalid JSON: {resolved_path}: {exc}") from exc
    return _ensure_mapping(payload, context=context)


def default_pilot_output_root(
    adequacy_id: str,
    *,
    repo_root: Path | None = None,
) -> Path:
    return (
        (repo_root or _repo_root()).expanduser().resolve()
        / "outputs"
        / "research"
        / "adequacy"
        / adequacy_id
        / "pilot"
    )


def _ensure_supported_configuration(*, adequacy_id: str, device: str) -> None:
    if adequacy_id != _SUPPORTED_ADEQUACY_ID:
        raise RuntimeError(
            "the lean adequacy pilot currently supports only "
            f"{_SUPPORTED_ADEQUACY_ID!r}, got {adequacy_id!r}"
        )
    if str(device).strip().lower() != _SUPPORTED_DEVICE:
        raise RuntimeError(
            f"the lean adequacy pilot supports device={_SUPPORTED_DEVICE!r} only, got {device!r}"
        )


__all__ = ["default_pilot_output_root"]
