"""Dependency-light read-only helpers for hardware architecture baselines."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, FiniteFloat, StrictInt, StrictStr, ValidationError, field_validator

from tab_foundry.registry.common import copy_jsonable
from tab_foundry.registry.paths import (  # noqa: F401 - re-exported
    normalize_registry_path_value,
    resolve_registry_path_value,
)
from tab_foundry.registry.storage import load_json_object_payload
from tab_foundry.repo_paths import repo_root


REGISTRY_SCHEMA = "tab-foundry-hardware-architecture-baselines-v1"
REGISTRY_VERSION = 1
_TOP_LEVEL_KEYS = {"schema", "version", "baselines"}


class _RegistryPayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    @field_validator("*")
    @classmethod
    def _normalize_string(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class _PreferredArchitecturePayload(_RegistryPayloadModel):
    arch: StrictStr
    d_icl: StrictInt
    head_hidden_dim: StrictInt
    tficl_n_heads: StrictInt
    tficl_n_layers: StrictInt
    architecture: dict[StrictStr, Any] | None = None
    build_spec: dict[StrictStr, Any] | None = None


class _RuntimeSummaryPayload(_RegistryPayloadModel):
    peak_vram_allocated: StrictInt | None = None
    peak_vram_reserved: StrictInt | None = None
    throughput_examples_per_second: FiniteFloat | None = None
    throughput_tokens_per_second: FiniteFloat | None = None
    non_train_overhead_seconds: FiniteFloat | None = None


class _SurfaceLabelsPayload(_RegistryPayloadModel):
    model: StrictStr
    data: StrictStr
    preprocessing: StrictStr
    training: StrictStr | None = None


class _HardwareArchitectureBaselineEntryPayload(_RegistryPayloadModel):
    baseline_id: StrictStr
    hardware_profile_id: StrictStr
    gpu_class: StrictStr
    vram_class_gb: StrictInt
    track: StrictStr
    surface_role: StrictStr
    runtime_profile: StrictStr
    config_profile: StrictStr
    benchmark_manifest_path: StrictStr
    control_baseline_id: StrictStr | None = None
    sweep_id: StrictStr | None = None
    surface_labels: _SurfaceLabelsPayload | None = None
    formal_anchor_run_id: StrictStr
    baseline_run_id: StrictStr
    preferred_run_id: StrictStr
    preferred_delta_ref: StrictStr | None = None
    preferred_architecture: _PreferredArchitecturePayload
    objective_metric: StrictStr
    selection_rule: StrictStr
    evidence_run_ids: list[StrictStr]
    decision: StrictStr
    rationale: StrictStr
    preferred_runtime_summary: _RuntimeSummaryPayload | None = None


def default_hardware_architecture_registry_path() -> Path:
    """Return the repo-tracked hardware architecture registry path."""

    return repo_root() / "src" / "tab_foundry" / "bench" / "hardware_architecture_baselines_v1.json"


def _validate_baseline_entry(entry: Any, *, baseline_id: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise RuntimeError(
            f"hardware architecture baseline entry must be an object: baseline_id={baseline_id}"
        )
    entry_payload = {str(key): value for key, value in entry.items()}
    try:
        validated = _HardwareArchitectureBaselineEntryPayload.model_validate(entry_payload)
    except ValidationError as exc:
        raise RuntimeError(
            f"hardware architecture baseline entry {baseline_id!r} is invalid: {exc}"
        ) from exc
    if str(validated.baseline_id) != baseline_id:
        raise RuntimeError(
            "hardware architecture baseline entry baseline_id mismatch: "
            f"expected={baseline_id!r}, actual={validated.baseline_id!r}"
        )
    return entry_payload


def _empty_registry() -> dict[str, Any]:
    return {
        "schema": REGISTRY_SCHEMA,
        "version": REGISTRY_VERSION,
        "baselines": {},
    }


def load_hardware_architecture_registry(path: Path | None = None) -> dict[str, Any]:
    """Load and minimally validate the hardware architecture baseline registry."""

    registry_path = (path or default_hardware_architecture_registry_path()).expanduser().resolve()
    payload = load_json_object_payload(
        registry_path,
        allow_missing=False,
        empty_payload=_empty_registry(),
        payload_label="hardware architecture baseline registry",
    )
    actual_keys = set(payload.keys())
    if actual_keys != _TOP_LEVEL_KEYS:
        raise RuntimeError(
            "hardware architecture baseline registry keys mismatch: "
            f"missing={sorted(_TOP_LEVEL_KEYS - actual_keys)}, "
            f"extra={sorted(actual_keys - _TOP_LEVEL_KEYS)}"
        )
    if payload["schema"] != REGISTRY_SCHEMA:
        raise RuntimeError(
            "hardware architecture baseline registry schema mismatch: "
            f"expected={REGISTRY_SCHEMA!r}, actual={payload['schema']!r}"
        )
    if int(payload["version"]) != REGISTRY_VERSION:
        raise RuntimeError(
            "hardware architecture baseline registry version mismatch: "
            f"expected={REGISTRY_VERSION}, actual={payload['version']}"
        )
    baselines = payload["baselines"]
    if not isinstance(baselines, dict):
        raise RuntimeError("hardware architecture baseline registry baselines must be an object")
    for baseline_id, entry in baselines.items():
        if not isinstance(baseline_id, str) or not baseline_id.strip():
            raise RuntimeError(
                "hardware architecture baseline registry baseline_id ids must be non-empty strings"
            )
        _validate_baseline_entry(entry, baseline_id=str(baseline_id))
    return {
        "schema": REGISTRY_SCHEMA,
        "version": REGISTRY_VERSION,
        "baselines": {str(key): value for key, value in cast(dict[str, Any], baselines).items()},
    }


def load_hardware_architecture_baseline_entry(
    baseline_id: str,
    *,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Load one hardware architecture baseline entry by id."""

    registry = load_hardware_architecture_registry(registry_path)
    baselines = cast(dict[str, dict[str, Any]], registry["baselines"])
    entry = baselines.get(str(baseline_id))
    if entry is None:
        raise RuntimeError(f"unknown hardware architecture baseline id: {baseline_id}")
    return cast(dict[str, Any], copy_jsonable(entry))
