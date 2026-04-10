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
    sandwich_heads: StrictInt | None = None
    sandwich_layers: StrictInt | None = None
    architecture: dict[StrictStr, Any] | None = None
    build_spec: dict[StrictStr, Any] | None = None


class _RuntimeSummaryPayload(_RegistryPayloadModel):
    peak_vram_allocated: StrictInt | None = None
    peak_vram_reserved: StrictInt | None = None
    throughput_examples_per_second: FiniteFloat | None = None
    throughput_tokens_per_second: FiniteFloat | None = None
    non_train_overhead_seconds: FiniteFloat | None = None


class _BenchmarkTimingPayload(_RegistryPayloadModel):
    wall_elapsed_seconds: FiniteFloat | None = None
    mean_checkpoint_elapsed_seconds: FiniteFloat | None = None
    max_checkpoint_elapsed_seconds: FiniteFloat | None = None
    attempted_checkpoint_count: StrictInt | None = None
    successful_checkpoint_count: StrictInt | None = None
    failed_checkpoint_count: StrictInt | None = None
    requested_device: StrictStr | None = None
    resolved_device: StrictStr | None = None
    host_fingerprint: StrictStr | None = None


class _InferenceTimingPayload(_RegistryPayloadModel):
    fixture_id: StrictStr
    requested_device: StrictStr | None = None
    resolved_device: StrictStr | None = None
    device_type: StrictStr | None = None
    raw_device_name: StrictStr | None = None
    gpu_class: StrictStr | None = None
    vram_class_gb: StrictInt | None = None
    hardware_profile_id: StrictStr | None = None
    warmup_iterations: StrictInt
    measured_iterations: StrictInt
    n_train: StrictInt
    n_test: StrictInt
    n_features: StrictInt
    num_classes: StrictInt
    mean_ms: FiniteFloat | None = None
    p50_ms: FiniteFloat | None = None
    p95_ms: FiniteFloat | None = None
    max_ms: FiniteFloat | None = None
    total_measured_seconds: FiniteFloat | None = None


class _SurfaceLabelsPayload(_RegistryPayloadModel):
    model: StrictStr
    data: StrictStr
    preprocessing: StrictStr
    training: StrictStr | None = None


class _ConstraintFormulaPayload(_RegistryPayloadModel):
    expression: StrictStr
    fit_kind: StrictStr
    coefficients: dict[StrictStr, Any] | None = None
    evidence_run_ids: list[StrictStr]


class _ConstraintPointPayload(_RegistryPayloadModel):
    total_params: StrictInt | None = None
    reserved_vram_gb: FiniteFloat | None = None
    train_wall_seconds: FiniteFloat | None = None
    benchmark_wall_seconds: FiniteFloat | None = None
    inference_mean_ms: FiniteFloat | None = None
    inference_p50_ms: FiniteFloat | None = None
    inference_p95_ms: FiniteFloat | None = None


class _ConstraintObservedPayload(_ConstraintPointPayload):
    run_id: StrictStr | None = None
    delta_ref: StrictStr | None = None
    health: StrictStr | None = None


class _ConstraintHeadroomPayload(_RegistryPayloadModel):
    hardware_vram_ceiling_gb: FiniteFloat
    reserved_vram_gb_to_ceiling: FiniteFloat | None = None
    train_wall_seconds_delta_vs_baseline: FiniteFloat | None = None
    benchmark_wall_seconds_delta_vs_baseline: FiniteFloat | None = None
    inference_mean_ms_delta_vs_baseline: FiniteFloat | None = None
    inference_p50_ms_delta_vs_baseline: FiniteFloat | None = None
    inference_p95_ms_delta_vs_baseline: FiniteFloat | None = None


class _ConstraintRowPayload(_RegistryPayloadModel):
    row: StrictStr
    d_icl: StrictInt
    sandwich_layers: StrictInt
    effective_size: StrictInt
    predicted: _ConstraintPointPayload
    observed: _ConstraintObservedPayload | None = None
    headroom: _ConstraintHeadroomPayload


class _ConstraintModelPayload(_RegistryPayloadModel):
    effective_size_expression: StrictStr
    formulas: dict[StrictStr, _ConstraintFormulaPayload]
    evidence_run_ids: list[StrictStr]
    baseline_row: StrictStr
    rows: list[_ConstraintRowPayload]


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
    preferred_benchmark_timing: _BenchmarkTimingPayload | None = None
    preferred_inference_timing: _InferenceTimingPayload | None = None
    constraint_model: _ConstraintModelPayload | None = None


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
