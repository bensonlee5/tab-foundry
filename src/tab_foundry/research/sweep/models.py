"""Typed payload models for canonical system-delta sweep state."""

from __future__ import annotations

from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr, ValidationInfo, field_validator


CATALOG_SCHEMA: Final = "tab-foundry-system-delta-catalog-v1"
SWEEP_INDEX_SCHEMA: Final = "tab-foundry-system-delta-sweep-index-v2"
SWEEP_SCHEMA: Final = "tab-foundry-system-delta-sweep-v1"
SWEEP_QUEUE_SCHEMA: Final = "tab-foundry-system-delta-sweep-queue-v1"


def _require_non_empty_string(value: str, *, context: str) -> str:
    if not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _require_string_list(value: Any, *, field_name: str) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        raise ValueError(f"{field_name} must be a list")
    normalized: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise ValueError(f"{field_name}[{index}] must be a non-empty string")
        normalized.append(str(item))
    return normalized


class _SweepPayloadModel(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True, populate_by_name=True)


class CatalogPayload(_SweepPayloadModel):
    schema_name: Literal["tab-foundry-system-delta-catalog-v1"] = Field(alias="schema")
    deltas: dict[StrictStr, dict[StrictStr, Any]] = Field(min_length=1)


class SweepIndexEntryPayload(_SweepPayloadModel):
    parent_sweep_id: StrictStr | None = None
    status: StrictStr
    anchor_run_id: StrictStr | None = None
    complexity_level: StrictStr
    benchmark_bundle_path: StrictStr
    control_baseline_id: StrictStr
    external_benchmarks: list[str] | None = None

    @field_validator("status", "complexity_level", "benchmark_bundle_path", "control_baseline_id")
    @classmethod
    def _validate_required_strings(cls, value: str, info: ValidationInfo) -> str:
        return _require_non_empty_string(value, context=str(info.field_name))

    @field_validator("parent_sweep_id", "anchor_run_id")
    @classmethod
    def _validate_optional_strings(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        return _require_non_empty_string(value, context=str(info.field_name))

    @field_validator("external_benchmarks", mode="before")
    @classmethod
    def _validate_external_benchmarks(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        return _require_string_list(value, field_name="external_benchmarks")


class SweepIndexPayload(_SweepPayloadModel):
    schema_name: Literal["tab-foundry-system-delta-sweep-index-v2"] = Field(alias="schema")
    sweeps: dict[StrictStr, SweepIndexEntryPayload] = Field(min_length=1)


class SweepPayload(_SweepPayloadModel):
    schema_name: Literal["tab-foundry-system-delta-sweep-v1"] = Field(alias="schema")
    sweep_id: StrictStr
    parent_sweep_id: StrictStr | None = None
    status: StrictStr
    complexity_level: StrictStr
    anchor_run_id: StrictStr | None = None
    benchmark_bundle_path: StrictStr
    control_baseline_id: StrictStr
    external_benchmarks: list[str] | None = None
    training_experiment: StrictStr
    training_config_profile: StrictStr
    surface_role: StrictStr
    comparison_policy: StrictStr
    upstream_reference: dict[StrictStr, Any] = Field(default_factory=dict)
    anchor_surface: dict[StrictStr, Any] = Field(default_factory=dict)
    anchor_context: dict[StrictStr, Any] = Field(default_factory=dict)

    @field_validator(
        "sweep_id",
        "status",
        "complexity_level",
        "benchmark_bundle_path",
        "control_baseline_id",
        "training_experiment",
        "training_config_profile",
        "surface_role",
        "comparison_policy",
    )
    @classmethod
    def _validate_required_strings(cls, value: str, info: ValidationInfo) -> str:
        return _require_non_empty_string(value, context=str(info.field_name))

    @field_validator("parent_sweep_id", "anchor_run_id")
    @classmethod
    def _validate_optional_parent_sweep_id(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        return _require_non_empty_string(value, context=str(info.field_name))

    @field_validator("external_benchmarks", mode="before")
    @classmethod
    def _validate_external_benchmarks(cls, value: Any) -> list[str] | None:
        if value is None:
            return None
        return _require_string_list(value, field_name="external_benchmarks")


class QueueRowPayload(_SweepPayloadModel):
    order: StrictInt
    delta_ref: StrictStr
    status: StrictStr = "ready"
    rationale: StrictStr = ""
    hypothesis: StrictStr = ""
    anchor_delta: StrictStr = ""
    model: dict[StrictStr, Any] = Field(default_factory=dict)
    data: dict[StrictStr, Any] = Field(default_factory=dict)
    preprocessing: dict[StrictStr, Any] = Field(default_factory=dict)
    training: dict[StrictStr, Any] = Field(default_factory=dict)
    parameter_adequacy_plan: list[str] = Field(default_factory=list)
    run_id: StrictStr | None = None
    followup_run_ids: list[str] = Field(default_factory=list)
    decision: StrictStr | None = None
    interpretation_status: StrictStr = "pending"
    confounders: list[str] = Field(default_factory=list)
    next_action: StrictStr = ""
    notes: list[str] = Field(default_factory=list)

    @field_validator("delta_ref", "status", "anchor_delta", "interpretation_status")
    @classmethod
    def _validate_required_strings(cls, value: str, info: ValidationInfo) -> str:
        return _require_non_empty_string(value, context=str(info.field_name))

    @field_validator("run_id", "decision")
    @classmethod
    def _validate_optional_strings(cls, value: str | None, info: ValidationInfo) -> str | None:
        if value is None:
            return None
        return _require_non_empty_string(value, context=str(info.field_name))

    @field_validator("parameter_adequacy_plan", "followup_run_ids", "confounders", "notes", mode="before")
    @classmethod
    def _validate_string_lists(cls, value: Any, info: ValidationInfo) -> list[str]:
        assert info.field_name is not None
        return _require_string_list(value, field_name=str(info.field_name))


class SweepQueuePayload(_SweepPayloadModel):
    schema_name: Literal["tab-foundry-system-delta-sweep-queue-v1"] = Field(alias="schema")
    sweep_id: StrictStr
    rows: list[QueueRowPayload] = Field(min_length=1)

    @field_validator("sweep_id")
    @classmethod
    def _validate_sweep_id(cls, value: str) -> str:
        return _require_non_empty_string(value, context="sweep_id")
