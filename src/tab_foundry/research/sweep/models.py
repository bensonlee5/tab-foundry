"""Typed payload models for canonical system-delta sweep state."""

from __future__ import annotations

from typing import Any, Final, Literal

from pydantic import AliasChoices, BaseModel, ConfigDict, Field, StrictInt, StrictStr, ValidationInfo, field_validator

from tab_foundry.external_benchmarks import (
    EXTERNAL_BENCHMARK_NANOTABPFN,
    EXTERNAL_BENCHMARK_TABICLV2,
    normalize_external_benchmarks,
)


CATALOG_SCHEMA: Final = "tab-foundry-system-delta-catalog-v1"
SWEEP_INDEX_SCHEMA: Final = "tab-foundry-system-delta-sweep-index-v2"
SWEEP_SCHEMA: Final = "tab-foundry-system-delta-sweep-v1"
SWEEP_QUEUE_SCHEMA: Final = "tab-foundry-system-delta-sweep-queue-v1"
MATERIALIZED_QUEUE_SCHEMA: Final = "tab-foundry-system-delta-queue-v1"
RESOLVED_QUEUE_SCHEMA: Final = "tab-foundry-system-delta-resolved-queue-v1"
DEFAULT_LEGACY_SWEEP_EXTERNAL_BENCHMARKS: Final = (EXTERNAL_BENCHMARK_NANOTABPFN,)
DEFAULT_NEW_SWEEP_EXTERNAL_BENCHMARKS: Final = (EXTERNAL_BENCHMARK_TABICLV2,)


def _require_non_empty_string(value: str, *, context: str) -> str:
    if not value.strip():
        raise ValueError(f"{context} must be a non-empty string")
    return value


def _require_optional_string(value: str | None, *, context: str) -> str | None:
    if value is None:
        return None
    return _require_non_empty_string(value, context=context)


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


def _normalize_optional_external_benchmarks(
    value: Any,
    *,
    field_name: str,
) -> list[str] | None:
    if value is None:
        return None
    return list(
        normalize_external_benchmarks(
            value,
            context=field_name,
            allow_empty=True,
        )
    )


class _SweepPayloadModel(BaseModel):
    model_config = ConfigDict(extra="allow", strict=True, populate_by_name=True, validate_assignment=True)

    @classmethod
    def _field_name_for_key(cls, key: str) -> str | None:
        if key in cls.model_fields:
            return key
        for field_name, field_info in cls.model_fields.items():
            if field_info.alias == key:
                return field_name
        return None

    def to_payload_dict(self) -> dict[str, Any]:
        return self.model_dump(
            by_alias=True,
            exclude_none=False,
            exclude_unset=False,
        )

    def __getitem__(self, key: str) -> Any:
        field_name = self._field_name_for_key(key)
        if field_name is not None:
            return getattr(self, field_name)
        if self.model_extra is not None and key in self.model_extra:
            return self.model_extra[key]
        raise KeyError(key)

    def get(self, key: str, default: Any = None) -> Any:
        try:
            return self[key]
        except KeyError:
            return default

    def __setitem__(self, key: str, value: Any) -> None:
        field_name = self._field_name_for_key(key)
        if field_name is not None:
            setattr(self, field_name, value)
            return
        setattr(self, key, value)

    def keys(self) -> Any:
        return self.to_payload_dict().keys()

    def items(self) -> Any:
        return self.to_payload_dict().items()

    def values(self) -> Any:
        return self.to_payload_dict().values()

    def __len__(self) -> int:
        return len(self.to_payload_dict())


class CatalogDeltaPayload(_SweepPayloadModel):
    dimension_family: StrictStr | None = None
    family: StrictStr | None = None
    binary_applicable: bool = False
    description: StrictStr | None = None
    upstream_delta: StrictStr | None = None
    expected_effect: StrictStr | None = None
    legacy_stage_alias: StrictStr | None = None
    adequacy_knobs: list[str] = Field(default_factory=list)
    default_effective_surface: dict[StrictStr, Any] = Field(default_factory=dict)
    parameter_adequacy_policy: dict[StrictStr, Any] = Field(default_factory=dict)
    applicability_guards: list[dict[StrictStr, Any]] = Field(default_factory=list)
    default_initial_status: StrictStr | None = None
    default_initial_interpretation_status: StrictStr | None = None
    default_next_action: StrictStr | None = None

    @field_validator(
        "dimension_family",
        "family",
        "description",
        "upstream_delta",
        "expected_effect",
        "legacy_stage_alias",
        "default_initial_status",
        "default_initial_interpretation_status",
        "default_next_action",
    )
    @classmethod
    def _validate_optional_strings(cls, value: str | None, info: ValidationInfo) -> str | None:
        assert info.field_name is not None
        return _require_optional_string(value, context=str(info.field_name))

    @field_validator("adequacy_knobs", mode="before")
    @classmethod
    def _validate_adequacy_knobs(cls, value: Any) -> list[str]:
        return _require_string_list(value, field_name="adequacy_knobs")


class CatalogPayload(_SweepPayloadModel):
    schema_name: Literal["tab-foundry-system-delta-catalog-v1"] = Field(alias="schema")
    deltas: dict[StrictStr, CatalogDeltaPayload] = Field(min_length=1)


class SweepIndexEntryPayload(_SweepPayloadModel):
    parent_sweep_id: StrictStr | None = None
    status: StrictStr
    anchor_run_id: StrictStr | None = None
    complexity_level: StrictStr
    benchmark_manifest_path: StrictStr = Field(
        validation_alias=AliasChoices("benchmark_manifest_path", "benchmark_bundle_path")
    )
    control_baseline_id: StrictStr
    external_benchmarks: list[str] | None = None

    @field_validator("status", "complexity_level", "benchmark_manifest_path", "control_baseline_id")
    @classmethod
    def _validate_required_strings(cls, value: str, info: ValidationInfo) -> str:
        assert info.field_name is not None
        return _require_non_empty_string(value, context=str(info.field_name))

    @field_validator("parent_sweep_id", "anchor_run_id")
    @classmethod
    def _validate_optional_strings(cls, value: str | None, info: ValidationInfo) -> str | None:
        assert info.field_name is not None
        return _require_optional_string(value, context=str(info.field_name))

    @field_validator("external_benchmarks", mode="before")
    @classmethod
    def _validate_external_benchmarks(cls, value: Any) -> list[str] | None:
        return _normalize_optional_external_benchmarks(value, field_name="external_benchmarks")


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
    benchmark_manifest_path: StrictStr = Field(
        validation_alias=AliasChoices("benchmark_manifest_path", "benchmark_bundle_path")
    )
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
        "benchmark_manifest_path",
        "control_baseline_id",
        "training_experiment",
        "training_config_profile",
        "surface_role",
        "comparison_policy",
    )
    @classmethod
    def _validate_required_strings(cls, value: str, info: ValidationInfo) -> str:
        assert info.field_name is not None
        return _require_non_empty_string(value, context=str(info.field_name))

    @field_validator("parent_sweep_id", "anchor_run_id")
    @classmethod
    def _validate_optional_parent_sweep_id(cls, value: str | None, info: ValidationInfo) -> str | None:
        assert info.field_name is not None
        return _require_optional_string(value, context=str(info.field_name))

    @field_validator("external_benchmarks", mode="before")
    @classmethod
    def _validate_external_benchmarks(cls, value: Any) -> list[str] | None:
        return _normalize_optional_external_benchmarks(value, field_name="external_benchmarks")


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
    execution_policy: StrictStr = "benchmark_full"
    run_id: StrictStr | None = None
    followup_run_ids: list[str] = Field(default_factory=list)
    decision: StrictStr | None = None
    interpretation_status: StrictStr = "pending"
    confounders: list[str] = Field(default_factory=list)
    next_action: StrictStr = ""
    notes: list[str] = Field(default_factory=list)
    dynamic_model_overrides: dict[StrictStr, Any] | None = None
    screen_metrics: dict[StrictStr, Any] | None = None
    benchmark_metrics: dict[StrictStr, Any] | None = None
    parent_delta_ref: StrictStr | None = None

    @field_validator(
        "delta_ref",
        "status",
        "anchor_delta",
        "interpretation_status",
        "execution_policy",
    )
    @classmethod
    def _validate_required_strings(cls, value: str, info: ValidationInfo) -> str:
        assert info.field_name is not None
        return _require_non_empty_string(value, context=str(info.field_name))

    @field_validator("run_id", "decision", "parent_delta_ref")
    @classmethod
    def _validate_optional_strings(cls, value: str | None, info: ValidationInfo) -> str | None:
        assert info.field_name is not None
        return _require_optional_string(value, context=str(info.field_name))

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


class MaterializedQueueRowPayload(_SweepPayloadModel):
    order: StrictInt
    delta_id: StrictStr
    status: StrictStr
    rationale: StrictStr = ""
    hypothesis: StrictStr = ""
    anchor_delta: StrictStr = ""
    model: dict[StrictStr, Any] = Field(default_factory=dict)
    data: dict[StrictStr, Any] = Field(default_factory=dict)
    preprocessing: dict[StrictStr, Any] = Field(default_factory=dict)
    training: dict[StrictStr, Any] = Field(default_factory=dict)
    parameter_adequacy_plan: list[str] = Field(default_factory=list)
    execution_policy: StrictStr = "benchmark_full"
    run_id: StrictStr | None = None
    followup_run_ids: list[str] = Field(default_factory=list)
    decision: StrictStr | None = None
    interpretation_status: StrictStr = "pending"
    confounders: list[str] = Field(default_factory=list)
    next_action: StrictStr = ""
    notes: list[str] = Field(default_factory=list)
    dynamic_model_overrides: dict[StrictStr, Any] | None = None
    screen_metrics: dict[StrictStr, Any] | None = None
    benchmark_metrics: dict[StrictStr, Any] | None = None
    parent_delta_ref: StrictStr | None = None
    dimension_family: StrictStr | None = None
    family: StrictStr | None = None
    binary_applicable: bool | None = None
    description: StrictStr | None = None
    upstream_delta: StrictStr | None = None
    entangled_legacy_stage: StrictStr | None = None
    expected_effect: StrictStr | None = None
    adequacy_knobs: list[str] = Field(default_factory=list)
    parameter_adequacy_policy: dict[StrictStr, Any] = Field(default_factory=dict)
    applicability_guards: list[dict[StrictStr, Any]] = Field(default_factory=list)

    @field_validator(
        "delta_id",
        "status",
        "execution_policy",
    )
    @classmethod
    def _validate_required_strings(cls, value: str, info: ValidationInfo) -> str:
        assert info.field_name is not None
        return _require_non_empty_string(value, context=str(info.field_name))

    @field_validator(
        "run_id",
        "decision",
        "parent_delta_ref",
        "dimension_family",
        "family",
        "description",
        "upstream_delta",
        "entangled_legacy_stage",
        "expected_effect",
    )
    @classmethod
    def _validate_optional_strings(cls, value: str | None, info: ValidationInfo) -> str | None:
        assert info.field_name is not None
        return _require_optional_string(value, context=str(info.field_name))

    @field_validator(
        "parameter_adequacy_plan",
        "followup_run_ids",
        "confounders",
        "notes",
        "adequacy_knobs",
        mode="before",
    )
    @classmethod
    def _validate_string_lists(cls, value: Any, info: ValidationInfo) -> list[str]:
        assert info.field_name is not None
        return _require_string_list(value, field_name=str(info.field_name))


class MaterializedQueuePayload(_SweepPayloadModel):
    schema_name: Literal["tab-foundry-system-delta-queue-v1"] = Field(alias="schema")
    generated_from_sweep_id: StrictStr
    catalog_path: StrictStr | None = None
    canonical_sweep_path: StrictStr
    canonical_queue_path: StrictStr
    canonical_matrix_path: StrictStr
    sweep_id: StrictStr
    parent_sweep_id: StrictStr | None = None
    sweep_status: StrictStr | None = None
    complexity_level: StrictStr | None = None
    anchor_run_id: StrictStr | None = None
    benchmark_manifest_path: StrictStr = Field(
        validation_alias=AliasChoices("benchmark_manifest_path", "benchmark_bundle_path")
    )
    control_baseline_id: StrictStr
    external_benchmarks: list[str] = Field(default_factory=list)
    training_experiment: StrictStr
    training_config_profile: StrictStr
    surface_role: StrictStr
    comparison_policy: StrictStr
    upstream_reference: dict[StrictStr, Any] = Field(default_factory=dict)
    anchor_surface: dict[StrictStr, Any] = Field(default_factory=dict)
    anchor_context: dict[StrictStr, Any] = Field(default_factory=dict)
    rows: list[MaterializedQueueRowPayload] = Field(min_length=1)

    @field_validator(
        "generated_from_sweep_id",
        "canonical_sweep_path",
        "canonical_queue_path",
        "canonical_matrix_path",
        "sweep_id",
        "benchmark_manifest_path",
        "control_baseline_id",
        "training_experiment",
        "training_config_profile",
        "surface_role",
        "comparison_policy",
    )
    @classmethod
    def _validate_required_strings(cls, value: str, info: ValidationInfo) -> str:
        assert info.field_name is not None
        return _require_non_empty_string(value, context=str(info.field_name))

    @field_validator("catalog_path", "parent_sweep_id", "sweep_status", "complexity_level", "anchor_run_id")
    @classmethod
    def _validate_optional_strings(cls, value: str | None, info: ValidationInfo) -> str | None:
        assert info.field_name is not None
        return _require_optional_string(value, context=str(info.field_name))

    @field_validator("external_benchmarks", mode="before")
    @classmethod
    def _validate_external_benchmarks(cls, value: Any) -> list[str]:
        if value is None:
            return []
        return list(
            normalize_external_benchmarks(
                value,
                context="external_benchmarks",
                allow_empty=True,
            )
        )


class ResolvedQueueRowPayload(MaterializedQueueRowPayload):
    resolved_surface: dict[StrictStr, Any] = Field(default_factory=dict)
    resolved_surface_fingerprint: StrictStr

    @field_validator("resolved_surface_fingerprint")
    @classmethod
    def _validate_resolved_surface_fingerprint(cls, value: str) -> str:
        return _require_non_empty_string(value, context="resolved_surface_fingerprint")


class ResolvedQueuePayload(MaterializedQueuePayload):
    schema_name: Literal["tab-foundry-system-delta-resolved-queue-v1"] = Field(alias="schema")  # type: ignore[assignment]
    canonical_resolved_queue_path: StrictStr
    inputs_fingerprint: StrictStr
    rows: list[ResolvedQueueRowPayload] = Field(min_length=1)  # type: ignore[assignment]

    @field_validator("canonical_resolved_queue_path", "inputs_fingerprint")
    @classmethod
    def _validate_resolved_payload_strings(cls, value: str, info: ValidationInfo) -> str:
        assert info.field_name is not None
        return _require_non_empty_string(value, context=str(info.field_name))
