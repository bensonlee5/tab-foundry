"""Dependency-light read-only helpers for the control-baseline registry."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StrictInt, StrictStr, ValidationError, field_validator

from tab_foundry.bench.registry.storage import load_versioned_registry_payload
from tab_foundry.bench.registry_common import copy_jsonable
from tab_foundry.bench.registry_paths import (  # noqa: F401 - re-exported
    normalize_registry_path_value,
    resolve_registry_path_value,
)
from tab_foundry.repo_paths import repo_root


REGISTRY_SCHEMA = "tab-foundry-control-baselines-v1"
REGISTRY_VERSION = 1
_TOP_LEVEL_KEYS = {"schema", "version", "baselines"}


class _ControlBaselinePayloadModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    @field_validator("*")
    @classmethod
    def _normalize_string(cls, value: Any) -> Any:
        if isinstance(value, str) and not value.strip():
            raise ValueError("must be a non-empty string")
        return value


class _BenchmarkBundlePayload(_ControlBaselinePayloadModel):
    name: StrictStr
    version: StrictInt
    source_path: StrictStr
    task_count: StrictInt
    task_ids: list[StrictInt]


class _TabFoundryMetricsPayload(_ControlBaselinePayloadModel):
    best_step: FiniteFloat
    best_training_time: FiniteFloat
    final_step: FiniteFloat
    final_training_time: FiniteFloat
    best_bpc: FiniteFloat | None = None
    best_bpf: FiniteFloat | None = None
    best_roc_auc: FiniteFloat | None = None
    best_log_loss: FiniteFloat | None = None
    best_brier_score: FiniteFloat | None = None
    best_crps: FiniteFloat | None = None
    best_avg_pinball_loss: FiniteFloat | None = None
    best_picp_90: FiniteFloat | None = None
    final_bpc: FiniteFloat | None = None
    final_bpf: FiniteFloat | None = None
    final_roc_auc: FiniteFloat | None = None
    final_log_loss: FiniteFloat | None = None
    final_brier_score: FiniteFloat | None = None
    final_crps: FiniteFloat | None = None
    final_avg_pinball_loss: FiniteFloat | None = None
    final_picp_90: FiniteFloat | None = None


class _ControlBaselineEntryPayload(_ControlBaselinePayloadModel):
    baseline_id: StrictStr
    experiment: StrictStr
    config_profile: StrictStr
    budget_class: StrictStr
    manifest_path: StrictStr
    seed_set: list[StrictInt] = Field(min_length=1)
    run_dir: StrictStr
    comparison_summary_path: StrictStr
    benchmark_bundle: _BenchmarkBundlePayload
    tab_foundry_metrics: _TabFoundryMetricsPayload


def default_control_baseline_registry_path() -> Path:
    """Return the repo-tracked control-baseline registry path."""

    return repo_root() / "src" / "tab_foundry" / "bench" / "control_baselines_v1.json"


def _validate_baseline_entry(entry: Any, *, baseline_id: str) -> dict[str, Any]:
    if not isinstance(entry, dict):
        raise RuntimeError(f"control baseline entry must be an object: baseline_id={baseline_id}")
    entry_payload = {str(key): value for key, value in entry.items()}
    try:
        validated = _ControlBaselineEntryPayload.model_validate(entry_payload)
    except ValidationError as exc:
        raise RuntimeError(f"control baseline entry {baseline_id!r} is invalid: {exc}") from exc
    if str(validated.baseline_id) != baseline_id:
        raise RuntimeError(
            "control baseline entry baseline_id mismatch: "
            f"expected={baseline_id!r}, actual={validated.baseline_id!r}"
        )
    return entry_payload


def _empty_registry() -> dict[str, Any]:
    return {
        "schema": REGISTRY_SCHEMA,
        "version": REGISTRY_VERSION,
        "baselines": {},
    }


def load_control_baseline_registry(path: Path | None = None) -> dict[str, Any]:
    """Load and minimally validate the control-baseline registry."""

    registry_path = (path or default_control_baseline_registry_path()).expanduser().resolve()
    payload = load_versioned_registry_payload(
        registry_path,
        allow_missing=False,
        empty_payload=_empty_registry(),
        top_level_keys=_TOP_LEVEL_KEYS,
        schema=REGISTRY_SCHEMA,
        version=REGISTRY_VERSION,
        entries_key="baselines",
        registry_label="control baseline registry",
        validate_entry_fn=_validate_baseline_entry,
        entry_label="baseline_id",
    )
    return cast(dict[str, Any], payload)


def load_control_baseline_entry(
    baseline_id: str,
    *,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Load one control-baseline entry by id."""

    registry = load_control_baseline_registry(registry_path)
    baselines = cast(dict[str, dict[str, Any]], registry["baselines"])
    entry = baselines.get(str(baseline_id))
    if entry is None:
        raise RuntimeError(f"unknown control baseline id: {baseline_id}")
    return cast(dict[str, Any], copy_jsonable(entry))
