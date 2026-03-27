"""Validation helpers for sweep-aware system-delta tooling."""

from __future__ import annotations

from typing import Any, Mapping, cast

from tab_foundry.external_benchmarks import (
    EXTERNAL_BENCHMARK_NANOTABPFN,
    EXTERNAL_BENCHMARK_TABICLV2,
    normalize_external_benchmarks as _normalize_external_benchmarks,
)


_QUEUE_PROSE_FIELDS = (
    "notes",
    "confounders",
    "parameter_adequacy_plan",
    "adequacy_knobs",
)
DEFAULT_LEGACY_SWEEP_EXTERNAL_BENCHMARKS = (EXTERNAL_BENCHMARK_NANOTABPFN,)
DEFAULT_NEW_SWEEP_EXTERNAL_BENCHMARKS = (EXTERNAL_BENCHMARK_TABICLV2,)


def ensure_non_empty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{context} must be a non-empty string")
    return str(value)


def ensure_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must be a mapping")
    return cast(dict[str, Any], value)


def ensure_rows(value: Any, *, context: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{context} must be a non-empty list")
    if not all(isinstance(item, dict) for item in value):
        raise RuntimeError(f"{context} must contain only mappings")
    return cast(list[dict[str, Any]], value)


def ensure_string_list(value: Any, *, context: str) -> list[str]:
    if not isinstance(value, list):
        raise RuntimeError(f"{context} must be a list")
    normalized: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise RuntimeError(f"{context}[{index}] must be a non-empty string")
        normalized.append(str(item))
    return normalized


def ensure_external_benchmarks(
    value: Any,
    *,
    context: str,
    default: tuple[str, ...],
    allow_empty: bool = False,
) -> list[str]:
    if value is None:
        return list(
            _normalize_external_benchmarks(
                default,
                context=context,
                allow_empty=allow_empty,
            )
        )
    if not isinstance(value, list):
        raise RuntimeError(f"{context} must be a list")
    return list(
        _normalize_external_benchmarks(
            value,
            context=context,
            allow_empty=allow_empty,
        )
    )


def resolve_sweep_external_benchmarks(sweep: Mapping[str, Any]) -> list[str]:
    return ensure_external_benchmarks(
        sweep.get("external_benchmarks"),
        context="sweep.external_benchmarks",
        default=DEFAULT_LEGACY_SWEEP_EXTERNAL_BENCHMARKS,
        allow_empty=True,
    )


def validate_prose_fields(
    payload: Mapping[str, Any],
    *,
    context: str,
    field_names: tuple[str, ...] = _QUEUE_PROSE_FIELDS,
) -> None:
    for field_name in field_names:
        _ = ensure_string_list(payload.get(field_name, []), context=f"{context}.{field_name}")
