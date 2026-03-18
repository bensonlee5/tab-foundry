"""Validation helpers for sweep-aware system-delta tooling."""

from __future__ import annotations

from typing import Any, Mapping, cast


_QUEUE_PROSE_FIELDS = (
    "notes",
    "confounders",
    "parameter_adequacy_plan",
    "adequacy_knobs",
)


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


def validate_prose_fields(
    payload: Mapping[str, Any],
    *,
    context: str,
    field_names: tuple[str, ...] = _QUEUE_PROSE_FIELDS,
) -> None:
    for field_name in field_names:
        _ = ensure_string_list(payload.get(field_name, []), context=f"{context}.{field_name}")
