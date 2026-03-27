"""Shared feature-type vocabulary and normalization helpers."""

from __future__ import annotations

from typing import Any


FEATURE_TYPE_BOOL = "bool"
FEATURE_TYPE_INTEGER = "integer"
FEATURE_TYPE_FLOATING = "floating"
FEATURE_TYPE_STRING_BINARY = "string_binary"
FEATURE_TYPE_UNKNOWN = "unknown"

DEFAULT_FEATURE_TYPE = FEATURE_TYPE_FLOATING
FEATURE_TYPE_VOCAB = (
    FEATURE_TYPE_BOOL,
    FEATURE_TYPE_INTEGER,
    FEATURE_TYPE_FLOATING,
    FEATURE_TYPE_STRING_BINARY,
    FEATURE_TYPE_UNKNOWN,
)
_FEATURE_TYPE_SET = set(FEATURE_TYPE_VOCAB)


def normalize_feature_types(
    feature_types: Any,
    *,
    expected_count: int | None,
    context: str,
) -> list[str]:
    """Validate one feature-type list against the shared vocabulary."""

    if not isinstance(feature_types, list):
        raise ValueError(f"{context} must be a list of feature-type strings")
    if expected_count is not None and len(feature_types) != int(expected_count):
        raise ValueError(
            f"{context} length must equal expected_count={int(expected_count)}, "
            f"got {len(feature_types)}"
        )
    normalized: list[str] = []
    for index, raw_value in enumerate(feature_types):
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError(f"{context}[{index}] must be a non-empty string")
        value = raw_value.strip()
        if value not in _FEATURE_TYPE_SET:
            raise ValueError(
                f"{context}[{index}] must be one of {list(FEATURE_TYPE_VOCAB)}, got {value!r}"
            )
        normalized.append(value)
    return normalized


def resolve_feature_types(
    feature_types: Any,
    *,
    expected_count: int,
    context: str,
) -> list[str]:
    """Return one validated feature-type list, defaulting to all-floating when absent."""

    if feature_types is None:
        return [DEFAULT_FEATURE_TYPE] * int(expected_count)
    return normalize_feature_types(
        feature_types,
        expected_count=expected_count,
        context=context,
    )


def collapse_arrow_feature_type(data_type: Any) -> str:
    """Collapse one Arrow type into the shared parquet-physical-group vocabulary."""

    import pyarrow.types as pa_types

    if pa_types.is_boolean(data_type):
        return FEATURE_TYPE_BOOL
    if pa_types.is_integer(data_type):
        return FEATURE_TYPE_INTEGER
    if pa_types.is_floating(data_type) or pa_types.is_decimal(data_type):
        return FEATURE_TYPE_FLOATING
    if pa_types.is_string(data_type) or pa_types.is_large_string(data_type):
        return FEATURE_TYPE_STRING_BINARY
    if pa_types.is_binary(data_type) or pa_types.is_large_binary(data_type):
        return FEATURE_TYPE_STRING_BINARY
    return FEATURE_TYPE_UNKNOWN


__all__ = [
    "DEFAULT_FEATURE_TYPE",
    "FEATURE_TYPE_BOOL",
    "FEATURE_TYPE_FLOATING",
    "FEATURE_TYPE_INTEGER",
    "FEATURE_TYPE_STRING_BINARY",
    "FEATURE_TYPE_UNKNOWN",
    "FEATURE_TYPE_VOCAB",
    "collapse_arrow_feature_type",
    "normalize_feature_types",
    "resolve_feature_types",
]
