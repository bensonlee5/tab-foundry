"""Shared feature-type vocabulary and normalization helpers."""

from __future__ import annotations

from typing import Any

import torch


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
_FEATURE_TYPE_TO_ID = {name: index for index, name in enumerate(FEATURE_TYPE_VOCAB)}
_FEATURE_TYPE_ALIASES = {
    "num": FEATURE_TYPE_FLOATING,
    # Dagzoo emits generic categorical columns as "cat"; tab-foundry only needs
    # a stable non-floating bucket for these manifest-backed features.
    "cat": FEATURE_TYPE_UNKNOWN,
}


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
        value = _FEATURE_TYPE_ALIASES.get(value, value)
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


def metadata_has_explicit_feature_types(metadata: dict[str, Any]) -> bool:
    """Return whether one task metadata payload carries explicit feature-type lists."""

    members = metadata.get("task_members")
    if isinstance(members, list) and members:
        return any(isinstance(member, dict) and "feature_types" in member for member in members)
    return "feature_types" in metadata


def feature_type_ids_from_resolved(
    resolved_types_by_task: list[list[str]],
    *,
    device: torch.device,
) -> torch.Tensor:
    """Map normalized feature-type strings into stable vocabulary ids."""

    feature_type_ids = [
        [int(_FEATURE_TYPE_TO_ID[value]) for value in feature_types]
        for feature_types in resolved_types_by_task
    ]
    return torch.tensor(feature_type_ids, device=device, dtype=torch.int64)


def feature_type_ids_from_task_metadata(
    metadata: dict[str, Any],
    *,
    batch_size: int,
    num_features: int,
    device: torch.device,
    context: str = "batch.metadata",
) -> torch.Tensor:
    """Resolve explicit feature types from task metadata into vocabulary ids."""

    members = metadata.get("task_members")
    resolved_types_by_task: list[list[str]]
    if isinstance(members, list) and members:
        if len(members) != int(batch_size):
            raise RuntimeError(
                "task-batched feature-type metadata must align with the tensor batch size: "
                f"expected={int(batch_size)}, got={len(members)}"
            )
        resolved_types_by_task = []
        for index, member in enumerate(members):
            if not isinstance(member, dict):
                raise RuntimeError(
                    "task-batched feature-type metadata members must be objects, "
                    f"got task_members[{index}]={type(member).__name__}"
                )
            feature_types = member.get("feature_types")
            if feature_types is None:
                raise RuntimeError(f"{context}.task_members[{index}].feature_types is required")
            try:
                resolved_types_by_task.append(
                    normalize_feature_types(
                        feature_types,
                        expected_count=int(num_features),
                        context=f"{context}.task_members[{index}].feature_types",
                    )
                )
            except ValueError as exc:
                raise RuntimeError(str(exc)) from exc
    else:
        if int(batch_size) != 1:
            raise RuntimeError(
                "task-batched feature-type metadata requires one feature_types list per task"
            )
        feature_types = metadata.get("feature_types")
        if feature_types is None:
            raise RuntimeError(f"{context}.feature_types is required")
        try:
            resolved_types_by_task = [
                normalize_feature_types(
                    feature_types,
                    expected_count=int(num_features),
                    context=f"{context}.feature_types",
                )
            ]
        except ValueError as exc:
            raise RuntimeError(str(exc)) from exc
    return feature_type_ids_from_resolved(
        resolved_types_by_task,
        device=device,
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
    "feature_type_ids_from_resolved",
    "feature_type_ids_from_task_metadata",
    "metadata_has_explicit_feature_types",
    "normalize_feature_types",
    "resolve_feature_types",
]
