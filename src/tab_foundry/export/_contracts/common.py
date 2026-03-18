"""Shared coercion and canonicalization helpers for export contracts."""

from __future__ import annotations

from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from tab_foundry.input_normalization import SUPPORTED_INPUT_NORMALIZATION_MODES
from tab_foundry.model.spec import SUPPORTED_MANY_CLASS_TRAIN_MODES

from .models import SCHEMA_VERSION_V3


_ContractsPayloadT = TypeVar("_ContractsPayloadT", bound=BaseModel)


def _validate_payload_model(
    payload_model: type[_ContractsPayloadT],
    payload: Any,
    *,
    context: str,
) -> _ContractsPayloadT:
    try:
        return payload_model.model_validate(payload)
    except ValidationError as exc:
        raise ValueError(f"{context} is invalid: {exc}") from exc


def read_json_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload at {path} must be an object")
    return payload


def canonicalize_v3_manifest_payload(payload: dict[str, Any]) -> bytes:
    schema_version = _as_str(payload.get("schema_version"), context="manifest.schema_version")
    if schema_version != SCHEMA_VERSION_V3:
        raise ValueError(
            "canonicalize_v3_manifest_payload requires a tab-foundry-export-v3 payload, "
            f"got {schema_version!r}"
        )
    canonical_payload = dict(payload)
    canonical_payload.pop("manifest_sha256", None)
    try:
        return json.dumps(
            canonical_payload,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except ValueError as exc:
        raise ValueError("v3 manifest contains non-canonical JSON values") from exc


def compute_v3_manifest_sha256(payload: dict[str, Any]) -> str:
    return sha256(canonicalize_v3_manifest_payload(payload)).hexdigest()


def _require_keys(
    payload: dict[str, Any],
    *,
    keys: set[str],
    context: str,
    optional_keys: set[str] | None = None,
) -> None:
    optional = optional_keys if optional_keys is not None else set()
    actual = set(payload.keys())
    missing = sorted(keys - actual)
    extra = sorted(actual - (keys | optional))
    if missing or extra:
        details: list[str] = []
        if missing:
            details.append(f"missing={missing}")
        if extra:
            details.append(f"extra={extra}")
        raise ValueError(f"{context} keys mismatch: {', '.join(details)}")


def _as_int(value: Any, *, context: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{context} must be int")
    return int(value)


def _as_str(value: Any, *, context: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{context} must be str")
    return value


def _as_bool(value: Any, *, context: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{context} must be bool")
    return value


def _as_float(value: Any, *, context: str) -> float:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{context} must be float-compatible")
    return float(value)


def _validate_hex_digest(value: Any, *, context: str) -> str:
    digest = _as_str(value, context=context)
    if len(digest) != 64:
        raise ValueError(f"{context} must be a 64-char hex digest")
    return digest


def _validate_input_normalization(value: Any, *, context: str) -> str:
    input_normalization = _as_str(value, context=context).strip().lower()
    if input_normalization not in SUPPORTED_INPUT_NORMALIZATION_MODES:
        raise ValueError(
            f"{context} must be one of {SUPPORTED_INPUT_NORMALIZATION_MODES}, "
            f"got {input_normalization!r}"
        )
    return input_normalization


def _validate_many_class_train_mode(value: Any, *, context: str) -> str:
    many_class_train_mode = _as_str(value, context=context).strip().lower()
    if many_class_train_mode not in SUPPORTED_MANY_CLASS_TRAIN_MODES:
        raise ValueError(
            f"{context} must be one of {SUPPORTED_MANY_CLASS_TRAIN_MODES}, "
            f"got {many_class_train_mode!r}"
        )
    return many_class_train_mode


def _validate_created_at_utc(value: Any) -> str:
    created_at_utc = _as_str(value, context="manifest.created_at_utc")
    try:
        datetime.fromisoformat(created_at_utc.replace("Z", "+00:00"))
    except ValueError as exc:
        raise ValueError("manifest.created_at_utc must be ISO8601") from exc
    return created_at_utc
