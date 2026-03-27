"""Shared canonicalization helpers for export contracts."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

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
    schema_version = payload.get("schema_version")
    if not isinstance(schema_version, str):
        raise ValueError("manifest.schema_version must be str")
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
