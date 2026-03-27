"""Catalog and sweep metadata loaders for system-delta sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import ValidationError

from .models import (
    CatalogPayload,
    SweepIndexPayload,
    SweepPayload,
    SweepQueuePayload,
)
from .paths_io import (
    default_catalog_path,
    default_sweep_index_path,
    load_yaml_mapping,
    sweep_metadata_path,
    sweep_queue_path,
)
from .validation import ensure_non_empty_string


def _validate_payload(payload_model: type[Any], payload: dict[str, Any], *, context: str) -> dict[str, Any]:
    try:
        validated = payload_model.model_validate(payload)
    except ValidationError as exc:
        raise RuntimeError(f"{context} is invalid: {exc}") from exc
    return validated.model_dump(
        by_alias=True,
        exclude_none=False,
        exclude_unset=True,
    )


def load_system_delta_catalog(path: Path | None = None) -> dict[str, Any]:
    catalog = load_yaml_mapping(path or default_catalog_path(), context="system delta catalog")
    return _validate_payload(CatalogPayload, catalog, context="system delta catalog")


def load_system_delta_index(path: Path | None = None) -> dict[str, Any]:
    index = load_yaml_mapping(path or default_sweep_index_path(), context="system delta sweep index")
    return _validate_payload(SweepIndexPayload, index, context="system delta sweep index")


def resolve_selected_sweep_id(
    sweep_id: str | None,
    *,
    index_path: Path | None = None,
) -> str:
    del index_path
    if sweep_id is None:
        raise RuntimeError("sweep_id is required; the repo no longer tracks an active sweep")
    return ensure_non_empty_string(sweep_id, context="sweep_id")


def load_system_delta_sweep(
    sweep_id: str | None = None,
    *,
    index_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    resolved_sweep_id = resolve_selected_sweep_id(sweep_id, index_path=index_path)
    raw_sweep = load_yaml_mapping(
        sweep_metadata_path(resolved_sweep_id, sweeps_root=sweeps_root),
        context=f"system delta sweep {resolved_sweep_id!r}",
    )
    sweep = _validate_payload(
        SweepPayload,
        raw_sweep,
        context=f"system delta sweep {resolved_sweep_id!r}",
    )
    if ensure_non_empty_string(sweep.get("sweep_id"), context="sweep.sweep_id") != resolved_sweep_id:
        raise RuntimeError(
            f"system delta sweep id mismatch: expected {resolved_sweep_id!r}, got {sweep.get('sweep_id')!r}"
        )
    return sweep


def load_system_delta_queue_instance(
    sweep_id: str | None = None,
    *,
    index_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    resolved_sweep_id = resolve_selected_sweep_id(sweep_id, index_path=index_path)
    raw_queue = load_yaml_mapping(
        sweep_queue_path(resolved_sweep_id, sweeps_root=sweeps_root),
        context=f"system delta queue instance {resolved_sweep_id!r}",
    )
    queue = _validate_payload(
        SweepQueuePayload,
        raw_queue,
        context=f"system delta queue instance {resolved_sweep_id!r}",
    )
    if ensure_non_empty_string(queue.get("sweep_id"), context="queue.sweep_id") != resolved_sweep_id:
        raise RuntimeError(
            f"system delta queue sweep id mismatch: expected {resolved_sweep_id!r}, got {queue.get('sweep_id')!r}"
        )
    return queue
