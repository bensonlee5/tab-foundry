"""Catalog and sweep metadata loaders for system-delta sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .paths_io import _load_yaml_mapping, default_catalog_path, default_sweep_index_path, sweep_metadata_path, sweep_queue_path
from .validation import ensure_non_empty_string, ensure_rows, validate_prose_fields


CATALOG_SCHEMA = "tab-foundry-system-delta-catalog-v1"
SWEEP_INDEX_SCHEMA = "tab-foundry-system-delta-sweep-index-v1"
SWEEP_SCHEMA = "tab-foundry-system-delta-sweep-v1"
SWEEP_QUEUE_SCHEMA = "tab-foundry-system-delta-sweep-queue-v1"


def load_system_delta_catalog(path: Path | None = None) -> dict[str, Any]:
    catalog = _load_yaml_mapping(path or default_catalog_path(), context="system delta catalog")
    if catalog.get("schema") != CATALOG_SCHEMA:
        raise RuntimeError(
            f"system delta catalog schema must be {CATALOG_SCHEMA!r}, got {catalog.get('schema')!r}"
        )
    deltas = catalog.get("deltas")
    if not isinstance(deltas, dict) or not deltas:
        raise RuntimeError("system delta catalog must include a non-empty deltas mapping")
    return catalog


def load_system_delta_index(path: Path | None = None) -> dict[str, Any]:
    index = _load_yaml_mapping(path or default_sweep_index_path(), context="system delta sweep index")
    if index.get("schema") != SWEEP_INDEX_SCHEMA:
        raise RuntimeError(
            f"system delta sweep index schema must be {SWEEP_INDEX_SCHEMA!r}, got {index.get('schema')!r}"
        )
    ensure_non_empty_string(index.get("active_sweep_id"), context="system delta sweep index.active_sweep_id")
    sweeps = index.get("sweeps")
    if not isinstance(sweeps, dict) or not sweeps:
        raise RuntimeError("system delta sweep index must include a non-empty sweeps mapping")
    return index


def resolve_selected_sweep_id(
    sweep_id: str | None,
    *,
    index_path: Path | None = None,
) -> str:
    if sweep_id is not None:
        return ensure_non_empty_string(sweep_id, context="sweep_id")
    index = load_system_delta_index(index_path)
    return ensure_non_empty_string(index.get("active_sweep_id"), context="active_sweep_id")


def load_system_delta_sweep(
    sweep_id: str | None = None,
    *,
    index_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    resolved_sweep_id = resolve_selected_sweep_id(sweep_id, index_path=index_path)
    sweep = _load_yaml_mapping(
        sweep_metadata_path(resolved_sweep_id, sweeps_root=sweeps_root),
        context=f"system delta sweep {resolved_sweep_id!r}",
    )
    if sweep.get("schema") != SWEEP_SCHEMA:
        raise RuntimeError(
            f"system delta sweep schema must be {SWEEP_SCHEMA!r}, got {sweep.get('schema')!r}"
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
    queue = _load_yaml_mapping(
        sweep_queue_path(resolved_sweep_id, sweeps_root=sweeps_root),
        context=f"system delta queue instance {resolved_sweep_id!r}",
    )
    if queue.get("schema") != SWEEP_QUEUE_SCHEMA:
        raise RuntimeError(
            f"system delta queue instance schema must be {SWEEP_QUEUE_SCHEMA!r}, got {queue.get('schema')!r}"
        )
    if ensure_non_empty_string(queue.get("sweep_id"), context="queue.sweep_id") != resolved_sweep_id:
        raise RuntimeError(
            f"system delta queue sweep id mismatch: expected {resolved_sweep_id!r}, got {queue.get('sweep_id')!r}"
        )
    rows = ensure_rows(queue.get("rows"), context="system delta queue instance rows")
    for index, row in enumerate(rows):
        validate_prose_fields(
            row,
            context=f"system delta queue instance rows[{index}]",
            field_names=("notes", "confounders", "parameter_adequacy_plan"),
        )
    return queue
