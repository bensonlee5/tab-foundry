"""Concrete queue loading and snapshot helpers for system-delta sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Mapping

from pydantic import ValidationError

from .catalog import (
    load_system_delta_catalog_payload,
    load_system_delta_queue_instance_payload,
    load_system_delta_sweep_payload,
)
from .models import (
    MaterializedQueuePayload,
    MaterializedQueueRowPayload,
    RESOLVED_QUEUE_SCHEMA,
    ResolvedQueuePayload,
    SWEEP_QUEUE_SCHEMA,
    SweepQueuePayload,
)
from .paths_io import load_yaml_mapping, sweep_resolved_queue_path, write_yaml
from .queue_materialization import (
    _resolved_queue_inputs_fingerprint,
    inspection_system_delta_queue,
    materialize_resolved_system_delta_queue,
    materialize_system_delta_queue,
)


def _drop_compatibility_only_row_fields(row: dict[str, Any]) -> None:
    training = row.get("training")
    if not isinstance(training, dict):
        return
    synthetic_epoch_budget = training.get("synthetic_epoch_budget")
    if isinstance(synthetic_epoch_budget, dict):
        synthetic_epoch_budget.pop("resolution_source", None)


def _drop_compatibility_only_payload_fields(payload: dict[str, Any]) -> None:
    payload.pop("catalog_path", None)
    payload.pop("canonical_sweep_path", None)
    payload.pop("canonical_queue_path", None)
    payload.pop("canonical_matrix_path", None)
    rows = payload.get("rows")
    if isinstance(rows, list):
        for row in rows:
            if isinstance(row, dict):
                _drop_compatibility_only_row_fields(row)


def _resolved_queue_semantically_matches(
    *,
    stored: ResolvedQueuePayload,
    regenerated: ResolvedQueuePayload,
) -> bool:
    stored_payload = stored.to_payload_dict()
    regenerated_payload = regenerated.to_payload_dict()
    stored_payload.pop("inputs_fingerprint", None)
    regenerated_payload.pop("inputs_fingerprint", None)
    for payload in (stored_payload, regenerated_payload):
        _drop_compatibility_only_payload_fields(payload)
    return stored_payload == regenerated_payload


def _resolved_queue_matches_materialized_inputs(
    *,
    stored: ResolvedQueuePayload,
    materialized: MaterializedQueuePayload,
) -> bool:
    stored_payload = stored.to_payload_dict()
    materialized_payload = materialized.to_payload_dict()
    stored_payload.pop("schema", None)
    materialized_payload.pop("schema", None)
    stored_payload.pop("canonical_resolved_queue_path", None)
    stored_payload.pop("inputs_fingerprint", None)
    _drop_compatibility_only_payload_fields(stored_payload)
    _drop_compatibility_only_payload_fields(materialized_payload)
    stored_rows = stored_payload.get("rows")
    if isinstance(stored_rows, list):
        for row in stored_rows:
            if not isinstance(row, dict):
                continue
            row.pop("resolved_surface", None)
            row.pop("resolved_surface_fingerprint", None)
    return stored_payload == materialized_payload


def write_resolved_system_delta_queue(
    *,
    sweep_id: str | None = None,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
    out_path: Path | None = None,
) -> Path:
    catalog = load_system_delta_catalog_payload(catalog_path)
    sweep = load_system_delta_sweep_payload(
        sweep_id,
        index_path=index_path,
        sweeps_root=sweeps_root,
    )
    queue_instance = load_system_delta_queue_instance_payload(
        sweep_id or sweep.sweep_id,
        index_path=index_path,
        sweeps_root=sweeps_root,
    )
    resolved_queue = materialize_resolved_system_delta_queue(
        catalog=catalog,
        sweep=sweep,
        queue_instance=queue_instance,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
    )
    resolved_path = (
        sweep_resolved_queue_path(sweep.sweep_id, sweeps_root=sweeps_root)
        if out_path is None
        else Path(out_path).expanduser().resolve()
    )
    write_yaml(resolved_path, resolved_queue.to_payload_dict())
    return resolved_path


def _load_materialized_queue_payload(path: Path) -> MaterializedQueuePayload:
    payload = load_yaml_mapping(path, context="system delta queue")
    return MaterializedQueuePayload.model_validate(payload)


def _load_resolved_queue_payload(path: Path) -> ResolvedQueuePayload:
    payload = load_yaml_mapping(path, context="system delta resolved queue")
    return ResolvedQueuePayload.model_validate(payload)


def _load_system_delta_queue_common(
    path: Path | None = None,
    *,
    sweep_id: str | None = None,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
    mode: Literal["materialized", "inspection"],
) -> MaterializedQueuePayload | ResolvedQueuePayload:
    def _materialize_or_fallback(
        *,
        queue_instance: SweepQueuePayload,
    ) -> MaterializedQueuePayload:
        catalog = load_system_delta_catalog_payload(catalog_path)
        sweep = load_system_delta_sweep_payload(
            sweep_id or queue_instance.sweep_id,
            index_path=index_path,
            sweeps_root=sweeps_root,
        )
        try:
            return materialize_system_delta_queue(
                catalog=catalog,
                sweep=sweep,
                queue_instance=queue_instance,
                catalog_path=catalog_path,
                sweeps_root=sweeps_root,
            )
        except RuntimeError as exc:
            if mode != "inspection" or "unknown delta_ref" not in str(exc):
                raise
            return inspection_system_delta_queue(
                sweep=sweep,
                queue_instance=queue_instance,
                sweeps_root=sweeps_root,
            )

    if path is None:
        catalog = load_system_delta_catalog_payload(catalog_path)
        sweep = load_system_delta_sweep_payload(sweep_id, index_path=index_path, sweeps_root=sweeps_root)
        queue_instance = load_system_delta_queue_instance_payload(
            sweep_id or sweep.sweep_id,
            index_path=index_path,
            sweeps_root=sweeps_root,
        )
        resolved_path = sweep_resolved_queue_path(sweep.sweep_id, sweeps_root=sweeps_root)
        if resolved_path.exists():
            resolved_queue = _load_resolved_queue_payload(resolved_path)
            expected_inputs_fingerprint = _resolved_queue_inputs_fingerprint(
                catalog=catalog,
                sweep=sweep,
                queue_instance=queue_instance,
            )
            if resolved_queue.inputs_fingerprint != expected_inputs_fingerprint:
                materialized_queue = _materialize_or_fallback(
                    queue_instance=queue_instance,
                )
                if _resolved_queue_matches_materialized_inputs(
                    stored=resolved_queue,
                    materialized=materialized_queue,
                ):
                    return resolved_queue
                regenerated_queue = materialize_resolved_system_delta_queue(
                    catalog=catalog,
                    sweep=sweep,
                    queue_instance=queue_instance,
                    catalog_path=catalog_path,
                    sweeps_root=sweeps_root,
                )
                if _resolved_queue_semantically_matches(
                    stored=resolved_queue,
                    regenerated=regenerated_queue,
                ):
                    return regenerated_queue
                raise RuntimeError(
                    "resolved_queue.yaml is stale for "
                    f"sweep {sweep.sweep_id!r}; regenerate it before inspection or execution"
                )
            return resolved_queue
        return _materialize_or_fallback(queue_instance=queue_instance)

    payload = load_yaml_mapping(path, context="system delta queue")
    schema = payload.get("schema")
    if schema == SWEEP_QUEUE_SCHEMA:
        try:
            queue_instance = SweepQueuePayload.model_validate(payload)
        except ValidationError as exc:
            raise RuntimeError(f"system delta queue instance is invalid: {exc}") from exc
        return _materialize_or_fallback(queue_instance=queue_instance)
    if schema == RESOLVED_QUEUE_SCHEMA:
        try:
            return ResolvedQueuePayload.model_validate(payload)
        except ValidationError as exc:
            raise RuntimeError(f"system delta resolved queue is invalid: {exc}") from exc
    try:
        materialized = MaterializedQueuePayload.model_validate(payload)
    except ValidationError as exc:
        raise RuntimeError(f"system delta queue is invalid: {exc}") from exc
    return materialized


def load_system_delta_queue(
    path: Path | None = None,
    *,
    sweep_id: str | None = None,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    return _load_system_delta_queue_common(
        path,
        sweep_id=sweep_id,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
        mode="materialized",
    ).to_payload_dict()


def load_system_delta_queue_for_inspection(
    path: Path | None = None,
    *,
    sweep_id: str | None = None,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    return _load_system_delta_queue_common(
        path,
        sweep_id=sweep_id,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
        mode="inspection",
    ).to_payload_dict()


def _ordered_row_models(queue: MaterializedQueuePayload | Mapping[str, Any]) -> list[MaterializedQueueRowPayload]:
    if isinstance(queue, MaterializedQueuePayload):
        rows = queue.rows
    else:
        raw_rows = queue.get("rows")
        if not isinstance(raw_rows, list):
            raise RuntimeError("materialized system delta queue must include rows")
        rows = [MaterializedQueueRowPayload.model_validate(row) for row in raw_rows]
    return sorted(rows, key=lambda row: (int(row.order), str(row.delta_id)))


def ordered_rows(queue: MaterializedQueuePayload | Mapping[str, Any]) -> list[dict[str, Any]]:
    return [row.to_payload_dict() for row in _ordered_row_models(queue)]


def next_ready_row(queue: MaterializedQueuePayload | Mapping[str, Any]) -> dict[str, Any] | None:
    for row in _ordered_row_models(queue):
        if str(row.status).strip().lower() == "ready":
            return row.to_payload_dict()
    return None
