"""Library entrypoint for system-delta sweep execution."""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import Any, Mapping, cast

from tab_foundry.benchmark_registry import load_benchmark_run_registry
from tab_foundry.external_benchmarks import (
    EXTERNAL_BENCHMARK_NANOTABPFN,
    normalize_external_benchmarks,
)

from .artifacts import ExecutionPaths, read_yaml, write_yaml
from .catalog import load_system_delta_sweep
from .device_policy import resolve_sweep_execution_device
from .models import DEFAULT_LEGACY_SWEEP_EXTERNAL_BENCHMARKS
from .queue_loading import write_resolved_system_delta_queue
from .paths_io import sweep_queue_path
from .promote import promote_anchor
from .queue_state import recover_completed_queue_row_from_registry_run
from . import row_dependencies as _row_dependencies
from . import row_sync as _row_sync
from .row_execution import ALLOWED_DECISIONS, DEFAULT_CONCLUSION, DEFAULT_DECISION, run_row
from .selection import select_queue_rows, sorted_rows


def _optional_non_empty_string(value: Any, *, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{context} must be a non-empty string when provided")
    return str(value).strip()


def _resolved_external_benchmarks(sweep_meta: Mapping[str, Any]) -> tuple[str, ...]:
    raw_values = sweep_meta.get("external_benchmarks")
    values = (
        list(cast(Sequence[str], raw_values))
        if isinstance(raw_values, Sequence) and not isinstance(raw_values, (str, bytes))
        else None
    )
    return normalize_external_benchmarks(
        values,
        context="sweep.external_benchmarks",
        default=DEFAULT_LEGACY_SWEEP_EXTERNAL_BENCHMARKS,
        allow_empty=True,
    )


def _queue_row_by_order(
    *,
    sweep_id: str,
    queue: Mapping[str, Any],
    order: int,
) -> dict[str, Any]:
    matching_rows = [row for row in sorted_rows(queue) if int(row["order"]) == order]
    if not matching_rows:
        raise RuntimeError(f"sweep {sweep_id!r} does not contain queue row {order}")
    return matching_rows[0]


def _registry_run_for_run_id(
    *,
    run_id: str,
    paths: ExecutionPaths,
) -> dict[str, Any] | None:
    registry = load_benchmark_run_registry(paths.registry_path)
    runs = cast(dict[str, dict[str, Any]], registry["runs"])
    return runs.get(run_id)


def _sync_terminal_queue_row(
    *,
    sweep_id: str,
    queue_path: Path,
    local_queue: Mapping[str, Any],
    order: int,
    run_id: str,
    expected_delta_ref: str,
    paths: ExecutionPaths,
) -> dict[str, Any]:
    queue = read_yaml(queue_path)
    local_queue_row = _queue_row_by_order(sweep_id=sweep_id, queue=local_queue, order=order)
    queue_row = _queue_row_by_order(sweep_id=sweep_id, queue=queue, order=order)
    queue_delta_ref = _optional_non_empty_string(
        queue_row.get("delta_ref"),
        context=f"sweep {sweep_id!r} queue row {order}.delta_ref",
    )
    local_delta_ref = _optional_non_empty_string(
        local_queue_row.get("delta_ref"),
        context=f"sweep {sweep_id!r} local queue row {order}.delta_ref",
    )
    if queue_delta_ref != expected_delta_ref:
        raise RuntimeError(
            f"sweep {sweep_id!r} queue row {order} delta_ref changed from "
            f"{expected_delta_ref!r} to {queue_delta_ref!r} during execution"
        )
    if local_delta_ref != expected_delta_ref:
        raise RuntimeError(
            f"sweep {sweep_id!r} local queue row {order} has delta_ref {local_delta_ref!r}, "
            f"expected {expected_delta_ref!r}"
        )

    local_status = str(local_queue_row.get("status", "")).strip().lower()
    run = _registry_run_for_run_id(run_id=run_id, paths=paths)
    if run is not None:
        recover_completed_queue_row_from_registry_run(
            queue_row=queue_row,
            run_id=run_id,
            run=run,
        )
        source = "benchmark_registry"
    elif local_status in {"completed", "screened"}:
        recovered_queue_row = cast(dict[str, Any], deepcopy(local_queue_row))
        recovered_queue_row["run_id"] = run_id
        queue_row.clear()
        queue_row.update(recovered_queue_row)
        source = "in_memory_queue"
    else:
        recovered_queue_row = cast(dict[str, Any], deepcopy(local_queue_row))
        recovered_queue_row["status"] = "completed"
        recovered_queue_row["run_id"] = run_id
        queue_row.clear()
        queue_row.update(recovered_queue_row)
        source = "in_memory_queue_fallback"

    write_yaml(queue_path, queue)
    _row_sync.sync_sweep_matrix(sweep_id=sweep_id, paths=paths)
    print(
        "Synchronized terminal queue row.",
        f"sweep_id={sweep_id}",
        f"order={order}",
        f"run_id={run_id}",
        f"status={queue_row.get('status')}",
        f"source={source}",
        flush=True,
    )
    return queue


def _recover_partial_anchor_promotion(
    *,
    sweep_id: str,
    sweep_meta: Mapping[str, Any],
    queue: dict[str, Any],
    queue_path: Path,
    paths: ExecutionPaths,
) -> None:
    anchor_run_id = _optional_non_empty_string(
        sweep_meta.get("anchor_run_id"),
        context=f"sweep {sweep_id!r}.anchor_run_id",
    )
    if anchor_run_id is None:
        return

    if any(
        str(row.get("status", "")).strip().lower() == "completed"
        and str(row.get("run_id", "")).strip() == anchor_run_id
        for row in sorted_rows(queue)
    ):
        return

    run = _registry_run_for_run_id(run_id=anchor_run_id, paths=paths)
    if run is None:
        raise RuntimeError(
            f"sweep {sweep_id!r} anchor_run_id {anchor_run_id!r} is missing from the benchmark registry"
        )

    raw_sweep_payload = run.get("sweep")
    if not isinstance(raw_sweep_payload, Mapping):
        return
    sweep_payload = cast(Mapping[str, Any], raw_sweep_payload)
    recovered_sweep_id_raw = sweep_payload.get("sweep_id")
    if not isinstance(recovered_sweep_id_raw, str) or not recovered_sweep_id_raw.strip():
        return
    recovered_sweep_id = str(recovered_sweep_id_raw).strip()
    if recovered_sweep_id != sweep_id:
        return

    queue_order_raw = sweep_payload.get("queue_order")
    if queue_order_raw is None:
        raise RuntimeError(f"benchmark registry run {anchor_run_id!r} is missing sweep.queue_order")
    queue_order = int(queue_order_raw)
    if queue_order <= 0:
        raise RuntimeError(
            f"benchmark registry run {anchor_run_id!r} has invalid sweep.queue_order={queue_order_raw!r}"
        )

    delta_id = _optional_non_empty_string(
        sweep_payload.get("delta_id"),
        context=f"benchmark registry run {anchor_run_id!r}.sweep.delta_id",
    )
    queue_row = _queue_row_by_order(sweep_id=sweep_id, queue=queue, order=queue_order)
    queue_delta_ref = _optional_non_empty_string(
        queue_row.get("delta_ref"),
        context=f"sweep {sweep_id!r} queue row {queue_order}.delta_ref",
    )
    if queue_delta_ref != delta_id:
        raise RuntimeError(
            f"benchmark registry run {anchor_run_id!r} points to delta_id {delta_id!r}, "
            f"but sweep {sweep_id!r} queue row {queue_order} is {queue_delta_ref!r}"
        )

    recover_completed_queue_row_from_registry_run(
        queue_row=queue_row,
        run_id=anchor_run_id,
        run=run,
    )
    write_yaml(queue_path, queue)
    _row_sync.sync_sweep_matrix(sweep_id=sweep_id, paths=paths)
    print(
        "Recovered completed anchor row from benchmark registry.",
        f"sweep_id={sweep_id}",
        f"order={queue_order}",
        f"run_id={anchor_run_id}",
        flush=True,
    )


def execute_sweep(
    *,
    sweep_id: str | None,
    prior_dump: Path | None,
    nanotabpfn_root: Path | None,
    device: str,
    fallback_python: Path,
    reuse_nanotabpfn_only: bool = False,
    orders: list[int] | None = None,
    start_order: int | None = None,
    stop_after_order: int | None = None,
    include_completed: bool = False,
    decision_default: str = DEFAULT_DECISION,
    conclusion_default: str = DEFAULT_CONCLUSION,
    decision_overrides: Mapping[int, str] | None = None,
    conclusion_overrides: Mapping[int, str] | None = None,
    promote_first_executed_row_to_anchor: bool = False,
    paths: ExecutionPaths | None = None,
) -> list[str]:
    resolved_paths = ExecutionPaths.default() if paths is None else paths
    resolved_device = resolve_sweep_execution_device(device)

    sweep_meta = load_system_delta_sweep(
        sweep_id,
        index_path=resolved_paths.index_path,
        sweeps_root=resolved_paths.sweeps_root,
    )
    external_benchmarks = _resolved_external_benchmarks(sweep_meta)
    if EXTERNAL_BENCHMARK_NANOTABPFN in external_benchmarks and nanotabpfn_root is None:
        raise RuntimeError(
            "--nanotabpfn-root is required when sweep external_benchmarks include 'nanotabpfn'"
        )
    resolved_sweep_id = str(sweep_meta["sweep_id"])
    queue_path = sweep_queue_path(resolved_sweep_id, sweeps_root=resolved_paths.sweeps_root)
    queue = read_yaml(queue_path)
    _recover_partial_anchor_promotion(
        sweep_id=resolved_sweep_id,
        sweep_meta=sweep_meta,
        queue=queue,
        queue_path=queue_path,
        paths=resolved_paths,
    )
    _ = write_resolved_system_delta_queue(
        sweep_id=resolved_sweep_id,
        index_path=resolved_paths.index_path,
        catalog_path=resolved_paths.catalog_path,
        sweeps_root=resolved_paths.sweeps_root,
    )
    materialized_rows = _row_sync.materialized_row_map(
        sweep_id=resolved_sweep_id,
        paths=resolved_paths,
    )
    selected_rows = select_queue_rows(
        queue,
        orders=orders,
        start_order=start_order,
        stop_after_order=stop_after_order,
        include_completed=include_completed,
    )
    if not selected_rows:
        print("No rows selected for execution.", f"sweep_id={resolved_sweep_id}", flush=True)
        return []
    selected_orders = [int(row["order"]) for row in selected_rows]

    comparison_policy = str(sweep_meta.get("comparison_policy", "anchor_only")).strip().lower()
    current_anchor_run_id = sweep_meta.get("anchor_run_id")
    active_anchor = (
        str(current_anchor_run_id)
        if isinstance(current_anchor_run_id, str) and current_anchor_run_id.strip()
        else None
    )
    first_queue_order = min(int(row["order"]) for row in sorted_rows(queue))
    executed_run_ids: list[str] = []
    decision_map = dict(decision_overrides or {})
    conclusion_map = dict(conclusion_overrides or {})

    for index, order in enumerate(selected_orders):
        queue = read_yaml(queue_path)
        queue_rows = sorted_rows(queue)
        queue_row = _queue_row_by_order(sweep_id=resolved_sweep_id, queue=queue, order=order)
        decision = str(decision_map.get(order, decision_default)).strip().lower()
        if decision not in ALLOWED_DECISIONS:
            raise RuntimeError(f"decision must be one of {sorted(ALLOWED_DECISIONS)}, got {decision!r}")
        conclusion = str(conclusion_map.get(order, conclusion_default)).strip()
        if not conclusion:
            raise RuntimeError("conclusion must be non-empty")

        promote_now = bool(
            promote_first_executed_row_to_anchor and index == 0 and order == first_queue_order
        )
        if comparison_policy == "anchor_only" and active_anchor is None and not promote_now:
            raise RuntimeError(
                "anchor_only sweeps require a resolved anchor before executing non-anchor rows; "
                f"sweep_id={resolved_sweep_id} order={order}"
            )
        materialized_row = materialized_rows[str(queue_row["delta_ref"])]
        run_id = run_row(
            sweep_id=resolved_sweep_id,
            sweep_meta=sweep_meta,
            queue_row=queue_row,
            materialized_row=materialized_row,
            anchor_run_id=None if promote_now else active_anchor,
            parent_run_id=(
                None
                if promote_now
                else _row_dependencies.resolve_parent_run_id(
                    queue_row=queue_row,
                    queue_rows=queue_rows,
                    active_anchor=active_anchor,
                )
            ),
            queue=queue,
            prior_dump=prior_dump,
            nanotabpfn_root=nanotabpfn_root,
            reuse_nanotabpfn_only=reuse_nanotabpfn_only,
            device=resolved_device,
            fallback_python=fallback_python,
            decision=decision,
            conclusion=conclusion,
            paths=resolved_paths,
        )
        if promote_now:
            _ = promote_anchor(
                sweep_id=resolved_sweep_id,
                anchor_run_id=run_id,
                render_matrix=False,
                paths=resolved_paths.promotion_paths(),
            )
            active_anchor = run_id
            sweep_meta = load_system_delta_sweep(
                resolved_sweep_id,
                index_path=resolved_paths.index_path,
                sweeps_root=resolved_paths.sweeps_root,
            )
        queue = _sync_terminal_queue_row(
            sweep_id=resolved_sweep_id,
            queue_path=queue_path,
            local_queue=queue,
            order=order,
            run_id=run_id,
            expected_delta_ref=str(queue_row["delta_ref"]),
            paths=resolved_paths,
        )
        executed_run_ids.append(run_id)

    return executed_run_ids
