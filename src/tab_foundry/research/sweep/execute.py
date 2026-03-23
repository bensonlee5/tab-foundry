"""Library entrypoint for system-delta sweep execution."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping

from .artifacts import ExecutionPaths, read_yaml, write_yaml
from . import core as sweep_core
from .promote import promote_anchor
from . import row_dependencies as _row_dependencies
from . import row_sync as _row_sync
from .row_execution import ALLOWED_DECISIONS, DEFAULT_CONCLUSION, DEFAULT_DECISION, run_row
from .selection import select_queue_rows, sorted_rows


def execute_sweep(
    *,
    sweep_id: str | None,
    prior_dump: Path,
    nanotabpfn_root: Path,
    device: str,
    fallback_python: Path,
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

    sweep_meta = sweep_core.load_system_delta_sweep(
        sweep_id,
        index_path=resolved_paths.index_path,
        sweeps_root=resolved_paths.sweeps_root,
    )
    resolved_sweep_id = str(sweep_meta["sweep_id"])
    queue_path = sweep_core.sweep_queue_path(resolved_sweep_id, sweeps_root=resolved_paths.sweeps_root)
    queue = read_yaml(queue_path)
    queue_rows = sorted_rows(queue)
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

    current_anchor_run_id = sweep_meta.get("anchor_run_id")
    active_anchor = (
        str(current_anchor_run_id)
        if isinstance(current_anchor_run_id, str) and current_anchor_run_id.strip()
        else None
    )
    executed_run_ids: list[str] = []
    decision_map = dict(decision_overrides or {})
    conclusion_map = dict(conclusion_overrides or {})

    for index, queue_row in enumerate(selected_rows):
        order = int(queue_row["order"])
        decision = str(decision_map.get(order, decision_default)).strip().lower()
        if decision not in ALLOWED_DECISIONS:
            raise RuntimeError(f"decision must be one of {sorted(ALLOWED_DECISIONS)}, got {decision!r}")
        conclusion = str(conclusion_map.get(order, conclusion_default)).strip()
        if not conclusion:
            raise RuntimeError("conclusion must be non-empty")

        promote_now = bool(promote_first_executed_row_to_anchor and index == 0)
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
            device=device,
            fallback_python=fallback_python,
            decision=decision,
            conclusion=conclusion,
            paths=resolved_paths,
        )
        write_yaml(queue_path, queue)
        if promote_now:
            _ = promote_anchor(
                sweep_id=resolved_sweep_id,
                anchor_run_id=run_id,
                set_active=False,
                paths=resolved_paths.promotion_paths(),
            )
            active_anchor = run_id
            sweep_meta = sweep_core.load_system_delta_sweep(
                resolved_sweep_id,
                index_path=resolved_paths.index_path,
                sweeps_root=resolved_paths.sweeps_root,
            )
        _row_sync.sync_sweep_matrix(sweep_id=resolved_sweep_id, paths=resolved_paths)
        _row_sync.sync_active_aliases_if_active(
            sweep_id=resolved_sweep_id,
            paths=resolved_paths,
        )
        executed_run_ids.append(run_id)

    return executed_run_ids
