"""Sweep row dependency resolution helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

from .paths_io import repo_root as shared_repo_root
from .queue_loading import load_system_delta_queue_for_inspection
from .queue_updates import append_note
from .screening import pick_screen_winner
from .surface_resolution import (
    build_lightweight_training_surface_record,
    resolve_queue_row_cfg_mapping,
)
from .training_state import training_surface_record_fingerprint
from .transfer import resolve_transfer_schedule


def _optional_non_empty_string(value: Any, *, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{context} must be a non-empty string when provided")
    return str(value.strip())


def _optional_positive_int(value: Any, *, context: str) -> int | None:
    if value is None:
        return None
    numeric = int(value)
    if numeric <= 0:
        raise RuntimeError(f"{context} must be a positive int when provided")
    return numeric


def _mapping_value(payload: Mapping[str, Any], key: str, *, context: str) -> Mapping[str, Any]:
    raw_value = payload.get(key)
    if not isinstance(raw_value, Mapping):
        raise RuntimeError(f"{context}.{key} must be a mapping")
    return cast(Mapping[str, Any], raw_value)


def _rows_from_queue(queue: Mapping[str, Any], *, context: str) -> list[Mapping[str, Any]]:
    rows_raw = queue.get("rows")
    if not isinstance(rows_raw, list):
        raise RuntimeError(f"{context}.rows must be a list")
    return [cast(Mapping[str, Any], row) for row in rows_raw if isinstance(row, Mapping)]


def _queue_for_sweep(
    *,
    queue: Mapping[str, Any],
    sweep_id: str,
) -> tuple[Mapping[str, Any], list[Mapping[str, Any]]]:
    current_sweep_id = str(queue.get("sweep_id", "")).strip()
    if sweep_id == current_sweep_id:
        rows = _rows_from_queue(queue, context=f"sweep {sweep_id}")
        return queue, rows
    source_queue = load_system_delta_queue_for_inspection(sweep_id=sweep_id)
    rows = _rows_from_queue(source_queue, context=f"sweep {sweep_id}")
    return source_queue, rows


def _find_row_by_order(
    *,
    rows: list[Mapping[str, Any]],
    order: int,
    context: str,
) -> Mapping[str, Any]:
    for row in rows:
        if int(row["order"]) == int(order):
            return row
    raise RuntimeError(f"{context} order {order} is missing")


def _candidate_label(
    *,
    candidate: Mapping[str, Any],
    row: Mapping[str, Any],
) -> str:
    for key in ("value", "candidate_label", "label"):
        value = candidate.get(key)
        if isinstance(value, str) and value.strip():
            return str(value).strip()
    for field_name in ("transfer_context", "transfer_resolution"):
        payload = row.get(field_name)
        if isinstance(payload, Mapping):
            raw_label = payload.get("candidate_label")
            if isinstance(raw_label, str) and raw_label.strip():
                return str(raw_label).strip()
    delta_id = row.get("delta_id", row.get("delta_ref"))
    if isinstance(delta_id, str) and delta_id.strip():
        return str(delta_id).strip()
    return f"order_{int(row['order']):02d}"


def _resolve_screen_winner(
    *,
    queue: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[dict[str, Any], Mapping[str, Any], Mapping[str, Any], str]:
    compare_orders = policy.get("compare_orders")
    if not isinstance(compare_orders, list) or not compare_orders:
        raise RuntimeError("screen winner policy compare_orders must be a non-empty list")

    source_sweep_ids = {
        _optional_non_empty_string(
            cast(Mapping[str, Any], candidate).get("sweep_id"),
            context="screen winner compare_orders[].sweep_id",
        )
        or str(queue.get("sweep_id", "")).strip()
        for candidate in compare_orders
        if isinstance(candidate, Mapping)
    }
    if not source_sweep_ids:
        raise RuntimeError("screen winner compare_orders did not contain any candidates")
    if len(source_sweep_ids) != 1:
        raise RuntimeError(
            "screen winner policies currently require candidates from exactly one sweep; "
            f"got {sorted(source_sweep_ids)}"
        )
    source_sweep_id = next(iter(source_sweep_ids))
    source_queue, source_rows = _queue_for_sweep(queue=queue, sweep_id=source_sweep_id)

    candidates: list[dict[str, Any]] = []
    for candidate_raw in compare_orders:
        if not isinstance(candidate_raw, Mapping):
            raise RuntimeError("screen winner compare_orders entries must be mappings")
        order = int(candidate_raw["order"])
        candidate_row = _find_row_by_order(
            rows=source_rows,
            order=order,
            context=f"screen winner sweep {source_sweep_id}",
        )
        candidates.append(
            {
                "order": order,
                "value": _candidate_label(candidate=candidate_raw, row=candidate_row),
                "screen_metrics": candidate_row.get("screen_metrics"),
            }
        )

    tie_break_preference = str(
        policy.get("tie_break_preference", candidates[0]["value"])
    )
    resolution = pick_screen_winner(
        candidates=candidates,
        tie_break_preference=tie_break_preference,
    )
    winning_order = int(resolution["winning_order"])
    winning_row = _find_row_by_order(
        rows=source_rows,
        order=winning_order,
        context=f"screen winner sweep {source_sweep_id}",
    )
    winning_candidate = next(
        candidate for candidate in candidates if int(candidate["order"]) == winning_order
    )
    return resolution, winning_row, source_queue, str(winning_candidate["value"])


def _resolve_shared_anchor(
    *,
    queue: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> tuple[Mapping[str, Any], Mapping[str, Any]]:
    anchor_order = _optional_positive_int(
        policy.get("anchor_order"),
        context="shared anchor policy anchor_order",
    )
    if anchor_order is None:
        raise RuntimeError("shared anchor policy anchor_order must be provided")
    source_sweep_id = (
        _optional_non_empty_string(
            policy.get("anchor_sweep_id"),
            context="shared anchor policy anchor_sweep_id",
        )
        or str(queue.get("sweep_id", "")).strip()
    )
    source_queue, source_rows = _queue_for_sweep(queue=queue, sweep_id=source_sweep_id)
    anchor_row = _find_row_by_order(
        rows=source_rows,
        order=int(anchor_order),
        context=f"shared anchor sweep {source_sweep_id}",
    )
    return anchor_row, source_queue


def _shared_anchor_provenance(
    *,
    anchor_row: Mapping[str, Any],
    source_queue: Mapping[str, Any],
) -> dict[str, Any]:
    provenance: dict[str, Any] = {
        "anchor_sweep_id": str(source_queue["sweep_id"]),
        "anchor_order": int(anchor_row["order"]),
    }
    delta_id = _optional_non_empty_string(
        anchor_row.get("delta_id", anchor_row.get("delta_ref")),
        context=f"shared anchor row {int(anchor_row['order'])}.delta_id",
    )
    if delta_id is not None:
        provenance["anchor_delta_id"] = delta_id
    reuse_artifact = anchor_row.get("reuse_train_artifact")
    if isinstance(reuse_artifact, Mapping):
        anchor_run_dir = _optional_non_empty_string(
            reuse_artifact.get("run_dir"),
            context=f"shared anchor row {int(anchor_row['order'])}.reuse_train_artifact.run_dir",
        )
        if anchor_run_dir is not None:
            provenance["anchor_run_dir"] = anchor_run_dir
    imported_baseline = anchor_row.get("imported_baseline_provenance")
    if isinstance(imported_baseline, Mapping):
        provenance["anchor_imported_baseline_provenance"] = dict(cast(Mapping[str, Any], imported_baseline))
    run_id = _optional_non_empty_string(
        anchor_row.get("run_id"),
        context=f"shared anchor row {int(anchor_row['order'])}.run_id",
    )
    if run_id is not None:
        provenance["anchor_run_id"] = run_id
    return provenance


def _first_schedule_stage(row: Mapping[str, Any]) -> Mapping[str, Any]:
    training = _mapping_value(row, "training", context=f"row {int(row['order'])}")
    overrides = _mapping_value(training, "overrides", context=f"row {int(row['order'])} training")
    schedule = _mapping_value(overrides, "schedule", context=f"row {int(row['order'])} training.overrides")
    stages = schedule.get("stages")
    if not isinstance(stages, list) or not stages or not isinstance(stages[0], Mapping):
        raise RuntimeError(f"row {int(row['order'])} training.overrides.schedule.stages must start with a mapping")
    return cast(Mapping[str, Any], stages[0])


def _training_overrides(row: Mapping[str, Any]) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any], int]:
    training = _mapping_value(row, "training", context=f"row {int(row['order'])}")
    task_batch_size = _optional_positive_int(
        training.get("task_batch_size"),
        context=f"row {int(row['order'])}.training.task_batch_size",
    )
    if task_batch_size is None:
        raise RuntimeError(f"row {int(row['order'])} is missing training.task_batch_size")
    overrides = _mapping_value(training, "overrides", context=f"row {int(row['order'])} training")
    runtime = _mapping_value(overrides, "runtime", context=f"row {int(row['order'])} training.overrides")
    optimizer = _mapping_value(overrides, "optimizer", context=f"row {int(row['order'])} training.overrides")
    stage = _first_schedule_stage(row)
    return runtime, optimizer, stage, int(task_batch_size)


def _mutable_overrides(row: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], int]:
    training = cast(dict[str, Any], row.setdefault("training", {}))
    task_batch_size = _optional_positive_int(
        training.get("task_batch_size"),
        context=f"row {int(row['order'])}.training.task_batch_size",
    )
    if task_batch_size is None:
        raise RuntimeError(f"row {int(row['order'])} is missing training.task_batch_size")
    overrides = cast(dict[str, Any], training.setdefault("overrides", {}))
    runtime = cast(dict[str, Any], overrides.setdefault("runtime", {}))
    optimizer = cast(dict[str, Any], overrides.setdefault("optimizer", {}))
    schedule = cast(dict[str, Any], overrides.setdefault("schedule", {}))
    stages = schedule.setdefault("stages", [])
    if not isinstance(stages, list):
        raise RuntimeError(f"row {int(row['order'])} training.overrides.schedule.stages must be a list")
    if not stages:
        stages.append({})
    if not isinstance(stages[0], dict):
        raise RuntimeError(f"row {int(row['order'])} training.overrides.schedule.stages[0] must be a mapping")
    return runtime, optimizer, cast(dict[str, Any], stages[0]), int(task_batch_size)


def _resolved_train_dir(
    *,
    sweep_id: str,
    delta_id: str,
    run_id: str,
) -> tuple[str, Path]:
    repo_root = shared_repo_root()
    absolute = (
        repo_root
        / "outputs"
        / "staged_ladder"
        / "research"
        / str(sweep_id)
        / str(delta_id)
        / str(run_id)
        / "train"
    )
    return absolute.relative_to(repo_root).as_posix(), absolute


def _resolve_parent_row(
    *,
    queue_row: Mapping[str, Any],
    queue_rows: list[dict[str, Any]],
) -> dict[str, Any] | None:
    current_order = int(queue_row["order"])
    current_delta_ref = str(queue_row["delta_ref"])
    parent_delta_ref = _optional_non_empty_string(
        queue_row.get("parent_delta_ref"),
        context=f"queue row {current_order}.parent_delta_ref",
    )
    if parent_delta_ref is None:
        return None

    matching_rows = [
        row for row in queue_rows if str(row.get("delta_ref", "")).strip() == parent_delta_ref
    ]
    if not matching_rows:
        raise RuntimeError(
            f"queue row {current_order} ({current_delta_ref}) parent_delta_ref "
            f"{parent_delta_ref!r} is missing from sweep {queue_row.get('sweep_id', '<unknown>')!r}"
        )

    earlier_rows = [row for row in matching_rows if int(row["order"]) < current_order]
    if earlier_rows:
        return max(earlier_rows, key=lambda row: int(row["order"]))

    matching_orders = [int(row["order"]) for row in matching_rows]
    if any(order == current_order for order in matching_orders):
        raise RuntimeError(
            f"queue row {current_order} ({current_delta_ref}) parent_delta_ref "
            f"{parent_delta_ref!r} must reference an earlier row, not itself; "
            f"matching orders={matching_orders}"
        )
    raise RuntimeError(
        f"queue row {current_order} ({current_delta_ref}) parent_delta_ref "
        f"{parent_delta_ref!r} must reference an earlier row; matching orders={matching_orders}"
    )


def resolve_parent_run_id(
    *,
    queue_row: Mapping[str, Any],
    queue_rows: list[dict[str, Any]],
    active_anchor: str | None,
) -> str | None:
    parent_row = _resolve_parent_row(queue_row=queue_row, queue_rows=queue_rows)
    if parent_row is None:
        return active_anchor

    parent_run_id = parent_row.get("run_id")
    if not isinstance(parent_run_id, str) or not parent_run_id.strip():
        raise RuntimeError(
            f"queue row {int(queue_row['order'])} ({queue_row['delta_ref']}) parent_delta_ref "
            f"{parent_row['delta_ref']!r} resolved to row {int(parent_row['order'])}, "
            "but that row does not have a completed run_id"
        )
    return str(parent_run_id)


def resolve_dynamic_model_overrides(
    *,
    queue: Mapping[str, Any],
    queue_row: dict[str, Any],
    materialized_row: dict[str, Any],
) -> None:
    dynamic_overrides = queue_row.get("dynamic_model_overrides")
    if not isinstance(dynamic_overrides, Mapping):
        return
    queue_rows_raw = queue.get("rows")
    if not isinstance(queue_rows_raw, list):
        raise RuntimeError("queue rows must be a list")
    rows_by_order = {
        int(raw_row["order"]): cast(dict[str, Any], raw_row)
        for raw_row in queue_rows_raw
        if isinstance(raw_row, dict)
    }
    queue_model = cast(dict[str, Any], queue_row.setdefault("model", {}))
    queue_module_overrides = cast(dict[str, Any], queue_model.setdefault("module_overrides", {}))
    materialized_model = cast(dict[str, Any], materialized_row.setdefault("model", {}))
    materialized_module_overrides = cast(
        dict[str, Any],
        materialized_model.setdefault("module_overrides", {}),
    )
    queue_notes = cast(list[str], queue_row.setdefault("notes", []))
    materialized_notes = cast(list[str], materialized_row.setdefault("notes", []))

    for override_key, policy_raw in dynamic_overrides.items():
        if not isinstance(policy_raw, dict):
            raise RuntimeError(f"dynamic_model_overrides.{override_key} must be a mapping")
        policy = cast(dict[str, Any], policy_raw)
        if str(policy.get("kind")) != "screen_winner":
            raise RuntimeError(f"unsupported dynamic override policy kind: {policy.get('kind')!r}")
        resolved_value = policy.get("resolved_value")
        if isinstance(resolved_value, str) and resolved_value.strip():
            queue_module_overrides[str(override_key)] = resolved_value
            materialized_module_overrides[str(override_key)] = resolved_value
            continue
        compare_orders = policy.get("compare_orders")
        if not isinstance(compare_orders, list) or not compare_orders:
            raise RuntimeError(
                f"dynamic_model_overrides.{override_key}.compare_orders must be a non-empty list"
            )
        candidates: list[dict[str, Any]] = []
        for candidate_raw in compare_orders:
            if not isinstance(candidate_raw, Mapping):
                raise RuntimeError("dynamic compare_orders entries must be mappings")
            order = int(candidate_raw["order"])
            value = str(candidate_raw["value"])
            candidate_row = rows_by_order.get(order)
            if candidate_row is None:
                raise RuntimeError(f"dynamic compare order {order} is missing from the queue")
            candidates.append(
                {
                    "order": order,
                    "value": value,
                    "screen_metrics": candidate_row.get("screen_metrics"),
                }
            )
        resolution = pick_screen_winner(
            candidates=candidates,
            tie_break_preference=str(policy.get("tie_break_preference", "rmsnorm")),
        )
        winning_value = str(resolution["winning_value"])
        policy["resolved_value"] = winning_value
        policy["resolved_from_order"] = int(resolution["winning_order"])
        policy["resolution_reason"] = str(resolution["reason"])
        queue_module_overrides[str(override_key)] = winning_value
        materialized_module_overrides[str(override_key)] = winning_value
        resolution_note = (
            f"Resolved `{override_key}` to `{winning_value}` from screen row "
            f"`{int(resolution['winning_order'])}` ({resolution['reason']})."
        )
        queue_row["notes"] = append_note(queue_notes, resolution_note)
        materialized_row["notes"] = append_note(materialized_notes, resolution_note)


def resolve_dynamic_training_overrides(
    *,
    queue: Mapping[str, Any],
    queue_row: dict[str, Any],
    materialized_row: dict[str, Any],
) -> None:
    dynamic_overrides = queue_row.get("dynamic_training_overrides")
    if not isinstance(dynamic_overrides, Mapping):
        return
    queue_notes = cast(list[str], queue_row.setdefault("notes", []))
    materialized_notes = cast(list[str], materialized_row.setdefault("notes", []))

    for override_key, policy_raw in dynamic_overrides.items():
        if not isinstance(policy_raw, Mapping):
            raise RuntimeError(f"dynamic_training_overrides.{override_key} must be a mapping")
        policy = cast(dict[str, Any], policy_raw)
        kind = str(policy.get("kind", "")).strip()
        if kind not in {"screen_winner_transfer", "shared_anchor_transfer"}:
            raise RuntimeError(f"unsupported dynamic training override policy kind: {kind!r}")

        if kind == "screen_winner_transfer":
            resolution, anchor_row, source_queue, anchor_label = _resolve_screen_winner(
                queue=queue,
                policy=policy,
            )
            resolution_note = (
                f"Resolved transfer training overrides `{override_key}` from screen row "
                f"`{int(resolution['winning_order'])}` in `{source_queue['sweep_id']}` "
                f"({resolution['reason']})."
            )
            shared_anchor_provenance = _shared_anchor_provenance(
                anchor_row=anchor_row,
                source_queue=source_queue,
            )
            shared_anchor_provenance["anchor_candidate_label"] = str(anchor_label)
            shared_anchor_provenance["resolution_reason"] = str(resolution["reason"])
        else:
            anchor_row, source_queue = _resolve_shared_anchor(
                queue=queue,
                policy=policy,
            )
            anchor_label = _optional_non_empty_string(
                policy.get("anchor_label"),
                context=f"dynamic_training_overrides.{override_key}.anchor_label",
            ) or f"order_{int(anchor_row['order']):02d}"
            resolution = {
                "winning_order": int(anchor_row["order"]),
                "reason": "shared_anchor",
            }
            resolution_note = (
                f"Resolved transfer training overrides `{override_key}` from shared anchor row "
                f"`{int(anchor_row['order'])}` in `{source_queue['sweep_id']}`."
            )
            shared_anchor_provenance = _shared_anchor_provenance(
                anchor_row=anchor_row,
                source_queue=source_queue,
            )
            shared_anchor_provenance["anchor_candidate_label"] = str(anchor_label)

        runtime, optimizer, schedule_stage, task_batch_size = _training_overrides(anchor_row)
        base_grad_accum_steps = _optional_positive_int(
            runtime.get("grad_accum_steps"),
            context=f"row {int(anchor_row['order'])} runtime.grad_accum_steps",
        )
        base_max_steps = _optional_positive_int(
            runtime.get("max_steps"),
            context=f"row {int(anchor_row['order'])} runtime.max_steps",
        )
        if base_grad_accum_steps is None or base_max_steps is None:
            raise RuntimeError(
                f"anchor row {int(anchor_row['order'])} omitted grad_accum_steps or max_steps"
            )
        base_effective_batch = int(task_batch_size * base_grad_accum_steps)
        base_effective_budget = int(base_max_steps * base_effective_batch)
        resolved_schedule = resolve_transfer_schedule(
            regime_label=str(policy["regime_label"]),
            base_lr_max=float(schedule_stage["lr_max"]),
            base_momentum=float(optimizer.get("momentum", 0.95)),
            base_effective_batch=base_effective_batch,
            base_effective_budget=int(
                _optional_positive_int(
                    policy.get("base_effective_budget"),
                    context=f"dynamic_training_overrides.{override_key}.base_effective_budget",
                )
                or base_effective_budget
            ),
            target_effective_budget=int(
                _optional_positive_int(
                    policy.get("target_effective_budget"),
                    context=f"dynamic_training_overrides.{override_key}.target_effective_budget",
                )
                or 0
            ),
            task_batch_size=task_batch_size,
            fixed_effective_batch=_optional_positive_int(
                policy.get("fixed_effective_batch"),
                context=f"dynamic_training_overrides.{override_key}.fixed_effective_batch",
            ),
            min_lr_ratio=float(policy.get("min_lr_ratio", 1.0e-3)),
            max_budget_drift=float(policy.get("max_budget_drift", 0.02)),
        )

        for target_row in (queue_row, materialized_row):
            target_runtime, target_optimizer, target_stage, target_task_batch_size = _mutable_overrides(target_row)
            if int(target_task_batch_size) != int(task_batch_size):
                raise RuntimeError(
                    "transfer rows must keep task_batch_size fixed relative to the T0 winner: "
                    f"source={task_batch_size}, target={target_task_batch_size}"
                )
            target_runtime["grad_accum_steps"] = int(resolved_schedule["grad_accum_steps"])
            target_runtime["max_steps"] = int(resolved_schedule["max_steps"])
            target_stage["steps"] = int(resolved_schedule["max_steps"])
            target_stage["lr_max"] = float(resolved_schedule["target_lr_max"])
            target_optimizer["min_lr"] = float(resolved_schedule["min_lr"])
            target_optimizer["momentum"] = float(resolved_schedule["target_momentum"])
            target_row["transfer_resolution"] = {
                **(
                    cast(dict[str, Any], target_row.get("transfer_context"))
                    if isinstance(target_row.get("transfer_context"), Mapping)
                    else {}
                ),
                **resolved_schedule,
                "resolved_from_order": int(resolution["winning_order"]),
                "resolved_from_sweep_id": str(source_queue["sweep_id"]),
                "resolved_candidate_label": str(anchor_label),
                "resolution_reason": str(resolution["reason"]),
                "shared_anchor_provenance": dict(shared_anchor_provenance),
            }

        policy["resolved_from_order"] = int(resolution["winning_order"])
        policy["resolved_from_sweep_id"] = str(source_queue["sweep_id"])
        policy["resolved_candidate_label"] = str(anchor_label)
        policy["resolution_reason"] = str(resolution["reason"])
        queue_row["notes"] = append_note(queue_notes, resolution_note)
        materialized_row["notes"] = append_note(materialized_notes, resolution_note)


def resolve_dynamic_reuse_train_artifact(
    *,
    queue: Mapping[str, Any],
    queue_row: dict[str, Any],
    materialized_row: dict[str, Any],
) -> None:
    dynamic_reuse = queue_row.get("dynamic_reuse_train_artifact")
    if not isinstance(dynamic_reuse, Mapping):
        return
    queue_notes = cast(list[str], queue_row.setdefault("notes", []))
    materialized_notes = cast(list[str], materialized_row.setdefault("notes", []))

    for override_key, policy_raw in dynamic_reuse.items():
        if not isinstance(policy_raw, Mapping):
            raise RuntimeError(f"dynamic_reuse_train_artifact.{override_key} must be a mapping")
        policy = cast(dict[str, Any], policy_raw)
        kind = str(policy.get("kind", "")).strip()
        if kind != "screen_winner_artifact":
            raise RuntimeError(f"unsupported dynamic reuse policy kind: {kind!r}")

        resolution, winning_row, source_queue, winning_value = _resolve_screen_winner(
            queue=queue,
            policy=policy,
        )
        run_id = _optional_non_empty_string(
            winning_row.get("run_id"),
            context=f"screen winner row {int(winning_row['order'])}.run_id",
        )
        if run_id is None:
            raise RuntimeError(
                f"screen winner row {int(winning_row['order'])} in `{source_queue['sweep_id']}` "
                "does not have a completed run_id yet"
            )
        delta_id = _optional_non_empty_string(
            winning_row.get("delta_id", winning_row.get("delta_ref")),
            context=f"screen winner row {int(winning_row['order'])}.delta_id",
        )
        if delta_id is None:
            raise RuntimeError(f"screen winner row {int(winning_row['order'])} is missing delta_id/delta_ref")
        configured_run_dir, absolute_train_dir = _resolved_train_dir(
            sweep_id=str(source_queue["sweep_id"]),
            delta_id=delta_id,
            run_id=run_id,
        )
        raw_cfg = resolve_queue_row_cfg_mapping(
            winning_row,
            run_dir=absolute_train_dir,
            training_experiment=str(source_queue["training_experiment"]),
            sweep_id=str(source_queue["sweep_id"]),
        )
        record = build_lightweight_training_surface_record(
            raw_cfg=raw_cfg,
            run_dir=absolute_train_dir,
        )
        reuse_payload = {
            "run_dir": configured_run_dir,
            "training_surface_fingerprint": training_surface_record_fingerprint(record),
        }
        for target_row in (queue_row, materialized_row):
            target_row["reuse_train_artifact"] = dict(reuse_payload)

        policy["resolved_from_order"] = int(resolution["winning_order"])
        policy["resolved_from_sweep_id"] = str(source_queue["sweep_id"])
        policy["resolved_candidate_label"] = str(winning_value)
        policy["resolution_reason"] = str(resolution["reason"])
        resolution_note = (
            f"Resolved reusable train artifact `{override_key}` from screen row "
            f"`{int(resolution['winning_order'])}` in `{source_queue['sweep_id']}`."
        )
        queue_row["notes"] = append_note(queue_notes, resolution_note)
        materialized_row["notes"] = append_note(materialized_notes, resolution_note)
