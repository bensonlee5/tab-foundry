"""Sweep row dependency resolution helpers."""

from __future__ import annotations

from typing import Any, Mapping, cast

from .queue_updates import append_note
from .screening import pick_screen_winner


def _optional_non_empty_string(value: Any, *, context: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{context} must be a non-empty string when provided")
    return str(value.strip())


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
