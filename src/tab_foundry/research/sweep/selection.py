"""Row-selection helpers for system-delta execution."""

from __future__ import annotations

from typing import Any, Mapping, cast


def parse_order_overrides(values: list[str] | None, *, arg_name: str) -> dict[int, str]:
    overrides: dict[int, str] = {}
    for raw in values or []:
        left, separator, right = str(raw).partition("=")
        if separator != "=" or not left.strip() or not right.strip():
            raise RuntimeError(f"{arg_name} values must look like <order>=<value>, got {raw!r}")
        try:
            order = int(left)
        except ValueError as exc:
            raise RuntimeError(f"{arg_name} order must be an integer, got {left!r}") from exc
        overrides[order] = right
    return overrides


def sorted_rows(queue: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = queue.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("queue rows must be a list")
    return sorted(cast(list[dict[str, Any]], rows), key=lambda row: int(row["order"]))


def select_queue_rows(
    queue: Mapping[str, Any],
    *,
    orders: list[int] | None = None,
    start_order: int | None = None,
    stop_after_order: int | None = None,
    include_completed: bool = False,
) -> list[dict[str, Any]]:
    rows = sorted_rows(queue)
    explicit_orders = list(orders or [])
    explicit_selection = bool(explicit_orders) or start_order is not None or stop_after_order is not None
    if not explicit_selection:
        return [row for row in rows if str(row.get("status", "")).strip().lower() == "ready"]

    known_orders = {int(row["order"]) for row in rows}
    selected_orders = set(explicit_orders)
    if start_order is not None or stop_after_order is not None:
        min_order = min(known_orders)
        max_order = max(known_orders)
        lower = min_order if start_order is None else int(start_order)
        upper = max_order if stop_after_order is None else int(stop_after_order)
        if lower > upper:
            raise RuntimeError("start_order must be less than or equal to stop_after_order")
        selected_orders.update(range(lower, upper + 1))
    missing = sorted(order for order in selected_orders if order not in known_orders)
    if missing:
        raise RuntimeError(f"unknown queue orders for selection: {missing}")

    selected = [row for row in rows if int(row["order"]) in selected_orders]
    if not include_completed:
        completed_orders = [
            int(row["order"])
            for row in selected
            if str(row.get("status", "")).strip().lower() in {"completed", "screened"}
        ]
        if completed_orders:
            raise RuntimeError(
                "explicitly selected completed rows require --include-completed; "
                f"got completed or screened orders {completed_orders}"
            )
    return selected
