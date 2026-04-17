"""Pareto reporting helpers for sweep summaries and selector interpretation."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any, Mapping, Sequence, cast


_MIN_KEPT_CONTRACT_FRONTIER_GEOMETRIES = 2


def _optional_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def _geometry_label(row: Mapping[str, Any]) -> str | None:
    existing = row.get("selector_geometry_label")
    if isinstance(existing, str) and existing.strip():
        return str(existing).strip()
    model = row.get("model")
    if not isinstance(model, Mapping):
        return None
    d_icl = model.get("d_icl")
    layers = model.get("sandwich_layers")
    if d_icl is None or layers is None:
        return None
    return f"{int(d_icl)}x{int(layers)}"


def _prescription_label(row: Mapping[str, Any]) -> str | None:
    existing = row.get("selector_prescription_label")
    if isinstance(existing, str) and existing.strip():
        return str(existing).strip()
    for key in ("delta_id", "delta_ref"):
        raw = row.get(key)
        if not isinstance(raw, str) or "_muon_" not in raw:
            continue
        suffix = raw.rsplit("_muon_", 1)[1]
        suffix = re.sub(r"_v\d+$", "", suffix)
        return suffix or None
    return None


def _pareto_orders(rows: Sequence[Mapping[str, Any]]) -> set[int]:
    eligible: list[tuple[int, float, float]] = []
    for row in rows:
        loss = _optional_float(row.get("final_log_loss"))
        wall = _optional_float(row.get("end_to_end_wall_seconds"))
        if loss is None or wall is None:
            continue
        eligible.append((int(row["order"]), loss, wall))
    frontier: set[int] = set()
    for order, loss, wall in eligible:
        dominated = False
        for other_order, other_loss, other_wall in eligible:
            if other_order == order:
                continue
            if (
                other_loss <= loss
                and other_wall <= wall
                and (other_loss < loss or other_wall < wall)
            ):
                dominated = True
                break
        if not dominated:
            frontier.add(order)
    return frontier


def _row_descriptor(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "order": int(row["order"]),
        "delta_id": str(row["delta_id"]),
        "geometry_label": _geometry_label(row),
        "prescription_label": _prescription_label(row),
        "final_log_loss": _optional_float(row.get("final_log_loss")),
        "end_to_end_wall_seconds": _optional_float(row.get("end_to_end_wall_seconds")),
    }


def annotate_rows_with_pareto(
    *,
    rows: list[dict[str, Any]],
    surface_role: str | None = None,
) -> dict[str, Any] | None:
    global_frontier = _pareto_orders(rows)
    geometry_groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        geometry_label = _geometry_label(row)
        prescription_label = _prescription_label(row)
        row["pareto_admissible"] = (
            None
            if _optional_float(row.get("final_log_loss")) is None
            or _optional_float(row.get("end_to_end_wall_seconds")) is None
            else int(row["order"]) in global_frontier
        )
        row["selector_geometry_label"] = geometry_label
        row["selector_prescription_label"] = prescription_label
        row["geometry_pareto_admissible"] = None
        if geometry_label is not None:
            geometry_groups[geometry_label].append(row)

    geometry_frontiers: dict[str, set[int]] = {
        geometry_label: _pareto_orders(group_rows)
        for geometry_label, group_rows in geometry_groups.items()
    }
    for row in rows:
        geometry_label = cast(str | None, row.get("selector_geometry_label"))
        if geometry_label is None:
            continue
        row["geometry_pareto_admissible"] = (
            None
            if _optional_float(row.get("final_log_loss")) is None
            or _optional_float(row.get("end_to_end_wall_seconds")) is None
            else int(row["order"]) in geometry_frontiers.get(geometry_label, set())
        )

    if surface_role != "classification_training_dynamics_selector":
        return None

    eligible_rows = [
        row
        for row in rows
        if row.get("selector_geometry_label") is not None
        and row.get("selector_prescription_label") is not None
        and _optional_float(row.get("final_log_loss")) is not None
        and _optional_float(row.get("end_to_end_wall_seconds")) is not None
    ]
    if not eligible_rows:
        return {
            "eligible_row_count": 0,
            "global_frontier_orders": [],
            "per_geometry_frontiers": {},
            "best_row": None,
            "kept_contract": None,
            "no_universal_kept_contract": True,
        }

    best_row_payload = _row_descriptor(
        min(
            eligible_rows,
            key=lambda row: (
                float(row["final_log_loss"]),
                float(row["end_to_end_wall_seconds"]),
                int(row["order"]),
            ),
        )
    )

    per_geometry_frontiers: dict[str, list[dict[str, Any]]] = {}
    prescription_rows: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for geometry_label, frontier_orders in geometry_frontiers.items():
        frontier_rows = [
            row
            for row in eligible_rows
            if row.get("selector_geometry_label") == geometry_label
            and int(row["order"]) in frontier_orders
        ]
        sorted_rows = sorted(frontier_rows, key=lambda row: int(row["order"]))
        per_geometry_frontiers[geometry_label] = [_row_descriptor(row) for row in sorted_rows]
        for row in sorted_rows:
            prescription_rows[str(row["selector_prescription_label"])].append(row)

    coverage: list[dict[str, Any]] = []
    for prescription_label, rows_for_prescription in prescription_rows.items():
        geometry_labels = sorted(
            {str(row["selector_geometry_label"]) for row in rows_for_prescription}
        )
        coverage.append(
            {
                "prescription_label": prescription_label,
                "geometry_count": len(geometry_labels),
                "geometry_labels": geometry_labels,
                "mean_end_to_end_wall_seconds": sum(
                    float(row["end_to_end_wall_seconds"]) for row in rows_for_prescription
                )
                / float(len(rows_for_prescription)),
                "mean_benchmark_log_loss": sum(
                    float(row["final_log_loss"]) for row in rows_for_prescription
                )
                / float(len(rows_for_prescription)),
                "orders": sorted(int(row["order"]) for row in rows_for_prescription),
            }
        )
    coverage.sort(
        key=lambda item: (
            -int(item["geometry_count"]),
            float(item["mean_end_to_end_wall_seconds"]),
            float(item["mean_benchmark_log_loss"]),
            str(item["prescription_label"]),
        )
    )

    kept_contract = None
    if (
        coverage
        and int(coverage[0]["geometry_count"])
        >= _MIN_KEPT_CONTRACT_FRONTIER_GEOMETRIES
    ):
        kept_contract = dict(coverage[0])

    return {
        "eligible_row_count": len(eligible_rows),
        "global_frontier_orders": sorted(global_frontier),
        "per_geometry_frontiers": per_geometry_frontiers,
        "best_row": best_row_payload,
        "prescription_coverage": coverage,
        "kept_contract": kept_contract,
        "no_universal_kept_contract": kept_contract is None,
    }
