"""Shared sweep/scaling navigation and contract summaries."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from tab_foundry.bench.openml_benchmark import (
    default_anchor_benchmark_summary,
    default_anchor_control_baseline_id,
    default_benchmark_manifest_path,
)
from tab_foundry.data.manifest_characteristics import compute_manifest_characteristics
from tab_foundry.repo_paths import normalize_repo_relative_path
from tab_foundry.research.scaling.study import ScalingStudyConfig, default_scaling_studies_root, load_scaling_study_config
from tab_foundry.research.sweep.catalog import load_system_delta_index_payload
from tab_foundry.research.sweep.materialize import load_system_delta_queue

_BINARY_CLASS_LIMIT = 2


def _optional_string(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return str(value).strip()


def _row_corpus_ref(row: Mapping[str, Any]) -> str | None:
    data = row.get("data")
    if not isinstance(data, Mapping):
        return None
    corpus_ref = _optional_string(data.get("corpus_ref"))
    if corpus_ref is not None:
        return corpus_ref
    surface_overrides = data.get("surface_overrides")
    if not isinstance(surface_overrides, Mapping):
        return None
    return _optional_string(surface_overrides.get("corpus_ref"))


def _unique_corpus_refs(rows: Sequence[Any]) -> list[str]:
    return sorted(
        {
            corpus_ref
            for row in rows
            if isinstance(row, Mapping)
            for corpus_ref in [_row_corpus_ref(row)]
            if corpus_ref is not None
        }
    )


def _default_anchor_benchmark_contract() -> dict[str, Any]:
    summary = default_anchor_benchmark_summary()
    return {
        "benchmark_manifest_path": normalize_repo_relative_path(default_benchmark_manifest_path()),
        "control_baseline_id": default_anchor_control_baseline_id(),
        "benchmark_bundle": dict(summary),
    }


def _uses_default_anchor_benchmark_contract(
    *,
    benchmark_manifest_path: str | Path,
    control_baseline_id: str | None,
) -> bool:
    resolved_manifest_path = Path(str(benchmark_manifest_path)).expanduser().resolve()
    default_manifest_path = default_benchmark_manifest_path().expanduser().resolve()
    if resolved_manifest_path != default_manifest_path:
        return False
    return _optional_string(control_baseline_id) == default_anchor_control_baseline_id()


def _default_anchor_manifest_contract_issues(manifest_path: Path) -> list[str]:
    if manifest_path.expanduser().resolve() != default_benchmark_manifest_path().expanduser().resolve():
        return []
    if not manifest_path.exists():
        return ["default anchor benchmark manifest is not materialized locally"]
    characteristics = compute_manifest_characteristics(manifest_path)
    expected_summary = default_anchor_benchmark_summary()
    expected_task_count = int(expected_summary["task_count"])
    issues: list[str] = []
    if int(characteristics["record_count"]) != expected_task_count:
        issues.append(
            "default anchor benchmark manifest record_count mismatch: "
            f"expected={expected_task_count} actual={int(characteristics['record_count'])}"
        )
    expected_allow_missing = bool(expected_summary.get("allow_missing_values"))
    actual_missing_policy = characteristics.get("missing_value_policy")
    if expected_allow_missing and str(actual_missing_policy) != "allow_any":
        issues.append(
            "default anchor benchmark manifest should preserve natural missingness: "
            f"expected missing_value_policy='allow_any' actual={actual_missing_policy!r}"
        )
    expected_selection = cast(Mapping[str, Any], expected_summary.get("selection", {}))
    expected_max_classes = expected_selection.get("max_classes")
    class_distribution = characteristics.get("class_count_distribution")
    if isinstance(class_distribution, Mapping):
        actual_max_classes = int(class_distribution["max"])
        if (
            expected_max_classes is not None
            and actual_max_classes <= _BINARY_CLASS_LIMIT
            and int(expected_max_classes) > _BINARY_CLASS_LIMIT
        ):
            issues.append(
                "default anchor benchmark manifest collapsed to a binary-only surface: "
                f"expected multiclass max_classes={int(expected_max_classes)} actual_max_classes={actual_max_classes}"
            )
    else:
        issues.append("default anchor benchmark manifest did not report class-count distribution")
    return issues


def _winner_from_rows(rows: Sequence[Any]) -> dict[str, Any] | None:
    candidates: list[Mapping[str, Any]] = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("status", "")).strip() != "completed":
            continue
        metrics = row.get("benchmark_metrics")
        if not isinstance(metrics, Mapping):
            continue
        if metrics.get("final_log_loss") is None:
            continue
        candidates.append(row)
    if not candidates:
        return None
    best = min(candidates, key=lambda row: float(cast(Mapping[str, Any], row["benchmark_metrics"])["final_log_loss"]))
    best_metrics = cast(Mapping[str, Any], best["benchmark_metrics"])
    model = cast(Mapping[str, Any], best.get("model", {}))
    geometry_label = None
    if model.get("d_icl") is not None and model.get("sandwich_layers") is not None:
        geometry_label = f"{int(model['d_icl'])}x{int(model['sandwich_layers'])}"
    return {
        "order": int(best["order"]),
        "delta_ref": str(best.get("delta_ref", best.get("delta_id", ""))),
        "run_id": _optional_string(best.get("run_id")),
        "geometry_label": geometry_label,
        "benchmark_log_loss": float(best_metrics["final_log_loss"]),
        "throughput_tokens_per_second": (
            None
            if best_metrics.get("throughput_tokens_per_second") is None
            else float(best_metrics["throughput_tokens_per_second"])
        ),
        "end_to_end_wall_seconds": (
            None
            if best_metrics.get("end_to_end_wall_seconds") is None
            else float(best_metrics["end_to_end_wall_seconds"])
        ),
        "tokens_per_step": (
            None if best_metrics.get("tokens_per_step") is None else float(best_metrics["tokens_per_step"])
        ),
        "steps": None if best_metrics.get("best_step") is None else int(best_metrics["best_step"]),
    }


def _winner_from_points(points: Sequence[Any]) -> dict[str, Any] | None:
    if not points:
        return None
    best = min(points, key=lambda point: float(point.benchmark_log_loss))
    return {
        "family": str(best.family),
        "sweep_id": str(best.sweep_id),
        "row_order": int(best.row_order),
        "row_label": str(best.row_label),
        "run_id": str(best.run_id),
        "geometry_label": f"{int(best.d_icl)}x{int(best.layers)}",
        "benchmark_log_loss": float(best.benchmark_log_loss),
        "validation_loss": None if best.validation_loss is None else float(best.validation_loss),
        "throughput_tokens_per_second": (
            None
            if getattr(best, "throughput_tokens_per_second", None) is None
            else float(best.throughput_tokens_per_second)
        ),
        "end_to_end_wall_seconds": (
            None
            if getattr(best, "end_to_end_wall_seconds", None) is None
            else float(best.end_to_end_wall_seconds)
        ),
        "tokens_per_step": float(best.tokens_per_step),
        "steps": int(best.steps),
    }


def sweep_lineage_entries(*, sweep_id: str, index_path: Path | None = None) -> list[dict[str, Any]]:
    index = load_system_delta_index_payload(index_path)
    lineage_reversed: list[dict[str, Any]] = []
    current_id = str(sweep_id)
    seen: set[str] = set()
    while current_id:
        if current_id in seen:
            raise RuntimeError(f"cycle detected in sweep lineage for {sweep_id!r}")
        seen.add(current_id)
        sweep_info = index.sweeps.get(current_id)
        if sweep_info is None:
            raise RuntimeError(f"unknown sweep in lineage: {current_id!r}")
        lineage_reversed.append(
            {
                "sweep_id": current_id,
                "parent_sweep_id": _optional_string(sweep_info.parent_sweep_id),
                "status": str(sweep_info.status),
                "complexity_level": str(sweep_info.complexity_level),
                "anchor_run_id": _optional_string(sweep_info.anchor_run_id),
            }
        )
        parent_id = _optional_string(sweep_info.parent_sweep_id)
        if parent_id is None:
            break
        current_id = parent_id
    lineage = list(reversed(lineage_reversed))
    for depth, entry in enumerate(lineage):
        entry["depth"] = depth
    return lineage


def list_sweep_tree_entries(*, index_path: Path | None = None) -> list[dict[str, Any]]:
    index = load_system_delta_index_payload(index_path)
    children: dict[str | None, list[str]] = defaultdict(list)
    for sweep_id, sweep_info in index.sweeps.items():
        children[_optional_string(sweep_info.parent_sweep_id)].append(str(sweep_id))
    for child_ids in children.values():
        child_ids.sort()
    ordered: list[dict[str, Any]] = []

    def _visit(sweep_id: str, *, depth: int) -> None:
        sweep_info = index.sweeps[sweep_id]
        lineage = sweep_lineage_entries(sweep_id=sweep_id, index_path=index_path)
        ordered.append(
            {
                "sweep_id": sweep_id,
                "parent_sweep_id": _optional_string(sweep_info.parent_sweep_id),
                "status": str(sweep_info.status),
                "complexity_level": str(sweep_info.complexity_level),
                "anchor_run_id": _optional_string(sweep_info.anchor_run_id),
                "benchmark_manifest_path": str(sweep_info.benchmark_manifest_path),
                "control_baseline_id": str(sweep_info.control_baseline_id),
                "depth": depth,
                "lineage": lineage,
                "child_sweep_ids": list(children.get(sweep_id, [])),
            }
        )
        for child_sweep_id in children.get(sweep_id, []):
            _visit(child_sweep_id, depth=depth + 1)

    for root_sweep_id in children.get(None, []):
        _visit(root_sweep_id, depth=0)
    return ordered


def _scan_linked_scaling_studies(*, sweep_id: str, studies_root: Path | None = None) -> list[str]:
    root = (
        studies_root.expanduser().resolve()
        if studies_root is not None
        else default_scaling_studies_root()
    )
    if not root.exists():
        return []
    linked: list[str] = []
    for study_path in sorted(root.glob("*.yaml")):
        try:
            config = load_scaling_study_config(study_path=study_path)
        except RuntimeError:
            continue
        if any(ref.sweep_id == sweep_id for ref in config.sweeps):
            linked.append(config.study_id)
            continue
        if config.phase1_reference_sweep_id == sweep_id:
            linked.append(config.study_id)
    return linked


def validate_sweep_contract(
    *,
    queue: Mapping[str, Any],
    index_path: Path | None = None,
) -> list[str]:
    issues: list[str] = []
    index = load_system_delta_index_payload(index_path)
    sweep_id = str(queue["sweep_id"])
    index_entry = index.sweeps.get(sweep_id)
    if index_entry is None:
        issues.append(f"sweep {sweep_id!r} is missing from the sweep index")
        return issues
    for key in (
        "anchor_run_id",
        "complexity_level",
        "benchmark_manifest_path",
        "control_baseline_id",
    ):
        queue_value = _optional_string(queue.get(key))
        index_value = _optional_string(getattr(index_entry, key))
        if queue_value != index_value:
            issues.append(
                f"{sweep_id}: {key} mismatch between queue metadata ({queue_value!r}) "
                f"and index ({index_value!r})"
            )
    anchor_context = queue.get("anchor_context")
    if isinstance(anchor_context, Mapping):
        anchor_context_run_id = _optional_string(anchor_context.get("run_id"))
        queue_anchor_run_id = _optional_string(queue.get("anchor_run_id"))
        if anchor_context_run_id != queue_anchor_run_id:
            issues.append(
                f"{sweep_id}: anchor_context.run_id {anchor_context_run_id!r} does not match "
                f"anchor_run_id {queue_anchor_run_id!r}"
            )
    benchmark_manifest_path = Path(str(queue["benchmark_manifest_path"]))
    issues.extend(
        f"{sweep_id}: {issue}"
        for issue in _default_anchor_manifest_contract_issues(benchmark_manifest_path)
    )
    return issues


def build_sweep_navigation_payload(
    *,
    queue: Mapping[str, Any],
    index_path: Path | None = None,
    studies_root: Path | None = None,
) -> dict[str, Any]:
    rows = cast(Sequence[Any], queue.get("rows", []))
    corpus_refs = _unique_corpus_refs(rows)
    return {
        "lineage": sweep_lineage_entries(sweep_id=str(queue["sweep_id"]), index_path=index_path),
        "linked_scaling_study_ids": _scan_linked_scaling_studies(
            sweep_id=str(queue["sweep_id"]),
            studies_root=studies_root,
        ),
        "contract": {
            "benchmark_manifest_path": str(queue["benchmark_manifest_path"]),
            "default_anchor_benchmark": _default_anchor_benchmark_contract(),
            "uses_default_anchor_benchmark": _uses_default_anchor_benchmark_contract(
                benchmark_manifest_path=str(queue["benchmark_manifest_path"]),
                control_baseline_id=_optional_string(queue.get("control_baseline_id")),
            ),
            "control_baseline_id": str(queue["control_baseline_id"]),
            "external_benchmarks": list(cast(Sequence[Any], queue.get("external_benchmarks", []))),
            "training_experiment": str(queue["training_experiment"]),
            "training_config_profile": str(queue["training_config_profile"]),
            "surface_role": str(queue["surface_role"]),
            "comparison_policy": str(queue["comparison_policy"]),
            "formal_external_reference": (
                dict(cast(Mapping[str, Any], queue["upstream_reference"]))
                if isinstance(queue.get("upstream_reference"), Mapping)
                else None
            ),
            "carried_in_family_baseline_run_id": _optional_string(queue.get("anchor_run_id")),
            "anchor_context_surface_labels": (
                dict(cast(Mapping[str, Any], cast(Mapping[str, Any], queue["anchor_context"])["surface_labels"]))
                if isinstance(queue.get("anchor_context"), Mapping)
                and isinstance(cast(Mapping[str, Any], queue["anchor_context"]).get("surface_labels"), Mapping)
                else None
            ),
            "corpus_ref": corpus_refs[0] if len(corpus_refs) == 1 else None,
            "corpus_refs": corpus_refs,
        },
        "winner": _winner_from_rows(rows),
        "contract_issues": validate_sweep_contract(queue=queue, index_path=index_path),
    }


def _expected_family_counts(config: ScalingStudyConfig) -> dict[str, int | None]:
    counts: dict[str, int | None] = {}
    for sweep_ref in config.sweeps:
        if sweep_ref.family == "ns_core":
            counts[sweep_ref.family] = len(config.geometry_row_labels) * len(config.step_ladder)
        elif sweep_ref.family == "batch_critical":
            counts[sweep_ref.family] = len(config.batch_grad_accum_ladder) * len(config.step_ladder)
        else:
            counts[sweep_ref.family] = None
    return counts


def build_scaling_navigation_payload(
    *,
    config: ScalingStudyConfig,
    points: Sequence[Any],
    index_path: Path,
    catalog_path: Path,
    sweeps_root: Path,
) -> dict[str, Any]:
    queues: dict[str, Mapping[str, Any]] = {}
    for sweep_ref in config.sweeps:
        queues[sweep_ref.sweep_id] = load_system_delta_queue(
            sweep_id=sweep_ref.sweep_id,
            index_path=index_path,
            catalog_path=catalog_path,
            sweeps_root=sweeps_root,
        )
    benchmark_paths = {str(queue["benchmark_manifest_path"]) for queue in queues.values()}
    control_baselines = {str(queue["control_baseline_id"]) for queue in queues.values()}
    anchor_run_ids = {
        anchor_run_id
        for queue in queues.values()
        for anchor_run_id in [_optional_string(queue.get("anchor_run_id"))]
        if anchor_run_id is not None
    }
    training_experiments = {str(queue["training_experiment"]) for queue in queues.values()}
    training_profiles = {str(queue["training_config_profile"]) for queue in queues.values()}
    corpus_refs = {
        corpus_ref
        for queue in queues.values()
        for corpus_ref in [build_sweep_navigation_payload(queue=queue, index_path=index_path)["contract"]["corpus_ref"]]
        if corpus_ref is not None
    }
    issues: list[str] = []
    for sweep_ref in config.sweeps:
        queue = queues[sweep_ref.sweep_id]
        issues.extend(validate_sweep_contract(queue=queue, index_path=index_path))
    if len(benchmark_paths) != 1:
        issues.append(f"scaling study {config.study_id}: benchmark manifest mismatch across sweeps {sorted(benchmark_paths)!r}")
    if len(control_baselines) != 1:
        issues.append(f"scaling study {config.study_id}: control baseline mismatch across sweeps {sorted(control_baselines)!r}")
    if len(training_experiments) != 1:
        issues.append(
            f"scaling study {config.study_id}: training_experiment mismatch across sweeps {sorted(training_experiments)!r}"
        )
    if len(training_profiles) != 1:
        issues.append(
            f"scaling study {config.study_id}: training_config_profile mismatch across sweeps {sorted(training_profiles)!r}"
        )
    if len(anchor_run_ids) > 1:
        issues.append(
            f"scaling study {config.study_id}: carried baseline / anchor mismatch across sweeps {sorted(anchor_run_ids)!r}"
        )
    if len(corpus_refs) > 1:
        issues.append(
            f"scaling study {config.study_id}: corpus_ref mismatch across sweeps {sorted(corpus_refs)!r}"
        )
    if config.phase1_reference_sweep_id is not None:
        for sweep_ref in config.sweeps:
            lineage_ids = {
                entry["sweep_id"]
                for entry in sweep_lineage_entries(sweep_id=sweep_ref.sweep_id, index_path=index_path)
            }
            if config.phase1_reference_sweep_id not in lineage_ids:
                issues.append(
                    f"scaling study {config.study_id}: phase1_reference_sweep_id "
                    f"{config.phase1_reference_sweep_id!r} is not in lineage for {sweep_ref.sweep_id!r}"
                )
    actual_counts = Counter(str(point.family) for point in points)
    expected_counts = _expected_family_counts(config)
    completeness: dict[str, Any] = {
        "expected_counts_by_family": dict(expected_counts),
        "actual_counts_by_family": dict(actual_counts),
        "missing_counts_by_family": {
            family: None if expected is None else max(int(expected) - int(actual_counts.get(family, 0)), 0)
            for family, expected in expected_counts.items()
        },
    }
    completeness["all_expected_points_present"] = all(
        expected is None or int(actual_counts.get(family, 0)) == int(expected)
        for family, expected in expected_counts.items()
    )
    completeness["ns_core_expected_points_present"] = all(
        family != "ns_core" or expected is None or int(actual_counts.get(family, 0)) == int(expected)
        for family, expected in expected_counts.items()
    )
    output_root = config.output_root_path()
    fit_summary_path = output_root / "fit_summary.json"
    audit_summary_path = output_root / "audit" / "audit_summary.json"
    validation_overlay_path = config.validation_overlay_resolved_path()
    linked_sweeps: list[dict[str, Any]] = []
    for sweep_ref in config.sweeps:
        queue = queues[sweep_ref.sweep_id]
        linked_sweeps.append(
            {
                "name": sweep_ref.name,
                "family": sweep_ref.family,
                "sweep_id": sweep_ref.sweep_id,
                "status": str(queue.get("status", "unknown")),
                "lineage": sweep_lineage_entries(sweep_id=sweep_ref.sweep_id, index_path=index_path),
                "winner": _winner_from_rows(cast(Sequence[Any], queue.get("rows", []))),
            }
        )
    first_queue = next(iter(queues.values()))
    return {
        "linked_sweeps": linked_sweeps,
        "contract": {
            "benchmark_manifest_path": next(iter(benchmark_paths)) if len(benchmark_paths) == 1 else None,
            "default_anchor_benchmark": _default_anchor_benchmark_contract(),
            "uses_default_anchor_benchmark": (
                len(benchmark_paths) == 1
                and len(control_baselines) == 1
                and _uses_default_anchor_benchmark_contract(
                    benchmark_manifest_path=next(iter(benchmark_paths)),
                    control_baseline_id=next(iter(control_baselines)),
                )
            ),
            "control_baseline_id": next(iter(control_baselines)) if len(control_baselines) == 1 else None,
            "training_experiment": next(iter(training_experiments)) if len(training_experiments) == 1 else None,
            "training_config_profile": next(iter(training_profiles)) if len(training_profiles) == 1 else None,
            "corpus_ref": next(iter(corpus_refs)) if len(corpus_refs) == 1 else None,
            "carried_in_family_baseline_run_id": next(iter(anchor_run_ids)) if len(anchor_run_ids) == 1 else None,
            "formal_external_reference": (
                dict(cast(Mapping[str, Any], first_queue["upstream_reference"]))
                if isinstance(first_queue.get("upstream_reference"), Mapping)
                else None
            ),
            "phase1_reference_sweep_id": config.phase1_reference_sweep_id,
            "historical_context_studies": list(config.historical_context_studies),
            "primary_fit": None if config.primary_fit is None else dict(config.primary_fit),
            "frozen_contract": None if config.frozen_contract is None else dict(config.frozen_contract),
        },
        "winner": _winner_from_points(points),
        "fit_audit_state": {
            "validation_overlay_exists": bool(
                validation_overlay_path is not None and validation_overlay_path.exists()
            ),
            "fit_summary_exists": fit_summary_path.exists(),
            "audit_summary_exists": audit_summary_path.exists(),
            "full_scope_ready": bool(completeness["all_expected_points_present"]),
            "ns_core_ready": bool(completeness["ns_core_expected_points_present"]),
            "fit_summary_path": str(fit_summary_path),
            "audit_summary_path": str(audit_summary_path),
        },
        "completeness": completeness,
        "contract_issues": issues,
    }
