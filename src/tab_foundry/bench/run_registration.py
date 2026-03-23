"""Canonical programmatic surface for benchmark-run registration."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import tab_foundry.benchmark_registry as read_benchmark_registry
from tab_foundry.bench.artifacts import write_json
from tab_foundry.bench.registry.paths import resolve_config_path as _resolve_config_path_impl
from tab_foundry.bench.registry.run_derivation import (
    comparison_delta as _comparison_delta_impl,
    derive_benchmark_run_entry as _derive_benchmark_run_entry_impl,
    derive_benchmark_run_record as _derive_benchmark_run_record_impl,
    empty_registry as _empty_registry_impl,
    validate_run_entry as _validate_run_entry_impl,
)
from tab_foundry.bench.registry.schema import ALLOWED_DECISIONS, DEFAULT_BUDGET_CLASS
from tab_foundry.bench.registry.storage import (
    ensure_registry_payload as _ensure_registry_payload_common,
    upsert_registry_entry as _upsert_registry_entry_common,
    utc_now as _utc_now_common,
)
from tab_foundry.bench.registry_common import copy_jsonable as _copy_jsonable
from tab_foundry.repo_paths import repo_root

__all__ = [
    "ALLOWED_DECISIONS",
    "DEFAULT_BUDGET_CLASS",
    "derive_benchmark_run_entry",
    "derive_benchmark_run_record",
    "register_benchmark_run",
    "upsert_benchmark_run_entry",
]


def _load_registry_payload(path: Path, *, allow_missing: bool) -> dict[str, Any]:
    resolved_path = path.expanduser().resolve()
    if allow_missing and not resolved_path.exists():
        return _empty_registry_impl()
    return read_benchmark_registry.load_benchmark_run_registry(resolved_path)


def _ensure_registry_payload(path: Path | None = None) -> tuple[Path, dict[str, Any]]:
    return _ensure_registry_payload_common(
        path,
        default_path=read_benchmark_registry.default_benchmark_run_registry_path(),
        load_registry_payload_fn=_load_registry_payload,
    )


def derive_benchmark_run_record(
    *,
    run_dir: Path,
    comparison_summary_path: Path,
    prior_dir: Path | None = None,
    benchmark_run_record_path: Path | None = None,
    sweep_id: str | None = None,
    delta_id: str | None = None,
    parent_sweep_id: str | None = None,
    queue_order: int | None = None,
    run_kind: str | None = None,
) -> dict[str, Any]:
    """Derive one machine-readable benchmark run record from current artifacts."""

    return _derive_benchmark_run_record_impl(
        run_dir=run_dir,
        comparison_summary_path=comparison_summary_path,
        prior_dir=prior_dir,
        benchmark_run_record_path=benchmark_run_record_path,
        sweep_id=sweep_id,
        delta_id=delta_id,
        parent_sweep_id=parent_sweep_id,
        queue_order=queue_order,
        run_kind=run_kind,
        normalize_path_value_fn=(
            lambda path: read_benchmark_registry.normalize_registry_path_value(
                path,
                root=repo_root(),
            )
        ),
        resolve_registry_path_value_fn=(
            lambda value: read_benchmark_registry.resolve_registry_path_value(
                value,
                root=repo_root(),
            )
        ),
        resolve_config_path_fn=(
            lambda raw_value: _resolve_config_path_impl(raw_value, root_fn=repo_root)
        ),
        utc_now_fn=_utc_now_common,
    )


def derive_benchmark_run_entry(
    *,
    run_id: str,
    track: str,
    experiment: str,
    config_profile: str,
    budget_class: str,
    run_dir: Path,
    comparison_summary_path: Path,
    decision: str,
    conclusion: str,
    parent_run_id: str | None = None,
    anchor_run_id: str | None = None,
    prior_dir: Path | None = None,
    control_baseline_id: str | None = None,
    sweep_id: str | None = None,
    delta_id: str | None = None,
    parent_sweep_id: str | None = None,
    queue_order: int | None = None,
    run_kind: str | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Derive one benchmark registry entry from benchmark artifacts and lineage."""

    return _derive_benchmark_run_entry_impl(
        run_id=run_id,
        track=track,
        experiment=experiment,
        config_profile=config_profile,
        budget_class=budget_class,
        run_dir=run_dir,
        comparison_summary_path=comparison_summary_path,
        decision=decision,
        conclusion=conclusion,
        parent_run_id=parent_run_id,
        anchor_run_id=anchor_run_id,
        prior_dir=prior_dir,
        control_baseline_id=control_baseline_id,
        sweep_id=sweep_id,
        delta_id=delta_id,
        parent_sweep_id=parent_sweep_id,
        queue_order=queue_order,
        run_kind=run_kind,
        registry_path=registry_path,
        ensure_registry_payload_fn=_ensure_registry_payload,
        derive_benchmark_run_record_fn=derive_benchmark_run_record,
        comparison_delta_fn=_comparison_delta_impl,
        validate_run_entry_fn=_validate_run_entry_impl,
        utc_now_fn=_utc_now_common,
        write_json_fn=write_json,
    )


def upsert_benchmark_run_entry(
    entry: Mapping[str, Any],
    *,
    registry_path: Path | None = None,
) -> Path:
    """Insert or replace one benchmark run entry in the registry."""

    return _upsert_registry_entry_common(
        entry,
        entry_id_key="run_id",
        validate_entry_fn=_validate_run_entry_impl,
        registry_path=registry_path,
        default_path=read_benchmark_registry.default_benchmark_run_registry_path(),
        load_registry_payload_fn=_load_registry_payload,
        entries_key="runs",
        write_json_fn=write_json,
        copy_jsonable_fn=_copy_jsonable,
    )


def register_benchmark_run(
    *,
    run_id: str,
    track: str,
    experiment: str,
    config_profile: str,
    budget_class: str,
    run_dir: Path,
    comparison_summary_path: Path,
    decision: str,
    conclusion: str,
    parent_run_id: str | None = None,
    anchor_run_id: str | None = None,
    prior_dir: Path | None = None,
    control_baseline_id: str | None = None,
    sweep_id: str | None = None,
    delta_id: str | None = None,
    parent_sweep_id: str | None = None,
    queue_order: int | None = None,
    run_kind: str | None = None,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    """Register one completed benchmark-facing run in the canonical registry."""

    entry = derive_benchmark_run_entry(
        run_id=run_id,
        track=track,
        experiment=experiment,
        config_profile=config_profile,
        budget_class=budget_class,
        run_dir=run_dir,
        comparison_summary_path=comparison_summary_path,
        decision=decision,
        conclusion=conclusion,
        parent_run_id=parent_run_id,
        anchor_run_id=anchor_run_id,
        prior_dir=prior_dir,
        control_baseline_id=control_baseline_id,
        sweep_id=sweep_id,
        delta_id=delta_id,
        parent_sweep_id=parent_sweep_id,
        queue_order=queue_order,
        run_kind=run_kind,
        registry_path=registry_path,
    )
    resolved_registry_path = upsert_benchmark_run_entry(entry, registry_path=registry_path)
    return {
        "registry_path": str(resolved_registry_path),
        "run": entry,
    }
