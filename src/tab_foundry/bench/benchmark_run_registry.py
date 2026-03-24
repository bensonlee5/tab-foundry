"""Canonical benchmark-run registry helpers."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Sequence

from tab_foundry.bench.artifacts import write_json
from tab_foundry.bench.registry.paths import (
    normalize_path_value as _normalize_path_value_impl,
    project_root as _project_root_impl,
    resolve_config_path as _resolve_config_path_impl,
    resolve_registry_path_value as _resolve_registry_path_value_impl,
)
from tab_foundry.bench.registry.run_derivation import (
    comparison_delta as _comparison_delta_impl,
    derive_benchmark_run_entry as _derive_benchmark_run_entry_impl,
    derive_benchmark_run_record as _derive_benchmark_run_record_impl,
    empty_registry as _empty_registry_impl,
    load_registry_payload as _load_registry_payload_impl,
    sweep_payload as _sweep_payload_impl,
    validate_record_payload as _validate_record_payload_impl,
    validate_run_entry as _validate_run_entry_impl,
)
from tab_foundry.bench.registry.schema import (
    ALLOWED_DECISIONS,
    DEFAULT_BUDGET_CLASS,
)
from tab_foundry.bench.registry.summary_metrics import (
    ensure_optional_finite_number as _ensure_optional_finite_number_impl,
)
from tab_foundry.bench.registry.storage import (
    ensure_registry_payload as _ensure_registry_payload_common,
    upsert_registry_entry as _upsert_registry_entry_common,
    utc_now as _utc_now_common,
)
from tab_foundry.bench.registry_common import copy_jsonable as _copy_jsonable


def project_root() -> Path:
    """Return the repository root for repo-relative artifact paths."""

    return _project_root_impl()


def _normalize_path_value(path: Path) -> str:
    return _normalize_path_value_impl(path, root_fn=project_root)


def resolve_registry_path_value(value: str) -> Path:
    """Resolve a registry path value to an absolute path."""

    return _resolve_registry_path_value_impl(value, root_fn=project_root)


def _resolve_config_path(raw_value: Any) -> Path:
    return _resolve_config_path_impl(raw_value, root_fn=project_root)


def default_benchmark_run_registry_path() -> Path:
    """Return the repo-tracked benchmark-run registry path."""

    return Path(__file__).resolve().with_name("benchmark_run_registry_v1.json")


def _utc_now() -> str:
    return _utc_now_common()


def _empty_registry() -> dict[str, Any]:
    return _empty_registry_impl()


def _sweep_payload(
    *,
    sweep_id: str | None,
    delta_id: str | None,
    parent_sweep_id: str | None,
    queue_order: int | None,
    run_kind: str | None,
) -> dict[str, Any] | None:
    return _sweep_payload_impl(
        sweep_id=sweep_id,
        delta_id=delta_id,
        parent_sweep_id=parent_sweep_id,
        queue_order=queue_order,
        run_kind=run_kind,
    )


def _load_registry_payload(path: Path, *, allow_missing: bool) -> dict[str, Any]:
    return _load_registry_payload_impl(path, allow_missing=allow_missing)


def load_benchmark_run_registry(path: Path | None = None) -> dict[str, Any]:
    """Load and validate the benchmark run registry."""

    return _load_registry_payload(path or default_benchmark_run_registry_path(), allow_missing=False)


def _ensure_registry_payload(path: Path | None = None) -> tuple[Path, dict[str, Any]]:
    return _ensure_registry_payload_common(
        path,
        default_path=default_benchmark_run_registry_path(),
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
        normalize_path_value_fn=_normalize_path_value,
        resolve_registry_path_value_fn=resolve_registry_path_value,
        resolve_config_path_fn=_resolve_config_path,
        utc_now_fn=_utc_now,
    )


def _validate_record_payload(payload: Any) -> None:
    _validate_record_payload_impl(payload)


def _validate_run_entry(entry: Any, *, run_id: str) -> None:
    _validate_run_entry_impl(entry, run_id=run_id)


def _comparison_delta(
    *,
    reference_run_id: str,
    current_metrics: Mapping[str, Any],
    reference_metrics: Mapping[str, Any],
) -> dict[str, Any]:
    return _comparison_delta_impl(
        reference_run_id=reference_run_id,
        current_metrics=current_metrics,
        reference_metrics=reference_metrics,
    )


def _ensure_optional_finite_number(value: Any, *, context: str) -> float | None:
    return _ensure_optional_finite_number_impl(value, context=context)


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
        comparison_delta_fn=_comparison_delta,
        validate_run_entry_fn=_validate_run_entry,
        utc_now_fn=_utc_now,
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
        validate_entry_fn=_validate_run_entry,
        registry_path=registry_path,
        default_path=default_benchmark_run_registry_path(),
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Register one completed benchmark-facing tab-foundry run"
    )
    parser.add_argument("--run-id", required=True, help="Canonical registry id for the run")
    parser.add_argument(
        "--track",
        required=True,
        help="Logical track label, e.g. binary_ladder or many_class_branch",
    )
    parser.add_argument("--run-dir", required=True, help="Completed tab-foundry run directory")
    parser.add_argument(
        "--comparison-summary",
        required=True,
        help="Benchmark comparison_summary.json for the same run",
    )
    parser.add_argument("--experiment", required=True, help="Logical experiment name stored in the registry")
    parser.add_argument(
        "--config-profile",
        default=None,
        help="Config profile stored in the registry entry; defaults to --experiment",
    )
    parser.add_argument(
        "--budget-class",
        default=DEFAULT_BUDGET_CLASS,
        help="Budget class label stored in the registry entry",
    )
    parser.add_argument(
        "--decision",
        required=True,
        choices=ALLOWED_DECISIONS,
        help="Human review decision stored with the run",
    )
    parser.add_argument("--conclusion", required=True, help="One-line keep/reject/defer conclusion")
    parser.add_argument("--parent-run-id", default=None, help="Optional previous-stage benchmark run id")
    parser.add_argument("--anchor-run-id", default=None, help="Optional frozen anchor run id")
    parser.add_argument("--prior-dir", default=None, help="Optional prior-training artifact directory")
    parser.add_argument(
        "--control-baseline-id",
        default=None,
        help="Optional frozen control baseline id associated with the run",
    )
    parser.add_argument("--sweep-id", default=None, help="Optional sweep id associated with the run")
    parser.add_argument("--delta-id", default=None, help="Optional delta id associated with the run")
    parser.add_argument(
        "--parent-sweep-id",
        default=None,
        help="Optional parent sweep id associated with the run",
    )
    parser.add_argument(
        "--queue-order",
        default=None,
        type=int,
        help="Optional positive queue order within the sweep",
    )
    parser.add_argument(
        "--run-kind",
        default=None,
        choices=("primary", "followup"),
        help="Optional sweep-local run kind",
    )
    parser.add_argument(
        "--registry-path",
        default=str(default_benchmark_run_registry_path()),
        help="Benchmark run registry JSON path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result = register_benchmark_run(
        run_id=str(args.run_id),
        track=str(args.track),
        experiment=str(args.experiment),
        config_profile=str(args.config_profile or args.experiment),
        budget_class=str(args.budget_class),
        run_dir=Path(str(args.run_dir)),
        comparison_summary_path=Path(str(args.comparison_summary)),
        decision=str(args.decision),
        conclusion=str(args.conclusion),
        parent_run_id=None if args.parent_run_id is None else str(args.parent_run_id),
        anchor_run_id=None if args.anchor_run_id is None else str(args.anchor_run_id),
        prior_dir=None if args.prior_dir is None else Path(str(args.prior_dir)),
        control_baseline_id=(
            None if args.control_baseline_id is None else str(args.control_baseline_id)
        ),
        sweep_id=None if args.sweep_id is None else str(args.sweep_id),
        delta_id=None if args.delta_id is None else str(args.delta_id),
        parent_sweep_id=None if args.parent_sweep_id is None else str(args.parent_sweep_id),
        queue_order=None if args.queue_order is None else int(args.queue_order),
        run_kind=None if args.run_kind is None else str(args.run_kind),
        registry_path=Path(str(args.registry_path)),
    )
    print("Benchmark run registered:")
    print(f"  registry_path={result['registry_path']}")
    print(f"  run={result['run']}")
    return 0
