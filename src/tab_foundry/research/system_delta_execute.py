"""Execute system-delta sweep rows through train, benchmark, and registration."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Mapping

from tab_foundry.research import system_delta
from tab_foundry.research.sweep import runner as _runner
from tab_foundry.research.sweep.artifacts import (
    ExecutionPaths,
    read_yaml,
    result_card_text,
    write_research_package,
    write_yaml,
)
from tab_foundry.research.sweep.queue_updates import (
    optional_metric as _optional_metric_impl,
    queue_metrics as _queue_metrics_impl,
    update_queue_row as _update_queue_row_impl,
)
from tab_foundry.research.sweep.selection import (
    parse_order_overrides as _parse_order_overrides_impl,
    select_queue_rows as _select_queue_rows_impl,
)
from tab_foundry.research.system_delta_promote import promote_anchor


DEFAULT_PRIOR_DUMP = _runner.DEFAULT_PRIOR_DUMP
DEFAULT_NANOTABPFN_ROOT = _runner.DEFAULT_NANOTABPFN_ROOT
DEFAULT_DEVICE = _runner.DEFAULT_DEVICE
DEFAULT_TRACK = _runner.DEFAULT_TRACK
DEFAULT_EXPERIMENT = _runner.DEFAULT_EXPERIMENT
DEFAULT_CONFIG_PROFILE = _runner.DEFAULT_CONFIG_PROFILE
DEFAULT_BUDGET_CLASS = _runner.DEFAULT_BUDGET_CLASS
DEFAULT_DECISION = _runner.DEFAULT_DECISION
DEFAULT_CONCLUSION = _runner.DEFAULT_CONCLUSION
_ALLOWED_DECISIONS = _runner.ALLOWED_DECISIONS

_row_id_for_order = _runner.row_id_for_order
_ensure_nanotabpfn_python = _runner.ensure_nanotabpfn_python
_completed_train_artifacts_exist = _runner.completed_train_artifacts_exist
_materialized_row_map = _runner.materialized_row_map
_apply_mapping = _runner.apply_mapping
_compose_cfg = _runner.compose_cfg
_sync_sweep_matrix = _runner.sync_sweep_matrix
_sync_active_aliases_if_active = _runner.sync_active_aliases_if_active
_optional_metric = _optional_metric_impl
_queue_metrics = _queue_metrics_impl
_read_yaml = read_yaml
_result_card_text = result_card_text
_update_queue_row = _update_queue_row_impl
_write_research_package = write_research_package
_write_yaml = write_yaml
parse_order_overrides = _parse_order_overrides_impl
select_queue_rows = _select_queue_rows_impl


def _run_row(**kwargs: Any) -> str:
    return _runner.run_row(**kwargs)


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
    return _runner.execute_sweep(
        sweep_id=sweep_id,
        prior_dump=prior_dump,
        nanotabpfn_root=nanotabpfn_root,
        device=device,
        fallback_python=fallback_python,
        orders=orders,
        start_order=start_order,
        stop_after_order=stop_after_order,
        include_completed=include_completed,
        decision_default=decision_default,
        conclusion_default=conclusion_default,
        decision_overrides=decision_overrides,
        conclusion_overrides=conclusion_overrides,
        promote_first_executed_row_to_anchor=promote_first_executed_row_to_anchor,
        paths=paths,
        run_row_fn=_run_row,
        sync_sweep_matrix_fn=_sync_sweep_matrix,
        sync_active_aliases_if_active_fn=_sync_active_aliases_if_active,
        promote_anchor_fn=promote_anchor,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Execute system-delta sweep rows")
    parser.add_argument("--sweep-id", default=None, help="Sweep id to execute; defaults to the active sweep")
    parser.add_argument(
        "--order",
        type=int,
        action="append",
        default=[],
        help="Explicit queue order to execute; repeatable",
    )
    parser.add_argument(
        "--start-order",
        type=int,
        default=None,
        help="Optional starting queue order for a contiguous range",
    )
    parser.add_argument(
        "--stop-after-order",
        type=int,
        default=None,
        help="Optional inclusive last queue order for a contiguous range",
    )
    parser.add_argument(
        "--include-completed",
        action="store_true",
        help="Allow explicitly selected completed rows to run again",
    )
    parser.add_argument(
        "--promote-first-executed-row-to-anchor",
        action="store_true",
        help="Promote the first executed row to the sweep anchor after it completes",
    )
    parser.add_argument("--prior-dump", default=str(DEFAULT_PRIOR_DUMP), help="Path to the nanoTabPFN prior dump")
    parser.add_argument(
        "--nanotabpfn-root",
        default=str(DEFAULT_NANOTABPFN_ROOT),
        help="Path to the nanoTabPFN checkout",
    )
    parser.add_argument("--device", default=DEFAULT_DEVICE, choices=("cpu", "cuda", "mps", "auto"))
    parser.add_argument(
        "--tab-foundry-python",
        default=str(system_delta.repo_root() / ".venv" / "bin" / "python"),
        help="Interpreter to expose under nanoTabPFN/.venv/bin/python",
    )
    parser.add_argument("--decision-default", default=DEFAULT_DECISION, choices=sorted(_ALLOWED_DECISIONS))
    parser.add_argument(
        "--conclusion-default",
        default=DEFAULT_CONCLUSION,
        help="Default conclusion recorded for executed rows",
    )
    parser.add_argument(
        "--decision-override",
        action="append",
        default=[],
        help="Per-order override like 7=keep",
    )
    parser.add_argument(
        "--conclusion-override",
        action="append",
        default=[],
        help="Per-order override like 7=Promote this surface.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    prior_dump = Path(str(args.prior_dump)).expanduser().resolve()
    nanotabpfn_root = Path(str(args.nanotabpfn_root)).expanduser().resolve()
    fallback_python = Path(str(args.tab_foundry_python)).expanduser().resolve()
    if not prior_dump.exists():
        raise RuntimeError(f"prior dump does not exist: {prior_dump}")
    if not fallback_python.exists():
        raise RuntimeError(f"tab-foundry interpreter does not exist: {fallback_python}")

    decision_overrides = parse_order_overrides(
        list(args.decision_override),
        arg_name="--decision-override",
    )
    conclusion_overrides = parse_order_overrides(
        list(args.conclusion_override),
        arg_name="--conclusion-override",
    )
    for decision in decision_overrides.values():
        if decision not in _ALLOWED_DECISIONS:
            raise RuntimeError(f"decision must be one of {sorted(_ALLOWED_DECISIONS)}, got {decision!r}")

    executed = execute_sweep(
        sweep_id=(None if args.sweep_id is None else str(args.sweep_id)),
        prior_dump=prior_dump,
        nanotabpfn_root=nanotabpfn_root,
        device=str(args.device),
        fallback_python=fallback_python,
        orders=list(args.order),
        start_order=(None if args.start_order is None else int(args.start_order)),
        stop_after_order=(None if args.stop_after_order is None else int(args.stop_after_order)),
        include_completed=bool(args.include_completed),
        decision_default=str(args.decision_default),
        conclusion_default=str(args.conclusion_default),
        decision_overrides=decision_overrides,
        conclusion_overrides=conclusion_overrides,
        promote_first_executed_row_to_anchor=bool(args.promote_first_executed_row_to_anchor),
    )
    target_sweep = "active" if args.sweep_id is None else str(args.sweep_id)
    print(
        "Queue execution complete.",
        f"sweep_id={target_sweep}",
        f"executed_rows={len(executed)}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
