"""CLI wiring for `tab-foundry research sweep execute`."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from tab_foundry.cli.click_utils import run_click_command, sweep_path_options
from tab_foundry.research.sweep.artifacts import ExecutionPaths
from tab_foundry.research.sweep.execute import execute_sweep
from tab_foundry.research.sweep.row_execution import (
    ALLOWED_DECISIONS,
    DEFAULT_CONCLUSION,
    DEFAULT_DECISION,
    DEFAULT_DEVICE,
)
from tab_foundry.research.sweep.runtime_env import absolute_path_without_resolving_symlinks
from tab_foundry.research.sweep.selection import parse_order_overrides

from tab_foundry.research.sweep.paths_io import repo_root as _repo_root


def _execute_command(
    *,
    sweep_id: str,
    order: tuple[int, ...],
    start_order: int | None,
    stop_after_order: int | None,
    include_completed: bool,
    promote_first_executed_row_to_anchor: bool,
    nanotabpfn_prior_dump: Path | None,
    nanotabpfn_root: Path | None,
    reuse_nanotabpfn_only: bool,
    device: str,
    tab_foundry_python: Path,
    decision_default: str,
    conclusion_default: str,
    decision_override: tuple[str, ...],
    conclusion_override: tuple[str, ...],
    catalog_path: Path,
    index_path: Path,
    sweeps_root: Path,
    registry_path: Path,
) -> int:
    prior_dump = (
        None if nanotabpfn_prior_dump is None else nanotabpfn_prior_dump.expanduser().resolve()
    )
    resolved_nanotabpfn_root = (
        None if nanotabpfn_root is None else nanotabpfn_root.expanduser().resolve()
    )
    fallback_python = absolute_path_without_resolving_symlinks(tab_foundry_python)
    default_paths = ExecutionPaths.default()
    paths = ExecutionPaths(
        repo_root=default_paths.repo_root,
        index_path=index_path.expanduser().resolve(),
        catalog_path=catalog_path.expanduser().resolve(),
        sweeps_root=sweeps_root.expanduser().resolve(),
        registry_path=registry_path.expanduser().resolve(),
        program_path=default_paths.program_path,
        control_baseline_registry_path=default_paths.control_baseline_registry_path,
    )
    if prior_dump is not None and not prior_dump.exists():
        raise RuntimeError(f"prior dump does not exist: {prior_dump}")
    if not fallback_python.exists():
        raise RuntimeError(f"tab-foundry interpreter does not exist: {fallback_python}")

    decision_overrides = parse_order_overrides(
        list(decision_override),
        arg_name="--decision-override",
    )
    conclusion_overrides = parse_order_overrides(
        list(conclusion_override),
        arg_name="--conclusion-override",
    )
    for decision in decision_overrides.values():
        if decision not in ALLOWED_DECISIONS:
            raise RuntimeError(
                f"decision must be one of {sorted(ALLOWED_DECISIONS)}, got {decision!r}"
            )

    executed = execute_sweep(
        sweep_id=sweep_id,
        prior_dump=prior_dump,
        nanotabpfn_root=resolved_nanotabpfn_root,
        reuse_nanotabpfn_only=reuse_nanotabpfn_only,
        device=device,
        fallback_python=fallback_python,
        orders=list(order),
        start_order=start_order,
        stop_after_order=stop_after_order,
        include_completed=include_completed,
        decision_default=decision_default,
        conclusion_default=conclusion_default,
        decision_overrides=decision_overrides,
        conclusion_overrides=conclusion_overrides,
        promote_first_executed_row_to_anchor=promote_first_executed_row_to_anchor,
        paths=paths,
    )
    print(
        "Queue execution complete.",
        f"sweep_id={sweep_id}",
        f"executed_rows={len(executed)}",
        flush=True,
    )
    return 0


@click.command(name="execute", help="Execute selected system-delta sweep rows")
@click.option("--sweep-id", required=True, help="Sweep id to execute")
@click.option(
    "--order", multiple=True, type=int, help="Explicit queue order to execute; repeatable"
)
@click.option(
    "--start-order",
    default=None,
    type=int,
    help="Optional starting queue order for a contiguous range",
)
@click.option(
    "--stop-after-order",
    default=None,
    type=int,
    help="Optional inclusive last queue order for a contiguous range",
)
@click.option(
    "--include-completed",
    is_flag=True,
    help="Allow explicitly selected completed rows to run again",
)
@click.option(
    "--promote-first-executed-row-to-anchor",
    is_flag=True,
    help="Promote the first executed row to the sweep anchor after it completes",
)
@click.option(
    "--nanotabpfn-prior-dump",
    default=None,
    type=click.Path(path_type=Path),
    help="Optional path to the nanoTabPFN prior dump",
)
@click.option(
    "--nanotabpfn-root",
    type=click.Path(path_type=Path),
    help="Path to the nanoTabPFN checkout. Required only when the selected sweep uses the nanoTabPFN comparator.",
)
@click.option(
    "--reuse-nanotabpfn-only",
    is_flag=True,
    help="Do not launch a fresh nanoTabPFN helper; reuse a cached curve/error when available and otherwise record a synthetic nanoTabPFN reuse-missing outcome.",
)
@click.option(
    "--device",
    default=DEFAULT_DEVICE,
    show_default=True,
    help="Sweep execution device: cpu, cuda, or auto. Sweeps do not support mps.",
)
@click.option(
    "--tab-foundry-python",
    default=str(_repo_root() / ".venv" / "bin" / "python"),
    show_default=True,
    type=click.Path(path_type=Path),
    help="Interpreter to expose under nanoTabPFN/.venv/bin/python",
)
@click.option(
    "--decision-default",
    default=DEFAULT_DECISION,
    show_default=True,
    type=click.Choice(sorted(ALLOWED_DECISIONS)),
)
@click.option(
    "--conclusion-default",
    default=DEFAULT_CONCLUSION,
    show_default=True,
    help="Default conclusion recorded for executed rows",
)
@click.option("--decision-override", multiple=True, help="Per-order override like 7=keep")
@click.option(
    "--conclusion-override", multiple=True, help="Per-order override like 7=Promote this surface."
)
@sweep_path_options(include_registry=True, include_sweeps_root=True)
def COMMAND(
    sweep_id: str,
    order: tuple[int, ...],
    start_order: int | None,
    stop_after_order: int | None,
    include_completed: bool,
    promote_first_executed_row_to_anchor: bool,
    nanotabpfn_prior_dump: Path | None,
    nanotabpfn_root: Path | None,
    reuse_nanotabpfn_only: bool,
    device: str,
    tab_foundry_python: Path,
    decision_default: str,
    conclusion_default: str,
    decision_override: tuple[str, ...],
    conclusion_override: tuple[str, ...],
    catalog_path: Path,
    index_path: Path,
    sweeps_root: Path,
    registry_path: Path,
) -> int:
    return _execute_command(
        sweep_id=sweep_id,
        order=order,
        start_order=start_order,
        stop_after_order=stop_after_order,
        include_completed=include_completed,
        promote_first_executed_row_to_anchor=promote_first_executed_row_to_anchor,
        nanotabpfn_prior_dump=nanotabpfn_prior_dump,
        nanotabpfn_root=nanotabpfn_root,
        reuse_nanotabpfn_only=reuse_nanotabpfn_only,
        device=device,
        tab_foundry_python=tab_foundry_python,
        decision_default=decision_default,
        conclusion_default=conclusion_default,
        decision_override=decision_override,
        conclusion_override=conclusion_override,
        catalog_path=catalog_path,
        index_path=index_path,
        sweeps_root=sweeps_root,
        registry_path=registry_path,
    )


def main(argv: list[str] | None = None) -> int:
    return run_click_command(COMMAND, argv, prog_name="tab-foundry research sweep execute")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
