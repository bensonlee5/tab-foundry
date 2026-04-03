"""CLI wiring for `tab-foundry bench registry register-run`."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from tab_foundry.benchmark_registry import default_benchmark_run_registry_path
from tab_foundry.bench.run_registration import DEFAULT_BUDGET_CLASS, ALLOWED_DECISIONS, register_benchmark_run
from tab_foundry.cli.click_utils import run_click_command


def _register_run_command(
    *,
    run_id: str,
    track: str,
    run_dir: Path,
    comparison_summary: Path,
    experiment: str,
    config_profile: str | None,
    budget_class: str,
    decision: str,
    conclusion: str,
    parent_run_id: str | None,
    anchor_run_id: str | None,
    prior_dir: Path | None,
    control_baseline_id: str | None,
    sweep_id: str | None,
    delta_id: str | None,
    parent_sweep_id: str | None,
    queue_order: int | None,
    run_kind: str | None,
    registry_path: Path,
) -> int:
    result = register_benchmark_run(
        run_id=run_id,
        track=track,
        experiment=experiment,
        config_profile=str(config_profile or experiment),
        budget_class=budget_class,
        run_dir=run_dir,
        comparison_summary_path=comparison_summary,
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
    print("Benchmark run registered:")
    print(f"  registry_path={result['registry_path']}")
    print(f"  run={result['run']}")
    return 0


@click.command(name="register-run", help="Register a benchmark run")
@click.option("--run-id", required=True, help="Canonical registry id for the run")
@click.option("--track", required=True, help="Logical track label, e.g. binary_ladder or many_class_branch")
@click.option("--run-dir", required=True, type=click.Path(path_type=Path), help="Completed tab-foundry run directory")
@click.option("--comparison-summary", required=True, type=click.Path(path_type=Path), help="Benchmark comparison_summary.json for the same run")
@click.option("--experiment", required=True, help="Logical experiment name stored in the registry")
@click.option("--config-profile", default=None, help="Config profile stored in the registry entry; defaults to --experiment")
@click.option("--budget-class", default=DEFAULT_BUDGET_CLASS, show_default=True, help="Budget class label stored in the registry entry")
@click.option("--decision", required=True, type=click.Choice(ALLOWED_DECISIONS), help="Human review decision stored with the run")
@click.option("--conclusion", required=True, help="One-line keep/reject/defer conclusion")
@click.option("--parent-run-id", default=None, help="Optional previous-stage benchmark run id")
@click.option("--anchor-run-id", default=None, help="Optional frozen anchor run id")
@click.option("--prior-dir", default=None, type=click.Path(path_type=Path), help="Optional prior-training artifact directory")
@click.option("--control-baseline-id", default=None, help="Optional frozen control baseline id associated with the run")
@click.option("--sweep-id", default=None, help="Optional sweep id associated with the run")
@click.option("--delta-id", default=None, help="Optional delta id associated with the run")
@click.option("--parent-sweep-id", default=None, help="Optional parent sweep id associated with the run")
@click.option("--queue-order", default=None, type=int, help="Optional positive queue order within the sweep")
@click.option("--run-kind", default=None, type=click.Choice(("primary", "followup")), help="Optional sweep-local run kind")
@click.option(
    "--registry-path",
    default=str(default_benchmark_run_registry_path()),
    show_default=True,
    type=click.Path(path_type=Path),
    help="Benchmark run registry JSON path",
)
def COMMAND(
    run_id: str,
    track: str,
    run_dir: Path,
    comparison_summary: Path,
    experiment: str,
    config_profile: str | None,
    budget_class: str,
    decision: str,
    conclusion: str,
    parent_run_id: str | None,
    anchor_run_id: str | None,
    prior_dir: Path | None,
    control_baseline_id: str | None,
    sweep_id: str | None,
    delta_id: str | None,
    parent_sweep_id: str | None,
    queue_order: int | None,
    run_kind: str | None,
    registry_path: Path,
) -> int:
    return _register_run_command(
        run_id=run_id,
        track=track,
        run_dir=run_dir,
        comparison_summary=comparison_summary,
        experiment=experiment,
        config_profile=config_profile,
        budget_class=budget_class,
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


def main(argv: list[str] | None = None) -> int:
    return run_click_command(COMMAND, argv, prog_name="tab-foundry bench registry register-run")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
