"""CLI wiring for `tab-foundry bench registry freeze-baseline`."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from tab_foundry.bench.control_baseline_freeze import (
    DEFAULT_BASELINE_ID,
    DEFAULT_BUDGET_CLASS,
    DEFAULT_CONFIG_PROFILE,
    DEFAULT_EXPERIMENT,
    freeze_control_baseline,
)
from tab_foundry.control_baseline_registry import default_control_baseline_registry_path
from tab_foundry.cli.click_utils import run_click_command


def _freeze_baseline_command(
    *,
    run_dir: Path,
    comparison_summary: Path,
    baseline_id: str,
    experiment: str,
    config_profile: str,
    budget_class: str,
    registry_path: Path,
) -> int:
    result = freeze_control_baseline(
        baseline_id=baseline_id,
        experiment=experiment,
        config_profile=config_profile,
        budget_class=budget_class,
        run_dir=run_dir,
        comparison_summary_path=comparison_summary,
        registry_path=registry_path,
    )
    print("Control baseline frozen:")
    print(f"  registry_path={result['registry_path']}")
    print(f"  baseline={result['baseline']}")
    return 0


@click.command(name="freeze-baseline", help="Freeze a control baseline")
@click.option("--run-dir", required=True, type=click.Path(path_type=Path), help="Completed tab-foundry run directory")
@click.option("--comparison-summary", required=True, type=click.Path(path_type=Path), help="Benchmark comparison_summary.json for the same run")
@click.option("--baseline-id", default=DEFAULT_BASELINE_ID, show_default=True, help="Registry id for the frozen baseline")
@click.option("--experiment", default=DEFAULT_EXPERIMENT, show_default=True, help="Logical experiment name stored in the registry entry")
@click.option("--config-profile", default=DEFAULT_CONFIG_PROFILE, show_default=True, help="Config profile name stored in the registry entry")
@click.option("--budget-class", default=DEFAULT_BUDGET_CLASS, show_default=True, help="Budget class label stored in the registry entry")
@click.option(
    "--registry-path",
    default=str(default_control_baseline_registry_path()),
    show_default=True,
    type=click.Path(path_type=Path),
    help="Control baseline registry JSON path",
)
def COMMAND(
    run_dir: Path,
    comparison_summary: Path,
    baseline_id: str,
    experiment: str,
    config_profile: str,
    budget_class: str,
    registry_path: Path,
) -> int:
    return _freeze_baseline_command(
        run_dir=run_dir,
        comparison_summary=comparison_summary,
        baseline_id=baseline_id,
        experiment=experiment,
        config_profile=config_profile,
        budget_class=budget_class,
        registry_path=registry_path,
    )


def main(argv: list[str] | None = None) -> int:
    return run_click_command(COMMAND, argv, prog_name="tab-foundry bench registry freeze-baseline")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
