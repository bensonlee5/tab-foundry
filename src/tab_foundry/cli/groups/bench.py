"""Benchmark CLI group."""

from __future__ import annotations

import click

import tab_foundry.cli.bench_bounce_diagnosis as bounce_diagnosis_cli
import tab_foundry.cli.bench_bundle_openml as bundle_openml_cli
import tab_foundry.cli.bench_compare as compare_cli
import tab_foundry.cli.bench_control_baseline_freeze as control_baseline_freeze_cli
import tab_foundry.cli.bench_env_bootstrap as env_bootstrap_cli
import tab_foundry.cli.bench_materialize_openml_bundle as materialize_openml_cli
import tab_foundry.cli.bench_run_registration as run_registration_cli
import tab_foundry.cli.bench_smoke_dagzoo as dagzoo_smoke_cli
import tab_foundry.cli.bench_smoke_iris as iris_smoke_cli
import tab_foundry.cli.bench_tune as tune_cli
from tab_foundry.cli.click_utils import GROUP_KWARGS


@click.group(name="bench", help="Benchmark workflows", **GROUP_KWARGS)
def GROUP() -> None:
    """Benchmark workflows."""


@click.group(name="smoke", help="Smoke harnesses", **GROUP_KWARGS)
def _smoke_group() -> None:
    """Smoke harnesses."""


_smoke_group.add_command(iris_smoke_cli.COMMAND)
_smoke_group.add_command(dagzoo_smoke_cli.COMMAND)


@click.group(name="env", help="Benchmark environment helpers", **GROUP_KWARGS)
def _env_group() -> None:
    """Benchmark environment helpers."""


_env_group.add_command(env_bootstrap_cli.COMMAND)


@click.group(name="bundle", help="Benchmark bundle workflows", **GROUP_KWARGS)
def _bundle_group() -> None:
    """Benchmark bundle workflows."""


_bundle_group.add_command(bundle_openml_cli.COMMAND)


@click.group(name="registry", help="Benchmark registry workflows", **GROUP_KWARGS)
def _registry_group() -> None:
    """Benchmark registry workflows."""


_registry_group.add_command(run_registration_cli.COMMAND)
_registry_group.add_command(control_baseline_freeze_cli.COMMAND)


@click.group(name="diagnose", help="Benchmark diagnosis flows", **GROUP_KWARGS)
def _diagnose_group() -> None:
    """Benchmark diagnosis flows."""


_diagnose_group.add_command(bounce_diagnosis_cli.COMMAND)


GROUP.add_command(_smoke_group)
GROUP.add_command(tune_cli.COMMAND)
GROUP.add_command(compare_cli.COMMAND)
GROUP.add_command(materialize_openml_cli.COMMAND)
GROUP.add_command(_env_group)
GROUP.add_command(_bundle_group)
GROUP.add_command(_registry_group)
GROUP.add_command(_diagnose_group)
