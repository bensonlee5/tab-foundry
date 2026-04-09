"""CLI wiring for `tab-foundry bench registry freeze-hardware-baseline`."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from tab_foundry.bench.hardware_architecture_freeze import (
    DEFAULT_SELECTION_RULE,
    freeze_hardware_architecture_baseline,
)
from tab_foundry.cli.click_utils import run_click_command
from tab_foundry.benchmark_registry import default_benchmark_run_registry_path
from tab_foundry.hardware_architecture_registry import default_hardware_architecture_registry_path


def _freeze_hardware_baseline_command(
    *,
    baseline_id: str,
    preferred_run_id: str,
    formal_anchor_run_id: str,
    baseline_run_id: str,
    evidence_run_ids: tuple[str, ...],
    rationale: str,
    decision: str,
    surface_role: str,
    runtime_profile: str | None,
    selection_rule: str,
    benchmark_registry_path: Path,
    registry_path: Path,
) -> int:
    result = freeze_hardware_architecture_baseline(
        baseline_id=baseline_id,
        preferred_run_id=preferred_run_id,
        formal_anchor_run_id=formal_anchor_run_id,
        baseline_run_id=baseline_run_id,
        evidence_run_ids=evidence_run_ids,
        rationale=rationale,
        decision=decision,
        surface_role=surface_role,
        runtime_profile=runtime_profile,
        selection_rule=selection_rule,
        benchmark_registry_path=benchmark_registry_path,
        registry_path=registry_path,
    )
    print("Hardware architecture baseline frozen:")
    print(f"  registry_path={result['registry_path']}")
    print(f"  baseline={result['baseline']}")
    return 0


@click.command(name="freeze-hardware-baseline", help="Freeze a hardware architecture baseline")
@click.option("--baseline-id", required=True, help="Registry id for the frozen hardware baseline")
@click.option("--preferred-run-id", required=True, help="Benchmark registry run id chosen as the preferred architecture")
@click.option("--formal-anchor-run-id", required=True, help="Formal lane-level anchor run id retained for interpretation")
@click.option("--baseline-run-id", required=True, help="Carried in-family baseline run id used for the decision")
@click.option("--evidence-run-id", "evidence_run_ids", multiple=True, required=True, help="Benchmark registry run ids used as evidence for the hardware baseline")
@click.option("--rationale", required=True, help="One-line rationale for the preferred architecture decision")
@click.option("--decision", required=True, type=click.Choice(("keep", "defer", "reject")), help="Human review decision stored with the hardware baseline")
@click.option("--surface-role", required=True, help="Surface role tied to the preferred architecture decision")
@click.option("--runtime-profile", default=None, help="Optional runtime profile label; defaults to the preferred run config profile")
@click.option("--selection-rule", default=DEFAULT_SELECTION_RULE, show_default=True, help="Selection rule recorded with the hardware baseline")
@click.option(
    "--benchmark-registry-path",
    default=str(default_benchmark_run_registry_path()),
    show_default=True,
    type=click.Path(path_type=Path),
    help="Benchmark run registry JSON path used as the evidence source",
)
@click.option(
    "--registry-path",
    default=str(default_hardware_architecture_registry_path()),
    show_default=True,
    type=click.Path(path_type=Path),
    help="Hardware architecture baseline registry JSON path",
)
def COMMAND(
    baseline_id: str,
    preferred_run_id: str,
    formal_anchor_run_id: str,
    baseline_run_id: str,
    evidence_run_ids: tuple[str, ...],
    rationale: str,
    decision: str,
    surface_role: str,
    runtime_profile: str | None,
    selection_rule: str,
    benchmark_registry_path: Path,
    registry_path: Path,
) -> int:
    return _freeze_hardware_baseline_command(
        baseline_id=baseline_id,
        preferred_run_id=preferred_run_id,
        formal_anchor_run_id=formal_anchor_run_id,
        baseline_run_id=baseline_run_id,
        evidence_run_ids=evidence_run_ids,
        rationale=rationale,
        decision=decision,
        surface_role=surface_role,
        runtime_profile=runtime_profile,
        selection_rule=selection_rule,
        benchmark_registry_path=benchmark_registry_path,
        registry_path=registry_path,
    )


def main(argv: list[str] | None = None) -> int:
    return run_click_command(
        COMMAND,
        argv,
        prog_name="tab-foundry bench registry freeze-hardware-baseline",
    )


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
