"""CLI wiring for `tab-foundry bench compare`."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

import click

from tab_foundry.bench.comparison_contract import (
    DEFAULT_NANOTABPFN_SEEDS,
    DEFAULT_NANOTABPFN_STEPS,
    DEFAULT_TABICL_CLASSIFIER_CHECKPOINT_VERSION,
    DEFAULT_TABICL_REGRESSOR_CHECKPOINT_VERSION,
    BenchmarkComparisonConfig,
)
from tab_foundry.bench.comparison_reporting import optional_non_empty_string
from tab_foundry.bench.comparison_runtime import (
    run_nanotabpfn_benchmark,
)
from tab_foundry.control_baseline_registry import default_control_baseline_registry_path
from tab_foundry.external_benchmarks import (
    ALLOWED_EXTERNAL_BENCHMARKS,
    DEFAULT_CLI_EXTERNAL_BENCHMARKS,
    EXTERNAL_BENCHMARK_NANOTABPFN,
    EXTERNAL_BENCHMARK_TABICLV2,
    normalize_external_benchmarks,
)
from tab_foundry.cli.click_utils import DEVICE_CHOICES, run_click_command

__all__ = [
    "COMMAND",
    "EXTERNAL_BENCHMARK_TABICLV2",
    "main",
]


def _default_out_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return Path("/tmp") / f"tab_foundry_benchmark_{stamp}"


def _compare_command(
    *,
    tab_foundry_run_dir: Path,
    nanotabpfn_root: Path | None,
    nanotabpfn_prior_dump: Path | None,
    external_benchmark: tuple[str, ...],
    tabicl_root: Path | None,
    tab_realdata_hub_root: Path | None,
    tabicl_classifier_checkpoint_version: str,
    tabicl_regressor_checkpoint_version: str,
    out_root: Path | None,
    device: str,
    nanotabpfn_steps: int,
    nanotabpfn_seeds: int,
    control_baseline_id: str | None,
    control_baseline_registry: Path,
    benchmark_manifest_path: Path | None,
) -> int:
    requested_external_benchmarks = normalize_external_benchmarks(
        list(external_benchmark),
        default=DEFAULT_CLI_EXTERNAL_BENCHMARKS,
        context="CLI external benchmarks",
    )
    summary = run_nanotabpfn_benchmark(
        BenchmarkComparisonConfig(
            tab_foundry_run_dir=tab_foundry_run_dir,
            out_root=_default_out_root() if out_root is None else out_root,
            nanotabpfn_root=nanotabpfn_root,
            nanotab_prior_dump=nanotabpfn_prior_dump,
            device=device,
            nanotabpfn_steps=nanotabpfn_steps,
            nanotabpfn_seeds=nanotabpfn_seeds,
            control_baseline_id=control_baseline_id,
            control_baseline_registry=control_baseline_registry,
            benchmark_manifest_path=benchmark_manifest_path,
            external_benchmarks=requested_external_benchmarks,
            tabicl_root=tabicl_root,
            tab_realdata_hub_root=tab_realdata_hub_root,
            tabicl_classifier_checkpoint_version=tabicl_classifier_checkpoint_version,
            tabicl_regressor_checkpoint_version=tabicl_regressor_checkpoint_version,
        )
    )
    print("benchmark comparison complete:")
    print(f"  dataset_count={summary['dataset_count']}")
    print(f"  tab_foundry={summary['tab_foundry']}")
    primary_external_benchmark = optional_non_empty_string(summary.get("primary_external_benchmark"))
    if primary_external_benchmark is not None:
        print(f"  primary_external_benchmark={primary_external_benchmark}")
    if "nanotabpfn" in summary:
        print(f"  nanotabpfn={summary['nanotabpfn']}")
    if "tabiclv2" in summary:
        print(f"  tabiclv2={summary['tabiclv2']}")
    print(f"  artifacts={summary['artifacts']}")
    return 0


@click.command(name="compare", help="Run the benchmark comparison against external baselines")
@click.option(
    "--tab-foundry-run-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Completed tab-foundry run directory with checkpoint snapshots",
)
@click.option(
    "--nanotabpfn-root",
    default=None,
    type=click.Path(path_type=Path),
    help="Local nanoTabPFN checkout",
)
@click.option(
    "--nanotabpfn-prior-dump",
    default=None,
    type=click.Path(path_type=Path),
    help="Path to nanoTabPFN prior dump (.h5)",
)
@click.option(
    "--external-benchmark",
    "external_benchmark",
    multiple=True,
    type=click.Choice(ALLOWED_EXTERNAL_BENCHMARKS),
    help="Ordered external benchmark to run; repeat to add a secondary comparator. Defaults to tabiclv2.",
)
@click.option(
    "--tabicl-root",
    default=None,
    type=click.Path(path_type=Path),
    help="Local TabICLv2 checkout used when tabiclv2 is selected",
)
@click.option(
    "--tab-realdata-hub-root",
    default=None,
    type=click.Path(path_type=Path),
    help="Explicit local tab-realdata-hub checkout used by external benchmark helpers",
)
@click.option(
    "--tabicl-classifier-checkpoint-version",
    default=DEFAULT_TABICL_CLASSIFIER_CHECKPOINT_VERSION,
    show_default=True,
    help="TabICLv2 classifier checkpoint version used when tabiclv2 is selected",
)
@click.option(
    "--tabicl-regressor-checkpoint-version",
    default=DEFAULT_TABICL_REGRESSOR_CHECKPOINT_VERSION,
    show_default=True,
    help="TabICLv2 regressor checkpoint version used when tabiclv2 is selected",
)
@click.option("--out-root", default=None, type=click.Path(path_type=Path), help="Output directory root")
@click.option(
    "--device",
    default="auto",
    show_default=True,
    type=click.Choice(DEVICE_CHOICES),
    help="Benchmark device",
)
@click.option(
    "--nanotabpfn-steps",
    default=DEFAULT_NANOTABPFN_STEPS,
    show_default=True,
    type=int,
    help="nanoTabPFN training steps",
)
@click.option(
    "--nanotabpfn-seeds",
    default=DEFAULT_NANOTABPFN_SEEDS,
    show_default=True,
    type=int,
    help="Number of nanoTabPFN random seeds",
)
@click.option(
    "--control-baseline-id",
    default=None,
    help="Optional frozen control baseline id to copy into comparison_summary.json",
)
@click.option(
    "--control-baseline-registry",
    default=str(default_control_baseline_registry_path()),
    show_default=True,
    type=click.Path(path_type=Path),
    help="Control baseline registry JSON path used with --control-baseline-id",
)
@click.option(
    "--benchmark-manifest-path",
    default=None,
    type=click.Path(path_type=Path),
    help="Optional repo-local manifest-backed benchmark surface path",
)
def COMMAND(
    tab_foundry_run_dir: Path,
    nanotabpfn_root: Path | None,
    nanotabpfn_prior_dump: Path | None,
    external_benchmark: tuple[str, ...],
    tabicl_root: Path | None,
    tab_realdata_hub_root: Path | None,
    tabicl_classifier_checkpoint_version: str,
    tabicl_regressor_checkpoint_version: str,
    out_root: Path | None,
    device: str,
    nanotabpfn_steps: int,
    nanotabpfn_seeds: int,
    control_baseline_id: str | None,
    control_baseline_registry: Path,
    benchmark_manifest_path: Path | None,
) -> int:
    requested_external_benchmarks = normalize_external_benchmarks(
        list(external_benchmark),
        default=DEFAULT_CLI_EXTERNAL_BENCHMARKS,
        context="CLI external benchmarks",
    )
    if EXTERNAL_BENCHMARK_NANOTABPFN in requested_external_benchmarks and nanotabpfn_root is None:
        raise click.UsageError(
            "--nanotabpfn-root is required when --external-benchmark nanotabpfn is selected"
        )
    if EXTERNAL_BENCHMARK_TABICLV2 in requested_external_benchmarks and tabicl_root is None:
        raise click.UsageError(
            "--tabicl-root is required when tabiclv2 benchmarking is enabled"
        )
    return _compare_command(
        tab_foundry_run_dir=tab_foundry_run_dir,
        nanotabpfn_root=nanotabpfn_root,
        nanotabpfn_prior_dump=nanotabpfn_prior_dump,
        external_benchmark=external_benchmark,
        tabicl_root=tabicl_root,
        tab_realdata_hub_root=tab_realdata_hub_root,
        tabicl_classifier_checkpoint_version=tabicl_classifier_checkpoint_version,
        tabicl_regressor_checkpoint_version=tabicl_regressor_checkpoint_version,
        out_root=out_root,
        device=device,
        nanotabpfn_steps=nanotabpfn_steps,
        nanotabpfn_seeds=nanotabpfn_seeds,
        control_baseline_id=control_baseline_id,
        control_baseline_registry=control_baseline_registry,
        benchmark_manifest_path=benchmark_manifest_path,
    )


def main(argv: list[str] | None = None) -> int:
    return run_click_command(COMMAND, argv, prog_name="tab-foundry bench compare")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
