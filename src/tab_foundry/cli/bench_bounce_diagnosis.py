"""CLI wiring for `tab-foundry bench diagnose bounce`."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import cast

import click

from tab_foundry.benchmark_registry import default_benchmark_run_registry_path
from tab_foundry.bench.bounce.config import (
    DEFAULT_BOUNCE_BOOTSTRAP_CONFIDENCE,
    DEFAULT_BOUNCE_BOOTSTRAP_SAMPLES,
    RerunMode,
    default_out_root,
)
from tab_foundry.bench.bounce.rerun import resolve_run_dir_from_registry
from tab_foundry.bench.bounce_diagnosis import BenchmarkBounceDiagnosisConfig, run_benchmark_bounce_diagnosis
from tab_foundry.cli.click_utils import DEVICE_CHOICES, run_click_command


def _bounce_diagnosis_command(
    *,
    run_dir: Path | None,
    run_id: str | None,
    registry_path: Path,
    out_root: Path | None,
    device: str,
    benchmark_manifest_path: Path | None,
    confirmation_benchmark_manifest_path: Path | None,
    bootstrap_samples: int,
    bootstrap_confidence: float,
    dense_checkpoint_every: int | None,
    dense_run_dir: Path | None,
    rerun_mode: str,
) -> int:
    if bool(run_dir) == bool(run_id):
        raise SystemExit("exactly one of --run-dir or --run-id must be provided")
    run_dir = (
        run_dir
        if run_dir is not None
        else resolve_run_dir_from_registry(
            str(run_id),
            registry_path=registry_path,
        )
    )
    summary = run_benchmark_bounce_diagnosis(
        BenchmarkBounceDiagnosisConfig(
            run_dir=run_dir,
            out_root=default_out_root(run_dir) if out_root is None else out_root,
            device=device,
            benchmark_manifest_path=benchmark_manifest_path,
            confirmation_benchmark_manifest_path=confirmation_benchmark_manifest_path,
            bootstrap_samples=bootstrap_samples,
            bootstrap_confidence=bootstrap_confidence,
            dense_checkpoint_every=dense_checkpoint_every,
            dense_run_dir=dense_run_dir,
            rerun_mode=cast(RerunMode, rerun_mode),
            run_id=run_id,
        )
    )
    print("benchmark bounce diagnosis complete:")
    print(f"  run_dir={summary['run_dir']}")
    print(f"  causes={summary['classification']['primary_causes']}")
    print(f"  artifacts={summary['artifacts']}")
    return 0


@click.command(name="bounce", help="Run the benchmark bounce diagnosis flow")
@click.option("--run-dir", default=None, type=click.Path(path_type=Path), help="Completed run directory to diagnose")
@click.option("--run-id", default=None, help="Benchmark registry run id to diagnose")
@click.option(
    "--registry-path",
    default=str(default_benchmark_run_registry_path()),
    show_default=True,
    type=click.Path(path_type=Path),
    help="Benchmark run registry used with --run-id",
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
    "--benchmark-manifest-path",
    default=None,
    type=click.Path(path_type=Path),
    help="Primary benchmark manifest path; defaults to the canonical medium manifest location",
)
@click.option(
    "--confirmation-benchmark-manifest-path",
    default=None,
    type=click.Path(path_type=Path),
    help="Optional confirmation benchmark manifest path; omit to stay on the primary no-missing surface only",
)
@click.option(
    "--bootstrap-samples",
    default=DEFAULT_BOUNCE_BOOTSTRAP_SAMPLES,
    show_default=True,
    type=int,
    help="Task-bootstrap samples per checkpoint",
)
@click.option(
    "--bootstrap-confidence",
    default=DEFAULT_BOUNCE_BOOTSTRAP_CONFIDENCE,
    show_default=True,
    type=float,
    help="Task-bootstrap confidence level",
)
@click.option(
    "--dense-checkpoint-every",
    default=None,
    type=int,
    help="Optional diagnosis-only rerun checkpoint cadence",
)
@click.option(
    "--dense-run-dir",
    default=None,
    type=click.Path(path_type=Path),
    help="Optional precomputed dense-checkpoint run directory",
)
@click.option(
    "--rerun-mode",
    default="none",
    show_default=True,
    type=click.Choice(("auto", "prior", "train", "none")),
    help="How to rerun when --dense-checkpoint-every is set",
)
def COMMAND(
    run_dir: Path | None,
    run_id: str | None,
    registry_path: Path,
    out_root: Path | None,
    device: str,
    benchmark_manifest_path: Path | None,
    confirmation_benchmark_manifest_path: Path | None,
    bootstrap_samples: int,
    bootstrap_confidence: float,
    dense_checkpoint_every: int | None,
    dense_run_dir: Path | None,
    rerun_mode: str,
) -> int:
    return _bounce_diagnosis_command(
        run_dir=run_dir,
        run_id=run_id,
        registry_path=registry_path,
        out_root=out_root,
        device=device,
        benchmark_manifest_path=benchmark_manifest_path,
        confirmation_benchmark_manifest_path=confirmation_benchmark_manifest_path,
        bootstrap_samples=bootstrap_samples,
        bootstrap_confidence=bootstrap_confidence,
        dense_checkpoint_every=dense_checkpoint_every,
        dense_run_dir=dense_run_dir,
        rerun_mode=rerun_mode,
    )


def main(argv: list[str] | None = None) -> int:
    return run_click_command(COMMAND, argv, prog_name="tab-foundry bench diagnose bounce")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
