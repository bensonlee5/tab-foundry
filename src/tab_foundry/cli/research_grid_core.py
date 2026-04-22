"""CLI wiring for `tab-foundry research grid-core ...` commands."""

from __future__ import annotations

from pathlib import Path

import click

from tab_foundry.bench.openml_benchmark import resolve_tab_foundry_best_checkpoint
from tab_foundry.cli.click_utils import (
    POSITIVE_INT,
    device_option,
    emit_payload,
    json_output_option,
    path_option,
    run_click_command,
)
from tab_foundry.research.grid_core_diagnostic import (
    GRID_CORE_CHUNK_SCOPES,
    GRID_CORE_INTERVENTION_MODES,
    render_grid_core_perturbation_text,
    run_grid_core_perturbation_diagnostic,
)


def _resolve_checkpoint_selection(
    *,
    run_dir: Path | None,
    checkpoint: Path | None,
) -> Path:
    if checkpoint is not None:
        return checkpoint.expanduser().resolve()
    if run_dir is None:
        raise click.UsageError("provide --checkpoint or --run-dir")
    return resolve_tab_foundry_best_checkpoint(run_dir)


def _perturb_checkpoint_command(
    *,
    run_dir: Path | None,
    checkpoint: Path | None,
    benchmark_manifest_path: Path | None,
    out_dir: Path | None,
    device: str,
    repeat_count: tuple[int, ...],
    chunk_scope: str,
    mode: tuple[str, ...],
    json_mode: bool,
) -> int:
    resolved_checkpoint = _resolve_checkpoint_selection(
        run_dir=run_dir,
        checkpoint=checkpoint,
    )
    modes = mode or GRID_CORE_INTERVENTION_MODES
    payload = run_grid_core_perturbation_diagnostic(
        checkpoint_path=resolved_checkpoint,
        benchmark_manifest_path=benchmark_manifest_path,
        out_dir=out_dir,
        device=device,
        repeat_counts=repeat_count,
        chunk_scope=chunk_scope,
        modes=modes,
    )
    emit_payload(
        payload,
        json_mode=json_mode,
        render_text=render_grid_core_perturbation_text,
    )
    return 0


@click.command(
    name="perturb-checkpoint",
    help="Evaluate ablate/repeat perturbations of contiguous grid-sandwich core chunks",
)
@path_option(
    "run-dir",
    required=False,
    help="Completed tab-foundry run directory; used to resolve checkpoints/best.pt",
)
@path_option(
    "checkpoint",
    required=False,
    help="Explicit grid_sandwich checkpoint to perturb; overrides --run-dir",
)
@path_option(
    "benchmark-manifest-path",
    required=False,
    help="Manifest-backed benchmark surface; defaults to the repo medium OpenML manifest",
)
@path_option(
    "out-dir",
    required=False,
    help="Output directory for JSON and Markdown diagnostic artifacts",
)
@device_option()
@click.option(
    "--repeat-count",
    default=(2, 4),
    show_default=True,
    multiple=True,
    type=POSITIVE_INT,
    help="Total repeat applications to evaluate for repeat_chunk; repeat the flag for multiple values.",
)
@click.option(
    "--chunk-scope",
    default="all",
    show_default=True,
    type=click.Choice(GRID_CORE_CHUNK_SCOPES),
    help="Contiguous chunks to evaluate",
)
@click.option(
    "--mode",
    "mode",
    multiple=True,
    type=click.Choice(GRID_CORE_INTERVENTION_MODES),
    help="Perturbation mode to run; repeat for multiple. Defaults to both modes.",
)
@json_output_option
def PERTURB_CHECKPOINT_COMMAND(
    run_dir: Path | None,
    checkpoint: Path | None,
    benchmark_manifest_path: Path | None,
    out_dir: Path | None,
    device: str,
    repeat_count: tuple[int, ...],
    chunk_scope: str,
    mode: tuple[str, ...],
    json_mode: bool,
) -> int:
    return _perturb_checkpoint_command(
        run_dir=run_dir,
        checkpoint=checkpoint,
        benchmark_manifest_path=benchmark_manifest_path,
        out_dir=out_dir,
        device=device,
        repeat_count=repeat_count,
        chunk_scope=chunk_scope,
        mode=mode,
        json_mode=json_mode,
    )


def main(argv: list[str] | None = None) -> int:
    return run_click_command(
        PERTURB_CHECKPOINT_COMMAND,
        argv,
        prog_name="tab-foundry research grid-core perturb-checkpoint",
    )
