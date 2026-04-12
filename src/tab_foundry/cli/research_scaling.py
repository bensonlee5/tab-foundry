"""CLI wiring for `tab-foundry research scaling ...` commands."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from tab_foundry.cli.click_utils import (
    POSITIVE_INT,
    emit_payload,
    json_output_option,
    path_option,
    run_click_command,
    sweep_path_options,
)
from tab_foundry.research.scaling.fit import (
    fit_scaling_study,
    inspect_scaling_study,
    render_scaling_study_text,
)
from tab_foundry.research.scaling.validation_backfill import (
    backfill_validation_study,
    render_validation_backfill_text,
)


def _study_options(func):
    return click.option("--study", required=False, help="Scaling study id")(  # type: ignore[misc]
        path_option(
            "study-path",
            required=False,
            help="Optional explicit scaling-study YAML path",
        )(
            path_option(
                "studies-root",
                required=False,
                help="Optional root for reference/scaling_studies/",
            )(func)
        )
    )


def _require_study_selection(*, study: str | None, study_path: Path | None) -> None:
    if study is None and study_path is None:
        raise click.UsageError("provide --study or --study-path")


@click.command(name="inspect", help="Inspect one scaling study and its completed points")
@_study_options
@json_output_option
@sweep_path_options(include_registry=True, include_sweeps_root=True)
def INSPECT_COMMAND(
    study: str | None,
    study_path: Path | None,
    studies_root: Path | None,
    json_mode: bool,
    catalog_path: Path,
    index_path: Path,
    sweeps_root: Path,
    registry_path: Path,
) -> int:
    _require_study_selection(study=study, study_path=study_path)
    payload = inspect_scaling_study(
        study_id=study,
        study_path=study_path,
        studies_root=studies_root,
        registry_path=registry_path.expanduser().resolve(),
        index_path=index_path.expanduser().resolve(),
        catalog_path=catalog_path.expanduser().resolve(),
        sweeps_root=sweeps_root.expanduser().resolve(),
    )
    emit_payload(payload, json_mode=json_mode, render_text=render_scaling_study_text)
    return 0


@click.command(name="fit", help="Fit one scaling study and write report artifacts")
@_study_options
@path_option(
    "out-root",
    required=False,
    help="Optional override for the study artifact root",
)
@json_output_option
@sweep_path_options(include_registry=True, include_sweeps_root=True)
def FIT_COMMAND(
    study: str | None,
    study_path: Path | None,
    studies_root: Path | None,
    out_root: Path | None,
    json_mode: bool,
    catalog_path: Path,
    index_path: Path,
    sweeps_root: Path,
    registry_path: Path,
) -> int:
    _require_study_selection(study=study, study_path=study_path)
    payload = fit_scaling_study(
        study_id=study,
        study_path=study_path,
        studies_root=studies_root,
        registry_path=registry_path.expanduser().resolve(),
        index_path=index_path.expanduser().resolve(),
        catalog_path=catalog_path.expanduser().resolve(),
        sweeps_root=sweeps_root.expanduser().resolve(),
        out_root=None if out_root is None else out_root.expanduser().resolve(),
    )
    emit_payload(payload, json_mode=json_mode, render_text=render_scaling_study_text)
    return 0


@click.command(
    name="backfill-validation",
    help="Backfill validation loss sidecars from completed scaling-study checkpoints",
)
@_study_options
@click.option(
    "--launch-gcs-root",
    "launch_gcs_roots",
    multiple=True,
    help="Launch root containing an artifacts/ subtree; may be gs:// or a local path",
)
@click.option(
    "--preseed-gcs-root",
    default=None,
    help="Optional preseed root containing repo-relative reusable train artifacts",
)
@path_option(
    "cache-root",
    required=False,
    help="Optional local overlay cache root for downloaded checkpoint artifacts",
)
@path_option(
    "out-root",
    required=False,
    help="Optional local output root for the validation backfill manifest",
)
@click.option("--val-batches", default=16, show_default=True, type=POSITIVE_INT)
@click.option("--device", default="cpu", show_default=True, type=click.Choice(("cpu", "cuda")))
@click.option("--start-order", default=None, type=POSITIVE_INT)
@click.option("--stop-after-order", default=None, type=POSITIVE_INT)
@click.option("--force", is_flag=True, help="Recompute even when a matching sidecar exists")
@click.option("--dry-run", is_flag=True, help="Resolve candidate rows and artifact sources only")
@json_output_option
@sweep_path_options(include_registry=True, include_sweeps_root=True)
def BACKFILL_VALIDATION_COMMAND(
    study: str | None,
    study_path: Path | None,
    studies_root: Path | None,
    launch_gcs_roots: tuple[str, ...],
    preseed_gcs_root: str | None,
    cache_root: Path | None,
    out_root: Path | None,
    val_batches: int,
    device: str,
    start_order: int | None,
    stop_after_order: int | None,
    force: bool,
    dry_run: bool,
    json_mode: bool,
    catalog_path: Path,
    index_path: Path,
    sweeps_root: Path,
    registry_path: Path,
) -> int:
    _require_study_selection(study=study, study_path=study_path)
    payload = backfill_validation_study(
        study_id=study,
        study_path=study_path,
        studies_root=studies_root,
        registry_path=registry_path.expanduser().resolve(),
        index_path=index_path.expanduser().resolve(),
        catalog_path=catalog_path.expanduser().resolve(),
        sweeps_root=sweeps_root.expanduser().resolve(),
        launch_gcs_roots=launch_gcs_roots,
        preseed_gcs_root=preseed_gcs_root,
        cache_root=None if cache_root is None else cache_root.expanduser().resolve(),
        out_root=None if out_root is None else out_root.expanduser().resolve(),
        val_batches=val_batches,
        device=device,
        start_order=start_order,
        stop_after_order=stop_after_order,
        force=force,
        dry_run=dry_run,
    )
    emit_payload(payload, json_mode=json_mode, render_text=render_validation_backfill_text)
    return 0


def main(argv: list[str] | None = None) -> int:
    group = click.Group(
        name="research-scaling",
        commands={
            "backfill-validation": BACKFILL_VALIDATION_COMMAND,
            "fit": FIT_COMMAND,
            "inspect": INSPECT_COMMAND,
        },
    )
    return run_click_command(group, argv, prog_name="tab-foundry research scaling")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
