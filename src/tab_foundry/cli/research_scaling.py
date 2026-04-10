"""CLI wiring for `tab-foundry research scaling ...` commands."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from tab_foundry.cli.click_utils import (
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


def main(argv: list[str] | None = None) -> int:
    group = click.Group(
        name="research-scaling",
        commands={
            "fit": FIT_COMMAND,
            "inspect": INSPECT_COMMAND,
        },
    )
    return run_click_command(group, argv, prog_name="tab-foundry research scaling")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
