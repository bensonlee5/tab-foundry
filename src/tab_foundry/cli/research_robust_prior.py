"""CLI wiring for `tab-foundry research robust-prior ...` commands."""

from __future__ import annotations

import sys
from pathlib import Path

import click

from tab_foundry.cli.click_utils import (
    emit_payload,
    json_output_option,
    path_option,
    run_click_command,
)
from tab_foundry.research.robust_prior import (
    inspect_robust_prior_pilot,
    render_robust_prior_text,
    run_robust_prior_pilot,
)


def _study_options(func):
    return click.option("--study", required=False, help="Robust-prior study id")(  # type: ignore[misc]
        path_option(
            "study-path",
            required=False,
            help="Optional explicit robust-prior YAML path",
        )(
            path_option(
                "studies-root",
                required=False,
                help="Optional root for reference/robust_prior/",
            )(func)
        )
    )


def _require_study_selection(*, study: str | None, study_path: Path | None) -> None:
    if study is None and study_path is None:
        raise click.UsageError("provide --study or --study-path")


@click.command(name="run", help="Run one robust-prior pilot from an anchor checkpoint")
@_study_options
@path_option("dagzoo-root", required=True, help="Local dagzoo checkout root")
@json_output_option
def RUN_COMMAND(
    study: str | None,
    study_path: Path | None,
    studies_root: Path | None,
    dagzoo_root: Path,
    json_mode: bool,
) -> int:
    _require_study_selection(study=study, study_path=study_path)
    payload = run_robust_prior_pilot(
        study_id=study,
        study_path=study_path,
        studies_root=studies_root,
        dagzoo_root=dagzoo_root.expanduser().resolve(),
    )
    emit_payload(payload, json_mode=json_mode, render_text=render_robust_prior_text)
    return 0


@click.command(name="inspect", help="Inspect one robust-prior pilot and its completed rounds")
@_study_options
@json_output_option
def INSPECT_COMMAND(
    study: str | None,
    study_path: Path | None,
    studies_root: Path | None,
    json_mode: bool,
) -> int:
    _require_study_selection(study=study, study_path=study_path)
    payload = inspect_robust_prior_pilot(
        study_id=study,
        study_path=study_path,
        studies_root=studies_root,
    )
    emit_payload(payload, json_mode=json_mode, render_text=render_robust_prior_text)
    return 0


def main(argv: list[str] | None = None) -> int:
    group = click.Group(
        name="research-robust-prior",
        commands={
            "inspect": INSPECT_COMMAND,
            "run": RUN_COMMAND,
        },
    )
    return run_click_command(group, argv, prog_name="tab-foundry research robust-prior")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
