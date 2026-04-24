"""CLI wiring for `tab-foundry bench validate-openml-manifest`."""

from __future__ import annotations

from pathlib import Path
import sys

import click

from tab_foundry.bench.openml_benchmark.missingness_validation import (
    validate_openml_missingness_manifest,
)
from tab_foundry.cli.click_utils import emit_payload, json_output_option, run_click_command


def _validate_openml_manifest_command(
    *,
    manifest_path: Path,
    summary_out: Path | None,
    require_observed_missing: bool,
    json_mode: bool,
) -> int:
    summary = validate_openml_missingness_manifest(
        manifest_path,
        summary_out=summary_out,
        require_observed_missing=require_observed_missing,
    )
    if json_mode:
        emit_payload(summary, json_mode=True)
        return 0
    print(f"Validated OpenML benchmark manifest: {summary['manifest_path']}")
    print(f"Datasets: {summary['dataset_count']}")
    print(f"Missing feature cells: {summary['total_missing_feature_cells']}")
    print(f"Missing rows: {summary['total_missing_rows']}")
    if summary_out is not None:
        print(f"Summary: {summary_out.expanduser().resolve()}")
    return 0


@click.command(
    name="validate-openml-manifest",
    help="Validate observed missingness in a materialized OpenML benchmark manifest",
)
@click.option(
    "--manifest-path",
    required=True,
    type=click.Path(path_type=Path),
    help="Materialized OpenML benchmark manifest parquet",
)
@click.option(
    "--summary-out",
    default=None,
    type=click.Path(path_type=Path),
    help="Optional JSON summary artifact path",
)
@click.option(
    "--require-observed-missing/--no-require-observed-missing",
    default=True,
    show_default=True,
    help="Require every accepted benchmark dataset to contain observed feature NaNs",
)
@json_output_option
def COMMAND(
    manifest_path: Path,
    summary_out: Path | None,
    require_observed_missing: bool,
    json_mode: bool,
) -> int:
    return _validate_openml_manifest_command(
        manifest_path=manifest_path,
        summary_out=summary_out,
        require_observed_missing=require_observed_missing,
        json_mode=json_mode,
    )


def main(argv: list[str] | None = None) -> int:
    return run_click_command(COMMAND, argv, prog_name="tab-foundry bench validate-openml-manifest")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
