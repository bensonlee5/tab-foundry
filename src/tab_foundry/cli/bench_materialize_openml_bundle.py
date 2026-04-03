"""CLI wiring for `tab-foundry bench materialize-openml-bundle`."""

from __future__ import annotations

from pathlib import Path
import sys

import click

from tab_realdata_hub.openml import materialize_bundle
from tab_foundry.repo_paths import repo_root
from tab_foundry.cli.click_utils import run_click_command


def _default_out_root(bundle_path: Path) -> Path:
    bundle_stem = bundle_path.expanduser().resolve().stem
    return repo_root() / "data" / "manifests" / "bench" / bundle_stem


def _materialize_bundle_command(
    *,
    bundle_path: Path,
    out_root: Path | None,
    force: bool,
    split_seed: int,
    test_size: float,
) -> int:
    result = materialize_bundle(
        bundle_path,
        _default_out_root(bundle_path) if out_root is None else out_root,
        force=force,
        split_seed=split_seed,
        test_size=test_size,
    )
    print(f"Materialized benchmark manifest: {result.manifest_path}")
    print(f"Packed shards: {result.data_root}")
    print(f"Tasks: {len(result.task_summaries)}")
    return 0


@click.command(
    name="materialize-openml-bundle",
    help="Materialize an OpenML bundle into a manifest-backed benchmark surface",
)
@click.option("--bundle-path", required=True, type=click.Path(path_type=Path), help="OpenML benchmark bundle JSON path")
@click.option(
    "--out-root",
    default=None,
    type=click.Path(path_type=Path),
    help="Output root for the materialized benchmark manifest surface",
)
@click.option("--force", is_flag=True, help="Overwrite an existing output root")
@click.option("--split-seed", default=0, show_default=True, type=int, help="Deterministic split seed")
@click.option("--test-size", default=0.20, show_default=True, type=float, help="Holdout ratio for packed shards")
def COMMAND(
    bundle_path: Path,
    out_root: Path | None,
    force: bool,
    split_seed: int,
    test_size: float,
) -> int:
    return _materialize_bundle_command(
        bundle_path=bundle_path,
        out_root=out_root,
        force=force,
        split_seed=split_seed,
        test_size=test_size,
    )


def main(argv: list[str] | None = None) -> int:
    return run_click_command(COMMAND, argv, prog_name="tab-foundry bench materialize-openml-bundle")


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
