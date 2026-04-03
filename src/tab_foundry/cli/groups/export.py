"""Export CLI group."""

from __future__ import annotations

from pathlib import Path

import click

from tab_foundry.export.exporter import export_checkpoint, validate_export_bundle
from tab_foundry.cli.click_utils import GROUP_KWARGS


def _run_bundle(*, checkpoint: Path, out_dir: Path, artifact_version: str) -> int:
    result = export_checkpoint(
        checkpoint_path=checkpoint,
        out_dir=out_dir,
        artifact_version=artifact_version,
    )
    print(
        "Export complete:",
        f"bundle_dir={result.bundle_dir}",
        f"manifest={result.manifest_path}",
        f"schema={result.schema_version}",
    )
    return 0


def _run_validate(*, bundle_dir: Path) -> int:
    validated = validate_export_bundle(bundle_dir)
    print(
        "Export bundle valid:",
        f"schema={validated.manifest.schema_version}",
        f"task={validated.manifest.task}",
        f"model={validated.manifest.model.arch}",
    )
    return 0


@click.group(name="export", help="Export workflows", **GROUP_KWARGS)
def GROUP() -> None:
    """Export workflows."""


@click.command(name="bundle", help="Export checkpoint to inference bundle")
@click.option("--checkpoint", required=True, type=click.Path(path_type=Path), help="Input training checkpoint path")
@click.option("--out-dir", required=True, type=click.Path(path_type=Path), help="Output bundle directory")
@click.option(
    "--artifact-version",
    default="tab-foundry-export-v3",
    show_default=True,
    help="Inference artifact schema version",
)
def BUNDLE_COMMAND(checkpoint: Path, out_dir: Path, artifact_version: str) -> int:
    return _run_bundle(checkpoint=checkpoint, out_dir=out_dir, artifact_version=artifact_version)


@click.command(name="validate", help="Validate an inference export bundle")
@click.option("--bundle-dir", required=True, type=click.Path(path_type=Path), help="Bundle directory path")
def VALIDATE_COMMAND(bundle_dir: Path) -> int:
    return _run_validate(bundle_dir=bundle_dir)


GROUP.add_command(BUNDLE_COMMAND)
GROUP.add_command(VALIDATE_COMMAND)
