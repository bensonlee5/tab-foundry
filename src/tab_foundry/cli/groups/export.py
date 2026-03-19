"""Export CLI group."""

from __future__ import annotations

import argparse
from pathlib import Path

from tab_foundry.export.exporter import export_checkpoint, validate_export_bundle


def _run_bundle(args: argparse.Namespace) -> int:
    result = export_checkpoint(
        checkpoint_path=Path(str(args.checkpoint)),
        out_dir=Path(str(args.out_dir)),
        artifact_version=str(args.artifact_version),
    )
    print(
        "Export complete:",
        f"bundle_dir={result.bundle_dir}",
        f"manifest={result.manifest_path}",
        f"schema={result.schema_version}",
    )
    return 0


def _run_validate(args: argparse.Namespace) -> int:
    validated = validate_export_bundle(Path(str(args.bundle_dir)))
    print(
        "Export bundle valid:",
        f"schema={validated.manifest.schema_version}",
        f"task={validated.manifest.task}",
        f"model={validated.manifest.model.arch}",
    )
    return 0


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("export", help="Export workflows")
    nested = parser.add_subparsers(dest="export_command", required=True)

    bundle_parser = nested.add_parser("bundle", help="Export checkpoint to inference bundle")
    bundle_parser.add_argument("--checkpoint", required=True, help="Input training checkpoint path")
    bundle_parser.add_argument("--out-dir", required=True, help="Output bundle directory")
    bundle_parser.add_argument(
        "--artifact-version",
        default="tab-foundry-export-v3",
        help="Inference artifact schema version",
    )
    bundle_parser.set_defaults(func=_run_bundle)

    validate_parser = nested.add_parser("validate", help="Validate an inference export bundle")
    validate_parser.add_argument("--bundle-dir", required=True, help="Bundle directory path")
    validate_parser.set_defaults(func=_run_validate)
