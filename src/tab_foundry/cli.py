"""CLI entrypoints."""

from __future__ import annotations

import argparse
from typing import Sequence
from pathlib import Path

from tab_foundry.config import compose_config
from tab_foundry.data.manifest import build_manifest
from tab_foundry.export.exporter import export_checkpoint, validate_export_bundle
from tab_foundry.training.evaluate import evaluate_checkpoint
from tab_foundry.training.trainer import train


def _cmd_build_manifest(args: argparse.Namespace) -> int:
    roots = [Path(path).expanduser() for path in args.data_root]
    summary = build_manifest(
        data_roots=roots,
        out_path=Path(args.out_manifest),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        filter_policy=str(args.filter_policy),
    )
    print(
        "Manifest built:",
        f"path={summary.out_path}",
        f"filter_policy={summary.filter_policy}",
        f"discovered={summary.discovered_records}",
        f"excluded={summary.excluded_records}",
        f"total={summary.total_records}",
        f"train={summary.train_records}",
        f"val={summary.val_records}",
        f"test={summary.test_records}",
    )
    if summary.filter_status_counts:
        counts = ", ".join(
            f"{status}={count}" for status, count in summary.filter_status_counts.items()
        )
        print("Filter status counts:", counts)
    for warning in summary.warnings:
        print("Warning:", warning)
    return 0


def _cmd_train(args: argparse.Namespace) -> int:
    cfg = compose_config(args.overrides)
    result = train(cfg)
    print(
        "Training complete:",
        f"output_dir={result.output_dir}",
        f"best={result.best_checkpoint}",
        f"latest={result.latest_checkpoint}",
        f"step={result.global_step}",
        f"metrics={result.metrics}",
    )
    return 0


def _cmd_eval(args: argparse.Namespace) -> int:
    overrides = list(args.overrides)
    if args.checkpoint is not None:
        overrides.append(f"eval.checkpoint={args.checkpoint}")
    if args.split is not None:
        overrides.append(f"eval.split={args.split}")

    cfg = compose_config(overrides)
    result = evaluate_checkpoint(cfg)
    print("Evaluation complete:", f"checkpoint={result.checkpoint}", f"metrics={result.metrics}")
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
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


def _cmd_validate_export(args: argparse.Namespace) -> int:
    validated = validate_export_bundle(Path(str(args.bundle_dir)))
    print(
        "Export bundle valid:",
        f"schema={validated.manifest.schema_version}",
        f"task={validated.manifest.task}",
        f"model={validated.manifest.model.arch}",
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="tab-foundry tooling")
    sub = parser.add_subparsers(dest="command", required=True)

    p_manifest = sub.add_parser(
        "build-manifest",
        help="Build manifest parquet from dagzoo packed shard outputs",
    )
    p_manifest.add_argument(
        "--data-root",
        action="append",
        required=True,
        help="Input dagzoo data root",
    )
    p_manifest.add_argument("--out-manifest", required=True, help="Output manifest parquet path")
    p_manifest.add_argument("--train-ratio", type=float, default=0.90)
    p_manifest.add_argument("--val-ratio", type=float, default=0.05)
    p_manifest.add_argument(
        "--filter-policy",
        choices=("include_all", "accepted_only"),
        default="include_all",
        help="Dataset selection policy based on dagzoo filter metadata",
    )
    p_manifest.set_defaults(func=_cmd_build_manifest)

    p_train = sub.add_parser("train", help="Train from Hydra config")
    p_train.add_argument("overrides", nargs="*", help="Hydra override strings")
    p_train.set_defaults(func=_cmd_train)

    p_eval = sub.add_parser("eval", help="Evaluate checkpoint")
    p_eval.add_argument("--checkpoint", default=None, help="Checkpoint override")
    p_eval.add_argument("--split", default=None, help="Eval split override")
    p_eval.add_argument("overrides", nargs="*", help="Hydra override strings")
    p_eval.set_defaults(func=_cmd_eval)

    p_export = sub.add_parser("export", help="Export checkpoint to inference bundle")
    p_export.add_argument("--checkpoint", required=True, help="Input training checkpoint path")
    p_export.add_argument("--out-dir", required=True, help="Output bundle directory")
    p_export.add_argument(
        "--artifact-version",
        default="tab-foundry-export-v1",
        help="Inference artifact schema version",
    )
    p_export.set_defaults(func=_cmd_export)

    p_validate = sub.add_parser("validate-export", help="Validate an inference export bundle")
    p_validate.add_argument("--bundle-dir", required=True, help="Bundle directory path")
    p_validate.set_defaults(func=_cmd_validate_export)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
