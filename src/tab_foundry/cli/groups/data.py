"""Data CLI group."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess

from tab_foundry.cli.helpers import register_delegate_leaf
from tab_foundry.data.dagzoo_workflow import (
    DagzooGenerateManifestConfig,
    run_dagzoo_generate_manifest,
)
from tab_foundry.data.manifest import build_manifest


_DEVICE_CHOICES = ("auto", "cpu", "cuda", "mps")
_HARDWARE_POLICY_CHOICES = ("none", "cuda_tiered_v1")
_MISSINGNESS_MECHANISM_CHOICES = ("none", "mcar", "mar", "mnar")


def _positive_int(raw: str) -> int:
    value = int(raw)
    if value <= 0:
        raise argparse.ArgumentTypeError(f"Expected a positive integer, got {raw}.")
    return value


def _seed_32bit_int(raw: str) -> int:
    value = int(raw)
    if value < 0 or value > 4_294_967_295:
        raise argparse.ArgumentTypeError(f"Expected a 32-bit unsigned seed, got {raw}.")
    return value


def _finite_float(raw: str, *, flag: str) -> float:
    try:
        value = float(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"Invalid {flag} value {raw!r}.") from exc
    if value != value or value in (float("inf"), float("-inf")):
        raise argparse.ArgumentTypeError(f"Invalid {flag} value {raw!r}.")
    return value


def _missing_rate(raw: str) -> float:
    value = _finite_float(raw, flag="--missing-rate")
    if value < 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError("Expected --missing-rate in [0, 1].")
    return value


def _missing_fraction(raw: str) -> float:
    value = _finite_float(raw, flag="--missing-mar-observed-fraction")
    if value <= 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError("Expected --missing-mar-observed-fraction in (0, 1].")
    return value


def _positive_float(raw: str, *, flag: str) -> float:
    value = _finite_float(raw, flag=flag)
    if value <= 0.0:
        raise argparse.ArgumentTypeError(f"Expected {flag} > 0.")
    return value


def _validate_split_ratios(*, train_ratio: float, val_ratio: float) -> None:
    if train_ratio <= 0 or val_ratio < 0 or train_ratio + val_ratio >= 1:
        raise SystemExit(
            "invalid split ratios: expected --train-ratio > 0, "
            "--val-ratio >= 0, and --train-ratio + --val-ratio < 1"
        )


def _print_manifest_summary(summary) -> None:
    print(
        "Manifest built:",
        f"path={summary.out_path}",
        f"filter_policy={summary.filter_policy}",
        f"missing_value_policy={summary.missing_value_policy}",
        f"discovered={summary.discovered_records}",
        f"excluded={summary.excluded_records}",
        f"excluded_for_missing_values={summary.excluded_for_missing_values}",
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
    if summary.missing_value_status_counts:
        counts = ", ".join(
            f"{status}={count}" for status, count in summary.missing_value_status_counts.items()
        )
        print("Missing-value status counts:", counts)
    for warning in summary.warnings:
        print("Warning:", warning)


def _run_build_manifest(args: argparse.Namespace) -> int:
    _validate_split_ratios(train_ratio=float(args.train_ratio), val_ratio=float(args.val_ratio))
    roots = [Path(path).expanduser() for path in args.data_root]
    summary = build_manifest(
        data_roots=roots,
        out_path=Path(args.out_manifest),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        filter_policy=str(args.filter_policy),
        missing_value_policy=str(args.missing_value_policy),
    )
    _print_manifest_summary(summary)
    return 0


def _run_dagzoo_generate_manifest(args: argparse.Namespace) -> int:
    _validate_split_ratios(train_ratio=float(args.train_ratio), val_ratio=float(args.val_ratio))
    try:
        result = run_dagzoo_generate_manifest(
            DagzooGenerateManifestConfig(
                dagzoo_root=Path(str(args.dagzoo_root)),
                dagzoo_config=Path(str(args.dagzoo_config)),
                handoff_root=Path(str(args.handoff_root)),
                out_manifest=Path(str(args.out_manifest)),
                num_datasets=int(args.num_datasets),
                seed=None if args.seed is None else int(args.seed),
                rows=None if args.rows is None else str(args.rows),
                device=None if args.device is None else str(args.device),
                hardware_policy=str(args.hardware_policy),
                diagnostics=bool(args.diagnostics),
                diagnostics_out_dir=(
                    None if args.diagnostics_out_dir is None else Path(str(args.diagnostics_out_dir))
                ),
                missing_rate=None if args.missing_rate is None else float(args.missing_rate),
                missing_mechanism=(
                    None if args.missing_mechanism is None else str(args.missing_mechanism)
                ),
                missing_mar_observed_fraction=(
                    None
                    if args.missing_mar_observed_fraction is None
                    else float(args.missing_mar_observed_fraction)
                ),
                missing_mar_logit_scale=(
                    None
                    if args.missing_mar_logit_scale is None
                    else float(args.missing_mar_logit_scale)
                ),
                missing_mnar_logit_scale=(
                    None
                    if args.missing_mnar_logit_scale is None
                    else float(args.missing_mnar_logit_scale)
                ),
                train_ratio=float(args.train_ratio),
                val_ratio=float(args.val_ratio),
                filter_policy=str(args.filter_policy),
                missing_value_policy=str(args.missing_value_policy),
            )
        )
    except subprocess.CalledProcessError as exc:
        return int(exc.returncode)
    print(f"Dagzoo handoff manifest: {result.handoff.handoff_manifest_path}")
    print(f"Dagzoo generated dir: {result.handoff.generated_dir}")
    print(f"Output manifest: {result.summary.out_path}")
    _print_manifest_summary(result.summary)
    return 0


def _run_manifest_inspect(argv=None) -> int:
    from tab_foundry.cli.data_inspect import main as inspect_main

    return inspect_main(None if argv is None else list(argv))


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    parser = subparsers.add_parser("data", help="Data workflows")
    nested = parser.add_subparsers(dest="data_command", required=True)

    build_parser = nested.add_parser(
        "build-manifest",
        help="Build manifest parquet from dagzoo packed shard outputs",
    )
    build_parser.add_argument(
        "--data-root",
        action="append",
        required=True,
        help="Input dagzoo data root",
    )
    build_parser.add_argument("--out-manifest", required=True, help="Output manifest parquet path")
    build_parser.add_argument(
        "--train-ratio",
        type=lambda raw: _finite_float(raw, flag="--train-ratio"),
        default=0.90,
    )
    build_parser.add_argument(
        "--val-ratio",
        type=lambda raw: _finite_float(raw, flag="--val-ratio"),
        default=0.05,
    )
    build_parser.add_argument(
        "--filter-policy",
        choices=("include_all", "accepted_only"),
        default="include_all",
        help="Dataset selection policy based on dagzoo filter metadata",
    )
    build_parser.add_argument(
        "--missing-value-policy",
        choices=("allow_any", "forbid_any"),
        default="allow_any",
        help="Dataset selection policy for NaN/Inf-containing inputs",
    )
    build_parser.set_defaults(func=_run_build_manifest)

    register_delegate_leaf(
        nested,
        "manifest-inspect",
        help="Inspect one manifest parquet and optionally preflight compatibility",
        delegate=_run_manifest_inspect,
    )

    dagzoo_parser = nested.add_parser("dagzoo", help="dagzoo-backed data workflows")
    dagzoo_nested = dagzoo_parser.add_subparsers(dest="dagzoo_command", required=True)

    generate_manifest_parser = dagzoo_nested.add_parser(
        "generate-manifest",
        help="Generate a dagzoo corpus and build a tab-foundry manifest",
    )
    generate_manifest_parser.add_argument(
        "--dagzoo-root",
        required=True,
        help="Local dagzoo checkout root",
    )
    generate_manifest_parser.add_argument(
        "--dagzoo-config",
        required=True,
        help="dagzoo config path (absolute or relative to --dagzoo-root)",
    )
    generate_manifest_parser.add_argument(
        "--handoff-root",
        required=True,
        help="dagzoo handoff root written by `dagzoo generate --handoff-root`",
    )
    generate_manifest_parser.add_argument(
        "--out-manifest",
        required=True,
        help="Output manifest parquet path",
    )
    generate_manifest_parser.add_argument(
        "--num-datasets",
        type=_positive_int,
        default=10,
        help="Number of dagzoo datasets to generate",
    )
    generate_manifest_parser.add_argument(
        "--seed",
        type=_seed_32bit_int,
        default=None,
        help="Optional 32-bit run seed override",
    )
    generate_manifest_parser.add_argument(
        "--rows",
        default=None,
        help="Optional dagzoo rows override",
    )
    generate_manifest_parser.add_argument(
        "--device",
        choices=_DEVICE_CHOICES,
        default=None,
        help="Optional dagzoo device override",
    )
    generate_manifest_parser.add_argument(
        "--hardware-policy",
        choices=_HARDWARE_POLICY_CHOICES,
        default="none",
        help="Explicit dagzoo hardware policy",
    )
    generate_manifest_parser.add_argument(
        "--diagnostics",
        action="store_true",
        help="Enable dagzoo diagnostics coverage artifacts",
    )
    generate_manifest_parser.add_argument(
        "--diagnostics-out-dir",
        default=None,
        help="Optional dagzoo diagnostics artifact directory",
    )
    generate_manifest_parser.add_argument(
        "--missing-rate",
        type=_missing_rate,
        default=None,
        help="Optional dagzoo missing-rate override in [0, 1]",
    )
    generate_manifest_parser.add_argument(
        "--missing-mechanism",
        choices=_MISSINGNESS_MECHANISM_CHOICES,
        default=None,
        help="Optional dagzoo missingness mechanism override",
    )
    generate_manifest_parser.add_argument(
        "--missing-mar-observed-fraction",
        type=_missing_fraction,
        default=None,
        help="Optional dagzoo MAR observed-feature fraction override",
    )
    generate_manifest_parser.add_argument(
        "--missing-mar-logit-scale",
        type=lambda raw: _positive_float(raw, flag="--missing-mar-logit-scale"),
        default=None,
        help="Optional dagzoo MAR logit scale override",
    )
    generate_manifest_parser.add_argument(
        "--missing-mnar-logit-scale",
        type=lambda raw: _positive_float(raw, flag="--missing-mnar-logit-scale"),
        default=None,
        help="Optional dagzoo MNAR logit scale override",
    )
    generate_manifest_parser.add_argument(
        "--train-ratio",
        type=lambda raw: _finite_float(raw, flag="--train-ratio"),
        default=0.90,
    )
    generate_manifest_parser.add_argument(
        "--val-ratio",
        type=lambda raw: _finite_float(raw, flag="--val-ratio"),
        default=0.05,
    )
    generate_manifest_parser.add_argument(
        "--filter-policy",
        choices=("include_all", "accepted_only"),
        default="include_all",
        help="Dataset selection policy based on dagzoo filter metadata",
    )
    generate_manifest_parser.add_argument(
        "--missing-value-policy",
        choices=("allow_any", "forbid_any"),
        default="allow_any",
        help="Dataset selection policy for NaN/Inf-containing inputs",
    )
    generate_manifest_parser.set_defaults(func=_run_dagzoo_generate_manifest)
