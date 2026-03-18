"""Data CLI group."""

from __future__ import annotations

import argparse
from pathlib import Path

from tab_foundry.data.manifest import build_manifest


def _run_build_manifest(args: argparse.Namespace) -> int:
    roots = [Path(path).expanduser() for path in args.data_root]
    summary = build_manifest(
        data_roots=roots,
        out_path=Path(args.out_manifest),
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        filter_policy=str(args.filter_policy),
        missing_value_policy=str(args.missing_value_policy),
    )
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
    return 0


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
    build_parser.add_argument("--train-ratio", type=float, default=0.90)
    build_parser.add_argument("--val-ratio", type=float, default=0.05)
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
