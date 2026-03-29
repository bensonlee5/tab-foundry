"""CLI wiring for `tab-foundry bench materialize-openml-bundle`."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

from tab_realdata_hub.openml import materialize_bundle
from tab_foundry.repo_paths import repo_root


def _default_out_root(bundle_path: Path) -> Path:
    bundle_stem = bundle_path.expanduser().resolve().stem
    return repo_root() / "data" / "manifests" / "bench" / bundle_stem


def configure_parser(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--bundle-path", required=True, help="OpenML benchmark bundle JSON path")
    parser.add_argument(
        "--out-root",
        default=None,
        help="Output root for the materialized benchmark manifest surface",
    )
    parser.add_argument("--force", action="store_true", help="Overwrite an existing output root")
    parser.add_argument("--split-seed", type=int, default=0, help="Deterministic split seed")
    parser.add_argument("--test-size", type=float, default=0.20, help="Holdout ratio for packed shards")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Materialize an OpenML benchmark bundle into a manifest-backed benchmark surface"
    )
    configure_parser(parser)
    return parser


def run_from_args(args: argparse.Namespace) -> int:
    bundle_path = Path(str(args.bundle_path))
    result = materialize_bundle(
        bundle_path,
        _default_out_root(bundle_path) if args.out_root is None else Path(str(args.out_root)),
        force=bool(args.force),
        split_seed=int(args.split_seed),
        test_size=float(args.test_size),
    )
    print(f"Materialized benchmark manifest: {result.manifest_path}")
    print(f"Packed shards: {result.data_root}")
    print(f"Tasks: {len(result.task_summaries)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    return run_from_args(build_parser().parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
