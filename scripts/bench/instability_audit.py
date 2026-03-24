"""Standalone instability-audit entrypoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


def _repo_src_root() -> Path:
    return Path(__file__).resolve().parents[2] / "src"


if str(_repo_src_root()) not in sys.path:
    sys.path.insert(0, str(_repo_src_root()))

import tab_foundry.bench.instability_audit as audit_module  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Audit existing system-delta runs for instability.")
    parser.add_argument(
        "--staged-ladder-root",
        default="outputs/staged_ladder",
        help="Root directory containing staged ladder outputs.",
    )
    parser.add_argument(
        "--reports-root",
        default=None,
        help="Optional report output directory. Defaults to <staged-ladder-root>/reports.",
    )
    parser.add_argument(
        "--sweep-id",
        default=audit_module.DEFAULT_SWEEP_ID,
        help="Sweep identifier used to match sd_<sweep_id>_* run directories.",
    )
    parser.add_argument(
        "--anchor-run-id",
        default=audit_module.DEFAULT_ANCHOR_RUN_ID,
        help="Registry run id for the reference anchor row.",
    )
    parser.add_argument(
        "--registry-path",
        default=str(audit_module.default_benchmark_run_registry_path()),
        help="Benchmark run registry JSON used to resolve the anchor run.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    staged_ladder_root = Path(str(args.staged_ladder_root))
    payload = audit_module.build_instability_audit(
        staged_ladder_root=staged_ladder_root,
        sweep_id=str(args.sweep_id),
        anchor_run_id=str(args.anchor_run_id),
        registry_path=Path(str(args.registry_path)),
    )
    report_paths = audit_module.write_instability_audit(
        payload,
        out_root=(
            staged_ladder_root / "reports"
            if args.reports_root is None
            else Path(str(args.reports_root))
        ),
        sweep_id=str(args.sweep_id),
    )
    print("Instability audit complete:")
    print(f"  json={report_paths['json']}")
    print(f"  markdown={report_paths['markdown']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
