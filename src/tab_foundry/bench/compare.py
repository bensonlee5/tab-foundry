"""Manual nanoTabPFN comparison runner."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess
import sys
from typing import Any, Sequence, cast

from tab_foundry.bench.artifacts import load_jsonl, write_json, write_jsonl
from tab_foundry.bench.benchmark_run_registry import derive_benchmark_run_record
from tab_foundry.bench.control_baseline import (
    default_control_baseline_registry_path,
    load_control_baseline_entry,
)
from tab_foundry.bench.nanotabpfn import (
    default_benchmark_bundle_path,
    build_comparison_summary,
    evaluate_tab_foundry_run,
    load_benchmark_bundle,
    load_openml_benchmark_datasets,
    plot_comparison_curve,
    save_dataset_cache,
)


DEFAULT_NANOTABPFN_STEPS = 2500
DEFAULT_NANOTABPFN_SEEDS = 2
DEFAULT_NANOTABPFN_EVAL_EVERY = 25
DEFAULT_NANOTABPFN_BATCH_SIZE = 32
DEFAULT_NANOTABPFN_LR = 4.0e-3


@dataclass(slots=True)
class NanoTabPFNBenchmarkConfig:
    """Input configuration for the notebook-style nanoTabPFN comparison."""

    tab_foundry_run_dir: Path
    out_root: Path
    nanotabpfn_root: Path = Path("~/dev/nanoTabPFN")
    nanotab_prior_dump: Path | None = None
    device: str = "auto"
    nanotabpfn_steps: int = DEFAULT_NANOTABPFN_STEPS
    nanotabpfn_seeds: int = DEFAULT_NANOTABPFN_SEEDS
    nanotabpfn_eval_every: int = DEFAULT_NANOTABPFN_EVAL_EVERY
    nanotabpfn_batch_size: int = DEFAULT_NANOTABPFN_BATCH_SIZE
    nanotabpfn_lr: float = DEFAULT_NANOTABPFN_LR
    control_baseline_id: str | None = None
    control_baseline_registry: Path | None = None
    benchmark_bundle_path: Path | None = None
    reuse_nanotabpfn_curve_path: Path | None = None


def _default_out_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return Path("/tmp") / f"tab_foundry_nanotabpfn_benchmark_{stamp}"


def _nanotabpfn_python(root: Path) -> Path:
    return root.expanduser().resolve() / ".venv" / "bin" / "python"


def _nanotabpfn_prior_dump(root: Path, override: Path | None) -> Path:
    return (override or (root / "300k_150x5_2.h5")).expanduser().resolve()


def _helper_script_path() -> Path:
    return Path(__file__).resolve().with_name("nanotabpfn_helper.py")


def _src_root() -> Path:
    return Path(__file__).resolve().parents[2]


def _is_legacy_benchmark_record_compat_error(exc: Exception) -> bool:
    message = str(exc)
    return "persisted model.arch" in message or "omitted feature_group_size" in message


def _nanotabpfn_helper_command(
    *,
    config: NanoTabPFNBenchmarkConfig,
    dataset_cache: Path,
    out_path: Path,
) -> list[str]:
    nanotab_root = config.nanotabpfn_root.expanduser().resolve()
    return [
        str(_nanotabpfn_python(nanotab_root)),
        str(_helper_script_path()),
        "--tab-foundry-src",
        str(_src_root()),
        "--dataset-cache",
        str(dataset_cache),
        "--prior-dump",
        str(_nanotabpfn_prior_dump(nanotab_root, config.nanotab_prior_dump)),
        "--out-path",
        str(out_path),
        "--device",
        str(config.device),
        "--steps",
        str(int(config.nanotabpfn_steps)),
        "--eval-every",
        str(int(config.nanotabpfn_eval_every)),
        "--seeds",
        str(int(config.nanotabpfn_seeds)),
        "--batch-size",
        str(int(config.nanotabpfn_batch_size)),
        "--lr",
        str(float(config.nanotabpfn_lr)),
    ]


def _validate_config(config: NanoTabPFNBenchmarkConfig) -> tuple[Path, Path, Path]:
    tab_foundry_run_dir = config.tab_foundry_run_dir.expanduser().resolve()
    nanotabpfn_root = config.nanotabpfn_root.expanduser().resolve()
    nanotabpfn_python = _nanotabpfn_python(nanotabpfn_root)
    prior_dump = _nanotabpfn_prior_dump(nanotabpfn_root, config.nanotab_prior_dump)

    if not tab_foundry_run_dir.exists():
        raise RuntimeError(f"tab-foundry run dir does not exist: {tab_foundry_run_dir}")
    if not nanotabpfn_root.exists():
        raise RuntimeError(f"nanoTabPFN root does not exist: {nanotabpfn_root}")
    if not nanotabpfn_python.exists():
        raise RuntimeError(
            f"missing nanoTabPFN interpreter at {nanotabpfn_python}; run scripts/bootstrap_benchmark_envs.py first"
        )
    if not prior_dump.exists():
        raise RuntimeError(f"nanoTabPFN prior dump does not exist: {prior_dump}")
    return tab_foundry_run_dir, nanotabpfn_root, prior_dump


def run_nanotabpfn_benchmark(config: NanoTabPFNBenchmarkConfig) -> dict[str, Any]:
    """Run the notebook-style tab-foundry vs nanoTabPFN comparison."""

    tab_foundry_run_dir, nanotabpfn_root, _prior_dump = _validate_config(config)
    out_root = config.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    benchmark_tasks_path = out_root / "benchmark_tasks.json"
    dataset_cache_path = out_root / "benchmark_dataset_cache.npz"
    tab_foundry_curve_path = out_root / "tab_foundry_curve.jsonl"
    nanotabpfn_curve_path = out_root / "nanotabpfn_curve.jsonl"
    comparison_curve_path = out_root / "comparison_curve.png"
    comparison_summary_path = out_root / "comparison_summary.json"
    benchmark_run_record_path = out_root / "benchmark_run_record.json"
    training_surface_record_path = out_root / "training_surface_record.json"
    benchmark_bundle_path = (
        default_benchmark_bundle_path()
        if config.benchmark_bundle_path is None
        else config.benchmark_bundle_path.expanduser().resolve()
    )
    benchmark_bundle = load_benchmark_bundle(benchmark_bundle_path)
    benchmark_selection = cast(dict[str, Any], benchmark_bundle["selection"])
    benchmark_new_instances = int(benchmark_selection["new_instances"])
    control_baseline = None
    if config.control_baseline_id is not None:
        control_baseline = load_control_baseline_entry(
            str(config.control_baseline_id),
            registry_path=config.control_baseline_registry,
        )

    datasets, benchmark_tasks = load_openml_benchmark_datasets(
        new_instances=benchmark_new_instances,
        benchmark_bundle_path=benchmark_bundle_path,
    )
    write_json(benchmark_tasks_path, benchmark_bundle)
    save_dataset_cache(dataset_cache_path, datasets)

    tab_foundry_records = evaluate_tab_foundry_run(
        tab_foundry_run_dir,
        datasets=datasets,
        device=config.device,
    )
    write_jsonl(tab_foundry_curve_path, tab_foundry_records)

    nanotabpfn_records: list[dict[str, Any]]
    if config.reuse_nanotabpfn_curve_path is not None:
        reuse_path = config.reuse_nanotabpfn_curve_path.expanduser().resolve()
        nanotabpfn_records = load_jsonl(reuse_path)
        if not nanotabpfn_records:
            raise RuntimeError(f"reused nanoTabPFN curve is empty: {reuse_path}")
        write_jsonl(nanotabpfn_curve_path, nanotabpfn_records)
    else:
        subprocess.run(
            _nanotabpfn_helper_command(
                config=config,
                dataset_cache=dataset_cache_path,
                out_path=nanotabpfn_curve_path,
            ),
            cwd=nanotabpfn_root,
            check=True,
        )
        nanotabpfn_records = load_jsonl(nanotabpfn_curve_path)
        if not nanotabpfn_records:
            raise RuntimeError("nanoTabPFN benchmark produced no curve records")

    plot_comparison_curve(
        tab_foundry_records=tab_foundry_records,
        nanotabpfn_records=nanotabpfn_records,
        out_path=comparison_curve_path,
    )
    summary = build_comparison_summary(
        tab_foundry_records=tab_foundry_records,
        nanotabpfn_records=nanotabpfn_records,
        benchmark_tasks=benchmark_tasks,
        benchmark_bundle=benchmark_bundle,
        benchmark_bundle_path=benchmark_bundle_path,
        tab_foundry_run_dir=tab_foundry_run_dir,
        nanotabpfn_root=nanotabpfn_root,
        nanotabpfn_python=_nanotabpfn_python(nanotabpfn_root),
        control_baseline=control_baseline,
    )
    summary["artifacts"] = {
        "benchmark_tasks_json": str(benchmark_tasks_path),
        "tab_foundry_curve_jsonl": str(tab_foundry_curve_path),
        "nanotabpfn_curve_jsonl": str(nanotabpfn_curve_path),
        "comparison_curve_png": str(comparison_curve_path),
        "benchmark_dataset_cache": str(dataset_cache_path),
        "benchmark_run_record_json": str(benchmark_run_record_path),
        "training_surface_record_json": str(training_surface_record_path),
    }
    write_json(comparison_summary_path, summary)
    try:
        benchmark_run_record = derive_benchmark_run_record(
            run_dir=tab_foundry_run_dir,
            comparison_summary_path=comparison_summary_path,
            benchmark_run_record_path=benchmark_run_record_path,
        )
    except (RuntimeError, ValueError) as exc:
        if not _is_legacy_benchmark_record_compat_error(exc):
            raise
        print(
            "Skipping benchmark_run_record.json derivation for legacy checkpoint metadata: "
            f"{exc}",
            file=sys.stderr,
        )
        summary["artifacts"]["benchmark_run_record_json"] = None
        summary["artifacts"]["training_surface_record_json"] = None
        cast(dict[str, Any], summary["tab_foundry"])[
            "benchmark_run_record_warning"
        ] = str(exc)
        write_json(comparison_summary_path, summary)
        return summary
    tab_foundry_summary = cast(dict[str, Any], summary["tab_foundry"])
    tab_foundry_summary["manifest_path"] = str(benchmark_run_record["manifest_path"])
    tab_foundry_summary["seed_set"] = list(benchmark_run_record["seed_set"])
    tab_foundry_summary["training_diagnostics"] = dict(benchmark_run_record["training_diagnostics"])
    tab_foundry_summary["model_size"] = dict(benchmark_run_record["model_size"])
    summary["artifacts"]["training_surface_record_json"] = cast(
        dict[str, Any],
        benchmark_run_record["artifacts"],
    ).get("training_surface_record_path")
    if benchmark_run_record.get("surface_labels") is not None:
        tab_foundry_summary["surface_labels"] = dict(benchmark_run_record["surface_labels"])
    write_json(comparison_summary_path, summary)
    write_json(benchmark_run_record_path, benchmark_run_record)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compare a completed tab-foundry run against nanoTabPFN")
    parser.add_argument(
        "--tab-foundry-run-dir",
        required=True,
        help="Completed tab-foundry run directory with checkpoint snapshots",
    )
    parser.add_argument("--nanotabpfn-root", default="~/dev/nanoTabPFN", help="Local nanoTabPFN checkout")
    parser.add_argument("--nanotab-prior-dump", default=None, help="Path to nanoTabPFN prior dump (.h5)")
    parser.add_argument("--out-root", default=None, help="Output directory root")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("cpu", "cuda", "mps", "auto"),
        help="Benchmark device",
    )
    parser.add_argument(
        "--nanotabpfn-steps",
        type=int,
        default=DEFAULT_NANOTABPFN_STEPS,
        help="nanoTabPFN training steps",
    )
    parser.add_argument(
        "--nanotabpfn-seeds",
        type=int,
        default=DEFAULT_NANOTABPFN_SEEDS,
        help="Number of nanoTabPFN random seeds",
    )
    parser.add_argument(
        "--control-baseline-id",
        default=None,
        help="Optional frozen control baseline id to copy into comparison_summary.json",
    )
    parser.add_argument(
        "--control-baseline-registry",
        default=str(default_control_baseline_registry_path()),
        help="Control baseline registry JSON path used with --control-baseline-id",
    )
    parser.add_argument(
        "--benchmark-bundle-path",
        default=None,
        help="Optional repo-tracked OpenML benchmark bundle JSON path",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run_nanotabpfn_benchmark(
        NanoTabPFNBenchmarkConfig(
            tab_foundry_run_dir=Path(str(args.tab_foundry_run_dir)),
            out_root=_default_out_root() if args.out_root is None else Path(str(args.out_root)),
            nanotabpfn_root=Path(str(args.nanotabpfn_root)),
            nanotab_prior_dump=(Path(str(args.nanotab_prior_dump)) if args.nanotab_prior_dump else None),
            device=str(args.device),
            nanotabpfn_steps=int(args.nanotabpfn_steps),
            nanotabpfn_seeds=int(args.nanotabpfn_seeds),
            control_baseline_id=(
                str(args.control_baseline_id) if args.control_baseline_id else None
            ),
            control_baseline_registry=(
                Path(str(args.control_baseline_registry))
                if args.control_baseline_registry
                else None
            ),
            benchmark_bundle_path=(
                Path(str(args.benchmark_bundle_path))
                if args.benchmark_bundle_path
                else None
            ),
        )
    )
    print("nanoTabPFN comparison complete:")
    print(f"  dataset_count={summary['dataset_count']}")
    print(f"  tab_foundry={summary['tab_foundry']}")
    print(f"  nanotabpfn={summary['nanotabpfn']}")
    print(f"  artifacts={summary['artifacts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
