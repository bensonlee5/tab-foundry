"""Diagnosis helpers for checkpoint-level benchmark bounce."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence, cast

from omegaconf import DictConfig

from tab_foundry.bench.artifacts import load_history, write_json, write_jsonl
from tab_foundry.bench.benchmark_run_registry import default_benchmark_run_registry_path
from tab_foundry.bench.bounce.config import (
    BenchmarkBounceDiagnosisConfig,
    DIAGNOSIS_SCHEMA as _DIAGNOSIS_SCHEMA,
    RerunMode,
    default_out_root as _default_out_root_impl,
    resolve_positive_int as _resolve_positive_int_impl,
    resolve_probability as _resolve_probability_impl,
    utc_now as _utc_now_impl,
)
from tab_foundry.bench.bounce.execution import (
    evaluate_one_bundle as _evaluate_one_bundle_impl,
    run_benchmark_bounce_diagnosis as _run_benchmark_bounce_diagnosis_impl,
)
from tab_foundry.bench.bounce.rerun import (
    checkpoint_cfg_from_run as _checkpoint_cfg_from_run_impl,
    resolve_run_dir_from_registry as _resolve_run_dir_from_registry_impl,
    run_dense_checkpoint_rerun as _run_dense_checkpoint_rerun_impl,
)
from tab_foundry.bench.bounce.signals import (
    checkpoint_aliasing_signal as _checkpoint_aliasing_signal_impl,
    classify_causes as _classify_causes_impl,
    curve_summary_compat as _curve_summary_impl,
    shared_bundle_analysis as _shared_bundle_analysis_impl,
    task_tradeoff_signal as _task_tradeoff_signal_impl,
    training_signal as _training_signal_impl,
)
from tab_foundry.bench.nanotabpfn import (
    benchmark_bundle_summary,
    benchmark_bundle_task_type,
    curve_summary,
    default_benchmark_bundle_path,
    evaluate_tab_foundry_run,
    load_benchmark_bundle_for_execution,
    load_openml_benchmark_datasets,
    summarize_checkpoint_curve,
)
from tab_foundry.bench.prior_train import train_tabfoundry_simple_prior
from tab_foundry.training.trainer import train

DIAGNOSIS_SCHEMA = _DIAGNOSIS_SCHEMA


def _utc_now() -> str:
    return _utc_now_impl()


def _default_out_root(run_dir: Path) -> Path:
    return _default_out_root_impl(run_dir)


def _resolve_positive_int(value: int, *, name: str) -> int:
    return _resolve_positive_int_impl(value, name=name)


def _resolve_probability(value: float, *, name: str) -> float:
    return _resolve_probability_impl(value, name=name)


def resolve_run_dir_from_registry(
    run_id: str,
    *,
    registry_path: Path | None = None,
) -> Path:
    """Resolve a benchmark registry run id into its concrete run directory."""

    return _resolve_run_dir_from_registry_impl(run_id, registry_path=registry_path)


def _checkpoint_cfg_from_run(run_dir: Path) -> DictConfig:
    return _checkpoint_cfg_from_run_impl(run_dir)


def _run_dense_checkpoint_rerun(config: BenchmarkBounceDiagnosisConfig) -> Path:
    return _run_dense_checkpoint_rerun_impl(
        config,
        checkpoint_cfg_from_run_fn=_checkpoint_cfg_from_run,
        prior_train_fn=train_tabfoundry_simple_prior,
        train_fn=train,
    )


def _shared_bundle_analysis(
    primary_records: list[dict[str, Any]],
    confirmation_records: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    return _shared_bundle_analysis_impl(primary_records, confirmation_records)


def _curve_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    return _curve_summary_impl(records)


def _training_signal(
    *,
    history: list[dict[str, Any]],
    curve_records: list[dict[str, Any]],
) -> dict[str, Any]:
    return _training_signal_impl(history=history, curve_records=curve_records)


def _task_tradeoff_signal(records: list[dict[str, Any]]) -> dict[str, Any]:
    return _task_tradeoff_signal_impl(records)


def _checkpoint_aliasing_signal(
    *,
    coarse_records: list[dict[str, Any]],
    dense_records: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    return _checkpoint_aliasing_signal_impl(
        coarse_records=coarse_records,
        dense_records=dense_records,
    )


def _classify_causes(
    *,
    bundle_analysis: dict[str, Any],
    training_signal: dict[str, Any],
    task_tradeoff_signal: dict[str, Any],
    checkpoint_aliasing_signal: dict[str, Any],
    evaluation_failures: dict[str, Any],
) -> dict[str, Any]:
    return _classify_causes_impl(
        bundle_analysis=bundle_analysis,
        training_signal=training_signal,
        task_tradeoff_signal=task_tradeoff_signal,
        checkpoint_aliasing_signal=checkpoint_aliasing_signal,
        evaluation_failures=evaluation_failures,
    )


def _evaluate_one_bundle(
    *,
    run_dir: Path,
    bundle_path: Path,
    device: str,
    out_path: Path,
    bootstrap_samples: int,
    bootstrap_confidence: float,
) -> dict[str, Any]:
    return _evaluate_one_bundle_impl(
        run_dir=run_dir,
        bundle_path=bundle_path,
        device=device,
        out_path=out_path,
        bootstrap_samples=bootstrap_samples,
        bootstrap_confidence=bootstrap_confidence,
        load_benchmark_bundle_for_execution_fn=load_benchmark_bundle_for_execution,
        load_openml_benchmark_datasets_fn=load_openml_benchmark_datasets,
        evaluate_tab_foundry_run_fn=evaluate_tab_foundry_run,
        summarize_checkpoint_curve_fn=summarize_checkpoint_curve,
        benchmark_bundle_summary_fn=benchmark_bundle_summary,
        benchmark_bundle_task_type_fn=benchmark_bundle_task_type,
        curve_summary_fn=curve_summary,
        write_jsonl_fn=write_jsonl,
    )


def run_benchmark_bounce_diagnosis(config: BenchmarkBounceDiagnosisConfig) -> dict[str, Any]:
    """Benchmark one run on multiple bundles and classify likely bounce causes."""

    return _run_benchmark_bounce_diagnosis_impl(
        config,
        default_benchmark_bundle_path_fn=default_benchmark_bundle_path,
        evaluate_one_bundle_fn=_evaluate_one_bundle,
        run_dense_checkpoint_rerun_fn=_run_dense_checkpoint_rerun,
        load_history_fn=load_history,
        shared_bundle_analysis_fn=_shared_bundle_analysis,
        training_signal_fn=_training_signal,
        task_tradeoff_signal_fn=_task_tradeoff_signal,
        checkpoint_aliasing_signal_fn=_checkpoint_aliasing_signal,
        classify_causes_fn=_classify_causes,
        utc_now_fn=_utc_now,
        write_json_fn=write_json,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Diagnose checkpoint-level benchmark bounce for one run")
    parser.add_argument("--run-dir", default=None, help="Completed run directory to diagnose")
    parser.add_argument("--run-id", default=None, help="Benchmark registry run id to diagnose")
    parser.add_argument(
        "--registry-path",
        default=str(default_benchmark_run_registry_path()),
        help="Benchmark run registry used with --run-id",
    )
    parser.add_argument("--out-root", default=None, help="Output directory root")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("cpu", "cuda", "mps", "auto"),
        help="Benchmark device",
    )
    parser.add_argument(
        "--benchmark-bundle-path",
        default=None,
        help="Primary benchmark bundle path; defaults to the current medium bundle",
    )
    parser.add_argument(
        "--confirmation-benchmark-bundle-path",
        default=None,
        help="Optional confirmation benchmark bundle path; omit to stay on the primary no-missing bundle only",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=2000,
        help="Task-bootstrap samples per checkpoint",
    )
    parser.add_argument(
        "--bootstrap-confidence",
        type=float,
        default=0.95,
        help="Task-bootstrap confidence level",
    )
    parser.add_argument(
        "--dense-checkpoint-every",
        type=int,
        default=None,
        help="Optional diagnosis-only rerun checkpoint cadence",
    )
    parser.add_argument(
        "--dense-run-dir",
        default=None,
        help="Optional precomputed dense-checkpoint run directory",
    )
    parser.add_argument(
        "--rerun-mode",
        default="none",
        choices=("auto", "prior", "train", "none"),
        help="How to rerun when --dense-checkpoint-every is set",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if bool(args.run_dir) == bool(args.run_id):
        raise SystemExit("exactly one of --run-dir or --run-id must be provided")
    run_id = None if args.run_id is None else str(args.run_id)
    run_dir = (
        Path(str(args.run_dir))
        if args.run_dir is not None
        else resolve_run_dir_from_registry(
            str(args.run_id),
            registry_path=Path(str(args.registry_path)) if args.registry_path else None,
        )
    )
    summary = run_benchmark_bounce_diagnosis(
        BenchmarkBounceDiagnosisConfig(
            run_dir=run_dir,
            out_root=_default_out_root(run_dir)
            if args.out_root is None
            else Path(str(args.out_root)),
            device=str(args.device),
            benchmark_bundle_path=(
                None if args.benchmark_bundle_path is None else Path(str(args.benchmark_bundle_path))
            ),
            confirmation_benchmark_bundle_path=(
                None
                if args.confirmation_benchmark_bundle_path is None
                else Path(str(args.confirmation_benchmark_bundle_path))
            ),
            bootstrap_samples=int(args.bootstrap_samples),
            bootstrap_confidence=float(args.bootstrap_confidence),
            dense_checkpoint_every=(
                None if args.dense_checkpoint_every is None else int(args.dense_checkpoint_every)
            ),
            dense_run_dir=None if args.dense_run_dir is None else Path(str(args.dense_run_dir)),
            rerun_mode=cast(RerunMode, str(args.rerun_mode)),
            run_id=run_id,
        )
    )
    print("benchmark bounce diagnosis complete:")
    print(f"  run_dir={summary['run_dir']}")
    print(f"  causes={summary['classification']['primary_causes']}")
    print(f"  artifacts={summary['artifacts']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
