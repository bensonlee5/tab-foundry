"""Execution helpers for benchmark bounce diagnosis."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from tab_foundry.bench.bounce.config import (
    BenchmarkBounceDiagnosisConfig,
    DIAGNOSIS_SCHEMA,
    resolve_positive_int,
    resolve_probability,
)


def evaluate_one_bundle(
    *,
    run_dir: Path,
    bundle_path: Path,
    device: str,
    out_path: Path,
    bootstrap_samples: int,
    bootstrap_confidence: float,
    load_benchmark_bundle_for_execution_fn: Any,
    load_openml_benchmark_datasets_fn: Any,
    evaluate_tab_foundry_run_fn: Any,
    summarize_checkpoint_curve_fn: Any,
    benchmark_bundle_summary_fn: Any,
    benchmark_bundle_task_type_fn: Any,
    curve_summary_fn: Any,
    write_jsonl_fn: Any,
) -> dict[str, Any]:
    bundle, allow_missing_values = load_benchmark_bundle_for_execution_fn(bundle_path)
    selection = cast(dict[str, Any], bundle["selection"])
    datasets, benchmark_tasks = load_openml_benchmark_datasets_fn(
        new_instances=int(selection["new_instances"]),
        benchmark_bundle_path=bundle_path,
        allow_missing_values=allow_missing_values,
    )
    raw_records = evaluate_tab_foundry_run_fn(
        run_dir,
        datasets=datasets,
        task_type=benchmark_bundle_task_type_fn(bundle),
        device=device,
        allow_checkpoint_failures=True,
        allow_missing_values=allow_missing_values,
    )
    diagnostics = summarize_checkpoint_curve_fn(
        raw_records,
        bootstrap_samples=int(bootstrap_samples),
        bootstrap_confidence=float(bootstrap_confidence),
    )
    records = cast(list[dict[str, Any]], diagnostics["successful_records"])
    failed_records = cast(list[dict[str, Any]], diagnostics["failed_records"])
    write_jsonl_fn(
        out_path,
        cast(list[dict[str, Any]], diagnostics["records"]),
    )
    return {
        "bundle": benchmark_bundle_summary_fn(bundle, source_path=bundle_path),
        "benchmark_tasks": benchmark_tasks,
        "records": records,
        "records_path": str(out_path.resolve()),
        "summary": curve_summary_fn(records),
        "failure_count": int(len(failed_records)),
        "failed_checkpoints": failed_records,
    }


def run_benchmark_bounce_diagnosis(
    config: BenchmarkBounceDiagnosisConfig,
    *,
    default_benchmark_bundle_path_fn: Any,
    evaluate_one_bundle_fn: Any,
    run_dense_checkpoint_rerun_fn: Any,
    load_history_fn: Any,
    shared_bundle_analysis_fn: Any,
    training_signal_fn: Any,
    task_tradeoff_signal_fn: Any,
    checkpoint_aliasing_signal_fn: Any,
    classify_causes_fn: Any,
    utc_now_fn: Any,
    write_json_fn: Any,
) -> dict[str, Any]:
    """Benchmark one run on multiple bundles and classify likely bounce causes."""

    bootstrap_samples = resolve_positive_int(int(config.bootstrap_samples), name="bootstrap_samples")
    bootstrap_confidence = resolve_probability(
        float(config.bootstrap_confidence),
        name="bootstrap_confidence",
    )
    out_root = config.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = config.run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise RuntimeError(f"run_dir does not exist: {run_dir}")

    benchmark_bundle_path = (
        default_benchmark_bundle_path_fn()
        if config.benchmark_bundle_path is None
        else config.benchmark_bundle_path.expanduser().resolve()
    )

    primary = evaluate_one_bundle_fn(
        run_dir=run_dir,
        bundle_path=benchmark_bundle_path,
        device=config.device,
        out_path=out_root / "primary_bundle_curve.jsonl",
        bootstrap_samples=bootstrap_samples,
        bootstrap_confidence=bootstrap_confidence,
    )
    confirmation: dict[str, Any] | None = None
    if config.confirmation_benchmark_bundle_path is not None:
        confirmation = evaluate_one_bundle_fn(
            run_dir=run_dir,
            bundle_path=config.confirmation_benchmark_bundle_path.expanduser().resolve(),
            device=config.device,
            out_path=out_root / "confirmation_bundle_curve.jsonl",
            bootstrap_samples=bootstrap_samples,
            bootstrap_confidence=bootstrap_confidence,
        )

    dense_run_dir: Path | None = None
    dense_confirmation: dict[str, Any] | None = None
    if config.dense_run_dir is not None:
        dense_run_dir = config.dense_run_dir.expanduser().resolve()
    elif config.dense_checkpoint_every is not None:
        dense_run_dir = run_dense_checkpoint_rerun_fn(config)
    if dense_run_dir is not None:
        dense_bundle_path = (
            benchmark_bundle_path
            if confirmation is None
            else Path(str(cast(dict[str, Any], confirmation["bundle"])["source_path"]))
        )
        dense_confirmation = evaluate_one_bundle_fn(
            run_dir=dense_run_dir,
            bundle_path=dense_bundle_path,
            device=config.device,
            out_path=out_root / "dense_confirmation_bundle_curve.jsonl",
            bootstrap_samples=bootstrap_samples,
            bootstrap_confidence=bootstrap_confidence,
        )

    history = load_history_fn(
        run_dir / "train_history.jsonl"
        if (run_dir / "train_history.jsonl").exists()
        else run_dir / "train_outputs" / "train_history.jsonl"
    )
    signal_records = (
        cast(list[dict[str, Any]], primary["records"])
        if confirmation is None
        else cast(list[dict[str, Any]], confirmation["records"])
    )
    bundle_analysis = shared_bundle_analysis_fn(
        cast(list[dict[str, Any]], primary["records"]),
        None if confirmation is None else cast(list[dict[str, Any]], confirmation["records"]),
    )
    training_signal = training_signal_fn(
        history=history,
        curve_records=signal_records,
    )
    task_tradeoff_signal = task_tradeoff_signal_fn(signal_records)
    checkpoint_aliasing_signal = checkpoint_aliasing_signal_fn(
        coarse_records=signal_records,
        dense_records=None if dense_confirmation is None else cast(list[dict[str, Any]], dense_confirmation["records"]),
    )
    evaluation_failures = {
        "failure_count": int(primary.get("failure_count", 0))
        + (0 if confirmation is None else int(confirmation.get("failure_count", 0))),
        "primary_bundle_failures": list(primary.get("failed_checkpoints", [])),
        "confirmation_bundle_failures": []
        if confirmation is None
        else list(confirmation.get("failed_checkpoints", [])),
    }
    classification = classify_causes_fn(
        bundle_analysis=bundle_analysis,
        training_signal=training_signal,
        task_tradeoff_signal=task_tradeoff_signal,
        checkpoint_aliasing_signal=checkpoint_aliasing_signal,
        evaluation_failures=evaluation_failures,
    )

    summary = {
        "schema": DIAGNOSIS_SCHEMA,
        "generated_at_utc": utc_now_fn(),
        "run_id": config.run_id,
        "run_dir": str(run_dir),
        "artifacts": {
            "primary_bundle_curve_jsonl": primary["records_path"],
            "confirmation_bundle_curve_jsonl": None
            if confirmation is None
            else confirmation["records_path"],
            "dense_confirmation_bundle_curve_jsonl": None
            if dense_confirmation is None
            else dense_confirmation["records_path"],
        },
        "bundles": {
            "primary": {
                "benchmark_bundle": primary["bundle"],
                "summary": primary["summary"],
            },
            "confirmation": None
            if confirmation is None
            else {
                "benchmark_bundle": confirmation["bundle"],
                "summary": confirmation["summary"],
            },
        },
        "bundle_analysis": bundle_analysis,
        "training_signal": training_signal,
        "task_tradeoff_signal": task_tradeoff_signal,
        "checkpoint_aliasing_signal": checkpoint_aliasing_signal,
        "evaluation_failures": evaluation_failures,
        "classification": classification,
    }
    if dense_run_dir is not None:
        summary["dense_run"] = {
            "run_dir": str(dense_run_dir),
            "checkpoint_every": int(config.dense_checkpoint_every)
            if config.dense_checkpoint_every is not None
            else None,
        }
    write_json_fn(out_root / "benchmark_bounce_diagnosis.json", summary)
    return summary
