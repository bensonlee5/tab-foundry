"""Diagnosis helpers for checkpoint-level benchmark bounce."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from tab_foundry.bench.artifacts import load_history, write_json, write_jsonl
import tab_foundry.bench.bounce.config as bounce_config_module
import tab_foundry.bench.bounce.execution as execution_module
import tab_foundry.bench.bounce.rerun as rerun_module
import tab_foundry.bench.bounce.signals as signal_module
from tab_foundry.bench.nanotabpfn import (
    curve_summary,
    default_benchmark_manifest_path,
    evaluate_tab_foundry_run,
    load_benchmark_manifest_datasets,
    summarize_checkpoint_curve,
)
from tab_foundry.training.prior_train import train_tabfoundry_simple_prior
from tab_foundry.training.trainer import train

BenchmarkBounceDiagnosisConfig = bounce_config_module.BenchmarkBounceDiagnosisConfig
DIAGNOSIS_SCHEMA = bounce_config_module.DIAGNOSIS_SCHEMA


def _run_dense_checkpoint_rerun(config: BenchmarkBounceDiagnosisConfig) -> Path:
    return rerun_module.run_dense_checkpoint_rerun(
        config,
        checkpoint_cfg_from_run_fn=rerun_module.checkpoint_cfg_from_run,
        prior_train_fn=train_tabfoundry_simple_prior,
        train_fn=train,
    )


def _evaluate_one_bundle(
    *,
    run_dir: Path,
    manifest_path: Path,
    device: str,
    out_path: Path,
    bootstrap_samples: int,
    bootstrap_confidence: float,
) -> dict[str, Any]:
    return execution_module.evaluate_one_bundle(
        run_dir=run_dir,
        manifest_path=manifest_path,
        device=device,
        out_path=out_path,
        bootstrap_samples=bootstrap_samples,
        bootstrap_confidence=bootstrap_confidence,
        load_benchmark_manifest_datasets_fn=load_benchmark_manifest_datasets,
        evaluate_tab_foundry_run_fn=evaluate_tab_foundry_run,
        summarize_checkpoint_curve_fn=summarize_checkpoint_curve,
        curve_summary_fn=curve_summary,
        write_jsonl_fn=write_jsonl,
    )


def run_benchmark_bounce_diagnosis(config: BenchmarkBounceDiagnosisConfig) -> dict[str, Any]:
    """Benchmark one run on multiple bundles and classify likely bounce causes."""

    return execution_module.run_benchmark_bounce_diagnosis(
        config,
        default_benchmark_manifest_path_fn=default_benchmark_manifest_path,
        evaluate_one_bundle_fn=_evaluate_one_bundle,
        run_dense_checkpoint_rerun_fn=_run_dense_checkpoint_rerun,
        load_history_fn=load_history,
        shared_bundle_analysis_fn=signal_module.shared_bundle_analysis,
        training_signal_fn=signal_module.training_signal,
        task_tradeoff_signal_fn=signal_module.task_tradeoff_signal,
        checkpoint_aliasing_signal_fn=signal_module.checkpoint_aliasing_signal,
        classify_causes_fn=signal_module.classify_causes,
        utc_now_fn=bounce_config_module.utc_now,
        write_json_fn=write_json,
    )
