"""Diagnosis helpers for checkpoint-level benchmark bounce."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Literal, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from tab_foundry.bench.artifacts import load_history, write_json, write_jsonl
from tab_foundry.bench.benchmark_run_registry import (
    default_benchmark_run_registry_path,
    load_benchmark_run_registry,
    resolve_registry_path_value,
)
from tab_foundry.bench.nanotabpfn import (
    benchmark_bundle_task_type,
    benchmark_bundle_summary,
    curve_summary,
    default_benchmark_bundle_path,
    evaluate_tab_foundry_run,
    load_benchmark_bundle_for_execution,
    load_openml_benchmark_datasets,
    summarize_checkpoint_curve,
)
from tab_foundry.bench.prior_train import train_tabfoundry_simple_prior
from tab_foundry.training.trainer import train

DIAGNOSIS_SCHEMA = "benchmark_bounce_diagnosis_v1"

RerunMode = Literal["auto", "prior", "train", "none"]


@dataclass(slots=True)
class BenchmarkBounceDiagnosisConfig:
    """Input configuration for one benchmark-bounce diagnosis run."""

    run_dir: Path
    out_root: Path
    device: str = "auto"
    benchmark_bundle_path: Path | None = None
    confirmation_benchmark_bundle_path: Path | None = None
    bootstrap_samples: int = 2000
    bootstrap_confidence: float = 0.95
    dense_checkpoint_every: int | None = None
    dense_run_dir: Path | None = None
    rerun_mode: RerunMode = "none"
    run_id: str | None = None

def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _default_out_root(run_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return Path("/tmp") / f"{run_dir.expanduser().resolve().name}_benchmark_bounce_{stamp}"


def _resolve_positive_int(value: int, *, name: str) -> int:
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be > 0, got {resolved}")
    return resolved


def _resolve_probability(value: float, *, name: str) -> float:
    resolved = float(value)
    if not 0.0 < resolved < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {resolved!r}")
    return resolved


def resolve_run_dir_from_registry(
    run_id: str,
    *,
    registry_path: Path | None = None,
) -> Path:
    """Resolve a benchmark registry run id into its concrete run directory."""

    registry = load_benchmark_run_registry(registry_path or default_benchmark_run_registry_path())
    runs = cast(dict[str, Any], registry["runs"])
    try:
        run_payload = cast(dict[str, Any], runs[str(run_id)])
    except KeyError as exc:
        raise RuntimeError(f"unknown benchmark registry run_id: {run_id!r}") from exc
    artifacts = cast(dict[str, Any], run_payload["artifacts"])
    run_dir_raw = artifacts.get("run_dir")
    if not isinstance(run_dir_raw, str) or not run_dir_raw.strip():
        raise RuntimeError(f"benchmark registry run missing artifacts.run_dir: {run_id!r}")
    return resolve_registry_path_value(run_dir_raw)


def _resolve_latest_checkpoint(run_dir: Path) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    candidates = [
        resolved_run_dir / "checkpoints" / "latest.pt",
        resolved_run_dir / "train_outputs" / "checkpoints" / "latest.pt",
        resolved_run_dir / "checkpoints" / "best.pt",
        resolved_run_dir / "train_outputs" / "checkpoints" / "best.pt",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    expected = ", ".join(str(path) for path in candidates)
    raise RuntimeError(f"missing checkpoint config under {resolved_run_dir}; checked {expected}")


def _checkpoint_cfg_from_run(run_dir: Path) -> DictConfig:
    checkpoint_path = _resolve_latest_checkpoint(run_dir)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint payload must be a mapping: {checkpoint_path}")
    raw_cfg = payload.get("config")
    if not isinstance(raw_cfg, dict):
        raise RuntimeError(f"checkpoint config must be a mapping: {checkpoint_path}")
    return cast(DictConfig, OmegaConf.create(json.loads(json.dumps(raw_cfg))))


def _infer_rerun_mode(cfg: DictConfig) -> Literal["prior", "train"]:
    training_cfg = cfg.get("training")
    surface_label = ""
    if isinstance(training_cfg, Mapping):
        surface_label = str(training_cfg.get("surface_label", "")).strip().lower()
    optimizer_cfg = cfg.get("optimizer")
    optimizer_name = ""
    if isinstance(optimizer_cfg, Mapping):
        optimizer_name = str(optimizer_cfg.get("name", "")).strip().lower()
    runtime_cfg = cfg.get("runtime")
    val_batches = 0
    if isinstance(runtime_cfg, Mapping):
        raw_val_batches = runtime_cfg.get("val_batches", 0)
        if raw_val_batches is not None:
            val_batches = int(raw_val_batches)
    if surface_label.startswith("prior_"):
        return "prior"
    if optimizer_name == "schedulefree_adamw" and val_batches == 0:
        return "prior"
    return "train"


def _prepare_dense_rerun_cfg(
    cfg: DictConfig,
    *,
    dense_output_dir: Path,
    dense_checkpoint_every: int,
) -> DictConfig:
    updated = cast(DictConfig, OmegaConf.create(OmegaConf.to_container(cfg, resolve=True)))
    updated.runtime.output_dir = str(dense_output_dir.resolve())
    updated.runtime.checkpoint_every = int(dense_checkpoint_every)
    if getattr(updated.runtime, "eval_every", None) is not None:
        updated.runtime.eval_every = int(dense_checkpoint_every)
    if getattr(updated, "logging", None) is not None:
        updated.logging.use_wandb = False
        updated.logging.run_name = f"{dense_output_dir.name}"
        updated.logging.history_jsonl_path = str((dense_output_dir / "train_history.jsonl").resolve())
    return updated


def _run_dense_checkpoint_rerun(config: BenchmarkBounceDiagnosisConfig) -> Path:
    if config.dense_checkpoint_every is None:
        raise RuntimeError("dense_checkpoint_every must be set to run a dense rerun")
    dense_output_dir = (
        config.dense_run_dir.expanduser().resolve()
        if config.dense_run_dir is not None
        else (config.out_root.expanduser().resolve() / "dense_checkpoint_run").resolve()
    )
    cfg = _prepare_dense_rerun_cfg(
        _checkpoint_cfg_from_run(config.run_dir),
        dense_output_dir=dense_output_dir,
        dense_checkpoint_every=_resolve_positive_int(
            int(config.dense_checkpoint_every),
            name="dense_checkpoint_every",
        ),
    )
    rerun_mode: RerunMode = config.rerun_mode
    if rerun_mode == "none":
        raise RuntimeError("rerun_mode='none' does not allow dense_checkpoint_every reruns")
    if rerun_mode == "auto":
        rerun_mode = _infer_rerun_mode(cfg)
    if rerun_mode == "prior":
        _ = train_tabfoundry_simple_prior(cfg)
    elif rerun_mode == "train":
        _ = train(cfg)
    else:
        raise RuntimeError(f"unsupported rerun_mode: {rerun_mode!r}")
    return dense_output_dir


def _shared_bundle_analysis(
    primary_records: list[dict[str, Any]],
    confirmation_records: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    primary_by_step = {int(record["step"]): record for record in primary_records}
    confirmation_by_step = (
        {}
        if confirmation_records is None
        else {int(record["step"]): record for record in confirmation_records}
    )
    shared_steps = sorted(set(primary_by_step) & set(confirmation_by_step))
    primary_summary = curve_summary(primary_records)
    confirmation_summary = (
        None if confirmation_records is None else curve_summary(confirmation_records)
    )
    best_step_changed = bool(
        int(primary_summary["checkpoint_count"]) > 0
        and confirmation_summary is not None
        and int(confirmation_summary["checkpoint_count"]) > 0
        and int(primary_summary["best_step"]) != int(confirmation_summary["best_step"])
    )
    likely_benchmark_noise = bool(
        best_step_changed
        and primary_summary["adjacent_ci_overlap_fraction"] is not None
        and float(primary_summary["adjacent_ci_overlap_fraction"]) >= 0.5
        and confirmation_summary is not None
        and int(confirmation_summary["task_count"]) > int(primary_summary["task_count"])
    )
    return {
        "shared_step_count": int(len(shared_steps)),
        "shared_steps": shared_steps,
        "best_step_changed_between_bundles": bool(best_step_changed),
        "primary": primary_summary,
        "confirmation": confirmation_summary,
        "likely_benchmark_noise": likely_benchmark_noise,
    }


def _curve_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compatibility wrapper around the shared checkpoint-curve summary helper."""

    return curve_summary(records)


def _history_variance(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    mean = sum(values) / float(len(values))
    return sum((value - mean) ** 2 for value in values) / float(len(values))


def _training_signal(
    *,
    history: list[dict[str, Any]],
    curve_records: list[dict[str, Any]],
) -> dict[str, Any]:
    grad_norms = [
        float(record["grad_norm"])
        for record in history
        if record.get("grad_norm") is not None and math.isfinite(float(record["grad_norm"]))
    ]
    train_losses = [
        float(record["train_loss"])
        for record in history
        if record.get("train_loss") is not None and math.isfinite(float(record["train_loss"]))
    ]
    median_grad_norm = float(np.median(np.asarray(grad_norms, dtype=np.float64))) if grad_norms else 0.0
    max_grad_norm = float(max(grad_norms)) if grad_norms else 0.0
    grad_spike_ratio = (
        float(max_grad_norm / median_grad_norm)
        if grad_norms and median_grad_norm > 0.0
        else float("inf") if max_grad_norm > 0.0 else 0.0
    )
    sorted_records = sorted(curve_records, key=lambda record: int(record["step"]))
    worst_drop: dict[str, Any] | None = None
    for previous, current in zip(sorted_records, sorted_records[1:], strict=False):
        delta = float(current["roc_auc"]) - float(previous["roc_auc"])
        if worst_drop is None or delta < float(worst_drop["roc_auc_delta"]):
            window = [
                record
                for record in history
                if int(previous["step"]) < int(record["step"]) <= int(current["step"])
            ]
            window_grad_norms = [
                float(record["grad_norm"])
                for record in window
                if record.get("grad_norm") is not None and math.isfinite(float(record["grad_norm"]))
            ]
            window_losses = [
                float(record["train_loss"])
                for record in window
                if record.get("train_loss") is not None and math.isfinite(float(record["train_loss"]))
            ]
            worst_drop = {
                "from_step": int(previous["step"]),
                "to_step": int(current["step"]),
                "roc_auc_delta": float(delta),
                "window_max_grad_norm": None if not window_grad_norms else float(max(window_grad_norms)),
                "window_train_loss_var": _history_variance(window_losses),
            }
    likely_optimization_instability = bool(
        max_grad_norm >= 50.0
        or (
            worst_drop is not None
            and worst_drop["window_max_grad_norm"] is not None
            and float(worst_drop["window_max_grad_norm"]) >= max(50.0, 10.0 * median_grad_norm)
            and float(worst_drop["roc_auc_delta"]) < -0.02
        )
    )
    return {
        "history_step_count": int(len(history)),
        "median_grad_norm": float(median_grad_norm),
        "max_grad_norm": float(max_grad_norm),
        "grad_spike_ratio": float(grad_spike_ratio),
        "train_loss_variance": _history_variance(train_losses),
        "worst_checkpoint_drop": worst_drop,
        "likely_optimization_instability": likely_optimization_instability,
    }


def _task_tradeoff_signal(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {
            "positive_task_count": 0,
            "negative_task_count": 0,
            "top_quartile_abs_delta_share": 0.0,
            "likely_heterogeneous_task_tradeoff": False,
        }
    sorted_records = sorted(records, key=lambda record: int(record["step"]))
    best_record = max(sorted_records, key=lambda record: float(record["roc_auc"]))
    final_record = sorted_records[-1]
    best_dataset = cast(dict[str, float], best_record.get("dataset_roc_auc", {}))
    final_dataset = cast(dict[str, float], final_record.get("dataset_roc_auc", {}))
    shared_dataset_names = sorted(set(best_dataset) & set(final_dataset))
    deltas = {
        dataset_name: float(final_dataset[dataset_name]) - float(best_dataset[dataset_name])
        for dataset_name in shared_dataset_names
    }
    positive_count = sum(1 for value in deltas.values() if value > 0)
    negative_count = sum(1 for value in deltas.values() if value < 0)
    abs_deltas = sorted((abs(value) for value in deltas.values()), reverse=True)
    total_abs_delta = float(sum(abs_deltas))
    top_count = max(1, math.ceil(len(abs_deltas) / 4.0)) if abs_deltas else 0
    top_share = (
        float(sum(abs_deltas[:top_count]) / total_abs_delta)
        if total_abs_delta > 0.0 and top_count > 0
        else 0.0
    )
    likely_tradeoff = bool(positive_count > 0 and negative_count > 0 and top_share >= 0.5)
    return {
        "positive_task_count": int(positive_count),
        "negative_task_count": int(negative_count),
        "top_quartile_abs_delta_share": float(top_share),
        "likely_heterogeneous_task_tradeoff": likely_tradeoff,
    }


def _checkpoint_aliasing_signal(
    *,
    coarse_records: list[dict[str, Any]],
    dense_records: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    if not dense_records:
        return {
            "available": False,
            "likely_checkpoint_aliasing": False,
        }
    coarse_summary = curve_summary(coarse_records)
    dense_summary = curve_summary(dense_records)
    coarse_steps = {int(record["step"]) for record in coarse_records}
    dense_best_step = int(dense_summary["best_step"])
    likely_aliasing = bool(
        dense_best_step not in coarse_steps
        and float(dense_summary["best_roc_auc"]) > float(coarse_summary["best_roc_auc"])
    )
    dense_intervals = [
        int(current["step"]) - int(previous["step"])
        for previous, current in zip(
            sorted(dense_records, key=lambda record: int(record["step"])),
            sorted(dense_records, key=lambda record: int(record["step"]))[1:],
            strict=False,
        )
    ]
    return {
        "available": True,
        "coarse": coarse_summary,
        "dense": dense_summary,
        "dense_checkpoint_interval": None if not dense_intervals else min(dense_intervals),
        "likely_checkpoint_aliasing": likely_aliasing,
    }


def _classify_causes(
    *,
    bundle_analysis: dict[str, Any],
    training_signal: dict[str, Any],
    task_tradeoff_signal: dict[str, Any],
    checkpoint_aliasing_signal: dict[str, Any],
    evaluation_failures: dict[str, Any],
) -> dict[str, Any]:
    primary_causes: list[str] = []
    evidence: list[str] = []
    if int(evaluation_failures["failure_count"]) > 0:
        primary_causes.append("checkpoint_evaluation_failure")
        evidence.append(
            "One or more checkpoints could not be benchmarked cleanly on the selected bundle, "
            "which is itself diagnostic evidence rather than a plotting artifact."
        )
    if bool(bundle_analysis["likely_benchmark_noise"]):
        primary_causes.append("benchmark_noise")
        evidence.append(
            "The best checkpoint changes between the primary and confirmation bundles while adjacent "
            "primary-bundle confidence intervals overlap heavily."
        )
    if bool(checkpoint_aliasing_signal.get("likely_checkpoint_aliasing")):
        primary_causes.append("checkpoint_aliasing")
        evidence.append(
            "The denser checkpoint run finds a better checkpoint that is not present in the coarse "
            "25-step snapshot grid."
        )
    if bool(training_signal["likely_optimization_instability"]):
        primary_causes.append("optimization_instability")
        evidence.append(
            "Gradient norms spike sharply relative to the run median or the worst ROC AUC drop aligns "
            "with a high-gradient interval."
        )
    if bool(task_tradeoff_signal["likely_heterogeneous_task_tradeoff"]):
        primary_causes.append("heterogeneous_task_tradeoff")
        evidence.append(
            "A minority of datasets account for most of the best-to-final ROC AUC change while other "
            "datasets move in the opposite direction."
        )
    if not primary_causes:
        primary_causes.append("unclear")
        evidence.append(
            "The current traces do not isolate a single dominant cause; the next step is more repeated "
            "measurement rather than an optimizer change."
        )
    return {
        "primary_causes": primary_causes,
        "evidence": evidence,
    }


def _evaluate_one_bundle(
    *,
    run_dir: Path,
    bundle_path: Path,
    device: str,
    out_path: Path,
    bootstrap_samples: int,
    bootstrap_confidence: float,
) -> dict[str, Any]:
    bundle, allow_missing_values = load_benchmark_bundle_for_execution(bundle_path)
    selection = cast(dict[str, Any], bundle["selection"])
    datasets, benchmark_tasks = load_openml_benchmark_datasets(
        new_instances=int(selection["new_instances"]),
        benchmark_bundle_path=bundle_path,
        allow_missing_values=allow_missing_values,
    )
    raw_records = evaluate_tab_foundry_run(
        run_dir,
        datasets=datasets,
        task_type=benchmark_bundle_task_type(bundle),
        device=device,
        allow_checkpoint_failures=True,
        allow_missing_values=allow_missing_values,
    )
    diagnostics = summarize_checkpoint_curve(
        raw_records,
        bootstrap_samples=int(bootstrap_samples),
        bootstrap_confidence=float(bootstrap_confidence),
    )
    records = cast(list[dict[str, Any]], diagnostics["successful_records"])
    failed_records = cast(list[dict[str, Any]], diagnostics["failed_records"])
    write_jsonl(
        out_path,
        cast(list[dict[str, Any]], diagnostics["records"]),
    )
    return {
        "bundle": benchmark_bundle_summary(bundle, source_path=bundle_path),
        "benchmark_tasks": benchmark_tasks,
        "records": records,
        "records_path": str(out_path.resolve()),
        "summary": curve_summary(records),
        "failure_count": int(len(failed_records)),
        "failed_checkpoints": failed_records,
    }


def run_benchmark_bounce_diagnosis(config: BenchmarkBounceDiagnosisConfig) -> dict[str, Any]:
    """Benchmark one run on multiple bundles and classify likely bounce causes."""

    bootstrap_samples = _resolve_positive_int(int(config.bootstrap_samples), name="bootstrap_samples")
    bootstrap_confidence = _resolve_probability(
        float(config.bootstrap_confidence),
        name="bootstrap_confidence",
    )
    out_root = config.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    run_dir = config.run_dir.expanduser().resolve()
    if not run_dir.exists():
        raise RuntimeError(f"run_dir does not exist: {run_dir}")

    benchmark_bundle_path = (
        default_benchmark_bundle_path()
        if config.benchmark_bundle_path is None
        else config.benchmark_bundle_path.expanduser().resolve()
    )

    primary = _evaluate_one_bundle(
        run_dir=run_dir,
        bundle_path=benchmark_bundle_path,
        device=config.device,
        out_path=out_root / "primary_bundle_curve.jsonl",
        bootstrap_samples=bootstrap_samples,
        bootstrap_confidence=bootstrap_confidence,
    )
    confirmation: dict[str, Any] | None = None
    if config.confirmation_benchmark_bundle_path is not None:
        confirmation = _evaluate_one_bundle(
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
        dense_run_dir = _run_dense_checkpoint_rerun(config)
    if dense_run_dir is not None:
        dense_bundle_path = (
            benchmark_bundle_path
            if confirmation is None
            else Path(str(cast(dict[str, Any], confirmation["bundle"])["source_path"]))
        )
        dense_confirmation = _evaluate_one_bundle(
            run_dir=dense_run_dir,
            bundle_path=dense_bundle_path,
            device=config.device,
            out_path=out_root / "dense_confirmation_bundle_curve.jsonl",
            bootstrap_samples=bootstrap_samples,
            bootstrap_confidence=bootstrap_confidence,
        )

    history = load_history(
        run_dir / "train_history.jsonl"
        if (run_dir / "train_history.jsonl").exists()
        else run_dir / "train_outputs" / "train_history.jsonl"
    )
    bundle_analysis = _shared_bundle_analysis(
        cast(list[dict[str, Any]], primary["records"]),
        None if confirmation is None else cast(list[dict[str, Any]], confirmation["records"]),
    )
    training_signal = _training_signal(
        history=history,
        curve_records=(
            cast(list[dict[str, Any]], primary["records"])
            if confirmation is None
            else cast(list[dict[str, Any]], confirmation["records"])
        ),
    )
    task_tradeoff_signal = _task_tradeoff_signal(
        (
            cast(list[dict[str, Any]], primary["records"])
            if confirmation is None
            else cast(list[dict[str, Any]], confirmation["records"])
        ),
    )
    checkpoint_aliasing_signal = _checkpoint_aliasing_signal(
        coarse_records=(
            cast(list[dict[str, Any]], primary["records"])
            if confirmation is None
            else cast(list[dict[str, Any]], confirmation["records"])
        ),
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
    classification = _classify_causes(
        bundle_analysis=bundle_analysis,
        training_signal=training_signal,
        task_tradeoff_signal=task_tradeoff_signal,
        checkpoint_aliasing_signal=checkpoint_aliasing_signal,
        evaluation_failures=evaluation_failures,
    )

    summary = {
        "schema": DIAGNOSIS_SCHEMA,
        "generated_at_utc": _utc_now(),
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
    write_json(out_root / "benchmark_bounce_diagnosis.json", summary)
    return summary


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
