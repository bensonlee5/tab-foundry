"""Dense-rerun helpers for benchmark bounce diagnosis."""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal, cast

from omegaconf import DictConfig, OmegaConf
import torch

from tab_foundry.benchmark_registry import (
    default_benchmark_run_registry_path,
    load_benchmark_run_entry,
    resolve_registry_path_value,
)
from tab_foundry.bench.bounce.config import BenchmarkBounceDiagnosisConfig, RerunMode, resolve_positive_int
from tab_foundry.repo_paths import resolve_repo_relative_path
from tab_foundry.training.checkpoint_paths import resolve_latest_checkpoint_path
from tab_foundry.training.instability import telemetry_path
from tab_foundry.training.prior_train import train_tabfoundry_simple_prior
from tab_foundry.training.trainer import train


def resolve_run_dir_from_registry(
    run_id: str,
    *,
    registry_path: Path | None = None,
) -> Path:
    """Resolve a benchmark registry run id into its concrete run directory."""

    run_payload = load_benchmark_run_entry(
        run_id,
        path=registry_path or default_benchmark_run_registry_path(),
    )
    artifacts = cast(dict[str, Any], run_payload["artifacts"])
    run_dir_raw = artifacts.get("run_dir")
    if not isinstance(run_dir_raw, str) or not run_dir_raw.strip():
        raise RuntimeError(f"benchmark registry run missing artifacts.run_dir: {run_id!r}")
    return resolve_registry_path_value(run_dir_raw)


def resolve_latest_checkpoint(run_dir: Path) -> Path:
    resolved_run_dir = run_dir.expanduser().resolve()
    checkpoint_path = resolve_latest_checkpoint_path(
        resolved_run_dir,
        additional_run_dirs=(resolved_run_dir / "train_outputs",),
        include_best_fallback=True,
    )
    if checkpoint_path is not None:
        return checkpoint_path
    candidates = [
        resolved_run_dir / "checkpoints" / "latest.pt",
        resolved_run_dir / "train_outputs" / "checkpoints" / "latest.pt",
        resolved_run_dir / "checkpoints" / "best.pt",
        resolved_run_dir / "train_outputs" / "checkpoints" / "best.pt",
    ]
    expected = ", ".join(str(path) for path in candidates)
    raise RuntimeError(f"missing checkpoint config under {resolved_run_dir}; checked {expected}")


def checkpoint_cfg_from_run(run_dir: Path) -> DictConfig:
    checkpoint_path = resolve_latest_checkpoint(run_dir)
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError(f"checkpoint payload must be a mapping: {checkpoint_path}")
    raw_cfg = payload.get("config")
    if not isinstance(raw_cfg, dict):
        raise RuntimeError(f"checkpoint config must be a mapping: {checkpoint_path}")
    return cast(DictConfig, OmegaConf.create(json.loads(json.dumps(raw_cfg))))


def _load_run_telemetry(run_dir: Path) -> dict[str, Any] | None:
    resolved_run_dir = run_dir.expanduser().resolve()
    candidates = (
        telemetry_path(resolved_run_dir),
        telemetry_path(resolved_run_dir / "train_outputs"),
    )
    for candidate in candidates:
        if not candidate.exists():
            continue
        payload = json.loads(candidate.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"telemetry payload must be a mapping: {candidate}")
        return cast(dict[str, Any], payload)
    return None


def resolve_prior_dump_from_telemetry(run_dir: Path) -> Path:
    telemetry_payload = _load_run_telemetry(run_dir)
    if telemetry_payload is None:
        raise RuntimeError(
            "dense prior rerun requires telemetry.json with missingness.prior_dump.path to resolve the prior dump"
        )
    raw_missingness = telemetry_payload.get("missingness")
    if not isinstance(raw_missingness, Mapping):
        raise RuntimeError("telemetry missingness summary is required for dense prior reruns")
    raw_prior_dump = raw_missingness.get("prior_dump")
    if not isinstance(raw_prior_dump, Mapping):
        raise RuntimeError("telemetry missingness.prior_dump summary is required for dense prior reruns")
    raw_prior_dump_path = raw_prior_dump.get("path")
    if not isinstance(raw_prior_dump_path, str) or not raw_prior_dump_path.strip():
        raise RuntimeError("telemetry missingness.prior_dump.path is required for dense prior reruns")
    return resolve_repo_relative_path(str(raw_prior_dump_path).strip())


def infer_rerun_mode(cfg: DictConfig) -> Literal["prior", "train"]:
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


def prepare_dense_rerun_cfg(
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


def run_dense_checkpoint_rerun(
    config: BenchmarkBounceDiagnosisConfig,
) -> Path:
    if config.dense_checkpoint_every is None:
        raise RuntimeError("dense_checkpoint_every must be set to run a dense rerun")
    dense_output_dir = (
        config.dense_run_dir.expanduser().resolve()
        if config.dense_run_dir is not None
        else (config.out_root.expanduser().resolve() / "dense_checkpoint_run").resolve()
    )
    cfg = prepare_dense_rerun_cfg(
        checkpoint_cfg_from_run(config.run_dir),
        dense_output_dir=dense_output_dir,
        dense_checkpoint_every=resolve_positive_int(
            int(config.dense_checkpoint_every),
            name="dense_checkpoint_every",
        ),
    )
    rerun_mode: RerunMode = config.rerun_mode
    if rerun_mode == "none":
        raise RuntimeError("rerun_mode='none' does not allow dense_checkpoint_every reruns")
    if rerun_mode == "auto":
        rerun_mode = infer_rerun_mode(cfg)
    if rerun_mode == "prior":
        _ = train_tabfoundry_simple_prior(
            cfg,
            prior_dump_path=resolve_prior_dump_from_telemetry(config.run_dir),
        )
    elif rerun_mode == "train":
        _ = train(cfg)
    else:
        raise RuntimeError(f"unsupported rerun_mode: {rerun_mode!r}")
    return dense_output_dir
