"""Benchmark comparison config and shared defaults."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from tab_foundry.external_benchmarks import DEFAULT_EXTERNAL_BENCHMARKS


DEFAULT_NANOTABPFN_STEPS = 2500
DEFAULT_NANOTABPFN_SEEDS = 2
DEFAULT_NANOTABPFN_EVAL_EVERY = 250
DEFAULT_NANOTABPFN_BATCH_SIZE = 32
DEFAULT_NANOTABPFN_LR = 4.0e-3
DEFAULT_TABICL_CLASSIFIER_CHECKPOINT_VERSION = "tabicl-classifier-v2-20260212.ckpt"
DEFAULT_TABICL_REGRESSOR_CHECKPOINT_VERSION = "tabicl-regressor-v2-20260212.ckpt"


@dataclass(slots=True)
class BenchmarkComparisonConfig:
    """Input configuration for benchmark comparison against external baselines."""

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
    benchmark_manifest_path: Path | None = None
    external_benchmarks: tuple[str, ...] = DEFAULT_EXTERNAL_BENCHMARKS
    reuse_nanotabpfn_curve_path: Path | None = None
    reuse_nanotabpfn_error: Mapping[str, Any] | None = None
    reuse_nanotabpfn_metadata: Mapping[str, Any] | None = None
    tabicl_root: Path = Path("~/dev/tabicl")
    tab_realdata_hub_root: Path | None = None
    tabicl_classifier_checkpoint_version: str = DEFAULT_TABICL_CLASSIFIER_CHECKPOINT_VERSION
    tabicl_regressor_checkpoint_version: str = DEFAULT_TABICL_REGRESSOR_CHECKPOINT_VERSION
