"""Shared configuration and validation helpers for benchmark bounce diagnosis."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Literal

from tab_foundry.bench.nanotabpfn.curves import (
    DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_CONFIDENCE,
    DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_SAMPLES,
)
from tab_foundry.timestamps import utc_now as utc_now

DIAGNOSIS_SCHEMA = "benchmark_bounce_diagnosis_v1"

RerunMode = Literal["auto", "prior", "train", "none"]
DEFAULT_BOUNCE_BOOTSTRAP_SAMPLES = DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_SAMPLES
DEFAULT_BOUNCE_BOOTSTRAP_CONFIDENCE = DEFAULT_CHECKPOINT_DIAGNOSTIC_BOOTSTRAP_CONFIDENCE


@dataclass(slots=True)
class BenchmarkBounceDiagnosisConfig:
    """Input configuration for one benchmark-bounce diagnosis run."""

    run_dir: Path
    out_root: Path
    device: str = "auto"
    benchmark_manifest_path: Path | None = None
    confirmation_benchmark_manifest_path: Path | None = None
    bootstrap_samples: int = DEFAULT_BOUNCE_BOOTSTRAP_SAMPLES
    bootstrap_confidence: float = DEFAULT_BOUNCE_BOOTSTRAP_CONFIDENCE
    dense_checkpoint_every: int | None = None
    dense_run_dir: Path | None = None
    rerun_mode: RerunMode = "none"
    run_id: str | None = None

def default_out_root(run_dir: Path) -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return Path("/tmp") / f"{run_dir.expanduser().resolve().name}_benchmark_bounce_{stamp}"


def resolve_positive_int(value: int, *, name: str) -> int:
    resolved = int(value)
    if resolved <= 0:
        raise ValueError(f"{name} must be > 0, got {resolved}")
    return resolved


def resolve_probability(value: float, *, name: str) -> float:
    resolved = float(value)
    if not 0.0 < resolved < 1.0:
        raise ValueError(f"{name} must be in (0, 1), got {resolved!r}")
    return resolved
