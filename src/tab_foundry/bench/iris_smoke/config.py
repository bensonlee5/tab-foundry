"""Configuration helpers for the Iris smoke harness."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


DEFAULT_DEVICE = "cpu"
DEFAULT_SEED = 1
DEFAULT_INITIAL_NUM_TASKS = 64
DEFAULT_MAX_NUM_TASKS = 512
DEFAULT_TEST_SIZE = 0.5
DEFAULT_CHECKPOINT_EVERY = 2
DEFAULT_IRIS_BENCHMARK_SEEDS = 5
DEFAULT_TRAIN_RATIO = 0.9
DEFAULT_VAL_RATIO = 0.05
DEFAULT_FILTER_POLICY = "accepted_only"
DEFAULT_STAGE1_STEPS = 4
DEFAULT_STAGE1_LR_MAX = 8.0e-4
DEFAULT_STAGE2_STEPS = 2
DEFAULT_STAGE2_LR_MAX = 1.0e-4


@dataclass(slots=True)
class IrisSmokeConfig:
    """Input configuration for the Iris smoke harness."""

    out_root: Path
    device: str = DEFAULT_DEVICE
    seed: int = DEFAULT_SEED
    initial_num_tasks: int = DEFAULT_INITIAL_NUM_TASKS
    max_num_tasks: int = DEFAULT_MAX_NUM_TASKS
    test_size: float = DEFAULT_TEST_SIZE
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
    iris_benchmark_seeds: int = DEFAULT_IRIS_BENCHMARK_SEEDS
    train_ratio: float = DEFAULT_TRAIN_RATIO
    val_ratio: float = DEFAULT_VAL_RATIO
    filter_policy: str = DEFAULT_FILTER_POLICY


def default_out_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return Path("/tmp") / f"tab_foundry_iris_smoke_{stamp}"
