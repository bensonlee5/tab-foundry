"""Iris-backed smoke harness for CI and local verification."""

from tab_foundry.bench.iris import IrisEvalSummary
from .config import (
    DEFAULT_CHECKPOINT_EVERY,
    DEFAULT_DEVICE,
    DEFAULT_FILTER_POLICY,
    DEFAULT_INITIAL_NUM_TASKS,
    DEFAULT_IRIS_BENCHMARK_SEEDS,
    DEFAULT_MAX_NUM_TASKS,
    DEFAULT_SEED,
    DEFAULT_STAGE1_LR_MAX,
    DEFAULT_STAGE1_STEPS,
    DEFAULT_STAGE2_LR_MAX,
    DEFAULT_STAGE2_STEPS,
    DEFAULT_TEST_SIZE,
    DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO,
    IrisSmokeConfig,
)
from .runner import run_iris_smoke

__all__ = [
    "DEFAULT_CHECKPOINT_EVERY",
    "DEFAULT_DEVICE",
    "DEFAULT_FILTER_POLICY",
    "DEFAULT_INITIAL_NUM_TASKS",
    "DEFAULT_IRIS_BENCHMARK_SEEDS",
    "DEFAULT_MAX_NUM_TASKS",
    "DEFAULT_SEED",
    "DEFAULT_STAGE1_LR_MAX",
    "DEFAULT_STAGE1_STEPS",
    "DEFAULT_STAGE2_LR_MAX",
    "DEFAULT_STAGE2_STEPS",
    "DEFAULT_TEST_SIZE",
    "DEFAULT_TRAIN_RATIO",
    "DEFAULT_VAL_RATIO",
    "IrisEvalSummary",
    "IrisSmokeConfig",
    "run_iris_smoke",
]
