"""Iris-backed smoke harness for CI and local verification."""

from __future__ import annotations

from typing import Any

from tab_foundry.bench.artifacts import (
    checkpoint_snapshots_from_history,
    ensure_finite_metrics,
    plot_loss_curve,
    write_json,
)
from tab_foundry.bench.iris import (
    IrisEvalSummary as IrisEvalSummary,
    evaluate_iris_checkpoint,
)
from tab_foundry.bench.openml_benchmark import resolve_device
from tab_foundry.bench.smoke_common import (
    build_cls_smoke_eval_config,
    build_cls_smoke_train_config,
    build_manifest_payload,
)
from tab_realdata_hub.manifest import build_manifest
from tab_foundry.training.evaluate import evaluate_checkpoint
from tab_foundry.training.trainer import train

from .config import (
    DEFAULT_CHECKPOINT_EVERY,
    DEFAULT_DEVICE,
    DEFAULT_FILTER_POLICY as DEFAULT_FILTER_POLICY,
    DEFAULT_INITIAL_NUM_TASKS,
    DEFAULT_IRIS_BENCHMARK_SEEDS,
    DEFAULT_MAX_NUM_TASKS,
    DEFAULT_SEED,
    DEFAULT_STAGE1_LR_MAX as DEFAULT_STAGE1_LR_MAX,
    DEFAULT_STAGE1_STEPS as DEFAULT_STAGE1_STEPS,
    DEFAULT_STAGE2_LR_MAX as DEFAULT_STAGE2_LR_MAX,
    DEFAULT_STAGE2_STEPS as DEFAULT_STAGE2_STEPS,
    DEFAULT_TEST_SIZE as DEFAULT_TEST_SIZE,
    DEFAULT_TRAIN_RATIO as DEFAULT_TRAIN_RATIO,
    DEFAULT_VAL_RATIO as DEFAULT_VAL_RATIO,
    IrisSmokeConfig,
    default_out_root as _default_out_root_impl,
)
from .data_gen import write_iris_tasks as _write_iris_tasks_impl
from .report import (
    iris_benchmark_payload as _iris_benchmark_payload_impl,
    write_summary_markdown as _write_summary_markdown_impl,
)
from .runner import run_iris_smoke as _run_iris_smoke_impl


_default_out_root = _default_out_root_impl
_write_iris_tasks = _write_iris_tasks_impl
_iris_benchmark_payload = _iris_benchmark_payload_impl
_write_summary_markdown = _write_summary_markdown_impl

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


def run_iris_smoke(config: IrisSmokeConfig) -> dict[str, Any]:
    """Execute the end-to-end Iris smoke harness."""

    return _run_iris_smoke_impl(
        config,
        resolve_device_fn=resolve_device,
        write_iris_tasks_fn=_write_iris_tasks,
        build_manifest_fn=build_manifest,
        train_fn=train,
        evaluate_checkpoint_fn=evaluate_checkpoint,
        evaluate_iris_checkpoint_fn=evaluate_iris_checkpoint,
        ensure_finite_metrics_fn=ensure_finite_metrics,
        plot_loss_curve_fn=plot_loss_curve,
        checkpoint_snapshots_from_history_fn=checkpoint_snapshots_from_history,
        write_json_fn=write_json,
        write_summary_markdown_fn=_write_summary_markdown,
        iris_benchmark_payload_fn=_iris_benchmark_payload,
        build_cls_smoke_train_config_fn=build_cls_smoke_train_config,
        build_cls_smoke_eval_config_fn=build_cls_smoke_eval_config,
        build_manifest_payload_fn=build_manifest_payload,
    )
