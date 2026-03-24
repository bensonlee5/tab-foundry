"""Iris-backed smoke harness for CI and local verification."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

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
from tab_foundry.bench.nanotabpfn import resolve_device
from tab_foundry.bench.smoke_common import (
    build_cls_smoke_eval_config,
    build_cls_smoke_train_config,
    build_manifest_payload,
)
from tab_foundry.data.manifest import build_manifest
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
    "build_parser",
    "main",
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Iris-backed tab-foundry smoke harness")
    parser.add_argument("--out-root", default=None, help="Output directory root")
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=("cpu", "cuda", "mps", "auto"),
        help="Training and evaluation device",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Shared run seed")
    parser.add_argument(
        "--initial-num-tasks",
        type=int,
        default=DEFAULT_INITIAL_NUM_TASKS,
        help="Initial number of derived Iris tasks to materialize",
    )
    parser.add_argument(
        "--max-num-tasks",
        type=int,
        default=DEFAULT_MAX_NUM_TASKS,
        help="Maximum number of derived Iris tasks to materialize",
    )
    parser.add_argument(
        "--iris-benchmark-seeds",
        type=int,
        default=DEFAULT_IRIS_BENCHMARK_SEEDS,
        help="Number of binary Iris benchmark splits for the final checkpoint",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help="Checkpoint snapshot cadence in steps",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    out_root = _default_out_root() if args.out_root is None else Path(str(args.out_root))
    telemetry = run_iris_smoke(
        IrisSmokeConfig(
            out_root=out_root,
            device=str(args.device),
            seed=int(args.seed),
            initial_num_tasks=int(args.initial_num_tasks),
            max_num_tasks=int(args.max_num_tasks),
            iris_benchmark_seeds=int(args.iris_benchmark_seeds),
            checkpoint_every=int(args.checkpoint_every),
        )
    )
    print("iris smoke complete:")
    print(f"  out_root={out_root.resolve()}")
    print(f"  best_checkpoint={telemetry['artifacts']['best_checkpoint']}")
    print(f"  eval_metrics={telemetry['eval_metrics']}")
    print(f"  iris_benchmark_means={telemetry['iris_benchmark']['means']}")
    print(f"  timings_seconds={telemetry['timings_seconds']}")
    return 0
