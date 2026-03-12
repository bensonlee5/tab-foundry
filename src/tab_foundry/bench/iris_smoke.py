"""Iris-backed smoke harness for CI and local verification."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time
from typing import Any, Sequence

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from omegaconf import DictConfig

from tab_foundry.bench.artifacts import (
    checkpoint_snapshots_from_history,
    ensure_finite_metrics,
    plot_loss_curve,
    write_json,
)
from tab_foundry.bench.iris import IrisEvalSummary, evaluate_iris_checkpoint
from tab_foundry.bench.nanotabpfn import resolve_device
from tab_foundry.config import compose_config
from tab_foundry.data.manifest import ManifestSummary, build_manifest
from tab_foundry.training.evaluate import evaluate_checkpoint
from tab_foundry.training.trainer import train


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


def _default_out_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return Path("/tmp") / f"tab_foundry_iris_smoke_{stamp}"


def _build_split_table(rows: list[tuple[int, np.ndarray, np.ndarray]]) -> pa.Table:
    dataset_indices: list[int] = []
    row_indices: list[int] = []
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    for dataset_index, x, y in rows:
        for row_index in range(int(x.shape[0])):
            dataset_indices.append(int(dataset_index))
            row_indices.append(int(row_index))
            x_rows.append(np.asarray(x[row_index], dtype=np.float32).tolist())
            y_rows.append(int(y[row_index]))

    return pa.table(
        {
            "dataset_index": pa.array(dataset_indices, type=pa.int64()),
            "row_index": pa.array(row_indices, type=pa.int64()),
            "x": pa.array(x_rows, type=pa.list_(pa.float32())),
            "y": pa.array(y_rows, type=pa.int64()),
        }
    )


def _binary_iris_arrays() -> tuple[np.ndarray, np.ndarray]:
    iris = load_iris()
    x = np.asarray(iris.data[iris.target != 0], dtype=np.float32)
    y = np.asarray(iris.target[iris.target != 0] - 1, dtype=np.int64)
    return x, y


def _write_iris_tasks(
    generated_dir: Path,
    *,
    num_tasks: int,
    seed: int,
    test_size: float,
) -> Path:
    if num_tasks <= 0:
        raise ValueError(f"num_tasks must be > 0, got {num_tasks}")
    if not 0.0 < test_size < 1.0:
        raise ValueError(f"test_size must be between 0 and 1, got {test_size}")

    if generated_dir.exists():
        shutil.rmtree(generated_dir)
    shard_dir = generated_dir / "shard_00000"
    shard_dir.mkdir(parents=True, exist_ok=True)

    x, y = _binary_iris_arrays()
    train_rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    test_rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    metadata_records: list[dict[str, Any]] = []
    for dataset_index in range(num_tasks):
        split_seed = int(seed + dataset_index)
        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=test_size,
            random_state=split_seed,
            stratify=y,
        )
        train_rows.append((dataset_index, x_train, y_train))
        test_rows.append((dataset_index, x_test, y_test))
        metadata_records.append(
            {
                "dataset_index": int(dataset_index),
                "n_train": int(x_train.shape[0]),
                "n_test": int(x_test.shape[0]),
                "n_features": int(x_train.shape[1]),
                "feature_types": ["num"] * int(x_train.shape[1]),
                "metadata": {
                    "n_features": int(x_train.shape[1]),
                    "n_classes": 2,
                    "seed": split_seed,
                    "filter": {"mode": "deferred", "status": "accepted", "accepted": True},
                    "config": {"dataset": {"task": "classification"}},
                    "source": {"name": "iris_binary_smoke"},
                },
            }
        )

    pq.write_table(_build_split_table(train_rows), shard_dir / "train.parquet")
    pq.write_table(_build_split_table(test_rows), shard_dir / "test.parquet")
    with (shard_dir / "metadata.ndjson").open("wb") as handle:
        for record in metadata_records:
            serialized = (json.dumps(record, sort_keys=True) + "\n").encode("utf-8")
            handle.write(serialized)
    return generated_dir


def _build_train_config(
    *,
    manifest_path: Path,
    output_dir: Path,
    history_path: Path,
    device: str,
    checkpoint_every: int,
) -> DictConfig:
    cfg = compose_config(["experiment=cls_smoke", "optimizer=adamw", "logging.use_wandb=false"])
    cfg.data.manifest_path = str(manifest_path)
    cfg.data.train_row_cap = None
    cfg.data.test_row_cap = None
    cfg.runtime.output_dir = str(output_dir)
    cfg.runtime.device = str(device)
    cfg.runtime.eval_every = 1
    cfg.runtime.checkpoint_every = int(checkpoint_every)
    cfg.runtime.val_batches = 1
    cfg.schedule.stages = [
        {"name": "stage1", "steps": DEFAULT_STAGE1_STEPS, "lr_max": DEFAULT_STAGE1_LR_MAX},
        {"name": "stage2", "steps": DEFAULT_STAGE2_STEPS, "lr_max": DEFAULT_STAGE2_LR_MAX},
    ]
    cfg.logging.history_jsonl_path = str(history_path)
    return cfg


def _build_eval_config(
    *,
    manifest_path: Path,
    checkpoint_path: Path,
    device: str,
) -> DictConfig:
    cfg = compose_config(["experiment=cls_smoke", "optimizer=adamw", "logging.use_wandb=false"])
    cfg.data.manifest_path = str(manifest_path)
    cfg.data.train_row_cap = None
    cfg.data.test_row_cap = None
    cfg.runtime.device = str(device)
    cfg.eval.checkpoint = str(checkpoint_path)
    cfg.eval.split = "test"
    return cfg


def _manifest_payload(summary: ManifestSummary) -> dict[str, Any]:
    return {
        "discovered_records": int(summary.discovered_records),
        "excluded_records": int(summary.excluded_records),
        "total_records": int(summary.total_records),
        "train_records": int(summary.train_records),
        "val_records": int(summary.val_records),
        "test_records": int(summary.test_records),
        "filter_policy": str(summary.filter_policy),
        "warnings": list(summary.warnings),
    }


def _iris_benchmark_payload(summary: IrisEvalSummary) -> dict[str, Any]:
    means = {name: float(np.mean(values)) for name, values in summary.results.items()}
    stddevs = {name: float(np.std(values)) for name, values in summary.results.items()}
    ensure_finite_metrics(means, context="iris benchmark mean")
    ensure_finite_metrics(stddevs, context="iris benchmark std")
    return {
        "checkpoint": str(summary.checkpoint),
        "means": means,
        "stddevs": stddevs,
        "raw": {name: [float(value) for value in values] for name, values in summary.results.items()},
    }


def _format_float(value: Any, digits: int = 3) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    if not math.isfinite(float(value)):
        return "-"
    return f"{float(value):.{digits}f}"


def _write_summary_markdown(path: Path, telemetry: dict[str, Any]) -> Path:
    config = telemetry["config"]
    manifest = telemetry["manifest"]
    train_metrics = telemetry["train_metrics"]
    eval_metrics = telemetry["eval_metrics"]
    benchmark = telemetry["iris_benchmark"]
    timings = telemetry["timings_seconds"]
    artifacts = telemetry["artifacts"]
    checkpoint_snapshots = telemetry["checkpoint_snapshots"]

    lines: list[str] = [
        "# Iris Smoke Report",
        "",
        f"- Generated at: `{telemetry.get('generated_at_utc', '-')}`",
        f"- Device: `{config['device']}`",
        f"- Task count: `{config['final_num_tasks']}`",
        f"- Task count attempts: `{', '.join(str(value) for value in config['task_count_attempts'])}`",
        f"- Manifest splits: `train={manifest['train_records']}, val={manifest['val_records']}, test={manifest['test_records']}`",
        f"- Best checkpoint: `{artifacts['best_checkpoint']}`",
        "",
        "## Timings",
        "",
        "| Stage | Seconds |",
        "|---|---:|",
    ]
    for key in (
        "generate_iris_tasks",
        "build_manifest",
        "train",
        "eval",
        "iris_benchmark",
        "total",
    ):
        lines.append(f"| {key} | {_format_float(timings.get(key), 3)} |")

    lines.extend(
        [
            "",
            "## Metrics",
            "",
            "| Category | Key | Value |",
            "|---|---|---:|",
        ]
    )
    for key, value in sorted(train_metrics.items()):
        lines.append(f"| train | {key} | {_format_float(value, 6)} |")
    for key, value in sorted(eval_metrics.items()):
        lines.append(f"| eval | {key} | {_format_float(value, 6)} |")

    lines.extend(
        [
            "",
            "## Iris Benchmark",
            "",
            "| Method | Mean ROC AUC | Std |",
            "|---|---:|---:|",
        ]
    )
    for name in sorted(benchmark["means"]):
        lines.append(
            "| "
            f"{name} | "
            f"{_format_float(benchmark['means'][name], 6)} | "
            f"{_format_float(benchmark['stddevs'][name], 6)} |"
        )

    lines.extend(
        [
            "",
            "## Checkpoints",
            "",
            "| Step | Train elapsed (s) |",
            "|---|---:|",
        ]
    )
    for snapshot in checkpoint_snapshots:
        lines.append(
            f"| {int(snapshot['step'])} | {_format_float(snapshot['train_elapsed_seconds'], 3)} |"
        )

    lines.extend(["", "## Artifacts", ""])
    for key in (
        "generated_dir",
        "manifest_path",
        "train_output_dir",
        "train_history_jsonl",
        "loss_curve_png",
        "telemetry_json",
        "summary_md",
    ):
        if key in artifacts:
            lines.append(f"- `{artifacts[key]}`")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def run_iris_smoke(config: IrisSmokeConfig) -> dict[str, Any]:
    """Execute the end-to-end Iris smoke harness."""

    out_root = config.out_root.expanduser().resolve()
    resolved_device = resolve_device(config.device)
    if config.initial_num_tasks <= 0:
        raise ValueError(f"initial_num_tasks must be > 0, got {config.initial_num_tasks}")
    if config.max_num_tasks < config.initial_num_tasks:
        raise ValueError(
            "max_num_tasks must be >= initial_num_tasks, "
            f"got max_num_tasks={config.max_num_tasks}, initial_num_tasks={config.initial_num_tasks}"
        )
    if config.checkpoint_every <= 0:
        raise ValueError(f"checkpoint_every must be > 0, got {config.checkpoint_every}")
    if config.iris_benchmark_seeds <= 0:
        raise ValueError(
            f"iris_benchmark_seeds must be > 0, got {config.iris_benchmark_seeds}"
        )

    out_root.mkdir(parents=True, exist_ok=True)
    generated_dir = out_root / "generated"
    manifest_path = out_root / "manifest.parquet"
    train_output_dir = out_root / "train_outputs"
    history_path = train_output_dir / "train_history.jsonl"
    loss_curve_path = train_output_dir / "loss_curve.png"
    telemetry_path = out_root / "telemetry.json"
    summary_path = out_root / "summary.md"

    timings_seconds: dict[str, float] = {}
    attempted_task_counts: list[int] = []
    telemetry: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "success": False,
        "config": {
            "device": resolved_device,
            "seed": int(config.seed),
            "requested_num_tasks": int(config.initial_num_tasks),
            "max_num_tasks": int(config.max_num_tasks),
            "final_num_tasks": None,
            "task_count_attempts": attempted_task_counts,
            "test_size": float(config.test_size),
            "checkpoint_every": int(config.checkpoint_every),
            "iris_benchmark_seeds": int(config.iris_benchmark_seeds),
        },
        "artifacts": {
            "generated_dir": str(generated_dir),
            "manifest_path": str(manifest_path),
            "train_output_dir": str(train_output_dir),
            "train_history_jsonl": str(history_path),
            "loss_curve_png": str(loss_curve_path),
            "telemetry_json": str(telemetry_path),
            "summary_md": str(summary_path),
        },
        "timings_seconds": timings_seconds,
    }

    total_start = time.perf_counter()
    manifest_summary: ManifestSummary | None = None
    try:
        num_tasks = int(config.initial_num_tasks)
        timings_seconds["generate_iris_tasks"] = 0.0
        timings_seconds["build_manifest"] = 0.0
        while True:
            attempted_task_counts.append(num_tasks)

            stage_start = time.perf_counter()
            _write_iris_tasks(
                generated_dir,
                num_tasks=num_tasks,
                seed=int(config.seed),
                test_size=float(config.test_size),
            )
            timings_seconds["generate_iris_tasks"] += time.perf_counter() - stage_start

            stage_start = time.perf_counter()
            manifest_summary = build_manifest(
                data_roots=[generated_dir],
                out_path=manifest_path,
                train_ratio=config.train_ratio,
                val_ratio=config.val_ratio,
                filter_policy=config.filter_policy,
            )
            timings_seconds["build_manifest"] += time.perf_counter() - stage_start

            if (
                manifest_summary.train_records > 0
                and manifest_summary.val_records > 0
                and manifest_summary.test_records > 0
            ):
                break
            if num_tasks >= config.max_num_tasks:
                raise RuntimeError(
                    "iris smoke could not populate all manifest splits up to "
                    f"max_num_tasks={config.max_num_tasks}: "
                    f"train={manifest_summary.train_records}, "
                    f"val={manifest_summary.val_records}, "
                    f"test={manifest_summary.test_records}"
                )
            num_tasks = min(int(config.max_num_tasks), int(num_tasks * 2))

        telemetry["config"]["final_num_tasks"] = num_tasks

        stage_start = time.perf_counter()
        train_result = train(
            _build_train_config(
                manifest_path=manifest_path,
                output_dir=train_output_dir,
                history_path=history_path,
                device=resolved_device,
                checkpoint_every=config.checkpoint_every,
            )
        )
        timings_seconds["train"] = time.perf_counter() - stage_start
        if train_result.best_checkpoint is None:
            raise RuntimeError("training did not produce a best checkpoint")

        stage_start = time.perf_counter()
        eval_result = evaluate_checkpoint(
            _build_eval_config(
                manifest_path=manifest_path,
                checkpoint_path=train_result.best_checkpoint,
                device=resolved_device,
            )
        )
        timings_seconds["eval"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        iris_summary = evaluate_iris_checkpoint(
            train_result.best_checkpoint,
            device=resolved_device,
            seeds=int(config.iris_benchmark_seeds),
        )
        timings_seconds["iris_benchmark"] = time.perf_counter() - stage_start

        ensure_finite_metrics(train_result.metrics, context="train")
        ensure_finite_metrics(eval_result.metrics, context="eval")

        plot_loss_curve(history_path, loss_curve_path, title="tab-foundry iris smoke loss curve")
        checkpoint_snapshots = checkpoint_snapshots_from_history(
            history_path,
            train_output_dir / "checkpoints",
        )

        telemetry["success"] = True
        telemetry["manifest"] = _manifest_payload(manifest_summary)
        telemetry["checkpoint_snapshots"] = checkpoint_snapshots
        telemetry["train_metrics"] = {
            "global_step": int(train_result.global_step),
            **{key: float(value) for key, value in train_result.metrics.items()},
        }
        telemetry["eval_metrics"] = {
            key: float(value) for key, value in eval_result.metrics.items()
        }
        telemetry["iris_benchmark"] = _iris_benchmark_payload(iris_summary)
        telemetry["artifacts"].update(
            {
                "best_checkpoint": str(train_result.best_checkpoint),
                "latest_checkpoint": (
                    str(train_result.latest_checkpoint) if train_result.latest_checkpoint else None
                ),
            }
        )
        _write_summary_markdown(summary_path, telemetry)
        return telemetry
    except Exception as exc:
        telemetry["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        timings_seconds["total"] = time.perf_counter() - total_start
        write_json(telemetry_path, telemetry)


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
