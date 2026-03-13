"""dagzoo-backed smoke harness for tab-foundry."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import subprocess
import time
from typing import Any, Sequence

from tab_foundry.bench.artifacts import (
    checkpoint_snapshots_from_history,
    ensure_finite_metrics,
    plot_loss_curve,
    write_json,
)
from tab_foundry.bench.smoke_common import (
    build_cls_smoke_eval_config,
    build_cls_smoke_train_config,
    build_manifest_payload,
)
from tab_foundry.data.manifest import build_manifest
from tab_foundry.training.evaluate import evaluate_checkpoint
from tab_foundry.training.trainer import train


DEFAULT_NUM_DATASETS = 128
DEFAULT_ROWS = 1024
DEFAULT_SEED = 1
DEFAULT_DEVICE = "cpu"
DEFAULT_TRAIN_STEPS = 250
DEFAULT_TRAIN_RATIO = 0.6
DEFAULT_VAL_RATIO = 0.2
DEFAULT_FILTER_POLICY = "include_all"
DEFAULT_STAGE_LR_MAX = 8.0e-4
DEFAULT_CHECKPOINT_EVERY = 25


@dataclass(slots=True)
class SmokeConfig:
    """Input configuration for the dagzoo smoke harness."""

    dagzoo_root: Path
    out_root: Path
    num_datasets: int = DEFAULT_NUM_DATASETS
    rows: int = DEFAULT_ROWS
    device: str = DEFAULT_DEVICE
    seed: int = DEFAULT_SEED
    train_steps: int = DEFAULT_TRAIN_STEPS
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
    train_ratio: float = DEFAULT_TRAIN_RATIO
    val_ratio: float = DEFAULT_VAL_RATIO
    filter_policy: str = DEFAULT_FILTER_POLICY


def _default_out_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return Path("/tmp") / f"tab_foundry_dagzoo_smoke_{stamp}"


def _dagzoo_generate_command(config: SmokeConfig, *, generated_dir: Path) -> list[str]:
    return [
        "uv",
        "run",
        "dagzoo",
        "generate",
        "--config",
        "configs/default.yaml",
        "--num-datasets",
        str(config.num_datasets),
        "--rows",
        str(config.rows),
        "--seed",
        str(config.seed),
        "--device",
        config.device,
        "--out",
        str(generated_dir),
    ]


def run_dagzoo_smoke(config: SmokeConfig) -> dict[str, Any]:
    """Execute the end-to-end dagzoo-backed smoke harness."""

    dagzoo_root = config.dagzoo_root.expanduser().resolve()
    out_root = config.out_root.expanduser().resolve()
    if config.rows <= 0:
        raise ValueError(f"rows must be > 0, got {config.rows}")
    if config.num_datasets <= 0:
        raise ValueError(f"num_datasets must be > 0, got {config.num_datasets}")
    if config.train_steps <= 0:
        raise ValueError(f"train_steps must be > 0, got {config.train_steps}")
    if config.checkpoint_every <= 0:
        raise ValueError(f"checkpoint_every must be > 0, got {config.checkpoint_every}")
    if not dagzoo_root.exists():
        raise RuntimeError(f"dagzoo root does not exist: {dagzoo_root}")
    if not (dagzoo_root / "configs" / "default.yaml").exists():
        raise RuntimeError(f"dagzoo config missing at {dagzoo_root / 'configs' / 'default.yaml'}")

    out_root.mkdir(parents=True, exist_ok=True)
    generated_dir = out_root / "generated"
    manifest_path = out_root / "manifest.parquet"
    train_output_dir = out_root / "train_outputs"
    history_path = train_output_dir / "train_history.jsonl"
    loss_curve_path = train_output_dir / "loss_curve.png"
    telemetry_path = out_root / "telemetry.json"

    timings_seconds: dict[str, float] = {}
    telemetry: dict[str, Any] = {
        "success": False,
        "config": {
            "dagzoo_root": str(dagzoo_root),
            "num_datasets": int(config.num_datasets),
            "rows": int(config.rows),
            "seed": int(config.seed),
            "device": str(config.device),
            "train_steps": int(config.train_steps),
            "checkpoint_every": int(config.checkpoint_every),
        },
        "artifacts": {
            "generated_dir": str(generated_dir),
            "manifest_path": str(manifest_path),
            "train_output_dir": str(train_output_dir),
            "train_history_jsonl": str(history_path),
            "loss_curve_png": str(loss_curve_path),
        },
        "timings_seconds": timings_seconds,
    }

    total_start = time.perf_counter()
    try:
        stage_start = time.perf_counter()
        subprocess.run(
            _dagzoo_generate_command(config, generated_dir=generated_dir),
            cwd=dagzoo_root,
            check=True,
        )
        timings_seconds["dagzoo_generate"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        manifest_summary = build_manifest(
            data_roots=[generated_dir],
            out_path=manifest_path,
            train_ratio=config.train_ratio,
            val_ratio=config.val_ratio,
            filter_policy=config.filter_policy,
        )
        timings_seconds["build_manifest"] = time.perf_counter() - stage_start

        stage_start = time.perf_counter()
        train_result = train(
            build_cls_smoke_train_config(
                manifest_path=manifest_path,
                output_dir=train_output_dir,
                history_path=history_path,
                schedule_stages=[
                    {
                        "name": "stage1",
                        "steps": int(config.train_steps),
                        "lr_max": DEFAULT_STAGE_LR_MAX,
                    }
                ],
                checkpoint_every=config.checkpoint_every,
                device=config.device,
                clear_row_caps=False,
            )
        )
        timings_seconds["train"] = time.perf_counter() - stage_start
        if train_result.best_checkpoint is None:
            raise RuntimeError("training did not produce a best checkpoint")

        stage_start = time.perf_counter()
        eval_result = evaluate_checkpoint(
            build_cls_smoke_eval_config(
                manifest_path=manifest_path,
                checkpoint_path=train_result.best_checkpoint,
                device=config.device,
                clear_row_caps=False,
            )
        )
        timings_seconds["eval"] = time.perf_counter() - stage_start
        ensure_finite_metrics(train_result.metrics, context="train")
        ensure_finite_metrics(eval_result.metrics, context="eval")

        plot_loss_curve(history_path, loss_curve_path)
        checkpoint_snapshots = checkpoint_snapshots_from_history(
            history_path,
            train_output_dir / "checkpoints",
        )

        telemetry["success"] = True
        telemetry["manifest"] = build_manifest_payload(manifest_summary)
        telemetry["checkpoint_snapshots"] = checkpoint_snapshots
        telemetry["train_metrics"] = {
            "global_step": int(train_result.global_step),
            **{key: float(value) for key, value in train_result.metrics.items()},
        }
        telemetry["eval_metrics"] = {key: float(value) for key, value in eval_result.metrics.items()}
        telemetry["artifacts"].update(
            {
                "best_checkpoint": str(train_result.best_checkpoint),
                "latest_checkpoint": (
                    str(train_result.latest_checkpoint) if train_result.latest_checkpoint else None
                ),
            }
        )
        return telemetry
    except Exception as exc:
        telemetry["error"] = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        timings_seconds["total"] = time.perf_counter() - total_start
        write_json(telemetry_path, telemetry)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the dagzoo-backed tab-foundry smoke harness")
    parser.add_argument("--dagzoo-root", default="~/dev/dagzoo", help="Local dagzoo checkout root")
    parser.add_argument("--out-root", default=None, help="Output directory root")
    parser.add_argument(
        "--num-datasets",
        type=int,
        default=DEFAULT_NUM_DATASETS,
        help="Number of dagzoo datasets to generate",
    )
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS, help="Rows per generated dataset")
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=("cpu", "cuda", "mps", "auto"),
        help="Generation and training device",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Shared run seed")
    parser.add_argument(
        "--train-steps",
        type=int,
        default=DEFAULT_TRAIN_STEPS,
        help="Training steps for the smoke harness",
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
    telemetry = run_dagzoo_smoke(
        SmokeConfig(
            dagzoo_root=Path(str(args.dagzoo_root)),
            out_root=out_root,
            num_datasets=int(args.num_datasets),
            rows=int(args.rows),
            device=str(args.device),
            seed=int(args.seed),
            train_steps=int(args.train_steps),
            checkpoint_every=int(args.checkpoint_every),
        )
    )
    print("dagzoo smoke complete:")
    print(f"  out_root={out_root.resolve()}")
    print(f"  best_checkpoint={telemetry['artifacts']['best_checkpoint']}")
    print(f"  eval_metrics={telemetry['eval_metrics']}")
    print(f"  timings_seconds={telemetry['timings_seconds']}")
    return 0
