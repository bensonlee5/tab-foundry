"""Fast nanoTabPFN-prior pretraining harness for tab-foundry."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import time
from typing import Any, Sequence

from omegaconf import DictConfig, OmegaConf

from tab_foundry.config import compose_config
from tab_foundry.data.nanoprior import inspect_nano_prior_dump
from tab_foundry.nanotabpfn_benchmark import collect_checkpoint_snapshots
from tab_foundry.smoke import plot_loss_curve
from tab_foundry.training.evaluate import evaluate_checkpoint
from tab_foundry.training.trainer import train


DEFAULT_PRIOR_DUMP = Path("~/dev/nanoTabPFN/300k_150x5_2.h5")
DEFAULT_DEVICE = "auto"
DEFAULT_TARGET_TRAIN_SECONDS = 330.0
DEFAULT_MAX_STEPS = 2500
DEFAULT_EVAL_EVERY = 25
DEFAULT_CHECKPOINT_EVERY = 25
DEFAULT_SEED = 0
DEFAULT_VAL_SIZE = 4096


@dataclass(slots=True)
class NanoPriorTrainConfig:
    """Input configuration for the nano-prior fast training harness."""

    prior_dump: Path
    out_root: Path
    device: str = DEFAULT_DEVICE
    target_train_seconds: float = DEFAULT_TARGET_TRAIN_SECONDS
    max_steps: int = DEFAULT_MAX_STEPS
    eval_every: int = DEFAULT_EVAL_EVERY
    checkpoint_every: int = DEFAULT_CHECKPOINT_EVERY
    seed: int = DEFAULT_SEED


def _default_out_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return Path("/tmp") / f"tab_foundry_nanoprior_{stamp}"


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _load_history(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        records = [json.loads(line) for line in handle if line.strip()]
    if not records:
        raise RuntimeError(f"history file contains no records: {path}")
    return records


def _clone_cfg(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def _resolved_val_size(total_tasks: int, requested: int = DEFAULT_VAL_SIZE) -> int:
    if total_tasks < 4:
        raise RuntimeError(f"nano prior dump must contain at least 4 tasks, got {total_tasks}")
    fallback = max(1, total_tasks // 10)
    return min(int(requested), int(fallback if total_tasks <= requested else requested))


def _fallback_checkpoint_snapshots(
    *,
    history_records: list[dict[str, Any]],
    train_result: Any,
) -> list[dict[str, Any]]:
    if not history_records:
        raise RuntimeError("cannot build fallback checkpoint snapshot without training history")
    checkpoint_path = train_result.best_checkpoint or train_result.latest_checkpoint
    if checkpoint_path is None:
        raise RuntimeError("cannot build fallback checkpoint snapshot without a checkpoint path")
    last_record = history_records[-1]
    return [
        {
            "step": int(last_record["step"]),
            "path": str(Path(str(checkpoint_path)).expanduser().resolve()),
            "elapsed_seconds": float(
                last_record.get("train_elapsed_seconds", last_record.get("elapsed_seconds", 0.0))
            ),
        }
    ]


def _build_train_config(
    *,
    prior_dump: Path,
    summary_num_tasks: int,
    output_dir: Path,
    history_path: Path,
    device: str,
    target_train_seconds: float,
    max_steps: int,
    eval_every: int,
    checkpoint_every: int,
    seed: int,
) -> tuple[DictConfig, int, int]:
    val_size = _resolved_val_size(summary_num_tasks)
    train_size = summary_num_tasks - val_size
    cfg = compose_config(
        [
            "experiment=cls_nano_aligned",
            "optimizer=schedulefree_adamw",
            "logging.use_wandb=false",
        ]
    )
    cfg.data.source = "nanoprior"
    cfg.data.prior_dump_path = str(prior_dump)
    cfg.data.prior_train_offset = 0
    cfg.data.prior_train_size = int(train_size)
    cfg.data.prior_val_offset = int(train_size)
    cfg.data.prior_val_size = int(val_size)
    cfg.runtime.output_dir = str(output_dir)
    cfg.runtime.device = str(device)
    cfg.runtime.seed = int(seed)
    cfg.runtime.eval_every = int(eval_every)
    cfg.runtime.checkpoint_every = int(checkpoint_every)
    cfg.runtime.max_steps = int(max_steps)
    cfg.runtime.target_train_seconds = float(target_train_seconds)
    cfg.logging.history_jsonl_path = str(history_path)
    cfg.eval.max_batches = 64
    return cfg, train_size, val_size


def run_nanoprior_training(config: NanoPriorTrainConfig) -> dict[str, Any]:
    """Train tab-foundry on the nanoTabPFN prior dump with a short time budget."""

    prior_dump = config.prior_dump.expanduser().resolve()
    if not prior_dump.exists():
        raise RuntimeError(f"nano prior dump does not exist: {prior_dump}")
    if config.target_train_seconds <= 0:
        raise ValueError("target_train_seconds must be > 0")
    if config.max_steps <= 0:
        raise ValueError("max_steps must be > 0")
    if config.eval_every <= 0:
        raise ValueError("eval_every must be > 0")
    if config.checkpoint_every <= 0:
        raise ValueError("checkpoint_every must be > 0")

    prior_summary = inspect_nano_prior_dump(prior_dump)
    out_root = config.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    train_output_dir = out_root / "train_outputs"
    history_path = train_output_dir / "train_history.jsonl"
    loss_curve_path = train_output_dir / "loss_curve.png"
    telemetry_path = out_root / "telemetry.json"

    timings_seconds: dict[str, float] = {}
    telemetry: dict[str, Any] = {
        "success": False,
        "config": {
            "prior_dump": str(prior_dump),
            "device": str(config.device),
            "target_train_seconds": float(config.target_train_seconds),
            "max_steps": int(config.max_steps),
            "eval_every": int(config.eval_every),
            "checkpoint_every": int(config.checkpoint_every),
            "seed": int(config.seed),
        },
        "artifacts": {
            "train_output_dir": str(train_output_dir),
            "train_history_jsonl": str(history_path),
            "loss_curve_png": str(loss_curve_path),
        },
        "prior": {
            "num_tasks": int(prior_summary.num_tasks),
            "max_rows": int(prior_summary.max_rows),
            "max_features": int(prior_summary.max_features),
            "max_num_classes": int(prior_summary.max_num_classes),
        },
        "timings_seconds": timings_seconds,
    }

    total_start = time.perf_counter()
    try:
        train_cfg, train_size, val_size = _build_train_config(
            prior_dump=prior_dump,
            summary_num_tasks=prior_summary.num_tasks,
            output_dir=train_output_dir,
            history_path=history_path,
            device=config.device,
            target_train_seconds=config.target_train_seconds,
            max_steps=config.max_steps,
            eval_every=config.eval_every,
            checkpoint_every=config.checkpoint_every,
            seed=config.seed,
        )
        telemetry["prior"].update({"train_tasks": int(train_size), "val_tasks": int(val_size)})

        stage_start = time.perf_counter()
        train_result = train(train_cfg)
        timings_seconds["train"] = time.perf_counter() - stage_start
        if train_result.best_checkpoint is None:
            raise RuntimeError("training did not produce a best checkpoint")

        stage_start = time.perf_counter()
        eval_cfg = _clone_cfg(train_cfg)
        eval_cfg.eval.checkpoint = str(train_result.best_checkpoint)
        eval_cfg.eval.split = "val"
        eval_result = evaluate_checkpoint(eval_cfg)
        timings_seconds["eval"] = time.perf_counter() - stage_start

        plot_loss_curve(history_path, loss_curve_path, title="tab-foundry nano-prior loss curve")
        history_records = _load_history(history_path)
        try:
            checkpoint_snapshots = collect_checkpoint_snapshots(out_root)
        except RuntimeError as exc:
            if "no step checkpoints found" not in str(exc):
                raise
            checkpoint_snapshots = _fallback_checkpoint_snapshots(
                history_records=history_records,
                train_result=train_result,
            )
        final_history = history_records[-1]
        train_elapsed_seconds = float(
            final_history.get("train_elapsed_seconds", final_history.get("elapsed_seconds", 0.0))
        )

        telemetry["success"] = True
        telemetry["checkpoint_snapshots"] = checkpoint_snapshots
        telemetry["train_metrics"] = {
            "global_step": int(train_result.global_step),
            **{key: float(value) for key, value in train_result.metrics.items()},
            "final_history_train_elapsed_seconds": train_elapsed_seconds,
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
        _write_json(telemetry_path, telemetry)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train tab-foundry on the nanoTabPFN prior dump")
    parser.add_argument(
        "--prior-dump",
        default=str(DEFAULT_PRIOR_DUMP),
        help="Path to the nanoTabPFN HDF5 prior dump",
    )
    parser.add_argument("--out-root", default=None, help="Output directory root")
    parser.add_argument(
        "--device",
        default=DEFAULT_DEVICE,
        choices=("cpu", "cuda", "mps", "auto"),
        help="Training device",
    )
    parser.add_argument(
        "--target-train-seconds",
        type=float,
        default=DEFAULT_TARGET_TRAIN_SECONDS,
        help="Train-only time budget in seconds",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=DEFAULT_MAX_STEPS,
        help="Maximum optimizer steps",
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=DEFAULT_EVAL_EVERY,
        help="Validation cadence in optimizer steps",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=DEFAULT_CHECKPOINT_EVERY,
        help="Step checkpoint cadence",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Run seed")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    telemetry = run_nanoprior_training(
        NanoPriorTrainConfig(
            prior_dump=Path(str(args.prior_dump)),
            out_root=_default_out_root() if args.out_root is None else Path(str(args.out_root)),
            device=str(args.device),
            target_train_seconds=float(args.target_train_seconds),
            max_steps=int(args.max_steps),
            eval_every=int(args.eval_every),
            checkpoint_every=int(args.checkpoint_every),
            seed=int(args.seed),
        )
    )
    print("nano-prior training complete:")
    print(f"  best_checkpoint={telemetry['artifacts']['best_checkpoint']}")
    print(f"  train_elapsed_seconds={telemetry['train_metrics']['train_elapsed_seconds']}")
    print(f"  eval_metrics={telemetry['eval_metrics']}")
    print(f"  telemetry={Path(str(telemetry['artifacts']['train_output_dir'])).parent / 'telemetry.json'}")
    return 0
