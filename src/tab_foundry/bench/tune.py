"""Internal tuning runner for benchmark-oriented tab-foundry training."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from itertools import product
import json
import math
from pathlib import Path
from typing import Any, Sequence, cast

from omegaconf import OmegaConf

from tab_foundry.config import compose_config
from tab_foundry.training.schedule import build_stage_configs, warmup_steps_for_stage
from tab_foundry.training.trainer import train


DEFAULT_LR_MAX_VALUES: tuple[float, ...] = (4.0e-4, 8.0e-4, 1.2e-3)
DEFAULT_WARMUP_RATIOS: tuple[float, ...] = (0.0, 0.05, 0.1)
DEFAULT_GRAD_CLIP_VALUES: tuple[float, ...] = (0.5, 1.0)


@dataclass(slots=True)
class TuneConfig:
    """Input configuration for the internal benchmark-profile sweep."""

    manifest_path: Path
    out_root: Path
    device: str = "auto"
    seed: int = 1
    experiment: str = "cls_benchmark_linear"
    lr_max_values: tuple[float, ...] = DEFAULT_LR_MAX_VALUES
    warmup_ratios: tuple[float, ...] = DEFAULT_WARMUP_RATIOS
    grad_clip_values: tuple[float, ...] = DEFAULT_GRAD_CLIP_VALUES


def _default_out_root() -> Path:
    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return Path("/tmp") / f"tab_foundry_tune_{stamp}"


def _parse_float_list(value: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in str(value).split(",") if item.strip())
    if not values:
        raise ValueError("expected at least one numeric value")
    return values


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "rank",
        "status",
        "trial_id",
        "lr_max",
        "warmup_ratio",
        "grad_clip",
        "best_val_loss",
        "final_val_loss",
        "post_warmup_train_loss_var",
        "mean_grad_norm",
        "max_grad_norm",
        "final_grad_norm",
        "train_elapsed_seconds",
        "wall_elapsed_seconds",
        "run_dir",
        "error",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _load_history(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            payload = json.loads(stripped)
            if not isinstance(payload, dict):
                raise RuntimeError(f"history record must be an object: path={path}")
            records.append(payload)
    if not records:
        raise RuntimeError(f"history file contains no records: {path}")
    return records


def _post_warmup_variance(history: list[dict[str, Any]], *, warmup_steps: int) -> float:
    losses = [
        float(record["train_loss"])
        for record in history
        if int(record["step"]) > warmup_steps and math.isfinite(float(record["train_loss"]))
    ]
    if len(losses) < 2:
        return float("inf")
    mean = sum(losses) / float(len(losses))
    return sum((loss - mean) ** 2 for loss in losses) / float(len(losses))


def _finite_history_values(history: list[dict[str, Any]], key: str) -> list[float]:
    values: list[float] = []
    for record in history:
        raw = record.get(key)
        if raw is None:
            continue
        value = float(raw)
        if math.isfinite(value):
            values.append(value)
    return values


def _slug_float(value: float) -> str:
    return f"{value:.2g}".replace(".", "p").replace("-", "m").replace("+", "")


def _trial_dir_name(trial_id: int, *, lr_max: float, warmup_ratio: float, grad_clip: float) -> str:
    return (
        f"trial_{trial_id:02d}_"
        f"lr{_slug_float(lr_max)}_"
        f"warm{_slug_float(warmup_ratio)}_"
        f"clip{_slug_float(grad_clip)}"
    )


def _build_trial_config(
    config: TuneConfig,
    *,
    trial_root: Path,
    lr_max: float,
    warmup_ratio: float,
    grad_clip: float,
) -> Any:
    cfg = compose_config([f"experiment={config.experiment}", "logging.use_wandb=false"])
    cfg.data.manifest_path = str(config.manifest_path.expanduser().resolve())
    cfg.runtime.output_dir = str((trial_root / "train_outputs").resolve())
    cfg.runtime.device = str(config.device)
    cfg.runtime.seed = int(config.seed)
    cfg.runtime.grad_clip = float(grad_clip)
    cfg.logging.run_name = trial_root.name
    cfg.logging.history_jsonl_path = str((trial_root / "train_outputs" / "train_history.jsonl").resolve())
    cfg.schedule.stages[0].lr_max = float(lr_max)
    cfg.schedule.stages[0].lr_schedule = "linear"
    cfg.schedule.stages[0].warmup_ratio = float(warmup_ratio)
    return cfg


def _summarize_trial(
    *,
    trial_id: int,
    trial_root: Path,
    lr_max: float,
    warmup_ratio: float,
    grad_clip: float,
    cfg: Any,
    train_metrics: dict[str, float],
) -> dict[str, Any]:
    history_path = trial_root / "train_outputs" / "train_history.jsonl"
    history = _load_history(history_path)
    raw_stage_payload = OmegaConf.to_container(cfg.schedule.stages, resolve=True)
    if raw_stage_payload is None:
        raw_stage_payload = []
    stage_configs = build_stage_configs(cast(list[dict[str, object]], raw_stage_payload))
    warmup_steps = warmup_steps_for_stage(stage_configs[0]) if stage_configs else 0
    val_losses = _finite_history_values(history, "val_loss")
    if not val_losses:
        raise RuntimeError(f"trial produced no finite validation losses: {trial_root}")
    grad_norms = _finite_history_values(history, "grad_norm")
    best_checkpoint = trial_root / "train_outputs" / "checkpoints" / "best.pt"

    return {
        "status": "ok",
        "trial_id": int(trial_id),
        "lr_max": float(lr_max),
        "warmup_ratio": float(warmup_ratio),
        "grad_clip": float(grad_clip),
        "best_val_loss": float(min(val_losses)),
        "final_val_loss": float(val_losses[-1]),
        "post_warmup_train_loss_var": float(_post_warmup_variance(history, warmup_steps=warmup_steps)),
        "mean_grad_norm": float(sum(grad_norms) / float(len(grad_norms))) if grad_norms else 0.0,
        "max_grad_norm": float(max(grad_norms)) if grad_norms else 0.0,
        "final_grad_norm": float(grad_norms[-1]) if grad_norms else 0.0,
        "train_elapsed_seconds": float(train_metrics.get("train_elapsed_seconds", 0.0)),
        "wall_elapsed_seconds": float(train_metrics.get("wall_elapsed_seconds", 0.0)),
        "best_val_step": float(train_metrics.get("best_val_step", 0.0)),
        "run_dir": str(trial_root.resolve()),
        "best_checkpoint": str(best_checkpoint.resolve()) if best_checkpoint.exists() else None,
        "error": None,
    }


def _trial_sort_key(trial: dict[str, Any]) -> tuple[float, float, float, float]:
    if trial.get("status") != "ok":
        return (float("inf"), float("inf"), float("inf"), float("inf"))
    return (
        float(trial["best_val_loss"]),
        float(trial["final_val_loss"]),
        float(trial["post_warmup_train_loss_var"]),
        float(trial["train_elapsed_seconds"]),
    )


def run_tuning(config: TuneConfig) -> dict[str, Any]:
    """Run the internal benchmark-profile sweep on a fixed manifest."""

    manifest_path = config.manifest_path.expanduser().resolve()
    if not manifest_path.exists():
        raise RuntimeError(f"manifest does not exist: {manifest_path}")
    out_root = config.out_root.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    trial_summaries: list[dict[str, Any]] = []
    grid = list(product(config.lr_max_values, config.warmup_ratios, config.grad_clip_values))
    for trial_id, (lr_max, warmup_ratio, grad_clip) in enumerate(grid, start=1):
        trial_root = out_root / _trial_dir_name(
            trial_id,
            lr_max=float(lr_max),
            warmup_ratio=float(warmup_ratio),
            grad_clip=float(grad_clip),
        )
        try:
            cfg = _build_trial_config(
                config,
                trial_root=trial_root,
                lr_max=float(lr_max),
                warmup_ratio=float(warmup_ratio),
                grad_clip=float(grad_clip),
            )
            result = train(cfg)
            summary = _summarize_trial(
                trial_id=trial_id,
                trial_root=trial_root,
                lr_max=float(lr_max),
                warmup_ratio=float(warmup_ratio),
                grad_clip=float(grad_clip),
                cfg=cfg,
                train_metrics=result.metrics,
            )
        except Exception as exc:
            summary = {
                "status": "error",
                "trial_id": int(trial_id),
                "lr_max": float(lr_max),
                "warmup_ratio": float(warmup_ratio),
                "grad_clip": float(grad_clip),
                "best_val_loss": float("inf"),
                "final_val_loss": float("inf"),
                "post_warmup_train_loss_var": float("inf"),
                "mean_grad_norm": 0.0,
                "max_grad_norm": 0.0,
                "final_grad_norm": 0.0,
                "train_elapsed_seconds": 0.0,
                "wall_elapsed_seconds": 0.0,
                "best_val_step": 0.0,
                "run_dir": str(trial_root.resolve()),
                "best_checkpoint": None,
                "error": str(exc),
            }
        trial_summaries.append(summary)

    ranked_trials = sorted(trial_summaries, key=_trial_sort_key)
    for rank, trial in enumerate(ranked_trials, start=1):
        trial["rank"] = int(rank)

    best_trial = next((trial for trial in ranked_trials if trial.get("status") == "ok"), None)
    summary = {
        "manifest_path": str(manifest_path),
        "out_root": str(out_root),
        "experiment": str(config.experiment),
        "device": str(config.device),
        "seed": int(config.seed),
        "ranking_policy": (
            "lowest best_val_loss, then lowest final_val_loss, then lowest post_warmup_train_loss_var"
        ),
        "grid": {
            "lr_max_values": [float(value) for value in config.lr_max_values],
            "warmup_ratios": [float(value) for value in config.warmup_ratios],
            "grad_clip_values": [float(value) for value in config.grad_clip_values],
        },
        "trial_count": int(len(ranked_trials)),
        "best_trial": best_trial,
        "trials": ranked_trials,
    }
    _write_json(out_root / "sweep_summary.json", summary)
    _write_csv(out_root / "sweep_results.csv", ranked_trials)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Tune tab-foundry against internal validation only")
    parser.add_argument("--manifest-path", required=True, help="Fixed manifest path used for every trial")
    parser.add_argument("--out-root", default=None, help="Output root for sweep artifacts")
    parser.add_argument(
        "--device",
        default="auto",
        choices=("cpu", "cuda", "mps", "auto"),
        help="Training device override",
    )
    parser.add_argument("--seed", type=int, default=1, help="Base random seed used for every trial")
    parser.add_argument(
        "--lr-max-values",
        default="4e-4,8e-4,1.2e-3",
        help="Comma-separated lr_max grid",
    )
    parser.add_argument(
        "--warmup-ratios",
        default="0.0,0.05,0.1",
        help="Comma-separated warmup_ratio grid",
    )
    parser.add_argument(
        "--grad-clip-values",
        default="0.5,1.0",
        help="Comma-separated grad_clip grid",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = run_tuning(
        TuneConfig(
            manifest_path=Path(str(args.manifest_path)),
            out_root=_default_out_root() if args.out_root is None else Path(str(args.out_root)),
            device=str(args.device),
            seed=int(args.seed),
            lr_max_values=_parse_float_list(str(args.lr_max_values)),
            warmup_ratios=_parse_float_list(str(args.warmup_ratios)),
            grad_clip_values=_parse_float_list(str(args.grad_clip_values)),
        )
    )
    print("tab-foundry tuning complete:")
    print(f"  trial_count={summary['trial_count']}")
    if summary["best_trial"] is not None:
        print(f"  best_trial={summary['best_trial']}")
    print(f"  artifacts={{'summary': '{Path(summary['out_root']) / 'sweep_summary.json'}', 'csv': '{Path(summary['out_root']) / 'sweep_results.csv'}'}}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
