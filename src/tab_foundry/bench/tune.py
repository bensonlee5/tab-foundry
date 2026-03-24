"""Internal tuning runner for benchmark-oriented tab-foundry training."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any, Mapping, Sequence

from tab_foundry.bench.artifacts import write_json
from tab_foundry.config import compose_config
from tab_foundry.bench.tune_metrics import (
    finite_history_values,
    load_history_summary,
    post_warmup_variance,
)
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
    cfg.model.feature_group_size = 1
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


def _optional_metric(metrics: Mapping[str, float | None], key: str) -> float | None:
    value = metrics.get(key)
    return None if value is None else float(value)


def _summarize_trial(
    *,
    trial_id: int,
    trial_root: Path,
    lr_max: float,
    warmup_ratio: float,
    grad_clip: float,
    cfg: Any,
    train_metrics: Mapping[str, float | None],
) -> dict[str, Any]:
    history_path = trial_root / "train_outputs" / "train_history.jsonl"
    history, history_summary = load_history_summary(history_path, raw_cfg=cfg)
    val_losses = finite_history_values(history, "val_loss")
    if not val_losses:
        raise RuntimeError(f"trial produced no finite validation losses: {trial_root}")
    best_checkpoint = trial_root / "train_outputs" / "checkpoints" / "best.pt"
    train_elapsed_seconds = _optional_metric(train_metrics, "train_elapsed_seconds")
    if train_elapsed_seconds is None:
        train_elapsed_seconds = _optional_metric(history_summary, "train_elapsed_seconds")
    wall_elapsed_seconds = _optional_metric(train_metrics, "wall_elapsed_seconds")
    if wall_elapsed_seconds is None:
        wall_elapsed_seconds = _optional_metric(history_summary, "wall_elapsed_seconds")
    best_val_step = _optional_metric(train_metrics, "best_val_step")

    return {
        "status": "ok",
        "trial_id": int(trial_id),
        "lr_max": float(lr_max),
        "warmup_ratio": float(warmup_ratio),
        "grad_clip": float(grad_clip),
        "best_val_loss": float(min(val_losses)),
        "final_val_loss": float(val_losses[-1]),
        "post_warmup_train_loss_var": float(post_warmup_variance(history, raw_cfg=cfg)),
        "mean_grad_norm": history_summary["mean_grad_norm"],
        "max_grad_norm": history_summary["max_grad_norm"],
        "final_grad_norm": history_summary["final_grad_norm"],
        "train_elapsed_seconds": 0.0 if train_elapsed_seconds is None else float(train_elapsed_seconds),
        "wall_elapsed_seconds": 0.0 if wall_elapsed_seconds is None else float(wall_elapsed_seconds),
        "best_val_step": 0.0 if best_val_step is None else float(best_val_step),
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
                "mean_grad_norm": None,
                "max_grad_norm": None,
                "final_grad_norm": None,
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
    write_json(out_root / "sweep_summary.json", summary)
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
