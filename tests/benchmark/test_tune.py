from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf

import tab_foundry.bench.tune as tune_module
from tab_foundry.types import TrainResult


def _base_cfg() -> object:
    return OmegaConf.create(
        {
            "task": "classification",
            "model": {},
            "data": {"manifest_path": "unused.parquet", "train_row_cap": None, "test_row_cap": None},
            "runtime": {
                "seed": 1,
                "num_workers": 0,
                "output_dir": "unused",
                "device": "cpu",
                "mixed_precision": "no",
                "grad_clip": 1.0,
                "grad_accum_steps": 1,
                "eval_every": 25,
                "checkpoint_every": 25,
                "val_batches": 16,
                "max_steps": 400,
                "target_train_seconds": 330,
            },
            "schedule": {
                "stages": [
                    {
                        "name": "stage1",
                        "steps": 400,
                        "lr_max": 8.0e-4,
                        "lr_schedule": "linear",
                        "warmup_ratio": 0.05,
                    }
                ]
            },
            "optimizer": {
                "name": "adamw",
                "weight_decay": 0.01,
                "betas": [0.9, 0.95],
                "require_requested": False,
                "muon_per_parameter_lr": False,
                "muon_lr_scale_base": 0.2,
                "muon_partition_non2d": True,
                "min_lr": 1.0e-6,
            },
            "logging": {
                "use_wandb": False,
                "project": "test",
                "run_name": "trial",
                "history_jsonl_path": None,
            },
        }
    )


def test_run_tuning_ranks_trials_from_internal_metrics(monkeypatch, tmp_path: Path) -> None:
    manifest_path = tmp_path / "manifest.parquet"
    manifest_path.write_text("manifest", encoding="utf-8")
    out_root = tmp_path / "sweep"

    monkeypatch.setattr(tune_module, "compose_config", lambda _overrides: _base_cfg())

    def _fake_train(cfg: object) -> TrainResult:
        history_path = Path(str(cfg.logging.history_jsonl_path))
        history_path.parent.mkdir(parents=True, exist_ok=True)
        lr_max = float(cfg.schedule.stages[0].lr_max)
        base_val = 0.4 if lr_max < 5.0e-4 else 0.6
        history_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "step": 1,
                            "stage": "stage1",
                            "train_loss": 1.0,
                            "lr": lr_max,
                            "grad_norm": 0.8,
                            "elapsed_seconds": 0.1,
                            "train_elapsed_seconds": 0.1,
                            "val_loss": base_val + 0.2,
                        }
                    ),
                    json.dumps(
                        {
                            "step": 25,
                            "stage": "stage1",
                            "train_loss": 0.8 if lr_max < 5.0e-4 else 1.1,
                            "lr": lr_max / 2.0,
                            "grad_norm": 0.6,
                            "elapsed_seconds": 0.2,
                            "train_elapsed_seconds": 0.2,
                            "val_loss": base_val,
                        }
                    ),
                    json.dumps(
                        {
                            "step": 50,
                            "stage": "stage1",
                            "train_loss": 0.7 if lr_max < 5.0e-4 else 1.0,
                            "lr": lr_max / 4.0,
                            "grad_norm": 0.5,
                            "elapsed_seconds": 0.3,
                            "train_elapsed_seconds": 0.3,
                            "val_loss": base_val + 0.05,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        checkpoint_dir = Path(str(cfg.runtime.output_dir)) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "best.pt").write_bytes(b"best")
        return TrainResult(
            output_dir=Path(str(cfg.runtime.output_dir)),
            best_checkpoint=checkpoint_dir / "best.pt",
            latest_checkpoint=checkpoint_dir / "best.pt",
            global_step=50,
            metrics={
                "best_val_loss": base_val,
                "best_val_step": 25.0,
                "final_val_loss": base_val + 0.05,
                "final_grad_norm": 0.5,
                "mean_grad_norm": 0.633,
                "max_grad_norm": 0.8,
                "train_elapsed_seconds": 0.3,
                "wall_elapsed_seconds": 0.35,
            },
        )

    monkeypatch.setattr(tune_module, "train", _fake_train)

    summary = tune_module.run_tuning(
        tune_module.TuneConfig(
            manifest_path=manifest_path,
            out_root=out_root,
            lr_max_values=(4.0e-4, 8.0e-4),
            warmup_ratios=(0.05,),
            grad_clip_values=(1.0,),
        )
    )

    assert summary["trial_count"] == 2
    assert summary["best_trial"] is not None
    assert summary["best_trial"]["lr_max"] == 4.0e-4
    assert (out_root / "sweep_summary.json").exists()
    assert (out_root / "sweep_results.csv").exists()
