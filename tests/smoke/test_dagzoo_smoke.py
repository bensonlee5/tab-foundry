from __future__ import annotations

import json
from pathlib import Path
import subprocess
from typing import Any

from omegaconf import OmegaConf

import tab_foundry.smoke as smoke_module
from tab_foundry.data.manifest import ManifestSummary
from tab_foundry.types import EvalResult, TrainResult


def _base_cfg() -> Any:
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
                "eval_every": 1,
                "checkpoint_every": None,
                "val_batches": 1,
            },
            "schedule": {"stages": [{"name": "stage1", "steps": 10, "lr_max": 1.0e-3}]},
            "optimizer": {
                "name": "adamw",
                "weight_decay": 0.0,
                "betas": [0.9, 0.95],
                "require_requested": False,
                "muon_per_parameter_lr": False,
                "muon_lr_scale_base": 0.2,
                "muon_partition_non2d": True,
                "min_lr": 1.0e-4,
            },
            "logging": {
                "use_wandb": False,
                "project": "test",
                "run_name": "test",
                "history_jsonl_path": None,
            },
            "eval": {"checkpoint": None, "split": "val", "max_batches": 128},
        }
    )


def test_build_parser_defaults_match_indicative_profile() -> None:
    args = smoke_module.build_parser().parse_args([])
    assert args.num_datasets == 128
    assert args.rows == 1024
    assert args.train_steps == 250
    assert args.checkpoint_every == 25
    assert args.device == "cpu"


def test_plot_loss_curve_writes_png(tmp_path: Path) -> None:
    history_path = tmp_path / "train_history.jsonl"
    history_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "step": 1,
                        "stage": "stage1",
                        "train_loss": 1.2,
                        "train_acc": 0.4,
                        "lr": 8.0e-4,
                        "elapsed_seconds": 0.1,
                        "val_loss": 1.0,
                        "val_acc": 0.5,
                    }
                ),
                json.dumps(
                    {
                        "step": 2,
                        "stage": "stage1",
                        "train_loss": 0.9,
                        "train_acc": 0.6,
                        "lr": 7.9e-4,
                        "elapsed_seconds": 0.2,
                        "val_loss": 0.8,
                        "val_acc": 0.7,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    out_path = smoke_module.plot_loss_curve(history_path, tmp_path / "loss_curve.png")

    assert out_path.exists()
    assert out_path.stat().st_size > 0


def test_run_dagzoo_smoke_writes_expected_telemetry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    dagzoo_root = tmp_path / "dagzoo"
    (dagzoo_root / "configs").mkdir(parents=True)
    (dagzoo_root / "configs" / "default.yaml").write_text("seed: 1\n", encoding="utf-8")
    out_root = tmp_path / "run"

    captured: dict[str, Any] = {}

    def _fake_run(cmd: list[str], *, cwd: Path, check: bool) -> subprocess.CompletedProcess[str]:
        captured["dagzoo_cmd"] = cmd
        captured["dagzoo_cwd"] = cwd
        captured["dagzoo_check"] = check
        return subprocess.CompletedProcess(cmd, 0)

    def _fake_compose_config(_overrides: list[str]) -> Any:
        return _base_cfg()

    def _fake_build_manifest(**_kwargs: Any) -> ManifestSummary:
        return ManifestSummary(
            out_path=out_root / "manifest.parquet",
            filter_policy="include_all",
            discovered_records=128,
            excluded_records=0,
            total_records=128,
            train_records=80,
            val_records=24,
            test_records=24,
            warnings=[],
        )

    def _fake_train(cfg: Any) -> TrainResult:
        captured["train_cfg"] = cfg
        history_path = Path(str(cfg.logging.history_jsonl_path))
        history_path.parent.mkdir(parents=True, exist_ok=True)
        history_path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "step": 25,
                            "stage": "stage1",
                            "train_loss": 1.0,
                            "train_acc": 0.5,
                            "lr": 8.0e-4,
                            "elapsed_seconds": 0.1,
                            "val_loss": 0.9,
                            "val_acc": 0.6,
                        }
                    ),
                    json.dumps(
                        {
                            "step": 250,
                            "stage": "stage1",
                            "train_loss": 0.8,
                            "train_acc": 0.7,
                            "lr": 7.5e-4,
                            "elapsed_seconds": 1.5,
                            "val_loss": 0.7,
                            "val_acc": 0.8,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        output_dir = Path(str(cfg.runtime.output_dir))
        checkpoint_dir = output_dir / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        (checkpoint_dir / "step_000025.pt").write_bytes(b"step25")
        (checkpoint_dir / "step_000250.pt").write_bytes(b"step250")
        best = checkpoint_dir / "best.pt"
        latest = checkpoint_dir / "latest_stage1.pt"
        best.write_bytes(b"best")
        latest.write_bytes(b"latest")
        return TrainResult(
            output_dir=output_dir,
            best_checkpoint=best,
            latest_checkpoint=latest,
            global_step=250,
            metrics={"best_val_loss": 0.75},
        )

    def _fake_eval(cfg: Any) -> EvalResult:
        captured["eval_cfg"] = cfg
        return EvalResult(checkpoint=Path(str(cfg.eval.checkpoint)), metrics={"loss": 0.7, "acc": 0.8})

    monkeypatch.setattr(smoke_module.subprocess, "run", _fake_run)
    monkeypatch.setattr(smoke_module, "compose_config", _fake_compose_config)
    monkeypatch.setattr(smoke_module, "build_manifest", _fake_build_manifest)
    monkeypatch.setattr(smoke_module, "train", _fake_train)
    monkeypatch.setattr(smoke_module, "evaluate_checkpoint", _fake_eval)

    telemetry = smoke_module.run_dagzoo_smoke(
        smoke_module.SmokeConfig(
            dagzoo_root=dagzoo_root,
            out_root=out_root,
        )
    )

    assert telemetry["success"] is True
    assert "--rows" in captured["dagzoo_cmd"]
    assert "1024" in captured["dagzoo_cmd"]
    assert "--num-datasets" in captured["dagzoo_cmd"]
    assert "128" in captured["dagzoo_cmd"]
    assert captured["dagzoo_cwd"] == dagzoo_root
    assert captured["dagzoo_check"] is True

    train_stage = captured["train_cfg"].schedule.stages[0]
    assert int(train_stage["steps"]) == 250
    assert str(captured["train_cfg"].runtime.device) == "cpu"
    assert int(captured["train_cfg"].runtime.checkpoint_every) == 25
    assert str(captured["eval_cfg"].eval.split) == "test"

    telemetry_path = out_root / "telemetry.json"
    assert telemetry_path.exists()
    payload = json.loads(telemetry_path.read_text(encoding="utf-8"))
    assert payload["artifacts"]["train_history_jsonl"].endswith("train_history.jsonl")
    assert payload["artifacts"]["loss_curve_png"].endswith("loss_curve.png")
    assert payload["manifest"]["train_records"] == 80
    assert payload["checkpoint_snapshots"][0]["step"] == 25
    assert payload["checkpoint_snapshots"][-1]["step"] == 250
    assert payload["train_metrics"]["global_step"] == 250
    assert payload["eval_metrics"]["acc"] == 0.8
