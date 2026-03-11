from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import h5py
import numpy as np

import tab_foundry.nanoprior_train as nanoprior_module
from tab_foundry.types import EvalResult, TrainResult


def _write_prior_dump(path: Path) -> None:
    x = np.arange(40 * 6 * 4, dtype=np.float32).reshape(40, 6, 4)
    y = np.tile(np.asarray([[0, 1, 0, 1, 0, 1]], dtype=np.float32), (40, 1))
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=x)
        handle.create_dataset("y", data=y)
        handle.create_dataset("num_features", data=np.full((40,), 4, dtype=np.int32))
        handle.create_dataset("num_datapoints", data=np.full((40,), 6, dtype=np.int32))
        handle.create_dataset("single_eval_pos", data=np.full((40,), 3, dtype=np.int32))
        handle.create_dataset("max_num_classes", data=np.asarray([2], dtype=np.int64))


def test_run_nanoprior_training_writes_expected_telemetry(monkeypatch, tmp_path: Path) -> None:
    prior_dump = tmp_path / "prior.h5"
    _write_prior_dump(prior_dump)
    out_root = tmp_path / "run"

    captured: dict[str, Any] = {}

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
                            "train_loss": 0.9,
                            "lr": 4.0e-3,
                            "elapsed_seconds": 10.0,
                            "train_elapsed_seconds": 6.0,
                            "val_loss": 0.8,
                            "val_acc": 0.7,
                        }
                    ),
                    json.dumps(
                        {
                            "step": 50,
                            "stage": "stage1",
                            "train_loss": 0.8,
                            "lr": 4.0e-3,
                            "elapsed_seconds": 15.0,
                            "train_elapsed_seconds": 9.0,
                            "val_loss": 0.7,
                            "val_acc": 0.8,
                        }
                    ),
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        checkpoint_dir = Path(str(cfg.runtime.output_dir)) / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best = checkpoint_dir / "best.pt"
        latest = checkpoint_dir / "latest_stage1.pt"
        (checkpoint_dir / "step_000025.pt").write_bytes(b"step25")
        (checkpoint_dir / "step_000050.pt").write_bytes(b"step50")
        best.write_bytes(b"best")
        latest.write_bytes(b"latest")
        return TrainResult(
            output_dir=Path(str(cfg.runtime.output_dir)),
            best_checkpoint=best,
            latest_checkpoint=latest,
            global_step=50,
            metrics={
                "best_val_loss": 0.7,
                "train_elapsed_seconds": 9.0,
                "wall_elapsed_seconds": 15.0,
            },
        )

    def _fake_eval(cfg: Any) -> EvalResult:
        captured["eval_cfg"] = cfg
        return EvalResult(checkpoint=Path(str(cfg.eval.checkpoint)), metrics={"loss": 0.7, "acc": 0.8})

    def _fake_plot_loss_curve(_history_path: Path, out_path: Path, *, title: str) -> Path:
        out_path.write_bytes(title.encode("utf-8"))
        return out_path

    monkeypatch.setattr(nanoprior_module, "train", _fake_train)
    monkeypatch.setattr(nanoprior_module, "evaluate_checkpoint", _fake_eval)
    monkeypatch.setattr(nanoprior_module, "plot_loss_curve", _fake_plot_loss_curve)

    telemetry = nanoprior_module.run_nanoprior_training(
        nanoprior_module.NanoPriorTrainConfig(
            prior_dump=prior_dump,
            out_root=out_root,
        )
    )

    assert telemetry["success"] is True
    assert str(captured["train_cfg"].data.source) == "nanoprior"
    assert str(captured["train_cfg"].optimizer.name) == "schedulefree_adamw"
    assert int(captured["train_cfg"].runtime.max_steps) == 2500
    assert float(captured["train_cfg"].runtime.target_train_seconds) == 330.0
    assert telemetry["prior"]["train_tasks"] == 36
    assert telemetry["prior"]["val_tasks"] == 4
    assert telemetry["checkpoint_snapshots"][0]["elapsed_seconds"] == 6.0
    assert telemetry["checkpoint_snapshots"][-1]["elapsed_seconds"] == 9.0
    assert telemetry["train_metrics"]["global_step"] == 50
    assert telemetry["artifacts"]["best_checkpoint"].endswith("best.pt")
    assert str(captured["eval_cfg"].eval.split) == "val"
    assert (out_root / "telemetry.json").exists()
