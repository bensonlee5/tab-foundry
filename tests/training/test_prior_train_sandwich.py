from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import pytest

import tab_foundry.training.prior_train as prior_train_module

h5py = pytest.importorskip("h5py")


def _write_prior_dump(
    path: Path,
    *,
    x: np.ndarray,
    y: np.ndarray,
    num_features: np.ndarray,
    num_datapoints: np.ndarray,
    single_eval_pos: np.ndarray,
    max_num_classes: int = 2,
) -> Path:
    with h5py.File(path, "w") as handle:
        handle.create_dataset("X", data=x)
        handle.create_dataset("y", data=y)
        handle.create_dataset("num_features", data=num_features)
        handle.create_dataset("num_datapoints", data=num_datapoints)
        handle.create_dataset("single_eval_pos", data=single_eval_pos)
        handle.create_dataset("max_num_classes", data=np.asarray([max_num_classes], dtype=np.int64))
    return path


def test_train_tabfoundry_sandwich_prior_smoke(tmp_path: Path) -> None:
    path = _write_prior_dump(
        tmp_path / "prior_sandwich.h5",
        x=np.asarray(
            [
                [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
                [[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]],
            ],
            dtype=np.float32,
        ),
        y=np.asarray(
            [
                [0, 1, 0, 1],
                [1, 0, 1, 0],
            ],
            dtype=np.int64,
        ),
        num_features=np.asarray([2, 2], dtype=np.int64),
        num_datapoints=np.asarray([4, 4], dtype=np.int64),
        single_eval_pos=np.asarray([2, 2], dtype=np.int64),
    )
    cfg = OmegaConf.create(
        {
            "task": "classification",
            "model": {
                "arch": "tabfoundry_sandwich",
                "d_icl": 32,
                "input_normalization": "train_zscore_clip",
                "many_class_base": 2,
                "head_hidden_dim": 64,
                "sandwich_latents": 12,
                "sandwich_layers": 2,
                "sandwich_heads": 4,
                "sandwich_ff_expansion": 2,
            },
            "runtime": {
                "seed": 1,
                "output_dir": str(tmp_path / "train_out"),
                "device": "cpu",
                "mixed_precision": "no",
                "grad_clip": 1.0,
                "max_steps": 2,
                "eval_every": 1,
                "checkpoint_every": 1,
            },
            "optimizer": {
                "name": "schedulefree_adamw",
                "require_requested": True,
                "weight_decay": 0.0,
                "min_lr": 4.0e-3,
                "betas": [0.9, 0.95],
                "muon_per_parameter_lr": False,
                "muon_lr_scale_base": 0.2,
                "muon_partition_non2d": True,
            },
            "logging": {
                "history_jsonl_path": str(tmp_path / "train_out" / "train_history.jsonl"),
            },
        }
    )

    result = prior_train_module.train_tabfoundry_simple_prior(
        cfg,
        prior_dump_path=path,
        batch_size=2,
    )

    assert result.global_step == 2
    assert (tmp_path / "train_out" / "checkpoints" / "latest.pt").exists()
    telemetry = json.loads((tmp_path / "train_out" / "telemetry.json").read_text(encoding="utf-8"))
    assert telemetry["success"] is True
    assert telemetry["artifacts"]["gradient_history_jsonl"].endswith("gradient_history.jsonl")
    training_surface = json.loads(
        (tmp_path / "train_out" / "training_surface_record.json").read_text(encoding="utf-8")
    )
    assert training_surface["model"]["arch"] == "tabfoundry_sandwich"
