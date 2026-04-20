from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf

import tab_foundry.training.prior_train as prior_train_module

from tests.training.test_prior_train_sandwich import _write_prior_dump


def _prior_dump_path(tmp_path: Path, *, name: str) -> Path:
    return _write_prior_dump(
        tmp_path / name,
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
        feature_types=np.asarray(
            [
                ["floating", "integer"],
                ["bool", "string_binary"],
            ],
            dtype=object,
        ),
    )


def _optimizer_cfg() -> dict[str, object]:
    return {
        "name": "schedulefree_adamw",
        "require_requested": True,
        "weight_decay": 0.0,
        "min_lr": 4.0e-3,
        "betas": [0.9, 0.95],
        "muon_per_parameter_lr": False,
        "muon_lr_scale_base": 0.2,
        "muon_partition_non2d": True,
    }


def test_train_routed_sandwich_prior_smoke(tmp_path: Path) -> None:
    path = _prior_dump_path(tmp_path, name="prior_routed_sandwich.h5")
    output_dir = tmp_path / "train_out_routed"
    cfg = OmegaConf.create(
        {
            "task": "classification",
            "model": {
                "arch": "routed_sandwich",
                "d_icl": 32,
                "input_normalization": "train_zscore_clip",
                "many_class_base": 2,
                "head_hidden_dim": 64,
                "sandwich_latents": 8,
                "sandwich_layers": 1,
                "sandwich_heads": 4,
                "sandwich_ff_expansion": 2,
                "routed_row_summary_tokens": 2,
                "routed_column_summary_tokens": 1,
                "routed_evidence_tokens": 4,
            },
            "runtime": {
                "seed": 1,
                "output_dir": str(output_dir),
                "device": "cpu",
                "mixed_precision": "no",
                "grad_clip": 1.0,
                "max_steps": 1,
                "eval_every": 1,
                "checkpoint_every": 1,
            },
            "optimizer": _optimizer_cfg(),
            "logging": {
                "history_jsonl_path": str(output_dir / "train_history.jsonl"),
            },
        }
    )

    result = prior_train_module.train_tabfoundry_simple_prior(
        cfg,
        prior_dump_path=path,
        batch_size=2,
    )

    assert result.global_step == 1
    training_surface = json.loads((output_dir / "training_surface_record.json").read_text(encoding="utf-8"))
    gradient_history = [
        json.loads(line)
        for line in (output_dir / "gradient_history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert training_surface["model"]["arch"] == "routed_sandwich"
    assert training_surface["training"]["loss_surface"] == "classification"
    assert training_surface["model"]["architecture"]["residual_routing"] == "dynamic_hyper"
    assert training_surface["model"]["architecture"]["evidence_tokens"] == 4
    module_names = set(gradient_history[0]["module_grad_norms"])
    assert {"evidence_builder", "latent_memory_router", "perceiver_stages.0", "direct_head"}.issubset(
        module_names
    )


def test_train_grid_sandwich_prior_smoke(tmp_path: Path) -> None:
    path = _prior_dump_path(tmp_path, name="prior_grid_sandwich.h5")
    output_dir = tmp_path / "train_out_grid"
    cfg = OmegaConf.create(
        {
            "task": "classification",
            "model": {
                "arch": "grid_sandwich",
                "d_icl": 32,
                "input_normalization": "train_zscore_clip",
                "many_class_base": 2,
                "head_hidden_dim": 64,
                "sandwich_layers": 1,
                "sandwich_heads": 4,
                "sandwich_ff_expansion": 2,
                "sandwich_pre_column_inducing_tokens": 8,
            },
            "runtime": {
                "seed": 1,
                "output_dir": str(output_dir),
                "device": "cpu",
                "mixed_precision": "no",
                "grad_clip": 1.0,
                "max_steps": 1,
                "eval_every": 1,
                "checkpoint_every": 1,
            },
            "optimizer": _optimizer_cfg(),
            "logging": {
                "history_jsonl_path": str(output_dir / "train_history.jsonl"),
            },
        }
    )

    result = prior_train_module.train_tabfoundry_simple_prior(
        cfg,
        prior_dump_path=path,
        batch_size=2,
    )

    assert result.global_step == 1
    training_surface = json.loads((output_dir / "training_surface_record.json").read_text(encoding="utf-8"))
    gradient_history = [
        json.loads(line)
        for line in (output_dir / "gradient_history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert training_surface["model"]["arch"] == "grid_sandwich"
    assert training_surface["training"]["loss_surface"] == "classification"
    assert training_surface["model"]["architecture"]["grid_core"] == "alternating_row_self_attention_and_column_row_isab"
    module_names = set(gradient_history[0]["module_grad_norms"])
    assert {"grid_layers.0", "row_pool", "direct_head"}.issubset(module_names)
