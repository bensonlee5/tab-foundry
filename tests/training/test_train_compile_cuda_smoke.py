from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

import tab_foundry.training.trainer as trainer_module

from tests.support.train_eval_smoke_cases import _write_task_batch_manifest


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_train_compile_cuda_smoke(tmp_path: Path) -> None:
    manifest_path = _write_task_batch_manifest(tmp_path)
    cfg = OmegaConf.create(
        {
            "task": "classification",
            "model": {
                "arch": "tabfoundry_sandwich",
                "d_icl": 16,
                "input_normalization": "train_zscore_clip",
                "many_class_base": 4,
                "head_hidden_dim": 32,
                "sandwich_latents": 12,
                "sandwich_layers": 1,
                "sandwich_heads": 4,
                "sandwich_ff_expansion": 2,
                "sandwich_summary_tokens_per_axis": 2,
                "sandwich_self_attention_per_cross": 1,
                "sandwich_pre_row_attention_layers": 1,
                "sandwich_pre_column_attention_layers": 1,
                "sandwich_pre_column_inducing_tokens": 4,
            },
            "data": {
                "source": "manifest",
                "manifest_path": str(manifest_path),
            },
            "runtime": {
                "seed": 1,
                "num_workers": 0,
                "output_dir": str(tmp_path / "outputs"),
                "device": "cuda",
                "mixed_precision": "no",
                "grad_clip": 1.0,
                "grad_accum_steps": 1,
                "compile_model": True,
                "trace_activations": False,
                "activation_checkpointing": True,
                "eval_every": 1,
                "checkpoint_every": 1,
                "val_batches": 0,
                "max_steps": 1,
            },
            "training": {
                "task_batch_size": 1,
                "loss_surface": "classification",
            },
            "schedule": {"stages": [{"name": "stage1", "steps": 1, "lr_max": 1.0e-3}]},
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
                "run_name": "train-compile-cuda-smoke",
            },
            "eval": {"checkpoint": None, "split": "val", "max_batches": 1},
        }
    )

    result = trainer_module.train(cfg)
    training_surface = json.loads(
        (result.output_dir / "training_surface_record.json").read_text(encoding="utf-8")
    )

    assert result.global_step == 1
    assert result.best_checkpoint is not None
    assert result.best_checkpoint.exists()
    assert training_surface["runtime"]["compile_model"] is True
    assert training_surface["runtime"]["activation_checkpointing"] is True
