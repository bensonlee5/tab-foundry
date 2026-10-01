from __future__ import annotations

from contextlib import nullcontext
from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

import tab_foundry.training.distributed as distributed_module
import tab_foundry.training.evaluate as evaluate_module
import tab_foundry.training.trainer_metrics as metrics_module
from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.model.spec import model_build_spec_from_mappings
from tests.support.train_eval_smoke_cases import _FakeAccelerator, _FakeTaskDataset


@pytest.mark.parametrize("checkpoint_surface", ["classification", "cell_bpc"])
def test_checkpoint_evaluation_restores_saved_loss_surface(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, checkpoint_surface: str
) -> None:
    model_cfg = {
        "arch": "tabfoundry_sandwich",
        "d_icl": 16,
        "head_hidden_dim": 16,
        "sandwich_latents": 4,
        "sandwich_layers": 1,
        "sandwich_heads": 2,
        "sandwich_self_attention_per_cross": 1,
        "sandwich_pre_row_attention_layers": 0,
        "sandwich_pre_column_attention_layers": 0,
        "many_class_base": 4,
    }
    model = build_model_from_spec(
        model_build_spec_from_mappings(task="classification", primary=model_cfg)
    )
    model.set_loss_surface(checkpoint_surface)
    batch = _FakeTaskDataset()[0]
    expected = metrics_module._evaluate_loader(
        model, [batch], accelerator=_FakeAccelerator(), task="classification", max_batches=1
    )
    checkpoint = tmp_path / "checkpoint.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "config": {
                "task": "classification",
                "model": model_cfg,
                "training": {"loss_surface": checkpoint_surface},
            },
        },
        checkpoint,
    )
    monkeypatch.setattr(evaluate_module, "build_task_dataset", lambda *a, **kw: object())
    monkeypatch.setattr(evaluate_module, "build_task_loader", lambda *a, **kw: [batch])
    monkeypatch.setattr(
        evaluate_module, "build_accelerator_from_runtime", lambda *a, **kw: _FakeAccelerator()
    )
    cfg = OmegaConf.create(
        {
            "task": "classification",
            "model": model_cfg,
            "training": {
                "loss_surface": "classification" if checkpoint_surface == "cell_bpc" else "cell_bpc"
            },
            "data": {"manifest_path": "unused.parquet"},
            "runtime": {"seed": 1, "num_workers": 0, "device": "cpu", "mixed_precision": "no"},
            "eval": {"checkpoint": str(checkpoint), "split": "val", "max_batches": 1},
        }
    )
    result = evaluate_module.evaluate_checkpoint(cfg)
    expected_keys = ("bpc", "bpf") if checkpoint_surface == "cell_bpc" else ("acc",)
    assert result.metrics == pytest.approx(
        {"loss": expected["val_loss"], **{key: expected[key] for key in expected_keys}}
    )


@pytest.mark.parametrize("has_local_batch", [True, False])
def test_evaluation_reduces_union_of_rank_metrics(
    monkeypatch: pytest.MonkeyPatch, has_local_batch: bool
) -> None:
    # A second rank supplies a different auxiliary metric, or all data when this
    # rank has no batches. All ranks must issue the same ordered collectives.
    remote = {"acc": (0.8, 1.0), "remote_only": (3.0, 1.0)}
    expected_keys = sorted({"acc", "remote_only"} | ({"local_only"} if has_local_batch else set()))
    monkeypatch.setattr(
        distributed_module, "gather_object", lambda keys: [keys, list(remote)]
    )
    monkeypatch.setattr(distributed_module, "broadcast_object_list", lambda *a, **kw: None)

    class RemoteAccelerator(_FakeAccelerator):
        def reduce(self, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
            assert reduction == "sum"
            if tensor.numel() == 1:
                # Loss sum and task count use their existing scalar reductions.
                return tensor + (2 if tensor.is_floating_point() else 1)
            assert tensor.numel() == 2 * len(expected_keys)
            remote_packed = torch.tensor(
                [value for key in expected_keys for value in remote.get(key, (0.0, 0.0))],
                dtype=tensor.dtype,
            )
            return tensor + remote_packed

    monkeypatch.setattr(
        metrics_module,
        "_compute_loss_and_metrics",
        lambda *a, **kw: (torch.tensor(1.0), {"local_only": 4.0, "acc": 0.2}),
    )
    result = metrics_module._evaluate_loader(
        torch.nn.Identity(),
        [_FakeTaskDataset()[0]] if has_local_batch else [],
        accelerator=RemoteAccelerator(),
        task="classification",
        max_batches=1,
    )
    expected = {"val_loss": 1.5, "acc": 0.5, "local_only": 4.0, "remote_only": 3.0}
    if not has_local_batch:
        expected = {"val_loss": 2.0, "acc": 0.8, "remote_only": 3.0}
    assert result == pytest.approx(expected)


@pytest.mark.parametrize("training", [True, False])
@pytest.mark.parametrize("fail_forward", [True, False])
def test_evaluation_restores_model_mode(
    monkeypatch: pytest.MonkeyPatch, training: bool, fail_forward: bool
) -> None:
    model = torch.nn.Identity().train(training)

    def compute(*args, **kwargs):
        assert not model.training
        if fail_forward:
            raise RuntimeError("evaluation failed")
        return torch.tensor(1.0), {"acc": 0.5}

    monkeypatch.setattr(metrics_module, "_compute_loss_and_metrics", compute)
    with pytest.raises(RuntimeError, match="evaluation failed") if fail_forward else nullcontext():
        metrics_module._evaluate_loader(
            model, [_FakeTaskDataset()[0]], accelerator=_FakeAccelerator(),
            task="classification", max_batches=1,
        )
    assert model.training is training
