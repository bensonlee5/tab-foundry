from __future__ import annotations

from contextlib import nullcontext
import json
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

import tab_foundry.training.evaluate as evaluate_module
import tab_foundry.training.trainer as trainer_module
from tab_foundry.model.architectures.tabfoundry import ClassificationOutput, RegressionOutput
from tab_foundry.training.schedule import build_stage_configs
from tab_foundry.types import TaskBatch


class _FakeAccelerator:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.is_main_process = True

    def prepare(self, *items: object) -> object:
        if len(items) == 1:
            return items[0]
        return items

    def prepare_optimizer(self, optimizer: torch.optim.Optimizer) -> torch.optim.Optimizer:
        return optimizer

    def autocast(self):
        return nullcontext()

    def accumulate(self, _model: nn.Module):
        return nullcontext()

    def backward(self, loss: torch.Tensor) -> None:
        loss.backward()

    def clip_grad_norm_(self, params, max_norm: float) -> torch.Tensor:
        return torch.nn.utils.clip_grad_norm_(list(params), max_norm)

    def reduce(self, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        if reduction != "sum":
            raise ValueError("only sum reduction is supported in fake accelerator")
        return tensor

    def unwrap_model(self, model: nn.Module) -> nn.Module:
        return model

    def get_state_dict(self, model: nn.Module) -> dict[str, torch.Tensor]:
        return model.state_dict()

    def print(self, *_args: object, **_kwargs: object) -> None:
        return None

    def wait_for_everyone(self) -> None:
        return None


class _FakeTaskDataset(Dataset[TaskBatch]):
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        super().__init__()

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> TaskBatch:
        seed = int(index) + 1
        torch.manual_seed(seed)
        x_train = torch.randn(6, 4)
        y_train = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64)
        return TaskBatch(
            x_train=x_train,
            y_train=y_train,
            x_test=torch.randn(3, 4),
            y_test=torch.tensor([0, 1, 2], dtype=torch.int64),
            metadata={"dataset_index": index},
            num_classes=3,
        )


class _FakeRegressionTaskDataset(Dataset[TaskBatch]):
    def __init__(self, *_args: object, **_kwargs: object) -> None:
        super().__init__()

    def __len__(self) -> int:
        return 2

    def __getitem__(self, index: int) -> TaskBatch:
        seed = int(index) + 11
        torch.manual_seed(seed)
        x_train = torch.randn(6, 4)
        y_train = torch.linspace(0.0, 1.0, steps=6, dtype=torch.float32)
        return TaskBatch(
            x_train=x_train,
            y_train=y_train,
            x_test=torch.randn(3, 4),
            y_test=torch.linspace(0.2, 0.8, steps=3, dtype=torch.float32),
            metadata={"dataset_index": index},
            num_classes=None,
        )


class _TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, batch: TaskBatch) -> ClassificationOutput:
        return ClassificationOutput(
            logits=self.linear(batch.x_test),
            num_classes=3,
        )


class _TinyRegressor(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 999)

    def forward(self, batch: TaskBatch) -> RegressionOutput:
        return RegressionOutput(
            quantiles=self.linear(batch.x_test),
        )


class _FakeWandbRun:
    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, object], int]] = []
        self.summary: dict[str, object] = {}
        self.finished = False

    def log(self, payload: dict[str, object], *, step: int) -> None:
        self.logged.append((dict(payload), int(step)))

    def finish(self) -> None:
        self.finished = True


def _classification_cfg(tmp_path: Path) -> object:
    return OmegaConf.create(
        {
            "task": "classification",
            "model": {},
            "data": {
                "source": "manifest",
                "manifest_path": "unused.parquet",
                "surface_label": "runtime_manifest",
                "allow_missing_values": False,
                "train_row_cap": None,
                "test_row_cap": None,
            },
            "preprocessing": {
                "surface_label": "runtime_default",
                "impute_missing": True,
                "all_nan_fill": 0.0,
                "label_mapping": "train_only_remap",
                "unseen_test_label_policy": "filter",
            },
            "runtime": {
                "seed": 1,
                "num_workers": 0,
                "output_dir": str(tmp_path / "outputs"),
                "device": "cpu",
                "mixed_precision": "no",
                "grad_clip": 1.0,
                "grad_accum_steps": 1,
                "eval_every": 1,
                "checkpoint_every": None,
                "val_batches": 1,
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
            "logging": {"use_wandb": False, "project": "test", "run_name": "test"},
            "eval": {"checkpoint": None, "split": "val", "max_batches": 1},
        }
    )


def _regression_cfg(tmp_path: Path) -> object:
    cfg = _classification_cfg(tmp_path)
    cfg.task = "regression"
    return cfg


def _install_classification_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_spec = SimpleNamespace(task="classification")
    monkeypatch.setattr(trainer_module, "build_task_dataset", lambda *_args, **_kwargs: _FakeTaskDataset())
    monkeypatch.setattr(evaluate_module, "build_task_dataset", lambda *_args, **_kwargs: _FakeTaskDataset())
    monkeypatch.setattr(
        trainer_module,
        "build_accelerator_from_runtime",
        lambda *_args, **_kwargs: _FakeAccelerator(),
    )
    monkeypatch.setattr(
        evaluate_module,
        "build_accelerator_from_runtime",
        lambda *_args, **_kwargs: _FakeAccelerator(),
    )
    monkeypatch.setattr(trainer_module, "init_wandb_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(trainer_module, "model_build_spec_from_mappings", lambda **_kwargs: fake_spec)
    monkeypatch.setattr(
        evaluate_module,
        "checkpoint_model_build_spec_from_mappings",
        lambda **_kwargs: fake_spec,
    )
    monkeypatch.setattr(trainer_module, "build_model_from_spec", lambda _spec: _TinyClassifier())
    monkeypatch.setattr(evaluate_module, "build_model_from_spec", lambda _spec: _TinyClassifier())


def _install_regression_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_spec = SimpleNamespace(task="regression")
    monkeypatch.setattr(
        trainer_module,
        "build_task_dataset",
        lambda *_args, **_kwargs: _FakeRegressionTaskDataset(),
    )
    monkeypatch.setattr(
        evaluate_module,
        "build_task_dataset",
        lambda *_args, **_kwargs: _FakeRegressionTaskDataset(),
    )
    monkeypatch.setattr(
        trainer_module,
        "build_accelerator_from_runtime",
        lambda *_args, **_kwargs: _FakeAccelerator(),
    )
    monkeypatch.setattr(
        evaluate_module,
        "build_accelerator_from_runtime",
        lambda *_args, **_kwargs: _FakeAccelerator(),
    )
    monkeypatch.setattr(trainer_module, "init_wandb_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(trainer_module, "model_build_spec_from_mappings", lambda **_kwargs: fake_spec)
    monkeypatch.setattr(
        evaluate_module,
        "checkpoint_model_build_spec_from_mappings",
        lambda **_kwargs: fake_spec,
    )
    monkeypatch.setattr(trainer_module, "build_model_from_spec", lambda _spec: _TinyRegressor())
    monkeypatch.setattr(evaluate_module, "build_model_from_spec", lambda _spec: _TinyRegressor())


def test_train_smoke_runs_end_to_end(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_classification_fakes(monkeypatch)
    cfg = _classification_cfg(tmp_path)

    result = trainer_module.train(cfg)

    assert result.global_step == 1
    assert result.best_checkpoint is not None
    assert result.latest_checkpoint is not None
    assert result.best_checkpoint.exists()
    assert result.latest_checkpoint.exists()
    assert result.metrics["best_val_loss"] >= 0.0
    assert result.metrics["final_val_loss"] >= 0.0
    assert result.metrics["max_grad_norm"] >= 0.0
    best_payload = torch.load(result.best_checkpoint, map_location="cpu", weights_only=False)
    assert "preprocessor_state" not in best_payload


def test_train_smoke_writes_step_snapshots(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_classification_fakes(monkeypatch)
    cfg = _classification_cfg(tmp_path)
    cfg.schedule.stages = [{"name": "stage1", "steps": 2, "lr_max": 1.0e-3}]
    cfg.runtime.checkpoint_every = 1

    result = trainer_module.train(cfg)

    checkpoint_dir = result.output_dir / "checkpoints"
    snapshots = sorted(checkpoint_dir.glob("step_*.pt"))
    assert [path.name for path in snapshots] == ["step_000001.pt", "step_000002.pt"]
    assert all(path.exists() for path in snapshots)


def test_evaluate_checkpoint_smoke(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_classification_fakes(monkeypatch)
    cfg = _classification_cfg(tmp_path)
    checkpoint = tmp_path / "tiny.pt"
    model = _TinyClassifier()
    torch.save(
        {"model": model.state_dict(), "config": {"task": "classification", "model": {}}},
        checkpoint,
    )
    cfg.eval.checkpoint = str(checkpoint)

    result = evaluate_module.evaluate_checkpoint(cfg)

    assert result.checkpoint == checkpoint.resolve()
    assert "loss" in result.metrics
    assert "acc" in result.metrics


def test_train_smoke_writes_history_jsonl(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    _install_classification_fakes(monkeypatch)
    cfg = _classification_cfg(tmp_path)
    history_path = tmp_path / "outputs" / "train_history.jsonl"
    cfg.logging.history_jsonl_path = str(history_path)

    _ = trainer_module.train(cfg)

    records = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(records) == 1
    assert records[0]["step"] == 1
    assert records[0]["stage"] == "stage1"
    assert records[0]["train_loss"] >= 0.0
    assert 0.0 <= records[0]["train_acc"] <= 1.0
    assert records[0]["val_loss"] >= 0.0
    assert 0.0 <= records[0]["val_acc"] <= 1.0
    assert records[0]["lr"] > 0.0
    assert records[0]["grad_norm"] >= 0.0
    assert records[0]["elapsed_seconds"] >= 0.0
    assert records[0]["train_elapsed_seconds"] >= 0.0
    assert records[0]["train_loss_delta"] is None
    assert records[0]["train_loss_ema"] >= 0.0
    assert records[0]["grad_clip_threshold"] == pytest.approx(1.0)
    assert isinstance(records[0]["grad_clip_triggered"], bool)


def test_train_rejects_non_empty_history_jsonl(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    cfg = _classification_cfg(tmp_path)
    history_path = tmp_path / "outputs" / "train_history.jsonl"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.write_text(json.dumps({"step": 25}) + "\n", encoding="utf-8")
    cfg.logging.history_jsonl_path = str(history_path)
    monkeypatch.setattr(
        trainer_module,
        "build_accelerator_from_runtime",
        lambda *_args, **_kwargs: pytest.fail("dirty-output guard should fail before accelerator setup"),
    )
    with pytest.raises(RuntimeError, match="not resume-safe"):
        _ = trainer_module.train(cfg)


def test_train_rejects_existing_checkpoint_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    cfg = _classification_cfg(tmp_path)
    checkpoint_dir = tmp_path / "outputs" / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    (checkpoint_dir / "step_000025.pt").write_bytes(b"stale")
    monkeypatch.setattr(
        trainer_module,
        "build_accelerator_from_runtime",
        lambda *_args, **_kwargs: pytest.fail("dirty-output guard should fail before accelerator setup"),
    )

    with pytest.raises(RuntimeError, match="not resume-safe"):
        _ = trainer_module.train(cfg)


def test_build_stage_configs_validates_payloads() -> None:
    stages = build_stage_configs([{"name": "warmup", "steps": 2, "lr_max": 5.0e-4}])
    assert len(stages) == 1
    assert stages[0].name == "warmup"
    assert stages[0].steps == 2
    assert stages[0].lr_max == pytest.approx(5.0e-4)


def test_build_stage_configs_rejects_non_int_steps() -> None:
    with pytest.raises(ValueError, match="stage steps must be int"):
        _ = build_stage_configs([{"name": "bad", "steps": 1.5, "lr_max": 1.0e-3}])


def test_build_stage_configs_rejects_non_numeric_lr() -> None:
    with pytest.raises(ValueError, match="stage lr_max must be float"):
        _ = build_stage_configs([{"name": "bad", "steps": 1, "lr_max": "fast"}])


def test_train_history_uses_linear_schedule_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_classification_fakes(monkeypatch)
    cfg = _classification_cfg(tmp_path)
    history_path = tmp_path / "outputs" / "linear_history.jsonl"
    cfg.logging.history_jsonl_path = str(history_path)
    cfg.runtime.eval_every = 10
    cfg.schedule.stages = [
        {
            "name": "stage1",
            "steps": 4,
            "lr_max": 1.0e-3,
            "lr_schedule": "linear",
            "warmup_ratio": 0.0,
        }
    ]

    _ = trainer_module.train(cfg)

    records = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert [record["lr"] for record in records] == pytest.approx([1.0e-3, 7.0e-4, 4.0e-4, 1.0e-4])


def test_train_logs_enriched_wandb_metrics_and_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_classification_fakes(monkeypatch)
    cfg = _classification_cfg(tmp_path)
    cfg.logging.use_wandb = True
    cfg.schedule.stages = [{"name": "stage1", "steps": 2, "lr_max": 1.0e-3}]
    fake_run = _FakeWandbRun()
    monkeypatch.setattr(trainer_module, "init_wandb_run", lambda *_args, **_kwargs: fake_run)

    result = trainer_module.train(cfg)

    train_logs = [
        (payload, step)
        for payload, step in fake_run.logged
        if "train/loss" in payload
    ]
    assert [step for _payload, step in train_logs] == [1, 2]
    assert "train/loss_delta" not in train_logs[0][0]
    assert train_logs[1][0]["train/stage"] == "stage1"
    assert "train/loss_delta" in train_logs[1][0]
    assert "train/loss_ema" in train_logs[1][0]
    assert "train/elapsed_seconds" in train_logs[1][0]
    assert "train/train_elapsed_seconds" in train_logs[1][0]
    assert "train/grad_clip_threshold" in train_logs[1][0]
    assert "train/grad_clip_triggered" in train_logs[1][0]
    assert "train/lr_adamw" in train_logs[1][0]
    val_logs = [
        (payload, step)
        for payload, step in fake_run.logged
        if "val/val_loss" in payload
    ]
    assert [step for _payload, step in val_logs] == [1, 2]
    assert fake_run.finished is True
    assert fake_run.summary["optimizer/requested_name"] == "adamw"
    assert fake_run.summary["optimizer/resolved_name"] == "adamw"
    assert fake_run.summary["run/output_dir"] == str(result.output_dir)
    assert fake_run.summary["run/global_step"] == 2
    assert fake_run.summary["run/best_checkpoint"] == str(result.best_checkpoint.resolve())
    assert fake_run.summary["run/latest_checkpoint"] == str(result.latest_checkpoint.resolve())
    assert fake_run.summary["metrics/best_val_loss"] >= 0.0
    assert fake_run.summary["metrics/final_train_loss"] >= 0.0
    assert 0.0 <= fake_run.summary["metrics/final_train_acc"] <= 1.0
    assert fake_run.summary["metrics/final_train_loss_ema"] >= 0.0
    assert fake_run.summary["metrics/final_grad_norm"] >= 0.0
    assert fake_run.summary["metrics/wall_elapsed_seconds"] >= 0.0


def test_evaluate_checkpoint_logs_wandb_metrics_for_classification(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_classification_fakes(monkeypatch)
    cfg = _classification_cfg(tmp_path)
    cfg.logging.use_wandb = True
    checkpoint = tmp_path / "tiny_cls.pt"
    model = _TinyClassifier()
    torch.save(
        {
            "model": model.state_dict(),
            "config": {"task": "classification", "model": {}},
            "global_step": 17,
        },
        checkpoint,
    )
    cfg.eval.checkpoint = str(checkpoint)
    fake_run = _FakeWandbRun()
    monkeypatch.setattr(evaluate_module, "init_wandb_run", lambda *_args, **_kwargs: fake_run)

    result = evaluate_module.evaluate_checkpoint(cfg)

    assert result.checkpoint == checkpoint.resolve()
    assert fake_run.logged == [
        (
            {
                "eval/loss": result.metrics["loss"],
                "eval/acc": result.metrics["acc"],
            },
            17,
        )
    ]
    assert fake_run.summary["run/checkpoint"] == str(checkpoint.resolve())
    assert fake_run.summary["run/split"] == "val"
    assert fake_run.summary["run/global_step"] == 17
    assert fake_run.summary["eval/max_batches"] == 1
    assert fake_run.summary["metrics/loss"] == result.metrics["loss"]
    assert fake_run.summary["metrics/acc"] == result.metrics["acc"]
    assert fake_run.finished is True


def test_evaluate_checkpoint_logs_wandb_metrics_for_regression(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_regression_fakes(monkeypatch)
    cfg = _regression_cfg(tmp_path)
    cfg.logging.use_wandb = True
    checkpoint = tmp_path / "tiny_reg.pt"
    model = _TinyRegressor()
    torch.save(
        {
            "model": model.state_dict(),
            "config": {"task": "regression", "model": {}},
        },
        checkpoint,
    )
    cfg.eval.checkpoint = str(checkpoint)
    fake_run = _FakeWandbRun()
    monkeypatch.setattr(evaluate_module, "init_wandb_run", lambda *_args, **_kwargs: fake_run)

    result = evaluate_module.evaluate_checkpoint(cfg)

    assert result.checkpoint == checkpoint.resolve()
    assert fake_run.logged == [
        (
            {
                "eval/loss": result.metrics["loss"],
                "eval/rmse": result.metrics["rmse"],
            },
            0,
        )
    ]
    assert fake_run.summary["run/checkpoint"] == str(checkpoint.resolve())
    assert fake_run.summary["run/split"] == "val"
    assert fake_run.summary["run/global_step"] == 0
    assert fake_run.summary["eval/max_batches"] == 1
    assert fake_run.summary["metrics/loss"] == result.metrics["loss"]
    assert fake_run.summary["metrics/rmse"] == result.metrics["rmse"]
    assert fake_run.finished is True
