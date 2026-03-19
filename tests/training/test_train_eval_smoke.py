from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

import tab_foundry.training.evaluate as evaluate_module
import tab_foundry.training.distributed as distributed_module
import tab_foundry.training.trainer as trainer_module
from tab_foundry.model.outputs import ClassificationOutput
from tab_foundry.training.schedule import build_stage_configs
from tab_foundry.types import TaskBatch


class _FakeAccelerator:
    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.is_main_process = True
        self.num_processes = 1

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


class _FakeMultiProcessActivationAccelerator(_FakeAccelerator):
    def __init__(
        self,
        *,
        remote_activation_trace_stats: dict[str, tuple[float, int]],
    ) -> None:
        super().__init__()
        self.num_processes = 2
        self.remote_activation_trace_stats = {
            str(key): (float(total_sum_sq), int(total_count))
            for key, (total_sum_sq, total_count) in remote_activation_trace_stats.items()
        }

    def reduce(self, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        if reduction != "sum":
            raise ValueError("only sum reduction is supported in fake accelerator")
        if int(tensor.numel()) == 2 * len(self.remote_activation_trace_stats):
            ordered_keys = sorted(self.remote_activation_trace_stats)
            remote_tensor = torch.zeros_like(tensor)
            for index, key in enumerate(ordered_keys):
                total_sum_sq, total_count = self.remote_activation_trace_stats[key]
                remote_tensor[2 * index] = float(total_sum_sq)
                remote_tensor[2 * index + 1] = float(total_count)
            return tensor + remote_tensor
        return tensor * 2


def _trace_activation_accumulate(
    buffer: dict[str, tuple[float, int]],
    name: str,
    tensor: torch.Tensor,
) -> None:
    trace_tensor = tensor.detach().to(torch.float32)
    trace_sum_sq = float(trace_tensor.square().sum().item())
    trace_count = int(trace_tensor.numel())
    total_sum_sq, total_count = buffer.get(name, (0.0, 0))
    buffer[name] = (total_sum_sq + trace_sum_sq, total_count + trace_count)


def _trace_activation_snapshot(buffer: dict[str, tuple[float, int]]) -> dict[str, float]:
    return {
        name: math.sqrt(total_sum_sq / float(total_count))
        for name, (total_sum_sq, total_count) in buffer.items()
        if total_count > 0
    }


def _trace_activation_stats_snapshot(
    buffer: dict[str, tuple[float, int]],
) -> dict[str, tuple[float, int]]:
    return {
        name: (float(total_sum_sq), int(total_count))
        for name, (total_sum_sq, total_count) in buffer.items()
        if total_count > 0
    }


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


class _VariableShapeTaskDataset(Dataset[TaskBatch]):
    def __init__(self, *, test_sizes: list[int]) -> None:
        super().__init__()
        self.test_sizes = [int(size) for size in test_sizes]

    def __len__(self) -> int:
        return len(self.test_sizes)

    def __getitem__(self, index: int) -> TaskBatch:
        torch.manual_seed(int(index) + 1)
        n_test = self.test_sizes[index]
        x_train = torch.randn(6, 4)
        y_train = torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64)
        y_test = torch.arange(n_test, dtype=torch.int64) % 3
        return TaskBatch(
            x_train=x_train,
            y_train=y_train,
            x_test=torch.randn(n_test, 4),
            y_test=y_test,
            metadata={"dataset_index": index},
            num_classes=3,
        )


class _TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, batch: TaskBatch) -> ClassificationOutput:
        return ClassificationOutput(logits=self.linear(batch.x_test), num_classes=3)


class _TraceableRowPool(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(self, encoded_cells: torch.Tensor, token_padding_mask=None) -> torch.Tensor:
        del token_padding_mask
        return self.linear(encoded_cells)


class _TraceableContextEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 4)

    def forward(
        self,
        rows: torch.Tensor,
        *,
        train_target_embeddings: torch.Tensor,
        train_test_split_index: int,
    ) -> torch.Tensor:
        del train_test_split_index
        if rows.ndim != 2:
            raise RuntimeError("traceable context encoder expects [R, D] rows")
        if train_target_embeddings.ndim != 2:
            raise RuntimeError("traceable context encoder expects [R_train, D] train_target_embeddings")
        context = train_target_embeddings.mean(dim=0, keepdim=True)
        return self.linear(rows + context)


class _TraceableStageLocalClassifier(nn.Module):
    def __init__(self, *, use_context: bool = True) -> None:
        super().__init__()
        self.feature_encoder = nn.Linear(4, 4)
        self.column_encoder = nn.Linear(4, 4)
        self.row_pool = _TraceableRowPool()
        self.context_label_embed = nn.Embedding(8, 4) if use_context else None
        self.context_encoder = _TraceableContextEncoder() if use_context else None
        self.direct_head = nn.Linear(4, 3)
        self._activation_trace: dict[str, tuple[float, int]] | None = None

    def enable_activation_trace(self) -> None:
        self._activation_trace = {}

    def disable_activation_trace(self) -> None:
        self._activation_trace = None

    def trace_activation(self, name: str, tensor: torch.Tensor) -> None:
        if self._activation_trace is None:
            return
        _trace_activation_accumulate(self._activation_trace, name, tensor)

    def flush_activation_trace_stats(self) -> dict[str, tuple[float, int]] | None:
        if self._activation_trace is None:
            return None
        snapshot = _trace_activation_stats_snapshot(self._activation_trace)
        self._activation_trace = {}
        return snapshot

    def flush_activation_trace(self) -> dict[str, float] | None:
        snapshot = self.flush_activation_trace_stats()
        if snapshot is None:
            return None
        return {
            name: math.sqrt(total_sum_sq / float(total_count))
            for name, (total_sum_sq, total_count) in snapshot.items()
            if total_count > 0
        }

    def forward(self, batch: TaskBatch) -> ClassificationOutput:
        features = self.feature_encoder(batch.x_test.to(torch.float32))
        self.trace_activation("post_feature_encoder", features)
        encoded = self.column_encoder(features)
        self.trace_activation("post_column_encoder", encoded)
        rows = self.row_pool(encoded, token_padding_mask=None)
        self.trace_activation("post_row_pool", rows)
        if self.context_encoder is not None and self.context_label_embed is not None:
            train_targets = self.context_label_embed(batch.y_train.clamp(max=7))
            rows = self.context_encoder(
                rows,
                train_target_embeddings=train_targets,
                train_test_split_index=int(batch.y_train.shape[0]),
            )
            self.trace_activation("post_context_encoder", rows)
        return ClassificationOutput(logits=self.direct_head(rows), num_classes=3)


class _DeterministicTraceClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.direct_head = nn.Linear(1, 3)
        self._activation_trace: dict[str, tuple[float, int]] | None = None

    def enable_activation_trace(self) -> None:
        self._activation_trace = {}

    def disable_activation_trace(self) -> None:
        self._activation_trace = None

    def trace_activation(self, name: str, tensor: torch.Tensor) -> None:
        if self._activation_trace is None:
            return
        _trace_activation_accumulate(self._activation_trace, name, tensor)

    def flush_activation_trace_stats(self) -> dict[str, tuple[float, int]] | None:
        if self._activation_trace is None:
            return None
        snapshot = _trace_activation_stats_snapshot(self._activation_trace)
        self._activation_trace = {}
        return snapshot

    def flush_activation_trace(self) -> dict[str, float] | None:
        snapshot = self.flush_activation_trace_stats()
        if snapshot is None:
            return None
        return {
            name: math.sqrt(total_sum_sq / float(total_count))
            for name, (total_sum_sq, total_count) in snapshot.items()
            if total_count > 0
        }

    def forward(self, batch: TaskBatch) -> ClassificationOutput:
        batch_size = int(batch.y_test.shape[0])
        self.trace_activation(
            "post_feature_encoder",
            torch.full((batch_size, 1), 2.0, dtype=torch.float32),
        )
        self.trace_activation(
            "post_column_encoder",
            torch.full((batch_size, 1), 4.0, dtype=torch.float32),
        )
        self.trace_activation(
            "post_row_pool",
            torch.full((batch_size, 1), 6.0, dtype=torch.float32),
        )
        self.trace_activation(
            "post_context_encoder",
            torch.full((batch_size, 1), 8.0, dtype=torch.float32),
        )
        logits = self.direct_head(torch.ones((batch_size, 1), dtype=torch.float32))
        return ClassificationOutput(logits=logits, num_classes=3)


class _FakeWandbRun:
    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, object], int]] = []
        self.summary: dict[str, object] = {}
        self.finished = False

    def log(self, payload: dict[str, object], *, step: int) -> None:
        self.logged.append((dict(payload), int(step)))

    def finish(self) -> None:
        self.finished = True


class _UnevenActivationTraceClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.direct_head = nn.Linear(1, 3)
        self._activation_trace: dict[str, tuple[float, int]] | None = None

    def enable_activation_trace(self) -> None:
        self._activation_trace = {}

    def disable_activation_trace(self) -> None:
        self._activation_trace = None

    def trace_activation(self, name: str, tensor: torch.Tensor) -> None:
        if self._activation_trace is None:
            return
        _trace_activation_accumulate(self._activation_trace, name, tensor)

    def flush_activation_trace_stats(self) -> dict[str, tuple[float, int]] | None:
        if self._activation_trace is None:
            return None
        snapshot = _trace_activation_stats_snapshot(self._activation_trace)
        self._activation_trace = {}
        return snapshot

    def flush_activation_trace(self) -> dict[str, float] | None:
        snapshot = self.flush_activation_trace_stats()
        if snapshot is None:
            return None
        return {
            name: math.sqrt(total_sum_sq / float(total_count))
            for name, (total_sum_sq, total_count) in snapshot.items()
            if total_count > 0
        }

    def forward(self, batch: TaskBatch) -> ClassificationOutput:
        batch_size = int(batch.y_test.shape[0])
        activation_value = 1.0 if batch_size == 1 else 10.0
        self.trace_activation(
            "post_feature_encoder",
            torch.full((batch_size + 1, 2), activation_value, dtype=torch.float32),
        )
        logits = self.direct_head(torch.ones((batch_size, 1), dtype=torch.float32))
        return ClassificationOutput(logits=logits, num_classes=3)


class _LegacyOnlyTraceClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.direct_head = nn.Linear(1, 3)
        self._trace_enabled = False

    def enable_activation_trace(self) -> None:
        self._trace_enabled = True

    def disable_activation_trace(self) -> None:
        self._trace_enabled = False

    def flush_activation_trace(self) -> dict[str, float] | None:
        if not self._trace_enabled:
            return None
        return {"post_feature_encoder": 2.0}

    def forward(self, batch: TaskBatch) -> ClassificationOutput:
        batch_size = int(batch.y_test.shape[0])
        logits = self.direct_head(torch.ones((batch_size, 1), dtype=torch.float32))
        return ClassificationOutput(logits=logits, num_classes=3)


def _classification_cfg(tmp_path: Path) -> object:
    return OmegaConf.create(
        {
            "task": "classification",
            "model": {},
            "data": {"manifest_path": "unused.parquet", "train_row_cap": None, "test_row_cap": None},
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


def _install_traceable_classifier(
    monkeypatch: pytest.MonkeyPatch,
    *,
    use_context: bool = True,
) -> None:
    monkeypatch.setattr(
        trainer_module,
        "build_model_from_spec",
        lambda _spec: _TraceableStageLocalClassifier(use_context=use_context),
    )


def _install_deterministic_trace_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trainer_module,
        "build_model_from_spec",
        lambda _spec: _DeterministicTraceClassifier(),
    )


def _install_uneven_trace_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trainer_module,
        "build_model_from_spec",
        lambda _spec: _UnevenActivationTraceClassifier(),
    )


def _install_legacy_trace_classifier(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        trainer_module,
        "build_model_from_spec",
        lambda _spec: _LegacyOnlyTraceClassifier(),
    )


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
    torch.save({"model": model.state_dict(), "config": {"task": "classification", "model": {}}}, checkpoint)
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
    assert fake_run.summary["telemetry/success"] is True
    assert fake_run.summary["artifacts/gradient_history_jsonl"].endswith("gradient_history.jsonl")
    assert fake_run.summary["artifacts/telemetry_json"].endswith("telemetry.json")


def test_train_writes_regular_gradient_history_and_telemetry_with_stage_local_traces(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_classification_fakes(monkeypatch)
    _install_traceable_classifier(monkeypatch)
    cfg = _classification_cfg(tmp_path)
    cfg.logging.use_wandb = True
    cfg.runtime.trace_activations = True
    cfg.schedule.stages = [{"name": "stage1", "steps": 2, "lr_max": 1.0e-3}]
    fake_run = _FakeWandbRun()
    monkeypatch.setattr(trainer_module, "init_wandb_run", lambda *_args, **_kwargs: fake_run)

    result = trainer_module.train(cfg)

    gradient_history_path = result.output_dir / "gradient_history.jsonl"
    telemetry_path = result.output_dir / "telemetry.json"
    gradient_history = [
        json.loads(line)
        for line in gradient_history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))

    assert len(gradient_history) == 2
    assert set(gradient_history[0]["module_grad_norms"]) == {
        "column_encoder",
        "context_encoder",
        "context_label_embed",
        "direct_head",
        "feature_encoder",
        "row_pool",
    }
    assert set(gradient_history[0]["activation_norms"]) == {
        "post_column_encoder",
        "post_context_encoder",
        "post_feature_encoder",
        "post_row_pool",
    }
    assert telemetry["artifacts"]["gradient_history_jsonl"] == str(gradient_history_path)
    assert telemetry["artifacts"]["telemetry_json"] == str(telemetry_path)
    assert telemetry["gradient_summary"]["modules"]["context_encoder"]["final_grad_norm"] >= 0.0
    assert (
        telemetry["diagnostics"]["stage_local_gradients"]["modules"]["row_pool"]["windows"]["final_10pct"][
            "mean_grad_norm"
        ]
        >= 0.0
    )
    assert (
        telemetry["diagnostics"]["activation_windows"]["tracked_activations"]["post_context_encoder"]["windows"][
            "final_10pct"
        ]["record_count"]
        == 1
    )
    assert fake_run.summary["telemetry/success"] is True
    assert (
        fake_run.summary[
            "diagnostics/stage_local_gradients/modules/column_encoder/windows/final_10pct/mean_grad_norm"
        ]
        >= 0.0
    )
    assert (
        fake_run.summary[
            "diagnostics/activation_windows/tracked_activations/post_row_pool/windows/final_10pct/mean"
        ]
        >= 0.0
    )


def test_train_trace_activations_handles_context_disabled_surface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_classification_fakes(monkeypatch)
    _install_traceable_classifier(monkeypatch, use_context=False)
    cfg = _classification_cfg(tmp_path)
    cfg.runtime.trace_activations = True

    result = trainer_module.train(cfg)

    gradient_history = [
        json.loads(line)
        for line in (result.output_dir / "gradient_history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    telemetry = json.loads((result.output_dir / "telemetry.json").read_text(encoding="utf-8"))

    assert "context_encoder" not in gradient_history[0]["module_grad_norms"]
    assert "post_context_encoder" not in gradient_history[0]["activation_norms"]
    assert (
        telemetry["diagnostics"]["stage_local_gradients"]["modules"]["context_encoder"]["windows"]["early_1_25"][
            "record_count"
        ]
        == 0
    )
    assert (
        telemetry["diagnostics"]["activation_windows"]["tracked_activations"]["post_context_encoder"]["windows"][
            "early_1_25"
        ]["record_count"]
        == 0
    )


def test_train_reduces_activation_norms_across_accelerator_ranks(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_classification_fakes(monkeypatch)
    _install_deterministic_trace_classifier(monkeypatch)
    remote_activation_trace_stats = {
        "post_column_encoder": (2000.0, 5),
        "post_context_encoder": (8000.0, 5),
        "post_feature_encoder": (500.0, 5),
        "post_row_pool": (4500.0, 5),
    }
    fake_run = _FakeWandbRun()
    monkeypatch.setattr(
        trainer_module,
        "build_accelerator_from_runtime",
        lambda *_args, **_kwargs: _FakeMultiProcessActivationAccelerator(
            remote_activation_trace_stats=remote_activation_trace_stats,
        ),
    )
    monkeypatch.setattr(trainer_module, "init_wandb_run", lambda *_args, **_kwargs: fake_run)
    monkeypatch.setattr(
        distributed_module,
        "gather_object",
        lambda local_keys: [list(local_keys), sorted(remote_activation_trace_stats)],
    )
    monkeypatch.setattr(
        distributed_module,
        "broadcast_object_list",
        lambda object_list, from_process=0: object_list,
    )
    cfg = _classification_cfg(tmp_path)
    cfg.runtime.trace_activations = True
    cfg.logging.use_wandb = True

    result = trainer_module.train(cfg)

    gradient_history = [
        json.loads(line)
        for line in (result.output_dir / "gradient_history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected_activation_norms = {
        "post_feature_encoder": 8.0,
        "post_column_encoder": 16.0,
        "post_row_pool": 24.0,
        "post_context_encoder": 32.0,
    }
    assert gradient_history[0]["activation_norms"] == pytest.approx(expected_activation_norms)
    train_payload = next(
        payload
        for payload, step in fake_run.logged
        if step == 1 and "train/activation_norm/post_feature_encoder" in payload
    )
    assert train_payload["train/activation_norm/post_feature_encoder"] == pytest.approx(8.0)
    assert train_payload["train/activation_norm/post_context_encoder"] == pytest.approx(32.0)


def test_train_skips_activation_rank_reduction_when_tracing_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_classification_fakes(monkeypatch)
    _install_deterministic_trace_classifier(monkeypatch)
    monkeypatch.setattr(
        trainer_module,
        "build_accelerator_from_runtime",
        lambda *_args, **_kwargs: _FakeMultiProcessActivationAccelerator(
            remote_activation_trace_stats={},
        ),
    )
    monkeypatch.setattr(
        distributed_module,
        "gather_object",
        lambda *_args, **_kwargs: pytest.fail("gather_object should not run when tracing is disabled"),
    )
    monkeypatch.setattr(
        distributed_module,
        "broadcast_object_list",
        lambda *_args, **_kwargs: pytest.fail(
            "broadcast_object_list should not run when tracing is disabled"
        ),
    )
    cfg = _classification_cfg(tmp_path)
    cfg.runtime.trace_activations = False

    result = trainer_module.train(cfg)

    gradient_history = [
        json.loads(line)
        for line in (result.output_dir / "gradient_history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert "activation_norms" not in gradient_history[0]


def test_train_aggregates_activation_norms_across_grad_accum_with_exact_trace_sizes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_classification_fakes(monkeypatch)
    _install_uneven_trace_classifier(monkeypatch)
    monkeypatch.setattr(
        trainer_module,
        "build_task_dataset",
        lambda *_args, **_kwargs: _VariableShapeTaskDataset(test_sizes=[1, 5]),
    )
    cfg = _classification_cfg(tmp_path)
    cfg.runtime.trace_activations = True
    cfg.runtime.grad_accum_steps = 2
    cfg.runtime.eval_every = 10

    result = trainer_module.train(cfg)

    gradient_history = [
        json.loads(line)
        for line in (result.output_dir / "gradient_history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    expected = math.sqrt(((4 * 1.0) + (12 * 100.0)) / 16.0)
    assert gradient_history[0]["activation_norms"]["post_feature_encoder"] == pytest.approx(expected)


def test_train_trace_activations_requires_raw_stats_for_grad_accum(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_classification_fakes(monkeypatch)
    _install_legacy_trace_classifier(monkeypatch)
    cfg = _classification_cfg(tmp_path)
    cfg.runtime.trace_activations = True
    cfg.runtime.grad_accum_steps = 2
    cfg.runtime.eval_every = 10

    with pytest.raises(RuntimeError, match="flush_activation_trace_stats"):
        _ = trainer_module.train(cfg)


@pytest.mark.parametrize(
    ("grad_norm_value", "expected_kind"),
    [
        (float("nan"), "nan"),
        (float("inf"), "pos_inf"),
        (-float("inf"), "neg_inf"),
    ],
)
def test_train_records_non_finite_global_grad_norm_kinds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    grad_norm_value: float,
    expected_kind: str,
) -> None:
    _install_classification_fakes(monkeypatch)
    cfg = _classification_cfg(tmp_path)
    cfg.runtime.eval_every = 10
    cfg.logging.use_wandb = True
    fake_run = _FakeWandbRun()
    monkeypatch.setattr(trainer_module, "init_wandb_run", lambda *_args, **_kwargs: fake_run)
    monkeypatch.setattr(
        trainer_module,
        "normalize_grad_norm_value",
        lambda *_args, **_kwargs: grad_norm_value,
    )

    result = trainer_module.train(cfg)

    gradient_history = [
        json.loads(line)
        for line in (result.output_dir / "gradient_history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    telemetry = json.loads((result.output_dir / "telemetry.json").read_text(encoding="utf-8"))

    assert gradient_history[0]["global_grad_norm"] is None
    assert gradient_history[0]["global_grad_norm_kind"] == expected_kind
    assert telemetry["gradient_summary"]["global"]["mean_grad_norm"] is None
    assert telemetry["gradient_summary"]["global"]["max_grad_norm"] is None
    assert telemetry["gradient_summary"]["global"]["final_grad_norm"] is None
    assert telemetry["gradient_summary"]["non_finite_global_grad_norm_counts"] == {
        "nan": 1 if expected_kind == "nan" else 0,
        "pos_inf": 1 if expected_kind == "pos_inf" else 0,
        "neg_inf": 1 if expected_kind == "neg_inf" else 0,
    }
    assert telemetry["gradient_summary"]["final_global_grad_norm_kind"] == expected_kind
    assert fake_run.summary["gradient_summary/final_global_grad_norm_kind"] == expected_kind
    assert (
        fake_run.summary[f"gradient_summary/non_finite_global_grad_norm_counts/{expected_kind}"]
        == 1
    )


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


def test_evaluate_checkpoint_rejects_regression_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    cfg = _classification_cfg(tmp_path)
    checkpoint = tmp_path / "tiny_reg.pt"
    torch.save({"model": {}, "config": {"task": "regression", "model": {}}}, checkpoint)
    cfg.eval.checkpoint = str(checkpoint)

    with pytest.raises(RuntimeError, match="classification checkpoints"):
        _ = evaluate_module.evaluate_checkpoint(cfg)
