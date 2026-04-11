from __future__ import annotations

from contextlib import nullcontext
import json
import math
from pathlib import Path
from types import SimpleNamespace

from omegaconf import OmegaConf
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
from torch import nn
from torch.utils.data import Dataset

import tab_foundry.training.evaluate as evaluate_module
import tab_foundry.training.artifacts as training_artifacts_module
import tab_foundry.training.distributed as distributed_module
import tab_foundry.training.runtime as training_runtime_module
import tab_foundry.training.trainer as trainer_module
import tab_foundry.training.trainer_loop as trainer_loop_module
import tab_foundry.training.trainer_metrics as trainer_metrics_module
from tab_foundry.config import compose_config
from tab_foundry.model.outputs import ClassificationOutput
from tab_foundry.training.optimizer import OptimizerSelection
from tab_foundry.training.schedule import build_stage_configs
from tab_foundry.types import TaskBatch

from tests.support.manifest_and_dataset_cases import (
    _classification_arrays,
    _classification_metadata,
    _write_packed_shard,
)
from tests.support.task_batching import write_task_batch_manifest_from_specs


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


class _GradAccumFakeAccelerator(_FakeAccelerator):
    def __init__(self, *, gradient_accumulation_steps: int) -> None:
        super().__init__()
        self.gradient_accumulation_steps = int(gradient_accumulation_steps)

    def backward(self, loss: torch.Tensor) -> None:
        (loss / float(self.gradient_accumulation_steps)).backward()


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


class _FakeMultiProcessNanGuardAccelerator(_FakeAccelerator):
    def __init__(self, *, remote_nan_detected: bool) -> None:
        super().__init__()
        self.num_processes = 2
        self.remote_nan_detected = bool(remote_nan_detected)

    def reduce(self, tensor: torch.Tensor, reduction: str = "sum") -> torch.Tensor:
        if reduction != "sum":
            raise ValueError("only sum reduction is supported in fake accelerator")
        if int(tensor.numel()) == 1 and tensor.dtype == torch.int64:
            remote_tensor = torch.tensor(
                1 if self.remote_nan_detected else 0,
                device=tensor.device,
                dtype=tensor.dtype,
            )
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
            metadata={"dataset_index": index, "feature_types": ["floating"] * 4},
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
            metadata={"dataset_index": index, "feature_types": ["floating"] * 4},
            num_classes=3,
        )


class _TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, batch: TaskBatch) -> ClassificationOutput:
        return ClassificationOutput(logits=self.linear(batch.x_test), num_classes=3)


class _TaskBatchAwareTinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, batch: TaskBatch) -> ClassificationOutput:
        logits = self.linear(batch.x_test.to(torch.float32))
        if logits.ndim == 3:
            logits = logits.reshape(int(logits.shape[0]) * int(logits.shape[1]), int(logits.shape[2]))
        return ClassificationOutput(logits=logits, num_classes=3)


class _MetricWeightingClassifier(nn.Module):
    def load_state_dict(self, _state: object) -> None:
        return None

    def forward(self, batch: TaskBatch) -> ClassificationOutput:
        n_rows = int(batch.y_test.reshape(-1).shape[0])
        return ClassificationOutput(
            logits=torch.zeros((n_rows, 3), dtype=torch.float32),
            num_classes=3,
        )


def _weighted_metric_batch(
    *,
    task_batch_size_actual: int,
    task_batch_size_requested: int,
) -> TaskBatch:
    mode = "singleton"
    if task_batch_size_requested > 1 and task_batch_size_actual == 1:
        mode = "singleton_fallback"
    elif task_batch_size_actual > 1:
        mode = "batched"
    if task_batch_size_actual > 1:
        return TaskBatch(
            x_train=torch.zeros((task_batch_size_actual, 6, 4), dtype=torch.float32),
            y_train=torch.zeros((task_batch_size_actual, 6), dtype=torch.int64),
            x_test=torch.zeros((task_batch_size_actual, 3, 4), dtype=torch.float32),
            y_test=torch.zeros((task_batch_size_actual, 3), dtype=torch.int64),
            metadata={
                "task_members": [{} for _ in range(task_batch_size_actual)],
                "task_batch_size_requested": task_batch_size_requested,
                "task_batch_size_actual": task_batch_size_actual,
                "task_batch_signature": "6x3x4x3",
                "task_batch_mode": mode,
            },
            num_classes=3,
        )
    return TaskBatch(
        x_train=torch.zeros((6, 4), dtype=torch.float32),
        y_train=torch.zeros((6,), dtype=torch.int64),
        x_test=torch.zeros((3, 4), dtype=torch.float32),
        y_test=torch.zeros((3,), dtype=torch.int64),
        metadata={
            "task_members": [{}],
            "task_batch_size_requested": task_batch_size_requested,
            "task_batch_size_actual": task_batch_size_actual,
            "task_batch_signature": "6x3x4x3",
            "task_batch_mode": mode,
        },
        num_classes=3,
    )


def _gradient_weighting_batch(
    *,
    task_batch_size_actual: int,
    task_batch_size_requested: int,
    seed: int,
) -> TaskBatch:
    generator = torch.Generator().manual_seed(int(seed))
    if task_batch_size_actual > 1:
        y_test = torch.stack(
            [
                torch.tensor(
                    [(seed + offset + index) % 3 for index in range(3)],
                    dtype=torch.int64,
                )
                for offset in range(task_batch_size_actual)
            ]
        )
        return TaskBatch(
            x_train=torch.randn((task_batch_size_actual, 6, 4), generator=generator, dtype=torch.float32),
            y_train=torch.randint(0, 3, (task_batch_size_actual, 6), generator=generator, dtype=torch.int64),
            x_test=torch.randn((task_batch_size_actual, 3, 4), generator=generator, dtype=torch.float32),
            y_test=y_test,
            metadata={
                "task_members": [{} for _ in range(task_batch_size_actual)],
                "task_batch_size_requested": task_batch_size_requested,
                "task_batch_size_actual": task_batch_size_actual,
                "task_batch_signature": "6x3x4x3",
                "task_batch_mode": "batched",
            },
            num_classes=3,
        )
    return TaskBatch(
        x_train=torch.randn((6, 4), generator=generator, dtype=torch.float32),
        y_train=torch.randint(0, 3, (6,), generator=generator, dtype=torch.int64),
        x_test=torch.randn((3, 4), generator=generator, dtype=torch.float32),
        y_test=torch.tensor([(seed + index) % 3 for index in range(3)], dtype=torch.int64),
        metadata={
            "task_members": [{}],
            "task_batch_size_requested": task_batch_size_requested,
            "task_batch_size_actual": task_batch_size_actual,
            "task_batch_signature": "6x3x4x3",
            "task_batch_mode": "singleton_fallback",
        },
        num_classes=3,
    )


def _combine_task_batches(*batches: TaskBatch) -> TaskBatch:
    x_train_parts: list[torch.Tensor] = []
    y_train_parts: list[torch.Tensor] = []
    x_test_parts: list[torch.Tensor] = []
    y_test_parts: list[torch.Tensor] = []
    for batch in batches:
        x_train_parts.append(batch.x_train.unsqueeze(0) if batch.x_train.ndim == 2 else batch.x_train)
        y_train_parts.append(batch.y_train.unsqueeze(0) if batch.y_train.ndim == 1 else batch.y_train)
        x_test_parts.append(batch.x_test.unsqueeze(0) if batch.x_test.ndim == 2 else batch.x_test)
        y_test_parts.append(batch.y_test.unsqueeze(0) if batch.y_test.ndim == 1 else batch.y_test)
    task_count = sum(int(part.shape[0]) for part in x_train_parts)
    return TaskBatch(
        x_train=torch.cat(x_train_parts, dim=0),
        y_train=torch.cat(y_train_parts, dim=0),
        x_test=torch.cat(x_test_parts, dim=0),
        y_test=torch.cat(y_test_parts, dim=0),
        metadata={
            "task_members": [{} for _ in range(task_count)],
            "task_batch_size_requested": 2,
            "task_batch_size_actual": task_count,
            "task_batch_signature": "6x3x4x3",
            "task_batch_mode": "batched",
        },
        num_classes=3,
    )


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
        self.activation_checkpointing_enabled = False
        self._activation_trace: dict[str, tuple[float, int]] | None = None

    def enable_activation_checkpointing(self) -> None:
        self.activation_checkpointing_enabled = True

    def disable_activation_checkpointing(self) -> None:
        self.activation_checkpointing_enabled = False

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
    def __init__(
        self,
        *,
        entity: str = "test-entity",
        project: str = "test",
        run_id: str = "wandb-run-123",
        name: str = "test",
        mode: str = "online",
    ) -> None:
        self.logged: list[tuple[dict[str, object], int]] = []
        self.summary: dict[str, object] = {}
        self.finished = False
        self.entity = entity
        self.project = project
        self.id = run_id
        self.name = name
        self.mode = mode

    def log(self, payload: dict[str, object], *, step: int) -> None:
        self.logged.append((dict(payload), int(step)))

    def finish(self) -> None:
        self.finished = True


class _CountingOptimizer:
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        self._optimizer = optimizer
        self.param_groups = optimizer.param_groups
        self.step_calls = 0

    def __getattr__(self, name: str) -> object:
        return getattr(self._optimizer, name)

    def zero_grad(self, set_to_none: bool = False) -> None:
        self._optimizer.zero_grad(set_to_none=set_to_none)

    def step(self, closure=None):
        self.step_calls += 1
        return self._optimizer.step(closure)


class _ModeTrackingOptimizer(_CountingOptimizer):
    def __init__(self, optimizer: torch.optim.Optimizer) -> None:
        super().__init__(optimizer)
        self.events: list[str] = []

    def train(self) -> None:
        self.events.append("train")

    def eval(self) -> None:
        self.events.append("eval")


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
            "data": {"manifest_path": "unused.parquet"},
            "runtime": {
                "seed": 1,
                "num_workers": 0,
                "output_dir": str(tmp_path / "outputs"),
                "device": "cpu",
                "mixed_precision": "no",
                "grad_clip": 1.0,
                "grad_accum_steps": 1,
                "activation_checkpointing": False,
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


def _write_task_batch_manifest(tmp_path: Path) -> Path:
    manifest_data_root = tmp_path / "manifest_data"
    shard_dir = manifest_data_root / "shard_00000"
    datasets: list[dict[str, object]] = []
    split_by_dataset_index = {1: "train", 2: "train", 3: "val"}
    for dataset_index, seed in ((1, 11), (2, 13), (3, 17)):
        x_train, y_train, x_test, y_test = _classification_arrays(
            n_train=6,
            n_test=3,
            n_features=4,
            n_classes=3,
            seed=seed,
        )
        datasets.append(
            {
                "dataset_index": dataset_index,
                "x_train": x_train,
                "y_train": y_train,
                "x_test": x_test,
                "y_test": y_test,
                "feature_types": ["floating"] * 4,
                "metadata": _classification_metadata(n_features=4, n_classes=3, seed=seed),
            }
        )
    offsets = _write_packed_shard(shard_dir, datasets=datasets)
    manifest_rows: list[dict[str, object]] = []
    for dataset in datasets:
        dataset_index = int(dataset["dataset_index"])
        offset, size, digest = offsets[dataset_index]
        manifest_rows.append(
            {
                "dataset_id": f"root_a/shard_00000/dataset_{dataset_index:06d}",
                "source_root_id": "root_a",
                "source_shard_relpath": "shard_00000",
                "split": split_by_dataset_index[dataset_index],
                "task": "classification",
                "dataset_index": dataset_index,
                "train_path": "manifest_data/shard_00000/train.parquet",
                "test_path": "manifest_data/shard_00000/test.parquet",
                "catalog_path": "manifest_data/shard_00000/metadata.ndjson",
                "catalog_offset_bytes": offset,
                "catalog_size_bytes": size,
                "catalog_sha256": digest,
                "n_train": int(dataset["x_train"].shape[0]),
                "n_test": int(dataset["x_test"].shape[0]),
                "n_features": int(dataset["x_train"].shape[1]),
                "n_classes": 3,
                "seed": 1,
                "filter_mode": "deferred",
                "filter_status": "not_run",
                "filter_accepted": None,
                "missing_value_policy": "allow_any",
                "missing_value_status": "clean",
            }
        )
    manifest_path = tmp_path / "manifest.parquet"
    pq.write_table(pa.Table.from_pylist(manifest_rows), manifest_path)
    return manifest_path


def _install_classification_fakes(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_spec = SimpleNamespace(task="classification", arch="tabfoundry_simple")
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
    training_surface_record = json.loads(
        (result.output_dir / 'training_surface_record.json').read_text(encoding='utf-8')
    )

    assert result.global_step == 1
    assert result.best_checkpoint is not None
    assert result.latest_checkpoint is not None
    assert result.best_checkpoint.exists()
    assert result.latest_checkpoint.exists()
    assert (result.output_dir / "checkpoints" / "latest.pt").exists()
    assert result.metrics["best_val_loss"] >= 0.0
    assert result.metrics["final_val_loss"] >= 0.0
    assert result.metrics["max_grad_norm"] >= 0.0
    best_payload = torch.load(result.best_checkpoint, map_location="cpu", weights_only=False)
    assert "preprocessor_state" not in best_payload
    assert training_surface_record['training']['backend'] == 'manifest'


def test_train_smoke_runs_end_to_end_with_tabfoundry_sandwich(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(trainer_module, "build_task_dataset", lambda *_args, **_kwargs: _FakeTaskDataset())
    monkeypatch.setattr(
        trainer_module,
        "build_accelerator_from_runtime",
        lambda *_args, **_kwargs: _FakeAccelerator(),
    )
    monkeypatch.setattr(trainer_module, "init_wandb_run", lambda *_args, **_kwargs: None)
    cfg = _classification_cfg(tmp_path)
    cfg.model.arch = "tabfoundry_sandwich"
    cfg.model.d_icl = 16
    cfg.model.input_normalization = "train_zscore_clip"
    cfg.model.many_class_base = 4
    cfg.model.head_hidden_dim = 32
    cfg.model.sandwich_latents = 12
    cfg.model.sandwich_layers = 2
    cfg.model.sandwich_heads = 4
    cfg.model.sandwich_ff_expansion = 2

    result = trainer_module.train(cfg)
    training_surface_record = json.loads(
        (result.output_dir / "training_surface_record.json").read_text(encoding="utf-8")
    )
    gradient_history = [
        json.loads(line)
        for line in (result.output_dir / "gradient_history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    telemetry = json.loads((result.output_dir / "telemetry.json").read_text(encoding="utf-8"))

    assert result.global_step == 1
    assert result.best_checkpoint is not None
    assert result.best_checkpoint.exists()
    assert training_surface_record["model"]["arch"] == "tabfoundry_sandwich"
    assert training_surface_record["training"]["loss_surface"] == "classification"
    assert training_surface_record["model"]["architecture"]["latents"] == 12
    assert (
        training_surface_record["model"]["architecture"]["initial_input_tokens"]
        == "full_cell_plus_row_col_summary_stream"
    )
    assert (
        training_surface_record["model"]["architecture"]["repeated_input_tokens"]
        == "row_col_summary_stream"
    )
    assert (
        training_surface_record["model"]["architecture"]["label_injection"]
        == "fused_into_row_summaries_and_feature_cells"
    )
    module_names = set(gradient_history[0]["module_grad_norms"])
    assert {
        "feature_encoder",
        "y_conditioner",
        "y_role_embedding",
        "token_type_embedding",
        "pre_row_attention_blocks.0",
        "pre_column_attention_blocks.0",
        "row_summary_builder",
        "column_summary_builder",
        "perceiver_stages.0",
        "perceiver_stages.1",
        "latent_readout",
        "cell_readout",
        "test_row_pool",
        "direct_head",
    }.issubset(module_names)
    assert "gaussian_head" not in module_names
    assert "feature_encoder_vs_direct_head" in telemetry["diagnostics"]["module_balance"]


def test_train_activation_checkpointing_enables_supported_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_classification_fakes(monkeypatch)
    fake_spec = SimpleNamespace(task="classification", arch="tabfoundry_staged")
    model = _TraceableStageLocalClassifier()
    monkeypatch.setattr(trainer_module, "model_build_spec_from_mappings", lambda **_kwargs: fake_spec)
    monkeypatch.setattr(trainer_module, "build_model_from_spec", lambda _spec: model)
    cfg = _classification_cfg(tmp_path)
    cfg.runtime.activation_checkpointing = True

    result = trainer_module.train(cfg)

    assert model.activation_checkpointing_enabled is True
    assert (result.output_dir / "training_surface_record.json").exists()
    assert result.best_checkpoint is not None
    assert result.best_checkpoint.exists()


def test_train_activation_checkpointing_requires_supported_model(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_classification_fakes(monkeypatch)
    cfg = _classification_cfg(tmp_path)
    cfg.runtime.activation_checkpointing = True

    with pytest.raises(RuntimeError, match="enable_activation_checkpointing"):
        _ = trainer_module.train(cfg)


def test_train_smoke_skips_validation_loader_when_val_batches_is_zero(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_classification_fakes(monkeypatch)

    def _build_task_dataset(*_args, split: str, **_kwargs):
        if split == 'val':
            raise AssertionError('val split should be skipped when runtime.val_batches == 0')
        return _FakeTaskDataset()

    monkeypatch.setattr(trainer_module, 'build_task_dataset', _build_task_dataset)
    cfg = _classification_cfg(tmp_path)
    cfg.runtime.val_batches = 0

    result = trainer_module.train(cfg)

    assert result.global_step == 1
    assert math.isinf(result.metrics['best_val_loss'])
    assert math.isinf(result.metrics['final_val_loss'])
    assert result.latest_checkpoint is not None


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


def test_train_smoke_saves_in_loop_checkpoints_in_eval_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_classification_fakes(monkeypatch)
    mode_tracking_optimizer: _ModeTrackingOptimizer | None = None
    save_events: list[tuple[str, str | None]] = []
    original_save_checkpoint = training_artifacts_module.save_checkpoint

    def _build_mode_tracking_optimizer(model, **_kwargs):
        nonlocal mode_tracking_optimizer
        base_optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        mode_tracking_optimizer = _ModeTrackingOptimizer(base_optimizer)
        return OptimizerSelection(
            optimizers=[("schedulefree_adamw", mode_tracking_optimizer)],
            requested_name="schedulefree_adamw",
            resolved_name="schedulefree_adamw",
            fallback_reason=None,
        )

    def _record_save(path: Path, *, model_state, global_step: int, cfg) -> None:
        _ = (model_state, global_step, cfg)
        mode = None if mode_tracking_optimizer is None or not mode_tracking_optimizer.events else mode_tracking_optimizer.events[-1]
        save_events.append((path.name, mode))
        original_save_checkpoint(path, model_state=model_state, global_step=global_step, cfg=cfg)

    monkeypatch.setattr(trainer_module, "build_optimizer", _build_mode_tracking_optimizer)
    monkeypatch.setattr(training_artifacts_module, "save_checkpoint", _record_save)

    cfg = _classification_cfg(tmp_path)
    cfg.schedule.stages = [{"name": "stage1", "steps": 1, "lr_max": 1.0e-3}]
    cfg.runtime.checkpoint_every = 1

    result = trainer_module.train(cfg)

    assert result.global_step == 1
    assert mode_tracking_optimizer is not None
    assert save_events == [
        ("best.pt", "eval"),
        ("step_000001.pt", "eval"),
        ("latest_stage1.pt", "eval"),
        ("latest.pt", "eval"),
    ]
    assert mode_tracking_optimizer.events == [
        "train",
        "eval",
        "eval",
        "train",
        "eval",
        "train",
        "eval",
        "train",
        "eval",
        "train",
    ]


def test_train_smoke_saves_fallback_best_checkpoint_in_eval_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_classification_fakes(monkeypatch)
    mode_tracking_optimizer: _ModeTrackingOptimizer | None = None
    save_events: list[tuple[str, str | None]] = []
    original_save_checkpoint = training_artifacts_module.save_checkpoint

    def _build_mode_tracking_optimizer(model, **_kwargs):
        nonlocal mode_tracking_optimizer
        base_optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        mode_tracking_optimizer = _ModeTrackingOptimizer(base_optimizer)
        return OptimizerSelection(
            optimizers=[("schedulefree_adamw", mode_tracking_optimizer)],
            requested_name="schedulefree_adamw",
            resolved_name="schedulefree_adamw",
            fallback_reason=None,
        )

    def _record_save(path: Path, *, model_state, global_step: int, cfg) -> None:
        _ = (model_state, global_step, cfg)
        mode = None if mode_tracking_optimizer is None or not mode_tracking_optimizer.events else mode_tracking_optimizer.events[-1]
        save_events.append((path.name, mode))
        original_save_checkpoint(path, model_state=model_state, global_step=global_step, cfg=cfg)

    monkeypatch.setattr(trainer_module, "build_optimizer", _build_mode_tracking_optimizer)
    monkeypatch.setattr(training_artifacts_module, "save_checkpoint", _record_save)

    cfg = _classification_cfg(tmp_path)
    cfg.schedule.stages = [{"name": "stage1", "steps": 1, "lr_max": 1.0e-3}]
    cfg.runtime.val_batches = 0
    cfg.runtime.checkpoint_every = None

    result = trainer_module.train(cfg)

    assert result.global_step == 1
    assert mode_tracking_optimizer is not None
    assert save_events == [
        ("latest_stage1.pt", "eval"),
        ("latest.pt", "eval"),
        ("best.pt", "eval"),
    ]
    assert mode_tracking_optimizer.events == [
        "train",
        "eval",
        "train",
        "eval",
        "train",
        "eval",
    ]


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


def test_evaluate_loader_weights_metrics_by_actual_task_batch_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = [
        _weighted_metric_batch(task_batch_size_actual=2, task_batch_size_requested=2),
        _weighted_metric_batch(task_batch_size_actual=1, task_batch_size_requested=2),
    ]

    def _fake_compute(_output: object, batch: TaskBatch, *, task: str) -> tuple[torch.Tensor, dict[str, float]]:
        assert task == "classification"
        actual_task_count = int(batch.metadata["task_batch_size_actual"])
        if actual_task_count == 2:
            return torch.tensor(1.0), {"acc": 0.2}
        return torch.tensor(3.0), {"acc": 0.8}

    monkeypatch.setattr(trainer_metrics_module, "_compute_loss_and_metrics", _fake_compute)

    metrics = trainer_metrics_module._evaluate_loader(
        _MetricWeightingClassifier(),
        batches,
        accelerator=_FakeAccelerator(),
        task="classification",
        max_batches=8,
    )

    assert metrics["val_loss"] == pytest.approx(5.0 / 3.0)
    assert metrics["acc"] == pytest.approx(0.4)


def test_evaluate_loader_caps_by_task_count_without_overshooting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = [
        _weighted_metric_batch(task_batch_size_actual=2, task_batch_size_requested=2),
        _weighted_metric_batch(task_batch_size_actual=2, task_batch_size_requested=2),
        _weighted_metric_batch(task_batch_size_actual=1, task_batch_size_requested=2),
    ]
    calls: list[int] = []

    def _fake_compute(_output: object, batch: TaskBatch, *, task: str) -> tuple[torch.Tensor, dict[str, float]]:
        assert task == "classification"
        calls.append(int(batch.metadata["task_batch_size_actual"]))
        return torch.tensor(1.0), {"acc": 0.25}

    monkeypatch.setattr(trainer_metrics_module, "_compute_loss_and_metrics", _fake_compute)

    metrics = trainer_metrics_module._evaluate_loader(
        _MetricWeightingClassifier(),
        batches,
        accelerator=_FakeAccelerator(),
        task="classification",
        max_batches=3,
    )

    assert calls == [2]
    assert metrics["val_loss"] == pytest.approx(1.0)
    assert metrics["acc"] == pytest.approx(0.25)


def test_evaluate_loader_processes_first_task_batch_even_when_it_exceeds_cap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    batches = [
        _weighted_metric_batch(task_batch_size_actual=2, task_batch_size_requested=2),
        _weighted_metric_batch(task_batch_size_actual=1, task_batch_size_requested=2),
    ]
    calls: list[int] = []

    def _fake_compute(_output: object, batch: TaskBatch, *, task: str) -> tuple[torch.Tensor, dict[str, float]]:
        assert task == "classification"
        calls.append(int(batch.metadata["task_batch_size_actual"]))
        return torch.tensor(2.0), {"acc": 0.5}

    monkeypatch.setattr(trainer_metrics_module, "_compute_loss_and_metrics", _fake_compute)

    metrics = trainer_metrics_module._evaluate_loader(
        _MetricWeightingClassifier(),
        batches,
        accelerator=_FakeAccelerator(),
        task="classification",
        max_batches=1,
    )

    assert calls == [2]
    assert metrics["val_loss"] == pytest.approx(2.0)
    assert metrics["acc"] == pytest.approx(0.5)


def test_evaluate_checkpoint_weights_metrics_by_actual_task_batch_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_classification_fakes(monkeypatch)
    batches = [
        _weighted_metric_batch(task_batch_size_actual=2, task_batch_size_requested=2),
        _weighted_metric_batch(task_batch_size_actual=1, task_batch_size_requested=2),
    ]

    def _fake_load(_path: Path, **_kwargs: object) -> dict[str, object]:
        return {
            "model": {},
            "config": {
                "task": "classification",
                "model": {},
                "training": {"task_batch_size": 2},
                "runtime": {"seed": 77},
            },
        }

    def _fake_compute(_output: object, batch: TaskBatch, *, task: str) -> tuple[torch.Tensor, dict[str, float]]:
        assert task == "classification"
        actual_task_count = int(batch.metadata["task_batch_size_actual"])
        if actual_task_count == 2:
            return torch.tensor(1.0), {"acc": 0.2}
        return torch.tensor(3.0), {"acc": 0.8}

    monkeypatch.setattr(evaluate_module.torch, "load", _fake_load)
    monkeypatch.setattr(evaluate_module, "build_task_dataset", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(evaluate_module, "build_task_loader", lambda *_args, **_kwargs: batches)
    monkeypatch.setattr(evaluate_module, "build_model_from_spec", lambda _spec: _MetricWeightingClassifier())
    monkeypatch.setattr(trainer_metrics_module, "_compute_loss_and_metrics", _fake_compute)

    cfg = _classification_cfg(tmp_path)
    cfg.eval.checkpoint = str(tmp_path / "weighted_eval.pt")
    cfg.eval.max_batches = 8

    result = evaluate_module.evaluate_checkpoint(cfg)

    assert result.metrics["loss"] == pytest.approx(5.0 / 3.0)
    assert result.metrics["acc"] == pytest.approx(0.4)


def test_evaluate_checkpoint_caps_by_task_count_without_overshooting(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_classification_fakes(monkeypatch)
    batches = [
        _weighted_metric_batch(task_batch_size_actual=2, task_batch_size_requested=2),
        _weighted_metric_batch(task_batch_size_actual=2, task_batch_size_requested=2),
        _weighted_metric_batch(task_batch_size_actual=1, task_batch_size_requested=2),
    ]
    calls: list[int] = []

    def _fake_load(_path: Path, **_kwargs: object) -> dict[str, object]:
        return {
            "model": {},
            "config": {
                "task": "classification",
                "model": {},
                "training": {"task_batch_size": 2},
                "runtime": {"seed": 77},
            },
        }

    def _fake_compute(_output: object, batch: TaskBatch, *, task: str) -> tuple[torch.Tensor, dict[str, float]]:
        assert task == "classification"
        calls.append(int(batch.metadata["task_batch_size_actual"]))
        return torch.tensor(1.0), {"acc": 0.25}

    monkeypatch.setattr(evaluate_module.torch, "load", _fake_load)
    monkeypatch.setattr(evaluate_module, "build_task_dataset", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(evaluate_module, "build_task_loader", lambda *_args, **_kwargs: batches)
    monkeypatch.setattr(evaluate_module, "build_model_from_spec", lambda _spec: _MetricWeightingClassifier())
    monkeypatch.setattr(trainer_metrics_module, "_compute_loss_and_metrics", _fake_compute)

    cfg = _classification_cfg(tmp_path)
    cfg.eval.checkpoint = str(tmp_path / "capped_eval.pt")
    cfg.eval.max_batches = 3

    result = evaluate_module.evaluate_checkpoint(cfg)

    assert calls == [2]
    assert result.metrics["loss"] == pytest.approx(1.0)
    assert result.metrics["acc"] == pytest.approx(0.25)


def test_evaluate_checkpoint_processes_first_task_batch_even_when_it_exceeds_cap(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_classification_fakes(monkeypatch)
    batches = [
        _weighted_metric_batch(task_batch_size_actual=2, task_batch_size_requested=2),
        _weighted_metric_batch(task_batch_size_actual=1, task_batch_size_requested=2),
    ]
    calls: list[int] = []

    def _fake_load(_path: Path, **_kwargs: object) -> dict[str, object]:
        return {
            "model": {},
            "config": {
                "task": "classification",
                "model": {},
                "training": {"task_batch_size": 2},
                "runtime": {"seed": 77},
            },
        }

    def _fake_compute(_output: object, batch: TaskBatch, *, task: str) -> tuple[torch.Tensor, dict[str, float]]:
        assert task == "classification"
        calls.append(int(batch.metadata["task_batch_size_actual"]))
        return torch.tensor(2.0), {"acc": 0.5}

    monkeypatch.setattr(evaluate_module.torch, "load", _fake_load)
    monkeypatch.setattr(evaluate_module, "build_task_dataset", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(evaluate_module, "build_task_loader", lambda *_args, **_kwargs: batches)
    monkeypatch.setattr(evaluate_module, "build_model_from_spec", lambda _spec: _MetricWeightingClassifier())
    monkeypatch.setattr(trainer_metrics_module, "_compute_loss_and_metrics", _fake_compute)

    cfg = _classification_cfg(tmp_path)
    cfg.eval.checkpoint = str(tmp_path / "first_batch_eval.pt")
    cfg.eval.max_batches = 1

    result = evaluate_module.evaluate_checkpoint(cfg)

    assert calls == [2]
    assert result.metrics["loss"] == pytest.approx(2.0)
    assert result.metrics["acc"] == pytest.approx(0.5)


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


def test_train_history_weights_microstep_metrics_by_actual_task_count(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_classification_fakes(monkeypatch)
    batches = [
        _weighted_metric_batch(task_batch_size_actual=2, task_batch_size_requested=2),
        _weighted_metric_batch(task_batch_size_actual=1, task_batch_size_requested=2),
    ]

    def _fake_compute(_output: object, batch: TaskBatch, *, task: str) -> tuple[torch.Tensor, dict[str, float]]:
        assert task == "classification"
        actual_task_count = int(batch.metadata["task_batch_size_actual"])
        if actual_task_count == 2:
            return torch.tensor(1.0, requires_grad=True), {"acc": 0.2}
        return torch.tensor(3.0, requires_grad=True), {"acc": 0.8}

    monkeypatch.setattr(trainer_module, "build_task_dataset", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(trainer_module, "build_task_loader", lambda *_args, **_kwargs: batches)
    monkeypatch.setattr(trainer_loop_module, "_compute_loss_and_metrics", _fake_compute)

    cfg = _classification_cfg(tmp_path)
    cfg.runtime.grad_accum_steps = 2
    cfg.runtime.val_batches = 0
    history_path = tmp_path / "outputs" / "weighted_train_history.jsonl"
    cfg.logging.history_jsonl_path = str(history_path)

    _ = trainer_module.train(cfg)

    records = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert records[0]["train_loss"] == pytest.approx(5.0 / 3.0)
    assert records[0]["train_acc"] == pytest.approx(0.4)


def test_train_task_batch_grad_accum_matches_all_tasks_reference_update(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    train_batches = [
        _gradient_weighting_batch(task_batch_size_actual=2, task_batch_size_requested=2, seed=11),
        _gradient_weighting_batch(task_batch_size_actual=1, task_batch_size_requested=2, seed=23),
    ]
    torch.manual_seed(7)
    trained_model = _TaskBatchAwareTinyClassifier()
    reference_model = _TaskBatchAwareTinyClassifier()
    reference_model.load_state_dict(trained_model.state_dict())

    monkeypatch.setattr(trainer_module, "build_task_dataset", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(trainer_module, "build_task_loader", lambda *_args, **_kwargs: train_batches)
    monkeypatch.setattr(trainer_module, "build_model_from_spec", lambda _spec: trained_model)
    monkeypatch.setattr(
        trainer_module,
        "build_accelerator_from_runtime",
        lambda *_args, **kwargs: _GradAccumFakeAccelerator(
            gradient_accumulation_steps=int(kwargs.get("grad_accum_steps_override", 1) or 1),
        ),
    )
    monkeypatch.setattr(trainer_module, "init_wandb_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "build_optimizer",
        lambda model, **_kwargs: OptimizerSelection(
            optimizers=[("sgd", torch.optim.SGD(model.parameters(), lr=0.1))],
            requested_name="sgd",
            resolved_name="sgd",
        ),
    )

    cfg = _classification_cfg(tmp_path)
    cfg.training = {"task_batch_size": 2}
    cfg.runtime.grad_accum_steps = 2
    cfg.runtime.val_batches = 0

    result = trainer_module.train(cfg)

    assert result.global_step == 1

    combined_batch = _combine_task_batches(*train_batches)
    reference_optimizer = torch.optim.SGD(reference_model.parameters(), lr=0.1)
    reference_optimizer.zero_grad(set_to_none=True)
    reference_output = reference_model(combined_batch)
    reference_loss, _ = trainer_metrics_module._compute_loss_and_metrics(
        reference_output,
        combined_batch,
        task="classification",
    )
    reference_loss.backward()
    reference_optimizer.step()

    for trained_param, reference_param in zip(
        trained_model.parameters(),
        reference_model.parameters(),
        strict=True,
    ):
        assert torch.allclose(trained_param, reference_param, atol=1.0e-6)


def test_train_grad_accum_streams_move_and_forward_in_lockstep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[str] = []
    train_batches = [
        _weighted_metric_batch(task_batch_size_actual=1, task_batch_size_requested=1),
        _weighted_metric_batch(task_batch_size_actual=1, task_batch_size_requested=1),
    ]

    class _EventOrderClassifier(_TaskBatchAwareTinyClassifier):
        def forward(self, batch: TaskBatch) -> ClassificationOutput:
            events.append("forward")
            return super().forward(batch)

    monkeypatch.setattr(trainer_module, "build_task_dataset", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(trainer_module, "build_task_loader", lambda *_args, **_kwargs: train_batches)
    monkeypatch.setattr(trainer_module, "build_model_from_spec", lambda _spec: _EventOrderClassifier())
    monkeypatch.setattr(
        trainer_module,
        "build_accelerator_from_runtime",
        lambda *_args, **kwargs: _GradAccumFakeAccelerator(
            gradient_accumulation_steps=int(kwargs.get("grad_accum_steps_override", 1) or 1),
        ),
    )
    monkeypatch.setattr(trainer_module, "init_wandb_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "build_optimizer",
        lambda model, **_kwargs: OptimizerSelection(
            optimizers=[("sgd", torch.optim.SGD(model.parameters(), lr=0.1))],
            requested_name="sgd",
            resolved_name="sgd",
        ),
    )
    monkeypatch.setattr(
        trainer_loop_module,
        "move_batch",
        lambda batch, _device, **_kwargs: events.append("move") or batch,
    )

    cfg = _classification_cfg(tmp_path)
    cfg.runtime.grad_accum_steps = 2
    cfg.runtime.val_batches = 0

    result = trainer_module.train(cfg)

    assert result.global_step == 1
    assert events == ["move", "forward", "move", "forward"]


def test_train_smoke_task_batching_manifest_loader_emits_batching_telemetry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = _write_task_batch_manifest(tmp_path)
    monkeypatch.setattr(
        trainer_module,
        "build_accelerator_from_runtime",
        lambda *_args, **_kwargs: _FakeAccelerator(),
    )
    monkeypatch.setattr(trainer_module, "init_wandb_run", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        trainer_module,
        "build_model_from_spec",
        lambda _spec: _TaskBatchAwareTinyClassifier(),
    )

    cfg = _classification_cfg(tmp_path)
    cfg.model.arch = "tabfoundry_simple"
    cfg.data.manifest_path = str(manifest_path)
    cfg.training = {"task_batch_size": 2}
    history_path = tmp_path / "outputs" / "task_batch_history.jsonl"
    cfg.logging.history_jsonl_path = str(history_path)

    result = trainer_module.train(cfg)

    records = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    telemetry = json.loads((result.output_dir / "telemetry.json").read_text(encoding="utf-8"))

    assert result.global_step == 1
    assert records[0]["task_batch_size_requested"] == 2
    assert records[0]["task_batch_size_actual"] == 2
    assert records[0]["task_batch_batched_count"] == 1
    assert records[0]["task_batch_singleton_fallback_count"] == 0
    assert records[0]["task_batch_singleton_fallback_fraction"] == 0.0
    assert records[0]["task_batch_signature_counts"] == {"6x3x4x3": 1}
    assert telemetry["diagnostics"]["task_batching"] == {
        "record_count": 1,
        "requested_task_batch_sizes": [2],
        "actual_task_batch_size_counts": {"2": 1},
        "batched_step_count": 1,
        "singleton_fallback_count": 0,
        "singleton_fallback_fraction": 0.0,
        "signature_counts": {"6x3x4x3": 1},
    }


def test_train_aggregates_task_batching_telemetry_across_grad_accum_steps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_classification_fakes(monkeypatch)
    batches = [
        _weighted_metric_batch(task_batch_size_actual=2, task_batch_size_requested=2),
        _weighted_metric_batch(task_batch_size_actual=1, task_batch_size_requested=2),
    ]

    def _fake_compute(
        _output: object,
        batch: TaskBatch,
        *,
        task: str,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        assert task == "classification"
        actual_task_count = int(batch.metadata["task_batch_size_actual"])
        return torch.tensor(float(actual_task_count), requires_grad=True), {"acc": 0.5}

    monkeypatch.setattr(trainer_module, "build_task_loader", lambda *_args, **_kwargs: batches)
    monkeypatch.setattr(
        trainer_module,
        "build_model_from_spec",
        lambda _spec: _TaskBatchAwareTinyClassifier(),
    )
    monkeypatch.setattr(trainer_loop_module, "_compute_loss_and_metrics", _fake_compute)

    cfg = _classification_cfg(tmp_path)
    cfg.training = {"task_batch_size": 2}
    cfg.runtime.grad_accum_steps = 2
    cfg.runtime.val_batches = 0
    history_path = tmp_path / "outputs" / "accumulated_task_batch_history.jsonl"
    cfg.logging.history_jsonl_path = str(history_path)

    result = trainer_module.train(cfg)

    history_records = [
        json.loads(line)
        for line in history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    gradient_records = [
        json.loads(line)
        for line in (result.output_dir / "gradient_history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    telemetry = json.loads((result.output_dir / "telemetry.json").read_text(encoding="utf-8"))

    assert history_records[0]["task_batch_size_requested"] == 2
    assert history_records[0]["task_batch_size_actual"] == 3
    assert history_records[0]["task_batch_batched_count"] == 1
    assert history_records[0]["task_batch_singleton_fallback_count"] == 1
    assert history_records[0]["task_batch_singleton_fallback_fraction"] == pytest.approx(0.5)
    assert history_records[0]["task_batch_signature_counts"] == {"6x3x4x3": 2}
    assert gradient_records[0]["task_batch_size_actual"] == 3
    assert gradient_records[0]["task_batch_batched_count"] == 1
    assert gradient_records[0]["task_batch_singleton_fallback_count"] == 1
    assert gradient_records[0]["task_batch_singleton_fallback_fraction"] == pytest.approx(0.5)
    assert gradient_records[0]["task_batch_signature_counts"] == {"6x3x4x3": 2}
    assert telemetry["diagnostics"]["task_batching"] == {
        "record_count": 1,
        "requested_task_batch_sizes": [2],
        "actual_task_batch_size_counts": {"3": 1},
        "batched_step_count": 1,
        "singleton_fallback_count": 1,
        "singleton_fallback_fraction": 0.5,
        "signature_counts": {"6x3x4x3": 2},
    }


def test_train_disables_even_batch_padding_for_task_batching(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _capture_accelerator(_runtime: object, **kwargs: object) -> None:
        captured["dataloader_even_batches_override"] = kwargs.get("dataloader_even_batches_override")
        raise RuntimeError("stop_after_accelerator")

    monkeypatch.setattr(trainer_module, "build_accelerator_from_runtime", _capture_accelerator)

    cfg = _classification_cfg(tmp_path)
    cfg.training = {"task_batch_size": 2}

    with pytest.raises(RuntimeError, match="stop_after_accelerator"):
        _ = trainer_module.train(cfg)

    assert captured["dataloader_even_batches_override"] is False


def test_train_rejects_task_batching_for_non_manifest_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_classification_fakes(monkeypatch)
    cfg = _classification_cfg(tmp_path)
    cfg.training = {"task_batch_size": 2}

    with pytest.raises(RuntimeError, match="requires a manifest-backed PackedParquetTaskDataset"):
        _ = trainer_module.train(cfg)


def test_train_rejects_explicit_mps_runtime_device(tmp_path: Path) -> None:
    cfg = _classification_cfg(tmp_path)
    cfg.runtime.device = "mps"

    with pytest.raises(ValueError, match="MPS is unsupported for training and checkpoint evaluation"):
        _ = trainer_module.train(cfg)


def test_train_rejects_auto_runtime_device_when_it_resolves_to_mps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(training_runtime_module, "resolve_device", lambda _device: "mps")
    cfg = _classification_cfg(tmp_path)
    cfg.runtime.device = "auto"

    with pytest.raises(ValueError, match="runtime.device='auto' resolved to 'mps'"):
        _ = trainer_module.train(cfg)


@pytest.mark.parametrize(
    ("experiment_name", "expected_runtime"),
    [
        (
            "cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_v1",
            {
                "num_workers": 0,
                "loader_pin_memory": False,
                "loader_persistent_workers": False,
                "loader_prefetch_factor": None,
                "non_blocking_device_transfer": False,
            },
        ),
        (
            "cls_benchmark_sandwich_classification_evolution_tf_rd_022_policy_train_speed_v1",
            {
                "num_workers": 2,
                "loader_pin_memory": True,
                "loader_persistent_workers": True,
                "loader_prefetch_factor": 2,
                "non_blocking_device_transfer": True,
            },
        ),
    ],
)
def test_tf_rd_022_experiment_training_route_uses_manifest_surface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    experiment_name: str,
    expected_runtime: dict[str, object],
) -> None:
    captured: dict[str, object] = {}

    def _capture_dataset(
        data_cfg: object,
        *,
        split: str,
        task: str,
        seed: int,
        preprocessing_cfg: object = None,
    ) -> object:
        del split, task, seed, preprocessing_cfg
        captured["data_cfg"] = data_cfg
        return object()

    def _capture_loader(
        _dataset: object,
        *,
        shuffle: bool,
        num_workers: int,
        seed: int,
        task_batch_size: int,
        pin_memory: bool,
        persistent_workers: bool,
        prefetch_factor: int | None,
    ) -> object:
        del shuffle, seed, task_batch_size
        captured["num_workers"] = num_workers
        captured["pin_memory"] = pin_memory
        captured["persistent_workers"] = persistent_workers
        captured["prefetch_factor"] = prefetch_factor
        raise RuntimeError("stop_after_loader")

    monkeypatch.setattr(trainer_module, "build_task_dataset", _capture_dataset)
    monkeypatch.setattr(trainer_module, "validate_task_batching_support", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(trainer_module, "build_task_loader", _capture_loader)

    cfg = compose_config([f"experiment={experiment_name}", "logging.use_wandb=false"])
    cfg.runtime.output_dir = str(tmp_path / experiment_name)

    with pytest.raises(RuntimeError, match="stop_after_loader"):
        _ = trainer_module.train(cfg)

    data_cfg = captured["data_cfg"]
    assert str(data_cfg.source) == "manifest"
    assert str(data_cfg.surface_label) == "tf_rd_010_dagzoo_medium_control"
    assert str(data_cfg.corpus_ref) == "tf_rd_010_dagzoo_medium_control_curated_v5"
    assert "legacy_prior" not in cfg
    assert captured["num_workers"] == expected_runtime["num_workers"]
    assert captured["pin_memory"] is expected_runtime["loader_pin_memory"]
    assert captured["persistent_workers"] is expected_runtime["loader_persistent_workers"]
    assert captured["prefetch_factor"] == expected_runtime["loader_prefetch_factor"]


def test_train_rejects_tensor_batched_true_many_class_surface_before_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = write_task_batch_manifest_from_specs(
        tmp_path,
        task_specs=[
            {"dataset_index": 1, "split": "train", "n_classes": 6, "seed": 11},
            {"dataset_index": 2, "split": "train", "n_classes": 6, "seed": 13},
        ],
    )
    monkeypatch.setattr(
        trainer_module,
        "build_accelerator_from_runtime",
        lambda *_args, **_kwargs: _FakeAccelerator(),
    )
    monkeypatch.setattr(
        trainer_module,
        "build_task_loader",
        lambda *_args, **_kwargs: pytest.fail("preflight should reject before loader construction"),
    )

    cfg = _classification_cfg(tmp_path)
    cfg.data.manifest_path = str(manifest_path)
    cfg.model = {
        "arch": "tabfoundry_staged",
        "stage": "many_class",
        "many_class_base": 4,
        "input_normalization": "none",
    }
    cfg.training = {"task_batch_size": 2}
    cfg.runtime.val_batches = 0

    with pytest.raises(RuntimeError, match="tensor-batched true-many-class execution is deferred"):
        _ = trainer_module.train(cfg)


def test_train_allows_task_batching_for_low_class_many_class_surface(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_classification_fakes(monkeypatch)
    monkeypatch.setattr(
        trainer_module,
        "model_build_spec_from_mappings",
        lambda **_kwargs: SimpleNamespace(task="classification", arch="tabfoundry_staged"),
    )
    monkeypatch.setattr(
        trainer_module,
        "build_model_from_spec",
        lambda _spec: _TaskBatchAwareTinyClassifier(),
    )
    monkeypatch.setattr(
        trainer_module,
        "build_task_loader",
        lambda *_args, **_kwargs: [
            _weighted_metric_batch(task_batch_size_actual=2, task_batch_size_requested=2),
        ],
    )
    cfg = _classification_cfg(tmp_path)
    cfg.model = {
        "arch": "tabfoundry_staged",
        "stage": "many_class",
        "many_class_base": 4,
        "input_normalization": "none",
    }
    cfg.training = {"task_batch_size": 2}
    cfg.runtime.val_batches = 0

    result = trainer_module.train(cfg)

    assert result.global_step == 1


def test_train_allows_singleton_true_many_class_fallback_preflight(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = write_task_batch_manifest_from_specs(
        tmp_path,
        task_specs=[
            {"dataset_index": 1, "split": "train", "n_classes": 6, "seed": 11},
            {"dataset_index": 2, "split": "train", "n_classes": 3, "seed": 13},
        ],
    )
    monkeypatch.setattr(
        trainer_module,
        "build_accelerator_from_runtime",
        lambda *_args, **_kwargs: _FakeAccelerator(),
    )

    def _stop_after_loader(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("stop_after_loader")

    monkeypatch.setattr(trainer_module, "build_task_loader", _stop_after_loader)

    cfg = _classification_cfg(tmp_path)
    cfg.data.manifest_path = str(manifest_path)
    cfg.model = {
        "arch": "tabfoundry_staged",
        "stage": "many_class",
        "many_class_base": 4,
        "input_normalization": "none",
    }
    cfg.training = {"task_batch_size": 2}
    cfg.runtime.val_batches = 0

    with pytest.raises(RuntimeError, match="stop_after_loader"):
        _ = trainer_module.train(cfg)


def test_train_passes_loader_overlap_runtime_knobs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    monkeypatch.setattr(
        trainer_module,
        "build_accelerator_from_runtime",
        lambda *_args, **_kwargs: _FakeAccelerator(),
    )
    monkeypatch.setattr(
        trainer_module,
        "model_build_spec_from_mappings",
        lambda **_kwargs: SimpleNamespace(task="classification", arch="tabfoundry_simple"),
    )
    monkeypatch.setattr(
        trainer_module,
        "resolve_training_loss_surface",
        lambda *_args, **_kwargs: "classification",
    )
    monkeypatch.setattr(trainer_module, "build_task_dataset", lambda *_args, **_kwargs: object())
    monkeypatch.setattr(trainer_module, "validate_task_batching_support", lambda *_args, **_kwargs: None)

    def _capture_loader(
        _dataset: object,
        *,
        shuffle: bool,
        num_workers: int,
        seed: int,
        task_batch_size: int,
        pin_memory: bool,
        persistent_workers: bool,
        prefetch_factor: int | None,
    ) -> None:
        captured["shuffle"] = shuffle
        captured["num_workers"] = num_workers
        captured["seed"] = seed
        captured["task_batch_size"] = task_batch_size
        captured["pin_memory"] = pin_memory
        captured["persistent_workers"] = persistent_workers
        captured["prefetch_factor"] = prefetch_factor
        raise RuntimeError("stop_after_loader")

    monkeypatch.setattr(trainer_module, "build_task_loader", _capture_loader)

    cfg = _classification_cfg(tmp_path)
    cfg.runtime.num_workers = 2
    cfg.runtime.loader_pin_memory = True
    cfg.runtime.loader_persistent_workers = True
    cfg.runtime.loader_prefetch_factor = 2
    cfg.runtime.val_batches = 0

    with pytest.raises(RuntimeError, match="stop_after_loader"):
        _ = trainer_module.train(cfg)

    assert captured == {
        "shuffle": True,
        "num_workers": 2,
        "seed": 1,
        "task_batch_size": 1,
        "pin_memory": True,
        "persistent_workers": True,
        "prefetch_factor": 2,
    }


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
    cfg.model = {
        "arch": "tabfoundry_staged",
        "stage": "row_cls_pool",
        "input_normalization": "train_zscore_clip",
        "tfrow_n_heads": 2,
        "tfrow_n_layers": 1,
        "tfrow_cls_tokens": 3,
        "tfrow_norm": "layernorm",
        "feature_group_size": 1,
        "many_class_base": 2,
    }
    cfg.logging.use_wandb = True
    cfg.schedule.stages = [{"name": "stage1", "steps": 2, "lr_max": 1.0e-3}]
    fake_run = _FakeWandbRun()
    monkeypatch.setattr(trainer_module, "init_wandb_run", lambda *_args, **_kwargs: fake_run)

    result = trainer_module.train(cfg)
    telemetry = json.loads((result.output_dir / "telemetry.json").read_text(encoding="utf-8"))

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
    assert fake_run.summary["runtime_summary/non_train_overhead_seconds"] >= 0.0
    assert fake_run.summary["runtime_summary/throughput_examples_per_second"] > 0.0
    assert fake_run.summary["runtime_summary/throughput_tokens_per_second"] > 0.0
    assert fake_run.summary["hardware_summary/device_type"] == "cpu"
    assert fake_run.summary["hardware_summary/gpu_class"] == "cpu"
    assert fake_run.summary["regime_budget/tokens_seen"] > 0
    assert fake_run.summary["regime_budget/token_budget"] == fake_run.summary["regime_budget/tokens_seen"]
    assert fake_run.summary["regime_budget/tokens_per_step"] > 0.0
    assert (
        fake_run.summary["regime_budget/objective_metric"]
        == "final_log_loss_at_matched_regime_budget"
    )
    assert fake_run.summary["surface/model/arch"] == "tabfoundry_staged"
    assert fake_run.summary["surface/model/module_selection/row_pool"] == "row_cls"
    assert fake_run.summary["surface/model/module_selection/context_encoder"] == "plain"
    assert fake_run.summary["surface/model/module_hyperparameters/row_pool/cls_tokens"] == 3
    assert telemetry["runtime_summary"] == {
        "peak_vram_allocated": None,
        "peak_vram_reserved": None,
        "throughput_examples_per_second": fake_run.summary["runtime_summary/throughput_examples_per_second"],
        "throughput_tokens_per_second": fake_run.summary["runtime_summary/throughput_tokens_per_second"],
        "non_train_overhead_seconds": fake_run.summary["runtime_summary/non_train_overhead_seconds"],
    }
    assert telemetry["hardware_summary"]["device_type"] == "cpu"
    assert telemetry["hardware_summary"]["hardware_profile_id"] == "cpu"
    assert telemetry["regime_budget"]["tokens_seen"] == fake_run.summary["regime_budget/tokens_seen"]
    assert telemetry["regime_budget"]["token_budget"] == telemetry["regime_budget"]["tokens_seen"]
    assert telemetry["regime_budget"]["tokens_per_step"] == fake_run.summary["regime_budget/tokens_per_step"]
    assert telemetry["regime_budget"]["objective_metric"] == "final_log_loss_at_matched_regime_budget"
    assert telemetry["wandb"] == {
        "entity": "test-entity",
        "project": "test",
        "run_id": "wandb-run-123",
        "run_name": "test",
        "mode": "online",
    }


def test_train_closes_wandb_and_writes_failure_telemetry_for_setup_errors(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_classification_fakes(monkeypatch)
    cfg = _classification_cfg(tmp_path)
    cfg.logging.use_wandb = True
    fake_run = _FakeWandbRun()
    monkeypatch.setattr(trainer_module, "init_wandb_run", lambda *_args, **_kwargs: fake_run)
    monkeypatch.setattr(
        trainer_module,
        "build_stage_configs",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("simulated setup failure")),
    )

    with pytest.raises(RuntimeError, match="simulated setup failure"):
        _ = trainer_module.train(cfg)

    telemetry_path = Path(str(cfg.runtime.output_dir)).expanduser().resolve() / "telemetry.json"
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))

    assert fake_run.finished is True
    assert telemetry["success"] is False
    assert telemetry["error"] == {
        "type": "RuntimeError",
        "message": "simulated setup failure",
    }
    assert fake_run.summary["telemetry/success"] is False
    assert fake_run.summary["error/type"] == "RuntimeError"
    assert fake_run.summary["error/message"] == "simulated setup failure"


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


def test_train_skips_optimizer_step_when_remote_rank_reports_nan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _install_classification_fakes(monkeypatch)
    fake_run = _FakeWandbRun()
    counting_optimizer: _CountingOptimizer | None = None

    def _build_counting_optimizer(model: nn.Module, **_kwargs: object) -> OptimizerSelection:
        nonlocal counting_optimizer
        base_optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=1.0e-3,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )
        counting_optimizer = _CountingOptimizer(base_optimizer)
        return OptimizerSelection(
            optimizers=[("adamw", counting_optimizer)],
            requested_name="adamw",
            resolved_name="adamw",
            fallback_reason=None,
        )

    monkeypatch.setattr(
        trainer_module,
        "build_accelerator_from_runtime",
        lambda *_args, **_kwargs: _FakeMultiProcessNanGuardAccelerator(remote_nan_detected=True),
    )
    monkeypatch.setattr(trainer_module, "build_optimizer", _build_counting_optimizer)
    monkeypatch.setattr(trainer_module, "init_wandb_run", lambda *_args, **_kwargs: fake_run)
    cfg = _classification_cfg(tmp_path)
    cfg.logging.use_wandb = True

    result = trainer_module.train(cfg)

    assert result.metrics["nan_skip_count"] == 1.0
    assert counting_optimizer is not None
    assert counting_optimizer.step_calls == 0
    assert any(
        step == 1 and payload.get("train/nan_guard_triggered") is True
        for payload, step in fake_run.logged
    )


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
    counting_optimizer: _CountingOptimizer | None = None

    def _build_counting_optimizer(model: nn.Module, **_kwargs: object) -> OptimizerSelection:
        nonlocal counting_optimizer
        base_optimizer = torch.optim.AdamW(
            [param for param in model.parameters() if param.requires_grad],
            lr=1.0e-3,
            weight_decay=0.0,
            betas=(0.9, 0.95),
        )
        counting_optimizer = _CountingOptimizer(base_optimizer)
        return OptimizerSelection(
            optimizers=[("adamw", counting_optimizer)],
            requested_name="adamw",
            resolved_name="adamw",
            fallback_reason=None,
        )

    cfg = _classification_cfg(tmp_path)
    cfg.runtime.eval_every = 10
    cfg.logging.use_wandb = True
    fake_run = _FakeWandbRun()
    monkeypatch.setattr(trainer_module, "init_wandb_run", lambda *_args, **_kwargs: fake_run)
    monkeypatch.setattr(trainer_module, "build_optimizer", _build_counting_optimizer)
    monkeypatch.setattr(
        trainer_loop_module,
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
    assert counting_optimizer is not None
    assert counting_optimizer.step_calls == 0
    assert result.metrics["mean_grad_norm"] is None
    assert result.metrics["max_grad_norm"] is None
    assert result.metrics["final_grad_norm"] is None
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
    assert any(
        step == 1 and payload.get("train/non_finite_grad_guard_triggered") is True
        for payload, step in fake_run.logged
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
