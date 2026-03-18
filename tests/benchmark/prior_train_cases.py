from __future__ import annotations

from functools import lru_cache
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

import tab_foundry.bench.checkpoint as checkpoint_module
import tab_foundry.bench.prior_train as prior_train_module
from tab_foundry.bench.nanotabpfn import evaluate_tab_foundry_run
from tab_foundry.bench.prior_dump import (
    PriorDumpBatchMissingness,
    PriorDumpNonFiniteInputError,
    PriorDumpTaskBatchReader,
)
from tab_foundry.model.architectures.tabfoundry_staged.model import TabFoundryStagedClassifier
from tab_foundry.model.architectures.tabfoundry_simple import TabFoundrySimpleClassifier
from tab_foundry.training.losses import classification_loss
from tab_foundry.training.optimizer import OptimizerSelection

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


def test_prior_dump_reader_slices_tasks_from_batch(tmp_path: Path) -> None:
    path = _write_prior_dump(
        tmp_path / "prior.h5",
        x=np.asarray(
            [
                [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0], [10.0, 11.0, 12.0]],
                [[13.0, 14.0, 15.0], [16.0, 17.0, 18.0], [19.0, 20.0, 21.0], [22.0, 23.0, 24.0]],
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
        num_features=np.asarray([2, 3], dtype=np.int64),
        num_datapoints=np.asarray([3, 4], dtype=np.int64),
        single_eval_pos=np.asarray([2, 2], dtype=np.int64),
    )

    step = next(iter(PriorDumpTaskBatchReader(path, num_steps=1, batch_size=2)))

    assert step.step_index == 1
    assert step.dataset_indices == (0, 1)
    assert step.train_test_split_index == 2
    assert step.x_batch is not None
    assert step.y_batch is not None
    assert tuple(step.x_batch.shape) == (2, 4, 3)
    assert tuple(step.y_batch.shape) == (2, 4)
    assert len(step.tasks) == 2
    task0, task1 = step.tasks
    assert task0.x_train.shape == (2, 2)
    assert task0.x_test.shape == (1, 2)
    assert task0.y_train.tolist() == [0, 1]
    assert task0.y_test.tolist() == [0]
    assert task1.x_train.shape == (2, 3)
    assert task1.x_test.shape == (2, 3)
    assert task1.y_train.tolist() == [1, 0]
    assert task1.y_test.tolist() == [1, 0]


def test_prior_dump_reader_uses_first_split_value_in_batch(tmp_path: Path) -> None:
    path = _write_prior_dump(
        tmp_path / "prior_bad_split.h5",
        x=np.zeros((2, 4, 2), dtype=np.float32),
        y=np.zeros((2, 4), dtype=np.int64),
        num_features=np.asarray([2, 2], dtype=np.int64),
        num_datapoints=np.asarray([4, 4], dtype=np.int64),
        single_eval_pos=np.asarray([2, 3], dtype=np.int64),
    )

    step = next(iter(PriorDumpTaskBatchReader(path, num_steps=1, batch_size=2)))

    assert step.train_test_split_index == 2
    assert step.x_batch is not None
    assert step.y_batch is not None
    assert tuple(step.x_batch.shape) == (2, 4, 2)
    assert tuple(step.y_batch.shape) == (2, 4)
    assert step.tasks[0].metadata["raw_single_eval_pos"] == 2
    assert step.tasks[1].metadata["raw_single_eval_pos"] == 3
    assert step.tasks[1].x_train.shape[0] == 2
    assert step.tasks[1].x_test.shape[0] == 2


def test_prior_dump_reader_rejects_non_binary_dump(tmp_path: Path) -> None:
    path = _write_prior_dump(
        tmp_path / "prior_multiclass.h5",
        x=np.zeros((1, 4, 2), dtype=np.float32),
        y=np.zeros((1, 4), dtype=np.int64),
        num_features=np.asarray([2], dtype=np.int64),
        num_datapoints=np.asarray([4], dtype=np.int64),
        single_eval_pos=np.asarray([2], dtype=np.int64),
        max_num_classes=3,
    )

    with pytest.raises(RuntimeError, match="binary classification"):
        _ = next(iter(PriorDumpTaskBatchReader(path, num_steps=1, batch_size=1)))


def test_prior_dump_reader_rejects_nan_or_inf_inputs_by_default(tmp_path: Path) -> None:
    path = _write_prior_dump(
        tmp_path / "prior_nonfinite.h5",
        x=np.asarray([[[1.0, np.nan], [2.0, 3.0], [4.0, np.inf]]], dtype=np.float32),
        y=np.asarray([[0, 1, 0]], dtype=np.int64),
        num_features=np.asarray([2], dtype=np.int64),
        num_datapoints=np.asarray([3], dtype=np.int64),
        single_eval_pos=np.asarray([2], dtype=np.int64),
    )

    with pytest.raises(PriorDumpNonFiniteInputError, match="contains NaN or Inf") as exc_info:
        _ = next(iter(PriorDumpTaskBatchReader(path, num_steps=1, batch_size=1)))
    assert exc_info.value.summary.non_finite_feature_count == 2
    assert exc_info.value.summary.non_finite_label_count == 0
    assert exc_info.value.summary.affected_dataset_indices == (0,)


def test_prior_dump_reader_rejects_nonfinite_padded_batch_cells_by_default(tmp_path: Path) -> None:
    path = _write_prior_dump(
        tmp_path / "prior_nonfinite_padding.h5",
        x=np.asarray(
            [
                [
                    [1.0, 2.0, 0.0],
                    [3.0, 4.0, np.nan],
                    [5.0, 6.0, 0.0],
                    [7.0, 8.0, 0.0],
                ],
                [
                    [9.0, 10.0, 11.0],
                    [12.0, 13.0, 14.0],
                    [15.0, 16.0, 17.0],
                    [18.0, 19.0, 20.0],
                ],
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
        num_features=np.asarray([2, 3], dtype=np.int64),
        num_datapoints=np.asarray([4, 4], dtype=np.int64),
        single_eval_pos=np.asarray([2, 2], dtype=np.int64),
    )

    with pytest.raises(PriorDumpNonFiniteInputError, match="contains NaN or Inf") as exc_info:
        _ = next(iter(PriorDumpTaskBatchReader(path, num_steps=1, batch_size=2)))
    assert exc_info.value.summary.non_finite_feature_count == 1
    assert exc_info.value.summary.non_finite_label_count == 0
    assert exc_info.value.summary.affected_dataset_indices == (0,)


def test_prior_dump_reader_reports_inf_labels_as_nonfinite(tmp_path: Path) -> None:
    path = _write_prior_dump(
        tmp_path / "prior_inf_labels.h5",
        x=np.asarray([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=np.float32),
        y=np.asarray([[0.0, np.inf, 1.0]], dtype=np.float32),
        num_features=np.asarray([2], dtype=np.int64),
        num_datapoints=np.asarray([3], dtype=np.int64),
        single_eval_pos=np.asarray([2], dtype=np.int64),
    )

    with pytest.raises(PriorDumpNonFiniteInputError, match="contains NaN or Inf") as exc_info:
        _ = next(iter(PriorDumpTaskBatchReader(path, num_steps=1, batch_size=1)))
    assert exc_info.value.summary.non_finite_feature_count == 0
    assert exc_info.value.summary.non_finite_label_count == 1
    assert exc_info.value.summary.affected_dataset_indices == (0,)


def test_prior_dump_reader_skips_nonfinite_batches_when_requested(tmp_path: Path) -> None:
    path = _write_prior_dump(
        tmp_path / "prior_skip_nonfinite.h5",
        x=np.asarray(
            [
                [[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]],
                [[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]],
            ],
            dtype=np.float32,
        ),
        y=np.asarray([[0, 1, 0], [1, 0, 1]], dtype=np.int64),
        num_features=np.asarray([2, 2], dtype=np.int64),
        num_datapoints=np.asarray([3, 3], dtype=np.int64),
        single_eval_pos=np.asarray([2, 2], dtype=np.int64),
    )
    seen: list[object] = []

    reader = PriorDumpTaskBatchReader(
        path,
        num_steps=1,
        batch_size=1,
        non_finite_policy="skip",
        on_non_finite_batch=seen.append,
    )
    step = next(iter(reader))

    assert step.step_index == 1
    assert step.dataset_indices == (1,)
    assert len(seen) == 1
    summary = seen[0]
    assert isinstance(summary, PriorDumpBatchMissingness)
    assert getattr(summary, "affected_dataset_indices") == (0,)
    assert getattr(summary, "non_finite_feature_count") == 1


def test_prior_dump_reader_skip_policy_errors_when_full_cycle_is_nonfinite(tmp_path: Path) -> None:
    path = _write_prior_dump(
        tmp_path / "prior_skip_all_bad.h5",
        x=np.asarray([[[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]]], dtype=np.float32),
        y=np.asarray([[0, 1, 0]], dtype=np.int64),
        num_features=np.asarray([2], dtype=np.int64),
        num_datapoints=np.asarray([3], dtype=np.int64),
        single_eval_pos=np.asarray([2], dtype=np.int64),
    )

    reader = PriorDumpTaskBatchReader(
        path,
        num_steps=1,
        batch_size=1,
        non_finite_policy="skip",
    )
    with pytest.raises(RuntimeError, match="yielded no valid batch during a full pass"):
        _ = next(iter(reader))


@lru_cache(maxsize=1)
def _nanotabpfn_module():
    model_path = Path("~/dev/nanoTabPFN/model.py").expanduser()
    if not model_path.exists():
        pytest.skip("local nanoTabPFN reference is not available")
    spec = importlib.util.spec_from_file_location("local_nanotabpfn_model", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load nanoTabPFN model module from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _ConstantLogitModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor([0.75, -0.25], dtype=torch.float32))

    def forward(self, *_args: object, **_kwargs: object) -> torch.Tensor:
        raise AssertionError("prior trainer should call forward_batched(), not forward()")

    def forward_batched(
        self,
        *,
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        train_test_split_index: int,
    ) -> torch.Tensor:
        _ = (y_train,)
        n_test = int(x_all.shape[1]) - train_test_split_index
        return self.bias.view(1, 1, 2).expand(int(x_all.shape[0]), n_test, 2)



class _CapturingConstantLogitModel(_ConstantLogitModel):
    def __init__(self) -> None:
        super().__init__()
        self.last_x_all: torch.Tensor | None = None

    def forward_batched(
        self,
        *,
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        train_test_split_index: int,
    ) -> torch.Tensor:
        self.last_x_all = x_all.detach().cpu().clone()
        return super().forward_batched(
            x_all=x_all,
            y_train=y_train,
            train_test_split_index=train_test_split_index,
        )


class _DeviceTrackingConstantLogitModel(_ConstantLogitModel):
    def __init__(self) -> None:
        super().__init__()
        self.to_device_types: list[str] = []

    def to(self, *args: object, **kwargs: object) -> "_DeviceTrackingConstantLogitModel":
        raw_device = kwargs.get("device")
        if raw_device is None and args:
            raw_device = args[0]
        if raw_device is not None:
            self.to_device_types.append(torch.device(raw_device).type)
        super().to(*args, **kwargs)
        return self


class _TracingConstantLogitModel(_ConstantLogitModel):
    def __init__(self) -> None:
        super().__init__()
        self._trace_enabled = False
        self._trace_step = 0

    def enable_activation_trace(self) -> None:
        self._trace_enabled = True

    def flush_activation_trace(self) -> dict[str, float] | None:
        if not self._trace_enabled:
            return None
        self._trace_step += 1
        step = float(self._trace_step)
        return {
            "post_feature_encoder": 1.0 + (0.1 * step),
            "pre_transformer": 2.0 + (0.2 * step),
        }


class _CountingOptimizer:
    def __init__(self) -> None:
        self.step_count = 0
        self.param_groups = [{"lr": 4.0e-3}]

    def zero_grad(self, set_to_none: bool = True) -> None:
        _ = set_to_none

    def step(self) -> None:
        self.step_count += 1

    def train(self) -> None:
        return None


class _ModeTrackingOptimizer(_CountingOptimizer):
    def __init__(self) -> None:
        super().__init__()
        self.events: list[str] = []

    def train(self) -> None:
        self.events.append("train")

    def eval(self) -> None:
        self.events.append("eval")


class _LrTrackingOptimizer(_CountingOptimizer):
    def __init__(self) -> None:
        super().__init__()
        self.step_lrs: list[float] = []

    def step(self) -> None:
        self.step_lrs.append(float(self.param_groups[0]["lr"]))
        super().step()


class _FakeWandbRun:
    def __init__(self) -> None:
        self.logged: list[tuple[dict[str, object], int]] = []
        self.summary: dict[str, object] = {}
        self.finished = False

    def log(self, payload: dict[str, object], *, step: int) -> None:
        self.logged.append((dict(payload), int(step)))

    def finish(self) -> None:
        self.finished = True


def _prior_cfg(
    tmp_path: Path,
    *,
    max_steps: int,
    training_cfg: dict[str, object] | None = None,
    schedule_cfg: dict[str, object] | None = None,
) -> object:
    payload: dict[str, object] = {
        "task": "classification",
        "model": {
            "arch": "tabfoundry_simple",
            "d_icl": 8,
            "input_normalization": "train_zscore_clip",
            "feature_group_size": 1,
            "many_class_train_mode": "path_nll",
            "max_mixed_radix_digits": 64,
            "norm_type": "layernorm",
            "tfcol_n_heads": 8,
            "tfcol_n_layers": 3,
            "tfcol_n_inducing": 128,
            "tfrow_n_heads": 8,
            "tfrow_n_layers": 3,
            "tfrow_cls_tokens": 4,
            "tfrow_norm": "layernorm",
            "tficl_n_heads": 2,
            "tficl_n_layers": 1,
            "tficl_ff_expansion": 2,
            "many_class_base": 2,
            "head_hidden_dim": 16,
            "use_digit_position_embed": True,
        },
        "runtime": {
            "seed": 1,
            "output_dir": str(tmp_path / "train_out"),
            "device": "cpu",
            "mixed_precision": "no",
            "grad_clip": 1.0,
            "max_steps": max_steps,
            "eval_every": 1,
            "checkpoint_every": 1,
        },
        "optimizer": {
            "name": "schedulefree_adamw",
            "require_requested": True,
            "weight_decay": 0.0,
            "min_lr": 4.0e-4,
            "betas": [0.9, 0.999],
            "muon_per_parameter_lr": False,
            "muon_lr_scale_base": 0.2,
            "muon_partition_non2d": True,
        },
        "logging": {
            "history_jsonl_path": str(tmp_path / "train_out" / "train_history.jsonl"),
        },
    }
    if training_cfg is not None:
        payload["training"] = training_cfg
    if schedule_cfg is not None:
        payload["schedule"] = schedule_cfg
    return OmegaConf.create(payload)


def _staged_prior_cfg(
    tmp_path: Path,
    *,
    max_steps: int,
    stage: str = "row_cls_pool",
    tfrow_n_layers: int = 3,
) -> object:
    cfg = _prior_cfg(tmp_path, max_steps=max_steps)
    cfg.model.arch = "tabfoundry_staged"
    cfg.model.stage = stage
    cfg.model.stage_label = stage
    cfg.model.tfrow_n_layers = tfrow_n_layers
    return cfg


def test_train_tabfoundry_simple_prior_averages_task_loss_and_steps_per_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _write_prior_dump(
        tmp_path / "prior_train.h5",
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
                [1, 0, 1, 1],
            ],
            dtype=np.int64,
        ),
        num_features=np.asarray([2, 2], dtype=np.int64),
        num_datapoints=np.asarray([4, 4], dtype=np.int64),
        single_eval_pos=np.asarray([2, 2], dtype=np.int64),
    )
    model = _ConstantLogitModel()
    optimizer = _CountingOptimizer()
    monkeypatch.setattr(prior_train_module, "build_model_from_spec", lambda _spec: model)
    monkeypatch.setattr(
        prior_train_module,
        "build_optimizer",
        lambda *args, **kwargs: OptimizerSelection(
            optimizers=[("schedulefree_adamw", optimizer)],
            requested_name="schedulefree_adamw",
            resolved_name="schedulefree_adamw",
            fallback_reason=None,
        ),
    )

    cfg = OmegaConf.create(
        {
            "task": "classification",
            "model": {
                "arch": "tabfoundry_simple",
                "d_icl": 8,
                "input_normalization": "train_zscore_clip",
                "feature_group_size": 1,
                "many_class_train_mode": "path_nll",
                "max_mixed_radix_digits": 64,
                "tfcol_n_heads": 8,
                "tfcol_n_layers": 3,
                "tfcol_n_inducing": 128,
                "tfrow_n_heads": 8,
                "tfrow_n_layers": 3,
                "tfrow_cls_tokens": 4,
                "tficl_n_heads": 2,
                "tficl_n_layers": 1,
                "tficl_ff_expansion": 2,
                "many_class_base": 2,
                "head_hidden_dim": 16,
                "use_digit_position_embed": True,
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

    expected_logits = model.forward_batched(
        x_all=torch.zeros((2, 4, 2), dtype=torch.float32),
        y_train=torch.zeros((2, 2), dtype=torch.float32),
        train_test_split_index=2,
    )
    expected_targets = torch.tensor([[0, 1], [1, 1]], dtype=torch.int64).reshape(-1)
    expected_mean_loss = float(
        classification_loss(
            expected_logits.reshape(-1, expected_logits.shape[-1]),
            expected_targets,
        ).item()
    )

    assert optimizer.step_count == 2
    assert result.global_step == 2
    history_path = tmp_path / "train_out" / "train_history.jsonl"
    assert history_path.exists()
    records = [json.loads(line) for line in history_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(records) == 2
    assert records[0]["train_loss"] == pytest.approx(expected_mean_loss, rel=1.0e-6, abs=1.0e-6)
    assert records[0]["train_loss_delta"] is None
    assert records[0]["train_loss_ema"] == pytest.approx(expected_mean_loss, rel=1.0e-6, abs=1.0e-6)
    assert records[0]["grad_clip_threshold"] == pytest.approx(1.0)
    assert isinstance(records[0]["grad_clip_triggered"], bool)
    assert (tmp_path / "train_out" / "checkpoints" / "step_000001.pt").exists()
    assert (tmp_path / "train_out" / "checkpoints" / "step_000002.pt").exists()
    assert (tmp_path / "train_out" / "checkpoints" / "latest.pt").exists()
    gradient_history_path = tmp_path / "train_out" / "gradient_history.jsonl"
    telemetry_path = tmp_path / "train_out" / "telemetry.json"
    assert gradient_history_path.exists()
    assert telemetry_path.exists()
    gradient_records = [
        json.loads(line)
        for line in gradient_history_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(gradient_records) == 2
    assert gradient_records[0]["module_grad_norms"] == {}
    telemetry = json.loads(telemetry_path.read_text(encoding="utf-8"))
    assert telemetry["success"] is True
    assert telemetry["missingness"]["prior_dump"]["non_finite_feature_count"] == 0
    assert telemetry["artifacts"]["gradient_history_jsonl"].endswith("gradient_history.jsonl")
    assert telemetry["gradient_summary"]["modules"] == {}


def test_train_tabfoundry_simple_prior_saves_checkpoints_in_eval_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _write_prior_dump(
        tmp_path / "prior_train_eval_mode.h5",
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
                [1, 0, 1, 1],
            ],
            dtype=np.int64,
        ),
        num_features=np.asarray([2, 2], dtype=np.int64),
        num_datapoints=np.asarray([4, 4], dtype=np.int64),
        single_eval_pos=np.asarray([2, 2], dtype=np.int64),
    )
    model = _ConstantLogitModel()
    optimizer = _ModeTrackingOptimizer()
    save_events: list[str] = []
    monkeypatch.setattr(prior_train_module, "build_model_from_spec", lambda _spec: model)
    monkeypatch.setattr(
        prior_train_module,
        "build_optimizer",
        lambda *args, **kwargs: OptimizerSelection(
            optimizers=[("schedulefree_adamw", optimizer)],
            requested_name="schedulefree_adamw",
            resolved_name="schedulefree_adamw",
            fallback_reason=None,
        ),
    )

    original_save_checkpoint = prior_train_module.save_checkpoint

    def _recording_save_checkpoint(path: Path, *, model_state, global_step: int, cfg) -> None:
        _ = (model_state, global_step, cfg)
        save_events.append(path.name)
        original_save_checkpoint(path, model_state=model_state, global_step=global_step, cfg=cfg)

    monkeypatch.setattr(prior_train_module, "save_checkpoint", _recording_save_checkpoint)

    cfg = OmegaConf.create(
        {
            "task": "classification",
            "model": {
                "arch": "tabfoundry_simple",
                "d_icl": 8,
                "input_normalization": "train_zscore_clip",
                "feature_group_size": 1,
                "many_class_train_mode": "path_nll",
                "max_mixed_radix_digits": 64,
                "tfcol_n_heads": 8,
                "tfcol_n_layers": 3,
                "tfcol_n_inducing": 128,
                "tfrow_n_heads": 8,
                "tfrow_n_layers": 3,
                "tfrow_cls_tokens": 4,
                "tficl_n_heads": 2,
                "tficl_n_layers": 1,
                "tficl_ff_expansion": 2,
                "many_class_base": 2,
                "head_hidden_dim": 16,
                "use_digit_position_embed": True,
            },
            "runtime": {
                "seed": 1,
                "output_dir": str(tmp_path / "train_out_eval_mode"),
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
                "betas": [0.9, 0.999],
                "muon_per_parameter_lr": False,
                "muon_lr_scale_base": 0.2,
                "muon_partition_non2d": True,
            },
            "logging": {
                "history_jsonl_path": str(tmp_path / "train_out_eval_mode" / "train_history.jsonl"),
            },
        }
    )

    _ = prior_train_module.train_tabfoundry_simple_prior(
        cfg,
        prior_dump_path=path,
        batch_size=2,
    )

    assert save_events == ["step_000001.pt", "step_000002.pt", "latest.pt"]
    assert optimizer.events == [
        "train",
        "eval",
        "train",
        "eval",
        "train",
        "eval",
    ]


def test_train_tabfoundry_simple_prior_keeps_constant_lr_when_schedule_is_disabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _write_prior_dump(
        tmp_path / "prior_constant_lr.h5",
        x=np.asarray(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]],
            dtype=np.float32,
        ),
        y=np.asarray([[0, 1, 0, 1]], dtype=np.int64),
        num_features=np.asarray([2], dtype=np.int64),
        num_datapoints=np.asarray([4], dtype=np.int64),
        single_eval_pos=np.asarray([2], dtype=np.int64),
    )
    model = _ConstantLogitModel()
    optimizer = _LrTrackingOptimizer()
    monkeypatch.setattr(prior_train_module, "build_model_from_spec", lambda _spec: model)
    monkeypatch.setattr(
        prior_train_module,
        "build_optimizer",
        lambda *args, **kwargs: OptimizerSelection(
            optimizers=[("schedulefree_adamw", optimizer)],
            requested_name="schedulefree_adamw",
            resolved_name="schedulefree_adamw",
            fallback_reason=None,
        ),
    )

    cfg = _prior_cfg(
        tmp_path,
        max_steps=2,
        training_cfg={"surface_label": "prior_constant_lr", "apply_schedule": False},
        schedule_cfg={
            "stages": [
                {
                    "name": "stage1",
                    "steps": 2,
                    "lr_max": 4.0e-3,
                    "lr_schedule": "linear",
                    "warmup_ratio": 0.5,
                }
            ]
        },
    )

    _ = prior_train_module.train_tabfoundry_simple_prior(cfg, prior_dump_path=path, batch_size=1)

    assert optimizer.step_lrs == [pytest.approx(4.0e-4), pytest.approx(4.0e-4)]


def test_train_tabfoundry_simple_prior_applies_linear_decay_schedule(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _write_prior_dump(
        tmp_path / "prior_linear_decay.h5",
        x=np.asarray(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]],
            dtype=np.float32,
        ),
        y=np.asarray([[0, 1, 0, 1]], dtype=np.int64),
        num_features=np.asarray([2], dtype=np.int64),
        num_datapoints=np.asarray([4], dtype=np.int64),
        single_eval_pos=np.asarray([2], dtype=np.int64),
    )
    model = _ConstantLogitModel()
    optimizer = _LrTrackingOptimizer()
    monkeypatch.setattr(prior_train_module, "build_model_from_spec", lambda _spec: model)
    monkeypatch.setattr(
        prior_train_module,
        "build_optimizer",
        lambda *args, **kwargs: OptimizerSelection(
            optimizers=[("schedulefree_adamw", optimizer)],
            requested_name="schedulefree_adamw",
            resolved_name="schedulefree_adamw",
            fallback_reason=None,
        ),
    )

    cfg = _prior_cfg(
        tmp_path,
        max_steps=3,
        training_cfg={"surface_label": "prior_linear_decay", "apply_schedule": True},
        schedule_cfg={
            "stages": [
                {
                    "name": "stage1",
                    "steps": 3,
                    "lr_max": 4.0e-3,
                    "lr_schedule": "linear",
                    "warmup_ratio": 0.0,
                }
            ]
        },
    )

    _ = prior_train_module.train_tabfoundry_simple_prior(cfg, prior_dump_path=path, batch_size=1)

    assert optimizer.step_lrs == [
        pytest.approx(4.0e-3),
        pytest.approx(2.2e-3),
        pytest.approx(4.0e-4),
    ]


def test_train_tabfoundry_simple_prior_applies_linear_warmup_decay_schedule(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _write_prior_dump(
        tmp_path / "prior_linear_warmup_decay.h5",
        x=np.asarray(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]],
            dtype=np.float32,
        ),
        y=np.asarray([[0, 1, 0, 1]], dtype=np.int64),
        num_features=np.asarray([2], dtype=np.int64),
        num_datapoints=np.asarray([4], dtype=np.int64),
        single_eval_pos=np.asarray([2], dtype=np.int64),
    )
    model = _ConstantLogitModel()
    optimizer = _LrTrackingOptimizer()
    monkeypatch.setattr(prior_train_module, "build_model_from_spec", lambda _spec: model)
    monkeypatch.setattr(
        prior_train_module,
        "build_optimizer",
        lambda *args, **kwargs: OptimizerSelection(
            optimizers=[("schedulefree_adamw", optimizer)],
            requested_name="schedulefree_adamw",
            resolved_name="schedulefree_adamw",
            fallback_reason=None,
        ),
    )

    cfg = _prior_cfg(
        tmp_path,
        max_steps=3,
        training_cfg={"surface_label": "prior_linear_warmup_decay", "apply_schedule": True},
        schedule_cfg={
            "stages": [
                {
                    "name": "stage1",
                    "steps": 3,
                    "lr_max": 4.0e-3,
                    "lr_schedule": "linear",
                    "warmup_ratio": 0.5,
                }
            ]
        },
    )

    _ = prior_train_module.train_tabfoundry_simple_prior(cfg, prior_dump_path=path, batch_size=1)

    assert optimizer.step_lrs == [
        pytest.approx(2.0e-3),
        pytest.approx(4.0e-3),
        pytest.approx(4.0e-4),
    ]


def test_train_tabfoundry_simple_prior_scales_lr_with_prior_dump_batch_size(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _write_prior_dump(
        tmp_path / "prior_batch_scaled_lr.h5",
        x=np.asarray(
            [[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]],
            dtype=np.float32,
        ),
        y=np.asarray([[0, 1, 0, 1]], dtype=np.int64),
        num_features=np.asarray([2], dtype=np.int64),
        num_datapoints=np.asarray([4], dtype=np.int64),
        single_eval_pos=np.asarray([2], dtype=np.int64),
    )
    model = _ConstantLogitModel()
    optimizer = _LrTrackingOptimizer()
    monkeypatch.setattr(prior_train_module, "build_model_from_spec", lambda _spec: model)
    monkeypatch.setattr(
        prior_train_module,
        "build_optimizer",
        lambda *args, **kwargs: OptimizerSelection(
            optimizers=[("schedulefree_adamw", optimizer)],
            requested_name="schedulefree_adamw",
            resolved_name="schedulefree_adamw",
            fallback_reason=None,
        ),
    )

    cfg = _prior_cfg(
        tmp_path,
        max_steps=2,
        training_cfg={
            "surface_label": "prior_linear_decay",
            "apply_schedule": True,
            "prior_dump_batch_size": 16,
            "prior_dump_lr_scale_rule": "sqrt",
            "prior_dump_batch_reference_size": 32,
        },
        schedule_cfg={
            "stages": [
                {
                    "name": "prior_dump",
                    "steps": 2,
                    "lr_max": 4.0e-3,
                    "lr_schedule": "linear",
                    "warmup_ratio": 0.0,
                }
            ]
        },
    )

    _ = prior_train_module.train_tabfoundry_simple_prior(cfg, prior_dump_path=path)

    scale = 2 ** -0.5
    assert optimizer.step_lrs == [
        pytest.approx(4.0e-3 * scale),
        pytest.approx(4.0e-4 * scale),
    ]

    training_surface_record = json.loads(
        (tmp_path / "train_out" / "training_surface_record.json").read_text(encoding="utf-8")
    )
    assert training_surface_record["training"]["prior_dump_batch_size"] == 16
    assert training_surface_record["training"]["prior_dump_lr_scale_rule"] == "sqrt"
    assert training_surface_record["training"]["prior_dump_batch_reference_size"] == 32
    assert training_surface_record["training"]["effective_lr_scale_factor"] == pytest.approx(scale)
    assert training_surface_record["training"]["optimizer_min_lr"] == pytest.approx(4.0e-4 * scale)
    assert training_surface_record["training"]["schedule_stages"][0]["lr_max"] == pytest.approx(
        4.0e-3 * scale
    )


def test_train_tabfoundry_simple_prior_matches_nanotabpfn_loss_for_one_batch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _write_prior_dump(
        tmp_path / "prior_exact.h5",
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
                [1, 0, 1, 1],
            ],
            dtype=np.float32,
        ),
        num_features=np.asarray([2, 2], dtype=np.int64),
        num_datapoints=np.asarray([4, 4], dtype=np.int64),
        single_eval_pos=np.asarray([2, 2], dtype=np.int64),
    )
    nano_module = _nanotabpfn_module()
    torch.manual_seed(0)
    nano_model = nano_module.NanoTabPFNModel(
        embedding_size=8,
        num_attention_heads=2,
        mlp_hidden_size=16,
        num_layers=1,
        num_outputs=2,
    )
    model = TabFoundrySimpleClassifier(
        d_icl=8,
        input_normalization="train_zscore_clip",
        many_class_base=2,
        tficl_n_heads=2,
        tficl_n_layers=1,
        head_hidden_dim=16,
    )
    model.load_state_dict(nano_model.state_dict(), strict=True)

    class _NoOpOptimizer:
        def __init__(self) -> None:
            self.param_groups = [{"lr": 4.0e-3}]

        def zero_grad(self, set_to_none: bool = True) -> None:
            _ = set_to_none

        def step(self) -> None:
            return None

        def train(self) -> None:
            return None

    monkeypatch.setattr(prior_train_module, "build_model_from_spec", lambda _spec: model)
    monkeypatch.setattr(
        prior_train_module,
        "build_optimizer",
        lambda *args, **kwargs: OptimizerSelection(
            optimizers=[("schedulefree_adamw", _NoOpOptimizer())],
            requested_name="schedulefree_adamw",
            resolved_name="schedulefree_adamw",
            fallback_reason=None,
        ),
    )

    cfg = OmegaConf.create(
        {
            "task": "classification",
            "model": {
                "arch": "tabfoundry_simple",
                "d_icl": 8,
                "input_normalization": "train_zscore_clip",
                "feature_group_size": 1,
                "many_class_train_mode": "path_nll",
                "max_mixed_radix_digits": 64,
                "tfcol_n_heads": 8,
                "tfcol_n_layers": 3,
                "tfcol_n_inducing": 128,
                "tfrow_n_heads": 8,
                "tfrow_n_layers": 3,
                "tfrow_cls_tokens": 4,
                "tficl_n_heads": 2,
                "tficl_n_layers": 1,
                "tficl_ff_expansion": 2,
                "many_class_base": 2,
                "head_hidden_dim": 16,
                "use_digit_position_embed": True,
            },
            "runtime": {
                "seed": 0,
                "output_dir": str(tmp_path / "train_exact"),
                "device": "cpu",
                "mixed_precision": "no",
                "grad_clip": 1.0,
                "max_steps": 1,
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
                "history_jsonl_path": str(tmp_path / "train_exact" / "train_history.jsonl"),
            },
        }
    )

    result = prior_train_module.train_tabfoundry_simple_prior(
        cfg,
        prior_dump_path=path,
        batch_size=2,
    )

    x_batch = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            [[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]],
        ],
        dtype=torch.float32,
    )
    y_batch = torch.tensor(
        [
            [0.0, 1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0, 1.0],
        ],
        dtype=torch.float32,
    )
    expected_logits = nano_model(
        (x_batch, y_batch[:, :2]),
        train_test_split_index=2,
    )
    expected_loss = float(
        classification_loss(
            expected_logits.reshape(-1, expected_logits.shape[-1]),
            y_batch[:, 2:].reshape(-1).to(torch.int64),
        ).item()
    )

    assert result.metrics["final_train_loss"] == pytest.approx(
        expected_loss,
        rel=1.0e-6,
        abs=1.0e-6,
    )
    gradient_history = [
        json.loads(line)
        for line in (tmp_path / "train_exact" / "gradient_history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert set(gradient_history[0]["module_grad_norms"]) == {
        "decoder",
        "feature_encoder",
        "target_encoder",
        "transformer_blocks.0",
    }


def test_train_tabfoundry_staged_prior_writes_staged_gradient_keys(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _write_prior_dump(
        tmp_path / "prior_staged.h5",
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
                [1, 0, 1, 1],
            ],
            dtype=np.float32,
        ),
        num_features=np.asarray([2, 2], dtype=np.int64),
        num_datapoints=np.asarray([4, 4], dtype=np.int64),
        single_eval_pos=np.asarray([2, 2], dtype=np.int64),
    )
    model = TabFoundryStagedClassifier(
        stage="nano_exact",
        d_icl=8,
        input_normalization="train_zscore_clip",
        many_class_base=2,
        tficl_n_heads=2,
        tficl_n_layers=1,
        head_hidden_dim=16,
    )
    monkeypatch.setattr(prior_train_module, "build_model_from_spec", lambda _spec: model)
    monkeypatch.setattr(
        prior_train_module,
        "build_optimizer",
        lambda *args, **kwargs: OptimizerSelection(
            optimizers=[("schedulefree_adamw", _CountingOptimizer())],
            requested_name="schedulefree_adamw",
            resolved_name="schedulefree_adamw",
            fallback_reason=None,
        ),
    )
    cfg = OmegaConf.create(
        {
            "task": "classification",
            "model": {
                "arch": "tabfoundry_staged",
                "stage": "nano_exact",
                "d_icl": 8,
                "input_normalization": "train_zscore_clip",
                "feature_group_size": 1,
                "many_class_train_mode": "path_nll",
                "max_mixed_radix_digits": 64,
                "tfcol_n_heads": 8,
                "tfcol_n_layers": 3,
                "tfcol_n_inducing": 128,
                "tfrow_n_heads": 8,
                "tfrow_n_layers": 3,
                "tfrow_cls_tokens": 4,
                "tficl_n_heads": 2,
                "tficl_n_layers": 1,
                "tficl_ff_expansion": 2,
                "many_class_base": 2,
                "head_hidden_dim": 16,
                "use_digit_position_embed": True,
            },
            "runtime": {
                "seed": 0,
                "output_dir": str(tmp_path / "train_staged"),
                "device": "cpu",
                "mixed_precision": "no",
                "grad_clip": 1.0,
                "max_steps": 1,
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
                "history_jsonl_path": str(tmp_path / "train_staged" / "train_history.jsonl"),
            },
        }
    )

    _ = prior_train_module.train_tabfoundry_simple_prior(
        cfg,
        prior_dump_path=path,
        batch_size=2,
    )

    gradient_history = [
        json.loads(line)
        for line in (tmp_path / "train_staged" / "gradient_history.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert set(gradient_history[0]["module_grad_norms"]) == {
        "column_encoder",
        "direct_head",
        "feature_encoder",
        "row_pool",
        "target_conditioner",
        "tokenizer",
        "transformer_blocks.0",
    }
    assert "activation_norms" not in gradient_history[0]
    assert "context_encoder" not in gradient_history[0]["module_grad_norms"]
    assert "context_label_embed" not in gradient_history[0]["module_grad_norms"]
    assert "digit_position_embed" not in gradient_history[0]["module_grad_norms"]


def test_train_tabfoundry_staged_prior_writes_activation_norms_when_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _write_prior_dump(
        tmp_path / "prior_staged_activation_trace.h5",
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
                [1, 0, 1, 1],
            ],
            dtype=np.float32,
        ),
        num_features=np.asarray([2, 2], dtype=np.int64),
        num_datapoints=np.asarray([4, 4], dtype=np.int64),
        single_eval_pos=np.asarray([2, 2], dtype=np.int64),
    )
    model = TabFoundryStagedClassifier(
        stage="nano_exact",
        module_overrides={"feature_encoder": "shared", "post_encoder_norm": "layernorm"},
        d_icl=8,
        input_normalization="train_zscore_clip",
        many_class_base=2,
        tficl_n_heads=2,
        tficl_n_layers=1,
        head_hidden_dim=16,
    )
    monkeypatch.setattr(prior_train_module, "build_model_from_spec", lambda _spec: model)
    monkeypatch.setattr(
        prior_train_module,
        "build_optimizer",
        lambda *args, **kwargs: OptimizerSelection(
            optimizers=[("schedulefree_adamw", _CountingOptimizer())],
            requested_name="schedulefree_adamw",
            resolved_name="schedulefree_adamw",
            fallback_reason=None,
        ),
    )
    cfg = OmegaConf.create(
        {
            "task": "classification",
            "model": {
                "arch": "tabfoundry_staged",
                "stage": "nano_exact",
                "stage_label": "delta_shared_norm_post_ln",
                "module_overrides": {
                    "feature_encoder": "shared",
                    "post_encoder_norm": "layernorm",
                },
                "d_icl": 8,
                "input_normalization": "train_zscore_clip",
                "feature_group_size": 1,
                "many_class_train_mode": "path_nll",
                "max_mixed_radix_digits": 64,
                "tfcol_n_heads": 8,
                "tfcol_n_layers": 3,
                "tfcol_n_inducing": 128,
                "tfrow_n_heads": 8,
                "tfrow_n_layers": 3,
                "tfrow_cls_tokens": 4,
                "tficl_n_heads": 2,
                "tficl_n_layers": 1,
                "tficl_ff_expansion": 2,
                "many_class_base": 2,
                "head_hidden_dim": 16,
                "use_digit_position_embed": True,
            },
            "runtime": {
                "seed": 0,
                "output_dir": str(tmp_path / "train_staged_activation_trace"),
                "device": "cpu",
                "mixed_precision": "no",
                "grad_clip": 1.0,
                "trace_activations": True,
                "max_steps": 1,
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
                "history_jsonl_path": str(
                    tmp_path / "train_staged_activation_trace" / "train_history.jsonl"
                ),
            },
        }
    )

    _ = prior_train_module.train_tabfoundry_simple_prior(
        cfg,
        prior_dump_path=path,
        batch_size=2,
    )

    gradient_history = [
        json.loads(line)
        for line in (
            tmp_path / "train_staged_activation_trace" / "gradient_history.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    telemetry = json.loads(
        (tmp_path / "train_staged_activation_trace" / "telemetry.json").read_text(encoding="utf-8")
    )

    assert set(gradient_history[0]["module_grad_norms"]) == {
        "column_encoder",
        "direct_head",
        "feature_encoder",
        "post_encoder_norm",
        "row_pool",
        "target_conditioner",
        "tokenizer",
        "transformer_blocks.0",
    }
    assert set(gradient_history[0]["activation_norms"]) == {
        "post_feature_encoder",
        "post_target_conditioner",
        "pre_transformer",
        "post_transformer_block_0",
        "post_column_encoder",
        "post_row_pool",
    }
    assert set(telemetry["gradient_summary"]["activations"]) == {
        "post_feature_encoder",
        "post_target_conditioner",
        "pre_transformer",
        "post_transformer_block_0",
        "post_column_encoder",
        "post_row_pool",
    }
    assert telemetry["gradient_summary"]["activations"]["pre_transformer"]["final"] >= 0.0


def test_train_tabfoundry_staged_prior_writes_context_gradient_keys_when_active(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _write_prior_dump(
        tmp_path / "prior_staged_context.h5",
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
                [1, 0, 1, 1],
            ],
            dtype=np.float32,
        ),
        num_features=np.asarray([2, 2], dtype=np.int64),
        num_datapoints=np.asarray([4, 4], dtype=np.int64),
        single_eval_pos=np.asarray([2, 2], dtype=np.int64),
    )
    model = TabFoundryStagedClassifier(
        stage="column_set",
        d_icl=8,
        input_normalization="train_zscore_clip",
        many_class_base=2,
        tficl_n_heads=2,
        tficl_n_layers=1,
        head_hidden_dim=16,
    )
    monkeypatch.setattr(prior_train_module, "build_model_from_spec", lambda _spec: model)
    monkeypatch.setattr(
        prior_train_module,
        "build_optimizer",
        lambda *args, **kwargs: OptimizerSelection(
            optimizers=[("schedulefree_adamw", _CountingOptimizer())],
            requested_name="schedulefree_adamw",
            resolved_name="schedulefree_adamw",
            fallback_reason=None,
        ),
    )
    cfg = OmegaConf.create(
        {
            "task": "classification",
            "model": {
                "arch": "tabfoundry_staged",
                "stage": "column_set",
                "d_icl": 8,
                "input_normalization": "train_zscore_clip",
                "feature_group_size": 1,
                "many_class_train_mode": "path_nll",
                "max_mixed_radix_digits": 64,
                "tfcol_n_heads": 8,
                "tfcol_n_layers": 3,
                "tfcol_n_inducing": 128,
                "tfrow_n_heads": 8,
                "tfrow_n_layers": 3,
                "tfrow_cls_tokens": 4,
                "tficl_n_heads": 2,
                "tficl_n_layers": 1,
                "tficl_ff_expansion": 2,
                "many_class_base": 2,
                "head_hidden_dim": 16,
                "use_digit_position_embed": True,
            },
            "runtime": {
                "seed": 0,
                "output_dir": str(tmp_path / "train_staged_context"),
                "device": "cpu",
                "mixed_precision": "no",
                "grad_clip": 1.0,
                "max_steps": 1,
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
                "history_jsonl_path": str(tmp_path / "train_staged_context" / "train_history.jsonl"),
            },
        }
    )

    _ = prior_train_module.train_tabfoundry_simple_prior(
        cfg,
        prior_dump_path=path,
        batch_size=2,
    )

    gradient_history = [
        json.loads(line)
        for line in (tmp_path / "train_staged_context" / "gradient_history.jsonl").read_text(
            encoding="utf-8"
        ).splitlines()
        if line.strip()
    ]
    assert set(gradient_history[0]["module_grad_norms"]) == {
        "column_encoder",
        "context_encoder",
        "context_label_embed",
        "direct_head",
        "feature_encoder",
        "row_pool",
        "target_conditioner",
        "tokenizer",
        "transformer_blocks.0",
    }
    assert "digit_position_embed" not in gradient_history[0]["module_grad_norms"]


def test_prior_train_main_passes_prior_dump_and_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    captured: dict[str, object] = {}
    cfg = OmegaConf.create({"runtime": {"output_dir": str(tmp_path / "train_out")}})

    def _fake_compose(overrides):
        captured["overrides"] = list(overrides)
        return cfg

    def _fake_train(resolved_cfg, *, prior_dump_path: Path, batch_size: int = prior_train_module.DEFAULT_BATCH_SIZE):
        captured["cfg"] = resolved_cfg
        captured["prior_dump_path"] = prior_dump_path
        captured["batch_size"] = batch_size
        return SimpleNamespace(
            output_dir=tmp_path / "train_out",
            latest_checkpoint=tmp_path / "train_out" / "checkpoints" / "latest.pt",
            global_step=25,
            metrics={"final_train_loss": 0.42},
        )

    monkeypatch.setattr(prior_train_module, "_compose_prior_cfg", _fake_compose)
    monkeypatch.setattr(prior_train_module, "train_tabfoundry_simple_prior", _fake_train)

    exit_code = prior_train_module.main(
        [
            "--prior-dump",
            str(tmp_path / "prior_dump.h5"),
            "model.stage=label_token",
            "runtime.output_dir=/tmp/cli-prior",
        ]
    )

    assert exit_code == 0
    assert captured["overrides"] == ["model.stage=label_token", "runtime.output_dir=/tmp/cli-prior"]
    assert captured["cfg"] is cfg
    assert captured["prior_dump_path"] == tmp_path / "prior_dump.h5"
    assert captured["batch_size"] == prior_train_module.DEFAULT_BATCH_SIZE
    assert "Training complete:" in capsys.readouterr().out


def test_train_tabfoundry_simple_prior_rejects_staged_many_class_before_io(
    tmp_path: Path,
) -> None:
    output_dir = tmp_path / "prior_train_many_class"
    cfg = OmegaConf.create(
        {
            "task": "classification",
            "model": {
                "arch": "tabfoundry_staged",
                "stage": "many_class",
                "d_icl": 8,
                "input_normalization": "none",
                "feature_group_size": 1,
                "many_class_train_mode": "path_nll",
                "max_mixed_radix_digits": 64,
                "tfcol_n_heads": 8,
                "tfcol_n_layers": 3,
                "tfcol_n_inducing": 128,
                "tfrow_n_heads": 8,
                "tfrow_n_layers": 3,
                "tfrow_cls_tokens": 4,
                "tficl_n_heads": 2,
                "tficl_n_layers": 1,
                "tficl_ff_expansion": 2,
                "many_class_base": 4,
                "head_hidden_dim": 16,
                "use_digit_position_embed": True,
            },
            "runtime": {
                "seed": 0,
                "output_dir": str(output_dir),
                "mixed_precision": "no",
                "max_steps": 1,
                "eval_every": 1,
                "checkpoint_every": 1,
                "grad_clip": 1.0,
                "device": "cpu",
            },
            "optimizer": {
                "name": "schedulefree_adamw",
                "weight_decay": 0.0,
                "require_requested": True,
                "muon_per_parameter_lr": False,
                "muon_lr_scale_base": 0.2,
                "muon_partition_non2d": True,
                "min_lr": 4.0e-3,
            },
            "logging": {
                "history_jsonl_path": str(output_dir / "train_history.jsonl"),
            },
        }
    )

    with pytest.raises(ValueError, match="forward_batched\\(\\) tensor logits"):
        _ = prior_train_module.train_tabfoundry_simple_prior(
            cfg,
            prior_dump_path=tmp_path / "missing_prior_dump.h5",
            batch_size=1,
        )

    assert not output_dir.exists()


def test_train_tabfoundry_simple_prior_rejects_mismatched_schedule_steps(
    tmp_path: Path,
) -> None:
    cfg = _prior_cfg(
        tmp_path,
        max_steps=3,
        training_cfg={"surface_label": "prior_linear_decay", "apply_schedule": True},
        schedule_cfg={
            "stages": [
                {
                    "name": "stage1",
                    "steps": 4,
                    "lr_max": 4.0e-3,
                    "lr_schedule": "linear",
                    "warmup_ratio": 0.0,
                }
            ]
        },
    )

    with pytest.raises(ValueError, match="requires schedule.stages\\[0\\]\\.steps to equal runtime.max_steps"):
        _ = prior_train_module.train_tabfoundry_simple_prior(
            cfg,
            prior_dump_path=tmp_path / "unused_prior_dump.h5",
            batch_size=1,
        )


def test_train_tabfoundry_simple_prior_writes_failure_telemetry_for_nonfinite_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _write_prior_dump(
        tmp_path / "prior_nonfinite_runtime.h5",
        x=np.asarray([[[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]]], dtype=np.float32),
        y=np.asarray([[0, 1, 0]], dtype=np.int64),
        num_features=np.asarray([2], dtype=np.int64),
        num_datapoints=np.asarray([3], dtype=np.int64),
        single_eval_pos=np.asarray([2], dtype=np.int64),
    )
    monkeypatch.setattr(prior_train_module, "build_model_from_spec", lambda _spec: _ConstantLogitModel())
    monkeypatch.setattr(
        prior_train_module,
        "build_optimizer",
        lambda *args, **kwargs: OptimizerSelection(
            optimizers=[("schedulefree_adamw", _CountingOptimizer())],
            requested_name="schedulefree_adamw",
            resolved_name="schedulefree_adamw",
            fallback_reason=None,
        ),
    )
    cfg = _prior_cfg(tmp_path, max_steps=1)

    with pytest.raises(PriorDumpNonFiniteInputError):
        _ = prior_train_module.train_tabfoundry_simple_prior(
            cfg,
            prior_dump_path=path,
            batch_size=1,
        )

    telemetry = json.loads((tmp_path / "train_out" / "telemetry.json").read_text(encoding="utf-8"))
    assert telemetry["success"] is False
    assert telemetry["error"]["type"] == "PriorDumpNonFiniteInputError"
    assert telemetry["missingness"]["prior_dump"]["affected_batch_count"] == 1
    assert telemetry["missingness"]["prior_dump"]["affected_dataset_indices"] == [0]
    assert telemetry["missingness"]["prior_dump"]["non_finite_feature_count"] == 1


def test_train_tabfoundry_simple_prior_writes_failure_telemetry_for_nonfinite_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _write_prior_dump(
        tmp_path / "prior_nonfinite_labels_runtime.h5",
        x=np.asarray([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=np.float32),
        y=np.asarray([[0.0, np.inf, 1.0]], dtype=np.float32),
        num_features=np.asarray([2], dtype=np.int64),
        num_datapoints=np.asarray([3], dtype=np.int64),
        single_eval_pos=np.asarray([2], dtype=np.int64),
    )
    monkeypatch.setattr(prior_train_module, "build_model_from_spec", lambda _spec: _ConstantLogitModel())
    monkeypatch.setattr(
        prior_train_module,
        "build_optimizer",
        lambda *args, **kwargs: OptimizerSelection(
            optimizers=[("schedulefree_adamw", _CountingOptimizer())],
            requested_name="schedulefree_adamw",
            resolved_name="schedulefree_adamw",
            fallback_reason=None,
        ),
    )
    cfg = _prior_cfg(tmp_path, max_steps=1)

    with pytest.raises(PriorDumpNonFiniteInputError):
        _ = prior_train_module.train_tabfoundry_simple_prior(
            cfg,
            prior_dump_path=path,
            batch_size=1,
        )

    telemetry = json.loads((tmp_path / "train_out" / "telemetry.json").read_text(encoding="utf-8"))
    assert telemetry["success"] is False
    assert telemetry["error"]["type"] == "PriorDumpNonFiniteInputError"
    assert telemetry["missingness"]["prior_dump"]["affected_batch_count"] == 1
    assert telemetry["missingness"]["prior_dump"]["affected_dataset_indices"] == [0]
    assert telemetry["missingness"]["prior_dump"]["non_finite_feature_count"] == 0
    assert telemetry["missingness"]["prior_dump"]["non_finite_label_count"] == 1


def test_resolve_prior_wandb_run_name_uses_stability_followup_output_slug(tmp_path: Path) -> None:
    cfg = _prior_cfg(tmp_path, max_steps=1)
    cfg.logging.run_name = "cls-benchmark-staged-prior"
    cfg.model.stage_label = "linear_warmup_decay_lr4e3_warm10"
    cfg.runtime.output_dir = str(
        tmp_path
        / "outputs"
        / "staged_ladder"
        / "sd_stability_followup_dpnb_12_row_cls_cls2_linear_warmup_decay_warm10_v1"
        / "train"
    )

    assert prior_train_module._resolve_prior_wandb_run_name(cfg) == "dpnb_row_cls_cls2_linear_warmup_decay_warm10"


def test_train_tabfoundry_simple_prior_logs_wandb_metrics_and_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _write_prior_dump(
        tmp_path / "prior_wandb.h5",
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
                [1, 0, 1, 1],
            ],
            dtype=np.int64,
        ),
        num_features=np.asarray([2, 2], dtype=np.int64),
        num_datapoints=np.asarray([4, 4], dtype=np.int64),
        single_eval_pos=np.asarray([2, 2], dtype=np.int64),
    )
    fake_run = _FakeWandbRun()
    monkeypatch.setattr(prior_train_module, "build_model_from_spec", lambda _spec: _TracingConstantLogitModel())
    monkeypatch.setattr(
        prior_train_module,
        "build_optimizer",
        lambda *args, **kwargs: OptimizerSelection(
            optimizers=[("schedulefree_adamw", _CountingOptimizer())],
            requested_name="schedulefree_adamw",
            resolved_name="schedulefree_adamw",
            fallback_reason=None,
        ),
    )
    monkeypatch.setattr(
        prior_train_module,
        "module_grad_norms",
        lambda _model: {"feature_encoder": 0.5, "direct_head": 2.0, "decoder": 0.25},
    )
    monkeypatch.setattr(prior_train_module, "init_wandb_run", lambda *_args, **_kwargs: fake_run)
    cfg = _prior_cfg(
        tmp_path,
        max_steps=2,
        training_cfg={"surface_label": "prior_linear_warmup_decay", "apply_schedule": True},
        schedule_cfg={
            "stages": [
                {
                    "name": "prior_dump",
                    "steps": 2,
                    "lr_max": 4.0e-3,
                    "lr_schedule": "linear",
                    "warmup_ratio": 0.5,
                }
            ]
        },
    )
    cfg.logging.use_wandb = True
    cfg.logging.project = "test-project"
    cfg.logging.run_name = "prior-wandb"
    cfg.runtime.trace_activations = True

    _ = prior_train_module.train_tabfoundry_simple_prior(
        cfg,
        prior_dump_path=path,
        batch_size=2,
    )

    train_logs = [
        (payload, step)
        for payload, step in fake_run.logged
        if "train/loss" in payload
    ]
    assert [step for _payload, step in train_logs] == [1, 2]
    assert "train/loss_delta" not in train_logs[0][0]
    assert train_logs[1][0]["train/stage"] == "prior_dump"
    assert "train/loss_delta" in train_logs[1][0]
    assert "train/loss_ema" in train_logs[1][0]
    assert "train/elapsed_seconds" in train_logs[1][0]
    assert "train/train_elapsed_seconds" in train_logs[1][0]
    assert "train/grad_clip_threshold" in train_logs[1][0]
    assert "train/grad_clip_triggered" in train_logs[1][0]
    assert "train/grad_clip_count_so_far" in train_logs[1][0]
    assert "train/grad_clip_fraction_so_far" in train_logs[1][0]
    assert train_logs[1][0]["train/module_grad_norm/decoder"] == pytest.approx(0.25)
    assert train_logs[1][0]["train/module_balance/direct_head_to_feature_encoder"] == pytest.approx(4.0)
    assert train_logs[1][0]["train/module_balance/feature_encoder_to_direct_head"] == pytest.approx(0.25)
    assert train_logs[1][0]["train/activation_norm/post_feature_encoder"] == pytest.approx(1.2)
    assert train_logs[1][0]["train/activation_norm/pre_transformer"] == pytest.approx(2.4)
    assert train_logs[1][0]["train/prior_dump_skipped_batch_count"] == 0
    assert fake_run.finished is True
    assert fake_run.summary["run/output_dir"] == str((tmp_path / "train_out").resolve())
    assert fake_run.summary["run/global_step"] == 2
    assert fake_run.summary["telemetry/success"] is True
    assert fake_run.summary["telemetry/checkpoint_snapshot_count"] == 2
    assert fake_run.summary["artifacts/latest_checkpoint"].endswith("latest.pt")
    assert fake_run.summary["loss_summary/final_train_loss"] >= 0.0
    assert fake_run.summary["gradient_summary/global/final_grad_norm"] >= 0.0
    assert fake_run.summary["diagnostics/grad_clip/clipped_step_fraction"] >= 0.0
    assert fake_run.summary[
        "diagnostics/module_balance/feature_encoder_vs_direct_head/windows/early_1_25/direct_head_to_feature_encoder_mean_ratio"
    ] == pytest.approx(4.0)
    assert fake_run.summary[
        "diagnostics/activation_windows/tracked_activations/post_feature_encoder/early_to_final_mean_delta"
    ] > 0.0
    assert fake_run.summary["missingness/prior_dump/non_finite_feature_count"] == 0


def test_train_tabfoundry_simple_prior_logs_wandb_failure_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _write_prior_dump(
        tmp_path / "prior_wandb_nonfinite.h5",
        x=np.asarray([[[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]]], dtype=np.float32),
        y=np.asarray([[0, 1, 0]], dtype=np.int64),
        num_features=np.asarray([2], dtype=np.int64),
        num_datapoints=np.asarray([3], dtype=np.int64),
        single_eval_pos=np.asarray([2], dtype=np.int64),
    )
    fake_run = _FakeWandbRun()
    monkeypatch.setattr(prior_train_module, "build_model_from_spec", lambda _spec: _ConstantLogitModel())
    monkeypatch.setattr(
        prior_train_module,
        "build_optimizer",
        lambda *args, **kwargs: OptimizerSelection(
            optimizers=[("schedulefree_adamw", _CountingOptimizer())],
            requested_name="schedulefree_adamw",
            resolved_name="schedulefree_adamw",
            fallback_reason=None,
        ),
    )
    monkeypatch.setattr(prior_train_module, "init_wandb_run", lambda *_args, **_kwargs: fake_run)
    cfg = _prior_cfg(tmp_path, max_steps=1)
    cfg.logging.use_wandb = True
    cfg.logging.project = "test-project"
    cfg.logging.run_name = "prior-wandb-failure"

    with pytest.raises(PriorDumpNonFiniteInputError):
        _ = prior_train_module.train_tabfoundry_simple_prior(
            cfg,
            prior_dump_path=path,
            batch_size=1,
        )

    assert fake_run.finished is True
    assert fake_run.summary["telemetry/success"] is False
    assert fake_run.summary["error/type"] == "PriorDumpNonFiniteInputError"
    assert "contains NaN or Inf" in str(fake_run.summary["error/message"])
    assert fake_run.summary["missingness/prior_dump/affected_batch_count"] == 1


def test_train_tabfoundry_simple_prior_skip_policy_preserves_successful_step_budget(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _write_prior_dump(
        tmp_path / "prior_skip_runtime.h5",
        x=np.asarray(
            [
                [[1.0, np.nan], [2.0, 3.0], [4.0, 5.0]],
                [[6.0, 7.0], [8.0, 9.0], [10.0, 11.0]],
            ],
            dtype=np.float32,
        ),
        y=np.asarray([[0, 1, 0], [1, 0, 1]], dtype=np.int64),
        num_features=np.asarray([2, 2], dtype=np.int64),
        num_datapoints=np.asarray([3, 3], dtype=np.int64),
        single_eval_pos=np.asarray([2, 2], dtype=np.int64),
    )
    optimizer = _CountingOptimizer()
    monkeypatch.setattr(prior_train_module, "build_model_from_spec", lambda _spec: _ConstantLogitModel())
    monkeypatch.setattr(
        prior_train_module,
        "build_optimizer",
        lambda *args, **kwargs: OptimizerSelection(
            optimizers=[("schedulefree_adamw", optimizer)],
            requested_name="schedulefree_adamw",
            resolved_name="schedulefree_adamw",
            fallback_reason=None,
        ),
    )
    cfg = _prior_cfg(tmp_path, max_steps=1)
    cfg.training = OmegaConf.create({"prior_dump_non_finite_policy": "skip"})

    result = prior_train_module.train_tabfoundry_simple_prior(
        cfg,
        prior_dump_path=path,
        batch_size=1,
    )

    telemetry = json.loads((tmp_path / "train_out" / "telemetry.json").read_text(encoding="utf-8"))
    assert result.global_step == 1
    assert optimizer.step_count == 1
    assert telemetry["success"] is True
    assert telemetry["missingness"]["prior_dump"]["non_finite_policy"] == "skip"
    assert telemetry["missingness"]["prior_dump"]["skipped_batch_count"] == 1
    assert telemetry["missingness"]["prior_dump"]["skipped_dataset_indices"] == [0]
    assert telemetry["missingness"]["prior_dump"]["skipped_non_finite_feature_count"] == 1


def test_train_tabfoundry_simple_prior_accepts_plain_adamw(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _write_prior_dump(
        tmp_path / "prior_adamw.h5",
        x=np.asarray([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=np.float32),
        y=np.asarray([[0, 1, 0]], dtype=np.int64),
        num_features=np.asarray([2], dtype=np.int64),
        num_datapoints=np.asarray([3], dtype=np.int64),
        single_eval_pos=np.asarray([2], dtype=np.int64),
    )
    optimizer = _CountingOptimizer()
    monkeypatch.setattr(prior_train_module, "build_model_from_spec", lambda _spec: _ConstantLogitModel())
    monkeypatch.setattr(
        prior_train_module,
        "build_optimizer",
        lambda *args, **kwargs: OptimizerSelection(
            optimizers=[("adamw", optimizer)],
            requested_name="adamw",
            resolved_name="adamw",
            fallback_reason=None,
        ),
    )
    cfg = _prior_cfg(tmp_path, max_steps=1)
    cfg.optimizer.name = "adamw"

    result = prior_train_module.train_tabfoundry_simple_prior(
        cfg,
        prior_dump_path=path,
        batch_size=1,
    )

    assert result.global_step == 1
    assert optimizer.step_count == 1


def test_resolve_prior_training_device_name_falls_back_for_multilayer_row_cls_on_mps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _staged_prior_cfg(tmp_path, max_steps=1, stage="row_cls_pool", tfrow_n_layers=3)
    cfg.runtime.device = "mps"
    spec = prior_train_module._model_spec_from_cfg(cfg)
    staged_surface = prior_train_module._validate_prior_training_model_spec(spec)
    monkeypatch.setattr(prior_train_module, "resolve_device", lambda _device: "mps")

    device_name = prior_train_module._resolve_prior_training_device_name(
        cfg,
        spec=spec,
        staged_surface=staged_surface,
    )

    assert device_name == "cpu"
    assert str(cfg.runtime.device) == "cpu"
    assert "row_pool='row_cls'" in capsys.readouterr().err


def test_resolve_prior_training_device_name_keeps_mps_for_target_column(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _staged_prior_cfg(tmp_path, max_steps=1, stage="nano_exact", tfrow_n_layers=3)
    cfg.runtime.device = "mps"
    spec = prior_train_module._model_spec_from_cfg(cfg)
    staged_surface = prior_train_module._validate_prior_training_model_spec(spec)
    monkeypatch.setattr(prior_train_module, "resolve_device", lambda _device: "mps")

    device_name = prior_train_module._resolve_prior_training_device_name(
        cfg,
        spec=spec,
        staged_surface=staged_surface,
    )

    assert device_name == "mps"
    assert str(cfg.runtime.device) == "mps"
    assert capsys.readouterr().err == ""


def test_resolve_prior_training_device_name_keeps_mps_for_single_layer_row_cls(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    cfg = _staged_prior_cfg(tmp_path, max_steps=1, stage="row_cls_pool", tfrow_n_layers=1)
    cfg.runtime.device = "mps"
    spec = prior_train_module._model_spec_from_cfg(cfg)
    staged_surface = prior_train_module._validate_prior_training_model_spec(spec)
    monkeypatch.setattr(prior_train_module, "resolve_device", lambda _device: "mps")

    device_name = prior_train_module._resolve_prior_training_device_name(
        cfg,
        spec=spec,
        staged_surface=staged_surface,
    )

    assert device_name == "mps"
    assert str(cfg.runtime.device) == "mps"
    assert capsys.readouterr().err == ""


def test_train_tabfoundry_staged_prior_falls_back_to_cpu_for_multilayer_row_cls_on_mps(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = _write_prior_dump(
        tmp_path / "prior_row_cls_fallback.h5",
        x=np.asarray([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=np.float32),
        y=np.asarray([[0, 1, 0]], dtype=np.int64),
        num_features=np.asarray([2], dtype=np.int64),
        num_datapoints=np.asarray([3], dtype=np.int64),
        single_eval_pos=np.asarray([2], dtype=np.int64),
    )
    model = _DeviceTrackingConstantLogitModel()
    optimizer = _CountingOptimizer()
    monkeypatch.setattr(prior_train_module, "build_model_from_spec", lambda _spec: model)
    monkeypatch.setattr(
        prior_train_module,
        "build_optimizer",
        lambda *args, **kwargs: OptimizerSelection(
            optimizers=[("schedulefree_adamw", optimizer)],
            requested_name="schedulefree_adamw",
            resolved_name="schedulefree_adamw",
            fallback_reason=None,
        ),
    )
    monkeypatch.setattr(prior_train_module, "resolve_device", lambda _device: "mps")
    cfg = _staged_prior_cfg(tmp_path, max_steps=1, stage="row_cls_pool", tfrow_n_layers=3)
    cfg.runtime.device = "mps"

    result = prior_train_module.train_tabfoundry_simple_prior(
        cfg,
        prior_dump_path=path,
        batch_size=1,
    )

    assert result.global_step == 1
    assert model.to_device_types == ["cpu"]
    assert str(cfg.runtime.device) == "cpu"
    assert "falling back to CPU" in capsys.readouterr().err


def test_evaluate_tab_foundry_run_supports_runs_without_best_checkpoint(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _FakeClassifier:
        def __init__(self, checkpoint_path: Path, *, device: str = "cpu") -> None:
            self.checkpoint_path = checkpoint_path
            self.device = device
            self.model_spec = SimpleNamespace(arch="tabfoundry", stage=None)
            self.model = SimpleNamespace(benchmark_profile=None)

        def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "_FakeClassifier":
            _ = (x_train, y_train)
            return self

        def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
            _ = x_test
            return np.tile(np.asarray([[0.25, 0.75]], dtype=np.float64), (x_test.shape[0], 1))

    monkeypatch.setattr(checkpoint_module, "TabFoundryClassifier", _FakeClassifier)

    run_dir = tmp_path / "prior_run"
    checkpoint_dir = run_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True)
    (checkpoint_dir / "step_000025.pt").write_bytes(b"step25")
    (checkpoint_dir / "step_000050.pt").write_bytes(b"step50")
    history_path = run_dir / "train_history.jsonl"
    history_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "step": 25,
                        "stage": "prior_dump",
                        "train_loss": 0.5,
                        "train_acc": 0.5,
                        "lr": 4.0e-3,
                        "grad_norm": 1.0,
                        "elapsed_seconds": 1.0,
                        "train_elapsed_seconds": 1.0,
                        "val_loss": None,
                        "val_acc": None,
                    }
                ),
                json.dumps(
                    {
                        "step": 50,
                        "stage": "prior_dump",
                        "train_loss": 0.4,
                        "train_acc": 0.6,
                        "lr": 4.0e-3,
                        "grad_norm": 1.1,
                        "elapsed_seconds": 2.0,
                        "train_elapsed_seconds": 2.0,
                        "val_loss": None,
                        "val_acc": None,
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    x = np.tile(np.arange(20, dtype=np.float32)[:, None], (1, 2))
    y = np.asarray([0, 1] * 10, dtype=np.int64)
    records = evaluate_tab_foundry_run(
        run_dir,
        datasets={"toy": (x, y)},
        task_type="supervised_classification",
        device="cpu",
    )

    assert [int(record["step"]) for record in records] == [25, 50]
    assert all("roc_auc" in record for record in records)


def test_tabfoundry_staged_nano_exact_matches_simple_prior_batch_loss() -> None:
    torch.manual_seed(0)
    simple = TabFoundrySimpleClassifier(
        d_icl=8,
        input_normalization="train_zscore_clip",
        many_class_base=2,
        tficl_n_heads=2,
        tficl_n_layers=1,
        head_hidden_dim=16,
    )
    staged = TabFoundryStagedClassifier(
        stage="nano_exact",
        d_icl=8,
        input_normalization="train_zscore_clip",
        many_class_base=2,
        tficl_n_heads=2,
        tficl_n_layers=1,
        head_hidden_dim=16,
    )
    staged.feature_encoder.load_state_dict(simple.feature_encoder.state_dict(), strict=True)
    staged.target_conditioner.encoder.load_state_dict(simple.target_encoder.state_dict(), strict=True)
    staged.direct_head.decoder.load_state_dict(simple.decoder.state_dict(), strict=True)
    for staged_block, simple_block in zip(staged.transformer_blocks, simple.transformer_blocks, strict=True):
        staged_block.block.load_state_dict(simple_block.state_dict(), strict=True)

    x_batch = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]],
            [[2.0, 1.0], [4.0, 3.0], [6.0, 5.0], [8.0, 7.0]],
        ],
        dtype=torch.float32,
    )
    y_train = torch.tensor([[0.0, 1.0], [1.0, 0.0]], dtype=torch.float32)
    targets = torch.tensor([[0, 1], [1, 1]], dtype=torch.int64).reshape(-1)

    simple_logits = simple.forward_batched(
        x_all=x_batch.clone(),
        y_train=y_train.clone(),
        train_test_split_index=2,
    )
    staged_logits = staged.forward_batched(
        x_all=x_batch.clone(),
        y_train=y_train.clone(),
        train_test_split_index=2,
    )

    simple_loss = classification_loss(
        simple_logits.reshape(-1, int(simple_logits.shape[-1])),
        targets,
    )
    staged_loss = classification_loss(
        staged_logits.reshape(-1, int(staged_logits.shape[-1])),
        targets,
    )

    assert staged_loss.item() == pytest.approx(simple_loss.item(), rel=1.0e-6, abs=1.0e-6)


def test_train_tabfoundry_simple_prior_injects_synthetic_missingness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    path = _write_prior_dump(
        tmp_path / "prior_synthetic_missingness.h5",
        x=np.asarray([[[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]]], dtype=np.float32),
        y=np.asarray([[0, 1, 0]], dtype=np.int64),
        num_features=np.asarray([2], dtype=np.int64),
        num_datapoints=np.asarray([3], dtype=np.int64),
        single_eval_pos=np.asarray([2], dtype=np.int64),
    )
    model = _CapturingConstantLogitModel()
    monkeypatch.setattr(prior_train_module, "build_model_from_spec", lambda _spec: model)
    monkeypatch.setattr(
        prior_train_module,
        "build_optimizer",
        lambda *args, **kwargs: OptimizerSelection(
            optimizers=[("schedulefree_adamw", _CountingOptimizer())],
            requested_name="schedulefree_adamw",
            resolved_name="schedulefree_adamw",
            fallback_reason=None,
        ),
    )
    cfg = _prior_cfg(
        tmp_path,
        max_steps=1,
        training_cfg={
            "surface_label": "prior_cosine_warmup",
            "apply_schedule": True,
            "overrides": {
                "prior_missingness": {
                    "enabled": True,
                    "min_rate": 1.0,
                    "max_rate": 1.0,
                }
            },
        },
        schedule_cfg={
            "stages": [
                {
                    "name": "prior_dump",
                    "steps": 1,
                    "lr_max": 4.0e-3,
                    "lr_schedule": "cosine",
                    "warmup_ratio": 0.0,
                }
            ]
        },
    )

    _ = prior_train_module.train_tabfoundry_simple_prior(cfg, prior_dump_path=path, batch_size=1)

    assert model.last_x_all is not None
    assert torch.isnan(model.last_x_all[0, :, :]).all()
    telemetry = json.loads((tmp_path / "train_out" / "telemetry.json").read_text(encoding="utf-8"))
    synthetic = telemetry["missingness"]["synthetic_prior"]
    assert synthetic["enabled"] is True
    assert synthetic["batches_seen"] == 1
    assert synthetic["affected_batch_count"] == 1
    assert synthetic["masked_feature_count"] == 6
    assert synthetic["affected_dataset_indices"] == [0]
