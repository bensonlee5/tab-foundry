from __future__ import annotations

from functools import lru_cache
import importlib.util
import json
from pathlib import Path
import runpy
import sys
from types import SimpleNamespace

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

import tab_foundry.bench.checkpoint as checkpoint_module
import tab_foundry.bench.prior_train as prior_train_module
from tab_foundry.bench.nanotabpfn import evaluate_tab_foundry_run
from tab_foundry.bench.prior_dump import PriorDumpTaskBatchReader
from tab_foundry.model.architectures.tabfoundry_staged.model import TabFoundryStagedClassifier
from tab_foundry.model.architectures.tabfoundry_simple import TabFoundrySimpleClassifier
from tab_foundry.training.losses import classification_loss
from tab_foundry.training.optimizer import OptimizerSelection

h5py = pytest.importorskip("h5py")
REPO_ROOT = Path(__file__).resolve().parents[2]


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
    assert (tmp_path / "train_out" / "checkpoints" / "step_000001.pt").exists()
    assert (tmp_path / "train_out" / "checkpoints" / "step_000002.pt").exists()
    assert (tmp_path / "train_out" / "checkpoints" / "latest.pt").exists()


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


def test_train_tabfoundry_staged_prior_script_injects_default_experiment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_main(argv=None):
        captured["argv"] = list(argv) if argv is not None else None
        return 0

    monkeypatch.setattr(prior_train_module, "main", _fake_main)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train_tabfoundry_staged_prior.py",
            "model.stage=label_token",
            "runtime.output_dir=/tmp/staged-prior",
        ],
    )

    with pytest.raises(SystemExit) as exc_info:
        runpy.run_path(str(REPO_ROOT / "scripts" / "train_tabfoundry_staged_prior.py"), run_name="__main__")

    assert exc_info.value.code == 0
    assert captured["argv"] == [
        "experiment=cls_benchmark_staged_prior",
        "model.stage=label_token",
        "runtime.output_dir=/tmp/staged-prior",
    ]


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
