from __future__ import annotations

from contextlib import nullcontext

import pytest
import torch

from tab_foundry.model.outputs import ClassificationOutput
from tab_foundry.training.compile_dispatch import SignatureFamilyCompileDispatcher
from tab_foundry.training.runtime import CompilePolicy, CudaGraphCapturePolicy
from tab_foundry.types import TaskBatch

from tests.support.train_eval_smoke_cases import _FakeAccelerator


class _TinyClassifier(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(4, 3)

    def forward(self, batch: TaskBatch) -> ClassificationOutput:
        return ClassificationOutput(logits=self.linear(batch.x_test.to(torch.float32)), num_classes=3)


def _classification_batch() -> TaskBatch:
    return TaskBatch(
        x_train=torch.randn(6, 4),
        y_train=torch.tensor([0, 1, 2, 0, 1, 2], dtype=torch.int64),
        x_test=torch.randn(3, 4),
        y_test=torch.tensor([0, 1, 2], dtype=torch.int64),
        metadata={"dataset_index": 1},
        num_classes=3,
    )


def test_signature_family_compile_dispatcher_records_cuda_graph_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch, "compile", lambda model, **_kwargs: model)
    dispatcher = SignatureFamilyCompileDispatcher(
        _TinyClassifier(),
        compile_policy=CompilePolicy(
            enabled=True,
            backend="eager",
            mode="default",
            dynamic=True,
        ),
        max_families=4,
        task="classification",
        grad_accum_steps=1,
        cuda_graph_policy=CudaGraphCapturePolicy(
            enabled=True,
            mode="signature_family",
            max_families=2,
        ),
        cuda_graph_autocast_context_factory=nullcontext,
    )
    monkeypatch.setattr(
        dispatcher,
        "_build_cuda_graph_replay",
        lambda **_kwargs: (_ for _ in ()).throw(RuntimeError("graph unsupported")),
    )

    loss, metrics = dispatcher.training_step(
        batch=_classification_batch(),
        task="classification",
        actual_task_count=1,
        accelerator=_FakeAccelerator(),
    )

    assert float(loss.item()) >= 0.0
    assert 0.0 <= metrics["acc"] <= 1.0
    assert dispatcher.summary()["cuda_graph_capture_failure_count"] == 1
    assert dispatcher.summary()["cuda_graph_fallback_call_count"] == 1


def test_signature_family_compile_dispatcher_reuses_captured_cuda_graph_runner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch, "compile", lambda model, **_kwargs: model)
    dispatcher = SignatureFamilyCompileDispatcher(
        _TinyClassifier(),
        compile_policy=CompilePolicy(
            enabled=True,
            backend="eager",
            mode="default",
            dynamic=True,
        ),
        max_families=4,
        task="classification",
        grad_accum_steps=1,
        cuda_graph_policy=CudaGraphCapturePolicy(
            enabled=True,
            mode="signature_family",
            max_families=2,
        ),
        cuda_graph_autocast_context_factory=nullcontext,
    )

    class _FakeGraphReplay:
        def replay(self, _batch: TaskBatch) -> tuple[torch.Tensor, dict[str, float]]:
            return torch.tensor(1.0), {"acc": 0.5}

    monkeypatch.setattr(dispatcher, "_build_cuda_graph_replay", lambda **_kwargs: _FakeGraphReplay())
    batch = _classification_batch()
    accelerator = _FakeAccelerator()

    first_loss, first_metrics = dispatcher.training_step(
        batch=batch,
        task="classification",
        actual_task_count=1,
        accelerator=accelerator,
    )
    second_loss, second_metrics = dispatcher.training_step(
        batch=batch,
        task="classification",
        actual_task_count=1,
        accelerator=accelerator,
    )

    assert float(first_loss.item()) == pytest.approx(1.0)
    assert float(second_loss.item()) == pytest.approx(1.0)
    assert first_metrics == {"acc": 0.5}
    assert second_metrics == {"acc": 0.5}
    summary = dispatcher.summary()
    assert summary["cuda_graph_captured_family_count"] == 1
    assert summary["cuda_graph_cache_misses"] == 1
    assert summary["cuda_graph_cache_hits"] == 1
