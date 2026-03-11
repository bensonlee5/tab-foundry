from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn

import tab_foundry.checkpoint_classifier as checkpoint_classifier
from tab_foundry.model.tabiclv2 import ClassificationOutput
from tab_foundry.types import TaskBatch


class _TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, batch: TaskBatch) -> ClassificationOutput:
        return ClassificationOutput(logits=self.linear(batch.x_test), num_classes=2)


def test_tab_foundry_classifier_predicts_probabilities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_spec = SimpleNamespace(task="classification")
    monkeypatch.setattr(checkpoint_classifier, "model_build_spec_from_mappings", lambda **_kwargs: fake_spec)
    monkeypatch.setattr(checkpoint_classifier, "build_model_from_spec", lambda _spec: _TinyClassifier())

    checkpoint = tmp_path / "tiny.pt"
    model = _TinyClassifier()
    torch.save({"model": model.state_dict(), "config": {"task": "classification", "model": {}}}, checkpoint)

    classifier = checkpoint_classifier.TabFoundryClassifier(checkpoint, device="cpu")
    classifier.fit(np.ones((6, 4), dtype=np.float32), np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64))
    probabilities = classifier.predict_proba(np.zeros((3, 4), dtype=np.float32))

    assert probabilities.shape == (3, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-6)
