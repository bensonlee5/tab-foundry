from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch
from torch import nn

import tab_foundry.bench.checkpoint as checkpoint_classifier
from tab_foundry.bench.nanotabpfn import evaluate_classifier, load_dataset_cache
from tab_foundry.input_normalization import normalize_train_test_arrays
from tab_foundry.model.architectures.tabfoundry import ClassificationOutput
from tab_foundry.model.factory import build_model
from tab_foundry.model.spec import model_build_spec_from_mappings
from tab_foundry.types import TaskBatch


class _TinyClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, batch: TaskBatch) -> ClassificationOutput:
        return ClassificationOutput(logits=self.linear(batch.x_test), num_classes=2)


class _CapturingClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_batch: TaskBatch | None = None

    def forward(self, batch: TaskBatch) -> ClassificationOutput:
        self.last_batch = batch
        logits = torch.zeros((batch.x_test.shape[0], 2), dtype=batch.x_test.dtype, device=batch.x_test.device)
        return ClassificationOutput(logits=logits, num_classes=2)


def _checkpoint_model_cfg(**overrides: object) -> dict[str, object]:
    return model_build_spec_from_mappings(task="classification", primary=overrides).to_dict()


def test_tab_foundry_classifier_predicts_probabilities(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    fake_spec = SimpleNamespace(task="classification")
    monkeypatch.setattr(
        checkpoint_classifier,
        "checkpoint_model_build_spec_from_mappings",
        lambda **_kwargs: fake_spec,
    )
    monkeypatch.setattr(checkpoint_classifier, "build_model_from_spec", lambda _spec: _TinyClassifier())

    checkpoint = tmp_path / "tiny.pt"
    model = _TinyClassifier()
    torch.save({"model": model.state_dict(), "config": {"task": "classification", "model": {}}}, checkpoint)

    classifier = checkpoint_classifier.TabFoundryClassifier(checkpoint, device="cpu")
    classifier.fit(np.ones((6, 4), dtype=np.float32), np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64))
    probabilities = classifier.predict_proba(np.zeros((3, 4), dtype=np.float32))

    assert probabilities.shape == (3, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-6)


@pytest.mark.parametrize(
    "mode",
    ["none", "train_zscore", "train_zscore_clip"],
)
def test_tab_foundry_classifier_honors_checkpoint_input_normalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    model = _CapturingClassifier()
    fake_spec = SimpleNamespace(task="classification", input_normalization=mode)
    monkeypatch.setattr(
        checkpoint_classifier,
        "checkpoint_model_build_spec_from_mappings",
        lambda **_kwargs: fake_spec,
    )
    monkeypatch.setattr(checkpoint_classifier, "build_model_from_spec", lambda _spec: model)

    checkpoint = tmp_path / f"{mode}.pt"
    torch.save({"model": model.state_dict(), "config": {"task": "classification", "model": {}}}, checkpoint)

    x_train = np.asarray(
        [
            [1.0, 3.0, 10.0, -5.0],
            [2.0, 3.0, 12.0, -5.0],
            [4.0, 3.0, 14.0, -5.0],
        ],
        dtype=np.float32,
    )
    x_test = np.asarray(
        [
            [3.0, 3.0, 16.0, -5.0],
            [5.0, 3.0, 8.0, -5.0],
        ],
        dtype=np.float32,
    )
    classifier = checkpoint_classifier.TabFoundryClassifier(checkpoint, device="cpu")
    classifier.fit(x_train, np.asarray([0, 1, 0], dtype=np.int64))
    _ = classifier.predict_proba(x_test)

    assert model.last_batch is not None
    observed_train = model.last_batch.x_train.cpu().numpy()
    observed_test = model.last_batch.x_test.cpu().numpy()
    if mode == "none":
        expected_train, expected_test = x_train, x_test
    else:
        expected_train, expected_test = normalize_train_test_arrays(x_train, x_test, mode=mode)
    assert np.allclose(observed_train, expected_train, atol=1.0e-6)
    assert np.allclose(observed_test, expected_test, atol=1.0e-6)


def test_tab_foundry_classifier_skips_external_normalization_for_tabfoundry_simple(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _CapturingClassifier()
    fake_spec = SimpleNamespace(
        task="classification",
        arch="tabfoundry_simple",
        input_normalization="train_zscore_clip",
    )
    monkeypatch.setattr(
        checkpoint_classifier,
        "checkpoint_model_build_spec_from_mappings",
        lambda **_kwargs: fake_spec,
    )
    monkeypatch.setattr(checkpoint_classifier, "build_model_from_spec", lambda _spec: model)

    checkpoint = tmp_path / "simple.pt"
    torch.save({"model": model.state_dict(), "config": {"task": "classification", "model": {}}}, checkpoint)

    x_train = np.asarray(
        [
            [1.0, 3.0, 10.0, -5.0],
            [2.0, 3.0, 12.0, -5.0],
            [4.0, 3.0, 14.0, -5.0],
        ],
        dtype=np.float32,
    )
    x_test = np.asarray(
        [
            [3.0, 3.0, 16.0, -5.0],
            [5.0, 3.0, 8.0, -5.0],
        ],
        dtype=np.float32,
    )
    classifier = checkpoint_classifier.TabFoundryClassifier(checkpoint, device="cpu")
    classifier.fit(x_train, np.asarray([0, 1, 0], dtype=np.int64))
    _ = classifier.predict_proba(x_test)

    assert model.last_batch is not None
    assert np.allclose(model.last_batch.x_train.cpu().numpy(), x_train, atol=1.0e-6)
    assert np.allclose(model.last_batch.x_test.cpu().numpy(), x_test, atol=1.0e-6)


def test_tab_foundry_classifier_skips_external_normalization_for_staged_nano_exact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _CapturingClassifier()
    fake_spec = SimpleNamespace(
        task="classification",
        arch="tabfoundry_staged",
        stage="nano_exact",
        input_normalization="train_zscore_clip",
    )
    monkeypatch.setattr(
        checkpoint_classifier,
        "checkpoint_model_build_spec_from_mappings",
        lambda **_kwargs: fake_spec,
    )
    monkeypatch.setattr(checkpoint_classifier, "build_model_from_spec", lambda _spec: model)

    checkpoint = tmp_path / "staged_nano_exact.pt"
    torch.save({"model": model.state_dict(), "config": {"task": "classification", "model": {}}}, checkpoint)

    x_train = np.asarray([[1.0, 3.0], [2.0, 4.0], [4.0, 8.0]], dtype=np.float32)
    x_test = np.asarray([[3.0, 5.0], [5.0, 9.0]], dtype=np.float32)
    classifier = checkpoint_classifier.TabFoundryClassifier(checkpoint, device="cpu")
    classifier.fit(x_train, np.asarray([0, 1, 0], dtype=np.int64))
    _ = classifier.predict_proba(x_test)

    assert model.last_batch is not None
    assert np.allclose(model.last_batch.x_train.cpu().numpy(), x_train, atol=1.0e-6)
    assert np.allclose(model.last_batch.x_test.cpu().numpy(), x_test, atol=1.0e-6)


def test_tab_foundry_classifier_uses_external_normalization_for_staged_shared_norm(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _CapturingClassifier()
    fake_spec = SimpleNamespace(
        task="classification",
        arch="tabfoundry_staged",
        stage="shared_norm",
        input_normalization="train_zscore_clip",
    )
    monkeypatch.setattr(
        checkpoint_classifier,
        "checkpoint_model_build_spec_from_mappings",
        lambda **_kwargs: fake_spec,
    )
    monkeypatch.setattr(checkpoint_classifier, "build_model_from_spec", lambda _spec: model)

    checkpoint = tmp_path / "staged_shared_norm.pt"
    torch.save({"model": model.state_dict(), "config": {"task": "classification", "model": {}}}, checkpoint)

    x_train = np.asarray([[1.0, 3.0], [2.0, 4.0], [4.0, 8.0]], dtype=np.float32)
    x_test = np.asarray([[3.0, 5.0], [5.0, 9.0]], dtype=np.float32)
    classifier = checkpoint_classifier.TabFoundryClassifier(checkpoint, device="cpu")
    classifier.fit(x_train, np.asarray([0, 1, 0], dtype=np.int64))
    _ = classifier.predict_proba(x_test)

    expected_train, expected_test = normalize_train_test_arrays(
        x_train,
        x_test,
        mode="train_zscore_clip",
    )
    assert model.last_batch is not None
    assert np.allclose(model.last_batch.x_train.cpu().numpy(), expected_train, atol=1.0e-6)
    assert np.allclose(model.last_batch.x_test.cpu().numpy(), expected_test, atol=1.0e-6)


def test_load_checkpoint_classifier_model_rejects_legacy_grouped_weights_without_override(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "legacy.pt"
    model = build_model(task="classification", feature_group_size=32)
    checkpoint_model_cfg = _checkpoint_model_cfg(missingness_mode="none")
    checkpoint_model_cfg.pop("feature_group_size")
    torch.save(
        {
            "model": model.state_dict(),
            "config": {"task": "classification", "model": checkpoint_model_cfg},
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="ambiguous across multiple tabfoundry layouts"):
        _ = checkpoint_classifier.load_checkpoint_classifier_model(
            checkpoint,
            device=torch.device("cpu"),
        )


def test_load_checkpoint_classifier_model_supports_explicit_override_for_legacy_weights(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "legacy.pt"
    model = build_model(task="classification", feature_group_size=32)
    checkpoint_model_cfg = _checkpoint_model_cfg(missingness_mode="none")
    checkpoint_model_cfg.pop("feature_group_size")
    torch.save(
        {
            "model": model.state_dict(),
            "config": {"task": "classification", "model": checkpoint_model_cfg},
        },
        checkpoint,
    )
    cfg = OmegaConf.create({"checkpoint_model_overrides": {"feature_group_size": 32}})

    loaded_model, spec = checkpoint_classifier.load_checkpoint_classifier_model(
        checkpoint,
        device=torch.device("cpu"),
        cfg=cfg,
    )

    assert spec.feature_group_size == 32
    assert getattr(loaded_model, "feature_group_size") == 32


def test_frozen_control_baseline_curve_matches_current_checkpoint_wrapper() -> None:
    benchmark_root = Path("outputs/control_baselines/cls_benchmark_linear_v1/benchmark")
    run_root = Path("outputs/control_baselines/cls_benchmark_linear_v1/train")
    dataset_cache_path = benchmark_root / "benchmark_dataset_cache.npz"
    curve_path = benchmark_root / "tab_foundry_curve.jsonl"
    step_100 = run_root / "checkpoints" / "step_000100.pt"
    step_400 = run_root / "checkpoints" / "step_000400.pt"
    if not all(path.exists() for path in (dataset_cache_path, curve_path, step_100, step_400)):
        pytest.skip("frozen control baseline artifacts are not available locally")
    payload = torch.load(step_100, map_location="cpu", weights_only=False)
    model_cfg = payload.get("config", {}).get("model", {})
    if not isinstance(model_cfg, dict) or model_cfg.get("arch") is None:
        pytest.skip("frozen control baseline artifacts predate persisted model.arch metadata")

    curve_by_step: dict[int, float] = {}
    for line in curve_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        payload = json.loads(line)
        curve_by_step[int(payload["step"])] = float(payload["roc_auc"])

    datasets = load_dataset_cache(dataset_cache_path)
    for step in (100, 400):
        checkpoint = run_root / "checkpoints" / f"step_{step:06d}.pt"
        classifier = checkpoint_classifier.TabFoundryClassifier(checkpoint, device="cpu")
        metrics = evaluate_classifier(classifier, datasets)
        assert metrics["ROC AUC"] == pytest.approx(curve_by_step[step], rel=2.0e-4, abs=2.0e-4)
