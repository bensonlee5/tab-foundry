from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json

import numpy as np
import pytest
import torch
from torch import nn

import tab_foundry.bench.checkpoint as checkpoint_classifier
import tab_foundry.bench.openml_benchmark.metrics as benchmark_metrics_module
from tab_foundry.bench.openml_benchmark import evaluate_classifier, load_dataset_cache
from tab_foundry.input_normalization import normalize_train_test_arrays
from tab_foundry.model.outputs import CellLikelihoodOutput, ClassificationOutput
from tab_foundry.preprocessing import preprocess_runtime_task_arrays
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
        logits = torch.zeros(
            (batch.x_test.shape[0], 2), dtype=batch.x_test.dtype, device=batch.x_test.device
        )
        return ClassificationOutput(logits=logits, num_classes=2)


class _CapturingSandwichClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_batches: list[TaskBatch] = []
        self.cell_likelihood_batches: list[TaskBatch] = []

    def forward(self, batch: TaskBatch) -> ClassificationOutput:
        self.forward_batches.append(batch)
        logits = torch.zeros(
            (batch.x_test.shape[0], 2), dtype=batch.x_test.dtype, device=batch.x_test.device
        )
        return ClassificationOutput(logits=logits, num_classes=2)

    def forward_cell_likelihood(self, batch: TaskBatch) -> CellLikelihoodOutput:
        self.cell_likelihood_batches.append(batch)
        per_cell_bits = torch.zeros(
            (1, batch.x_test.shape[0], batch.x_test.shape[1]),
            dtype=batch.x_test.dtype,
            device=batch.x_test.device,
        )
        return CellLikelihoodOutput(
            per_cell_bits=per_cell_bits,
            bpc=torch.tensor(1.25, dtype=batch.x_test.dtype, device=batch.x_test.device),
            bpf=torch.tensor(0.5, dtype=batch.x_test.dtype, device=batch.x_test.device),
            aux_metrics={
                "bpc_cell_count": float(max(batch.x_test.shape[0] * batch.x_test.shape[1] - 1, 1)),
                "bpf_feature_count": float(batch.x_test.shape[1]),
                "excluded_non_finite_cell_count": 1.0,
            },
        )


class _BatchedSandwichClassifier(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.forward_batches: list[TaskBatch] = []
        self.cell_likelihood_batches: list[TaskBatch] = []

    def forward(self, batch: TaskBatch) -> ClassificationOutput:
        self.forward_batches.append(batch)
        if batch.x_test.ndim == 2:
            labels = batch.x_test[:, 0].to(torch.int64)
            logits = torch.full(
                (int(labels.shape[0]), 2),
                -6.0,
                dtype=batch.x_test.dtype,
                device=batch.x_test.device,
            )
            logits[torch.arange(int(labels.shape[0]), device=labels.device), labels] = 6.0
            return ClassificationOutput(logits=logits, num_classes=2)

        labels = batch.x_test[:, :, 0].to(torch.int64)
        logits = torch.full(
            (int(labels.shape[0]), int(labels.shape[1]), 2),
            -6.0,
            dtype=batch.x_test.dtype,
            device=batch.x_test.device,
        )
        task_index = torch.arange(int(labels.shape[0]), device=labels.device).unsqueeze(1)
        row_index = torch.arange(int(labels.shape[1]), device=labels.device).unsqueeze(0)
        logits[task_index, row_index, labels] = 6.0
        return ClassificationOutput(
            logits=logits.reshape(int(labels.shape[0]) * int(labels.shape[1]), 2),
            num_classes=2,
        )

    def forward_cell_likelihood(self, batch: TaskBatch) -> CellLikelihoodOutput:
        self.cell_likelihood_batches.append(batch)
        if batch.x_test.ndim == 2:
            x_all = torch.cat([batch.x_train, batch.x_test], dim=0).unsqueeze(0)
        else:
            x_all = torch.cat([batch.x_train, batch.x_test], dim=1)
        per_cell_bits = torch.where(
            torch.isfinite(x_all),
            torch.full_like(x_all, 0.5),
            torch.full_like(x_all, float("nan")),
        )
        finite_cell_mask = torch.isfinite(per_cell_bits)
        feature_counts = finite_cell_mask.sum(dim=1)
        feature_sums = torch.where(
            finite_cell_mask,
            per_cell_bits,
            torch.zeros_like(per_cell_bits),
        ).sum(dim=1)
        feature_mean_bits = torch.full_like(feature_sums, float("nan"))
        valid_feature_mask = feature_counts > 0
        feature_mean_bits[valid_feature_mask] = (
            feature_sums[valid_feature_mask]
            / feature_counts[valid_feature_mask].to(dtype=feature_sums.dtype)
        )
        bpc = torch.where(
            finite_cell_mask,
            per_cell_bits,
            torch.zeros_like(per_cell_bits),
        ).sum() / finite_cell_mask.sum().to(dtype=per_cell_bits.dtype)
        bpf = torch.where(
            valid_feature_mask,
            feature_mean_bits,
            torch.zeros_like(feature_mean_bits),
        ).sum() / valid_feature_mask.sum().to(dtype=feature_mean_bits.dtype)
        return CellLikelihoodOutput(
            per_cell_bits=per_cell_bits,
            bpc=bpc,
            bpf=bpf,
            aux_metrics={
                "bpc_cell_count": float(finite_cell_mask.sum().item()),
                "bpf_feature_count": float(valid_feature_mask.sum().item()),
                "excluded_non_finite_cell_count": float((~torch.isfinite(x_all)).sum().item()),
            },
        )


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
    monkeypatch.setattr(
        checkpoint_classifier, "build_model_from_spec", lambda _spec: _TinyClassifier()
    )

    checkpoint = tmp_path / "tiny.pt"
    model = _TinyClassifier()
    torch.save(
        {"model": model.state_dict(), "config": {"task": "classification", "model": {}}}, checkpoint
    )

    classifier = checkpoint_classifier.TabFoundryClassifier(checkpoint, device="cpu")
    classifier.fit(
        np.ones((6, 4), dtype=np.float32), np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    )
    probabilities = classifier.predict_proba(np.zeros((3, 4), dtype=np.float32))

    assert probabilities.shape == (3, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1.0e-6)


def test_tab_foundry_classifier_rejects_underwidth_logits(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _UnderwidthLogitClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 2)

        def forward(self, batch: TaskBatch) -> ClassificationOutput:
            logits = torch.zeros(
                (batch.x_test.shape[0], 1), dtype=batch.x_test.dtype, device=batch.x_test.device
            )
            return ClassificationOutput(logits=logits, num_classes=2)

    fake_spec = SimpleNamespace(task="classification")
    monkeypatch.setattr(
        checkpoint_classifier,
        "checkpoint_model_build_spec_from_mappings",
        lambda **_kwargs: fake_spec,
    )
    monkeypatch.setattr(
        checkpoint_classifier,
        "build_model_from_spec",
        lambda _spec: _UnderwidthLogitClassifier(),
    )

    checkpoint = tmp_path / "underwidth_logits.pt"
    torch.save(
        {
            "model": _TinyClassifier().state_dict(),
            "config": {"task": "classification", "model": {}},
        },
        checkpoint,
    )

    classifier = checkpoint_classifier.TabFoundryClassifier(checkpoint, device="cpu")
    classifier.fit(
        np.ones((6, 4), dtype=np.float32), np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    )

    with pytest.raises(RuntimeError, match="logits width=1"):
        _ = classifier.predict_proba(np.zeros((3, 4), dtype=np.float32))


def test_tab_foundry_classifier_rejects_underwidth_class_probs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    class _UnderwidthProbClassifier(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.linear = nn.Linear(4, 2)

        def forward(self, batch: TaskBatch) -> ClassificationOutput:
            probs = torch.full(
                (batch.x_test.shape[0], 1),
                1.0,
                dtype=batch.x_test.dtype,
                device=batch.x_test.device,
            )
            return ClassificationOutput(logits=None, class_probs=probs, num_classes=2)

    fake_spec = SimpleNamespace(task="classification")
    monkeypatch.setattr(
        checkpoint_classifier,
        "checkpoint_model_build_spec_from_mappings",
        lambda **_kwargs: fake_spec,
    )
    monkeypatch.setattr(
        checkpoint_classifier,
        "build_model_from_spec",
        lambda _spec: _UnderwidthProbClassifier(),
    )

    checkpoint = tmp_path / "underwidth_probs.pt"
    torch.save(
        {
            "model": _TinyClassifier().state_dict(),
            "config": {"task": "classification", "model": {}},
        },
        checkpoint,
    )

    classifier = checkpoint_classifier.TabFoundryClassifier(checkpoint, device="cpu")
    classifier.fit(
        np.ones((6, 4), dtype=np.float32), np.asarray([0, 1, 0, 1, 0, 1], dtype=np.int64)
    )

    with pytest.raises(RuntimeError, match="class_probs width=1"):
        _ = classifier.predict_proba(np.zeros((3, 4), dtype=np.float32))


@pytest.mark.parametrize(
    "mode",
    ["none", "train_zscore", "train_zscore_clip", "train_zscore_tanh"],
)
def test_tab_foundry_classifier_honors_checkpoint_input_normalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    mode: str,
) -> None:
    model = _CapturingClassifier()
    fake_spec = SimpleNamespace(
        task="classification",
        arch="tabfoundry_staged",
        stage="shared_norm",
        input_normalization=mode,
    )
    monkeypatch.setattr(
        checkpoint_classifier,
        "checkpoint_model_build_spec_from_mappings",
        lambda **_kwargs: fake_spec,
    )
    monkeypatch.setattr(checkpoint_classifier, "build_model_from_spec", lambda _spec: model)

    checkpoint = tmp_path / f"{mode}.pt"
    torch.save(
        {"model": model.state_dict(), "config": {"task": "classification", "model": {}}}, checkpoint
    )

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
    torch.save(
        {"model": model.state_dict(), "config": {"task": "classification", "model": {}}}, checkpoint
    )

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
    torch.save(
        {"model": model.state_dict(), "config": {"task": "classification", "model": {}}}, checkpoint
    )

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
    torch.save(
        {"model": model.state_dict(), "config": {"task": "classification", "model": {}}}, checkpoint
    )

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


def test_tab_foundry_classifier_applies_runtime_preprocessing_before_normalization(
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

    checkpoint = tmp_path / "train" / "checkpoints" / "step_000100.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {"model": model.state_dict(), "config": {"task": "classification", "model": {}}}, checkpoint
    )

    x_train = np.asarray(
        [
            [np.nan, 10.0],
            [2.0, 12.0],
            [4.0, 14.0],
        ],
        dtype=np.float32,
    )
    x_test = np.asarray(
        [
            [3.0, np.nan],
            [5.0, 8.0],
        ],
        dtype=np.float32,
    )
    y_train = np.asarray([0, 1, 0], dtype=np.int64)

    classifier = checkpoint_classifier.TabFoundryClassifier(checkpoint, device="cpu")
    classifier.fit(x_train, y_train)
    _ = classifier.predict_proba(x_test)

    assert model.last_batch is not None
    expected = preprocess_runtime_task_arrays(
        task="classification",
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=None,
    )
    expected_train, expected_test = normalize_train_test_arrays(
        expected.x_train,
        expected.x_test,
        mode="train_zscore_clip",
    )
    observed_train = model.last_batch.x_train.cpu().numpy()
    observed_test = model.last_batch.x_test.cpu().numpy()
    assert np.all(np.isfinite(observed_train))
    assert np.all(np.isfinite(observed_test))
    assert np.allclose(observed_train, expected_train, atol=1.0e-6)
    assert np.allclose(observed_test, expected_test, atol=1.0e-6)


def test_tab_foundry_classifier_respects_training_surface_preprocessing_overrides(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _CapturingClassifier()
    fake_spec = SimpleNamespace(
        task="classification",
        arch="tabfoundry_staged",
        stage="shared_norm",
        input_normalization="none",
    )
    monkeypatch.setattr(
        checkpoint_classifier,
        "checkpoint_model_build_spec_from_mappings",
        lambda **_kwargs: fake_spec,
    )
    monkeypatch.setattr(checkpoint_classifier, "build_model_from_spec", lambda _spec: model)

    train_dir = tmp_path / "train"
    checkpoint = train_dir / "checkpoints" / "step_000100.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {"model": model.state_dict(), "config": {"task": "classification", "model": {}}}, checkpoint
    )
    (train_dir / "training_surface_record.json").write_text(
        json.dumps(
            {
                "preprocessing": {
                    "surface_label": "test_override",
                    "impute_missing": False,
                    "all_nan_fill": 7.0,
                    "label_mapping": "train_only_remap",
                    "unseen_test_label_policy": "filter",
                    "overrides": {},
                }
            }
        ),
        encoding="utf-8",
    )

    x_train = np.asarray([[np.nan, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32)
    x_test = np.asarray([[6.0, np.nan], [7.0, 8.0]], dtype=np.float32)
    classifier = checkpoint_classifier.TabFoundryClassifier(checkpoint, device="cpu")
    classifier.fit(x_train, np.asarray([0, 1, 0], dtype=np.int64))
    _ = classifier.predict_proba(x_test)

    assert model.last_batch is not None
    assert np.isnan(model.last_batch.x_train.cpu().numpy()).any()
    assert np.isnan(model.last_batch.x_test.cpu().numpy()).any()


def test_tab_foundry_classifier_requires_explicit_feature_types_for_sandwich(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _CapturingSandwichClassifier()
    fake_spec = SimpleNamespace(
        task="classification",
        arch="tabfoundry_sandwich",
        input_normalization="none",
    )
    monkeypatch.setattr(
        checkpoint_classifier,
        "checkpoint_model_build_spec_from_mappings",
        lambda **_kwargs: fake_spec,
    )
    monkeypatch.setattr(checkpoint_classifier, "build_model_from_spec", lambda _spec: model)

    checkpoint = tmp_path / "sandwich_missing_feature_types.pt"
    torch.save(
        {"model": model.state_dict(), "config": {"task": "classification", "model": {}}}, checkpoint
    )

    classifier = checkpoint_classifier.TabFoundryClassifier(checkpoint, device="cpu")
    with pytest.raises(
        RuntimeError,
        match="tabfoundry_sandwich benchmark evaluation requires explicit feature_types",
    ):
        classifier.fit(
            np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
            np.asarray([0, 1, 0], dtype=np.int64),
        )


def test_tab_foundry_classifier_preserves_missingness_and_feature_types_for_sandwich(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _CapturingSandwichClassifier()
    fake_spec = SimpleNamespace(
        task="classification",
        arch="tabfoundry_sandwich",
        input_normalization="train_zscore_clip",
    )
    monkeypatch.setattr(
        checkpoint_classifier,
        "checkpoint_model_build_spec_from_mappings",
        lambda **_kwargs: fake_spec,
    )
    monkeypatch.setattr(checkpoint_classifier, "build_model_from_spec", lambda _spec: model)

    checkpoint = tmp_path / "sandwich_missingness.pt"
    torch.save(
        {"model": model.state_dict(), "config": {"task": "classification", "model": {}}}, checkpoint
    )

    x_train = np.asarray(
        [[1.0, np.nan, np.inf, -np.inf], [2.0, 4.0, 6.0, 8.0], [4.0, 8.0, 10.0, 12.0]],
        dtype=np.float32,
    )
    x_test = np.asarray(
        [[3.0, np.nan, np.inf, -np.inf], [5.0, 9.0, 11.0, 13.0]],
        dtype=np.float32,
    )
    feature_types = ["floating", "integer", "floating", "integer"]
    classifier = checkpoint_classifier.TabFoundryClassifier(checkpoint, device="cpu")
    classifier.set_benchmark_feature_types(feature_types)
    classifier.fit(x_train, np.asarray([0, 1, 0], dtype=np.int64))
    _ = classifier.predict_proba(x_test)

    assert model.forward_batches
    batch = model.forward_batches[-1]
    assert batch.metadata["feature_types"] == feature_types
    assert np.isnan(batch.x_train.cpu().numpy()[0, 1])
    assert np.isnan(batch.x_test.cpu().numpy()[0, 1])
    assert np.isposinf(batch.x_train.cpu().numpy()[0, 2])
    assert np.isposinf(batch.x_test.cpu().numpy()[0, 2])
    assert np.isneginf(batch.x_train.cpu().numpy()[0, 3])
    assert np.isneginf(batch.x_test.cpu().numpy()[0, 3])


def test_tab_foundry_classifier_cell_likelihood_metrics_propagates_valid_counts(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _CapturingSandwichClassifier()
    fake_spec = SimpleNamespace(
        task="classification",
        arch="tabfoundry_sandwich",
        input_normalization="none",
    )
    monkeypatch.setattr(
        checkpoint_classifier,
        "checkpoint_model_build_spec_from_mappings",
        lambda **_kwargs: fake_spec,
    )
    monkeypatch.setattr(checkpoint_classifier, "build_model_from_spec", lambda _spec: model)

    checkpoint = tmp_path / "sandwich_cell_metrics.pt"
    torch.save(
        {"model": model.state_dict(), "config": {"task": "classification", "model": {}}}, checkpoint
    )

    classifier = checkpoint_classifier.TabFoundryClassifier(checkpoint, device="cpu")
    classifier.set_benchmark_feature_types(["floating", "integer"])
    classifier.fit(
        np.asarray([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=np.float32),
        np.asarray([0, 1, 0], dtype=np.int64),
    )

    metrics = classifier.cell_likelihood_metrics(
        np.asarray([[7.0, 8.0], [9.0, 10.0]], dtype=np.float32)
    )

    assert metrics["bpc"] == pytest.approx(1.25)
    assert metrics["bpf"] == pytest.approx(0.5)
    assert metrics["bpc_cell_count"] == pytest.approx(3.0)
    assert metrics["bpf_feature_count"] == pytest.approx(2.0)
    assert metrics["excluded_non_finite_cell_count"] == pytest.approx(1.0)


def test_tab_foundry_classifier_batched_hook_matches_serial_fold_results(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _BatchedSandwichClassifier()
    fake_spec = SimpleNamespace(
        task="classification",
        arch="tabfoundry_sandwich",
        input_normalization="none",
    )
    monkeypatch.setattr(
        checkpoint_classifier,
        "checkpoint_model_build_spec_from_mappings",
        lambda **_kwargs: fake_spec,
    )
    monkeypatch.setattr(checkpoint_classifier, "build_model_from_spec", lambda _spec: model)

    checkpoint = tmp_path / "sandwich_batched_metrics.pt"
    torch.save(
        {"model": model.state_dict(), "config": {"task": "classification", "model": {}}}, checkpoint
    )

    classifier = checkpoint_classifier.TabFoundryClassifier(checkpoint, device="cpu")
    folds = [
        benchmark_metrics_module.BenchmarkClassificationFold(
            fold_id=index,
            dataset_name="toy",
            x_train=np.asarray([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            y_train=np.asarray([0, 1, 0, 1], dtype=np.int64),
            x_test=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            y_test=np.asarray([0, 1], dtype=np.int64),
            labels=np.asarray([0, 1], dtype=np.int64),
            feature_types=["floating", "floating"],
        )
        for index in range(2)
    ]

    serial_results = [
        benchmark_metrics_module._evaluate_classifier_fold_serial(classifier, fold) for fold in folds
    ]
    model.forward_batches.clear()
    model.cell_likelihood_batches.clear()

    batched_results = classifier.evaluate_benchmark_folds_batched(folds)

    assert len(model.forward_batches) == 1
    assert len(model.cell_likelihood_batches) == 1
    assert model.forward_batches[0].x_test.ndim == 3
    assert model.forward_batches[0].x_test.shape[0] == 2
    for observed, expected in zip(batched_results, serial_results, strict=True):
        assert observed.classifier_labels is not None
        assert expected.classifier_labels is not None
        assert np.array_equal(observed.classifier_labels, expected.classifier_labels)
        assert np.allclose(observed.probabilities, expected.probabilities, atol=1.0e-12)
        assert observed.cell_likelihood_metrics == pytest.approx(
            expected.cell_likelihood_metrics,
            abs=1.0e-12,
            rel=1.0e-12,
        )


def test_tab_foundry_classifier_batched_hook_falls_back_to_serial_for_incompatible_folds(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _BatchedSandwichClassifier()
    fake_spec = SimpleNamespace(
        task="classification",
        arch="tabfoundry_sandwich",
        input_normalization="none",
    )
    monkeypatch.setattr(
        checkpoint_classifier,
        "checkpoint_model_build_spec_from_mappings",
        lambda **_kwargs: fake_spec,
    )
    monkeypatch.setattr(checkpoint_classifier, "build_model_from_spec", lambda _spec: model)

    checkpoint = tmp_path / "sandwich_batched_fallback.pt"
    torch.save(
        {"model": model.state_dict(), "config": {"task": "classification", "model": {}}}, checkpoint
    )

    classifier = checkpoint_classifier.TabFoundryClassifier(checkpoint, device="cpu")
    folds = [
        benchmark_metrics_module.BenchmarkClassificationFold(
            fold_id=0,
            dataset_name="wide2",
            x_train=np.asarray([[0.0, 0.0], [1.0, 1.0], [0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            y_train=np.asarray([0, 1, 0, 1], dtype=np.int64),
            x_test=np.asarray([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32),
            y_test=np.asarray([0, 1], dtype=np.int64),
            labels=np.asarray([0, 1], dtype=np.int64),
            feature_types=["floating", "floating"],
        ),
        benchmark_metrics_module.BenchmarkClassificationFold(
            fold_id=1,
            dataset_name="wide3",
            x_train=np.asarray(
                [[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [0.0, 0.0, 0.0], [1.0, 1.0, 1.0]],
                dtype=np.float32,
            ),
            y_train=np.asarray([0, 1, 0, 1], dtype=np.int64),
            x_test=np.asarray([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32),
            y_test=np.asarray([0, 1], dtype=np.int64),
            labels=np.asarray([0, 1], dtype=np.int64),
            feature_types=["floating", "floating", "floating"],
        ),
    ]

    _ = benchmark_metrics_module._evaluate_classifier_folds(classifier, folds)

    assert len(model.forward_batches) == 2
    assert len(model.cell_likelihood_batches) == 2
    assert all(batch.x_test.ndim == 2 for batch in model.forward_batches)


def test_tab_foundry_classifier_sandwich_benchmark_paths_honor_checkpoint_input_normalization(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _CapturingSandwichClassifier()
    fake_spec = SimpleNamespace(
        task="classification",
        arch="tabfoundry_sandwich",
        input_normalization="train_zscore_clip",
    )
    monkeypatch.setattr(
        checkpoint_classifier,
        "checkpoint_model_build_spec_from_mappings",
        lambda **_kwargs: fake_spec,
    )
    monkeypatch.setattr(checkpoint_classifier, "build_model_from_spec", lambda _spec: model)

    checkpoint = tmp_path / "sandwich_cell_metrics_norm.pt"
    torch.save(
        {"model": model.state_dict(), "config": {"task": "classification", "model": {}}},
        checkpoint,
    )

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
    expected_train, expected_test = normalize_train_test_arrays(
        x_train,
        x_test,
        mode="train_zscore_clip",
    )

    classifier = checkpoint_classifier.TabFoundryClassifier(checkpoint, device="cpu")
    classifier.set_benchmark_feature_types(["floating", "floating", "floating", "floating"])
    classifier.fit(x_train, np.asarray([0, 1, 0], dtype=np.int64))
    _ = classifier.predict_proba(x_test)
    _ = classifier.cell_likelihood_metrics(x_test)

    assert model.forward_batches
    assert model.cell_likelihood_batches
    predict_batch = model.forward_batches[-1]
    likelihood_batch = model.cell_likelihood_batches[-1]
    assert np.allclose(predict_batch.x_train.cpu().numpy(), expected_train, atol=1.0e-6)
    assert np.allclose(predict_batch.x_test.cpu().numpy(), expected_test, atol=1.0e-6)
    assert np.allclose(likelihood_batch.x_train.cpu().numpy(), expected_train, atol=1.0e-6)
    assert np.allclose(likelihood_batch.x_test.cpu().numpy(), expected_test, atol=1.0e-6)


def test_load_checkpoint_classifier_model_rejects_legacy_grouped_weights_without_override(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "legacy.pt"
    torch.save(
        {
            "model": {"group_linear.weight": torch.zeros((128, 96))},
            "config": {"task": "classification", "model": {}},
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="Legacy tabfoundry checkpoints"):
        _ = checkpoint_classifier.load_checkpoint_classifier_model(
            checkpoint,
            device=torch.device("cpu"),
        )


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
        assert float(metrics["Log Loss"]) >= 0.0


def test_tab_foundry_classifier_skips_external_normalization_for_staged_missingness_token(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _CapturingClassifier()
    fake_spec = SimpleNamespace(
        task="classification",
        arch="tabfoundry_staged",
        stage="shared_norm",
        input_normalization="train_zscore_clip",
        module_overrides={"tokenizer": "scalar_per_feature_nan_mask"},
    )
    monkeypatch.setattr(
        checkpoint_classifier,
        "checkpoint_model_build_spec_from_mappings",
        lambda **_kwargs: fake_spec,
    )
    monkeypatch.setattr(checkpoint_classifier, "build_model_from_spec", lambda _spec: model)

    checkpoint = tmp_path / "staged_missingness.pt"
    torch.save(
        {"model": model.state_dict(), "config": {"task": "classification", "model": {}}}, checkpoint
    )

    x_train = np.asarray(
        [[1.0, np.nan, np.inf, -np.inf], [2.0, 4.0, 6.0, 8.0], [4.0, 8.0, 10.0, 12.0]],
        dtype=np.float32,
    )
    x_test = np.asarray([[3.0, np.nan, np.inf, -np.inf], [5.0, 9.0, 11.0, 13.0]], dtype=np.float32)
    classifier = checkpoint_classifier.TabFoundryClassifier(checkpoint, device="cpu")
    classifier.fit(x_train, np.asarray([0, 1, 0], dtype=np.int64))
    _ = classifier.predict_proba(x_test)

    assert model.last_batch is not None
    assert np.isnan(model.last_batch.x_train.cpu().numpy()[0, 1])
    assert np.isnan(model.last_batch.x_test.cpu().numpy()[0, 1])
    assert np.isposinf(model.last_batch.x_train.cpu().numpy()[0, 2])
    assert np.isposinf(model.last_batch.x_test.cpu().numpy()[0, 2])
    assert np.isneginf(model.last_batch.x_train.cpu().numpy()[0, 3])
    assert np.isneginf(model.last_batch.x_test.cpu().numpy()[0, 3])


def test_evaluate_classifier_reports_sandwich_bpc_bpf_with_missing_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model = _CapturingSandwichClassifier()
    fake_spec = SimpleNamespace(
        task="classification",
        arch="tabfoundry_sandwich",
        input_normalization="none",
    )
    monkeypatch.setattr(
        checkpoint_classifier,
        "checkpoint_model_build_spec_from_mappings",
        lambda **_kwargs: fake_spec,
    )
    monkeypatch.setattr(checkpoint_classifier, "build_model_from_spec", lambda _spec: model)

    checkpoint = tmp_path / "run" / "checkpoints" / "best.pt"
    checkpoint.parent.mkdir(parents=True)
    torch.save(
        {"model": model.state_dict(), "config": {"task": "classification", "model": {}}}, checkpoint
    )

    classifier = checkpoint_classifier.TabFoundryClassifier(checkpoint, device="cpu")
    feature_types = ["floating", "integer"]
    x = np.asarray(
        [
            [np.nan, 0.0],
            [0.0, np.inf],
            [-np.inf, 1.0],
            [1.0, 0.0],
            [0.5, 0.0],
            [0.0, 0.5],
            [1.5, 0.0],
            [0.0, 1.5],
            [2.0, 0.0],
            [0.0, 2.0],
        ],
        dtype=np.float32,
    )
    y = np.asarray([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64)

    metrics = evaluate_classifier(
        classifier,
        {"sandwich": (x, y, feature_types)},
        allow_missing_values=True,
    )

    assert metrics["sandwich/BPC"] == pytest.approx(1.25)
    assert metrics["sandwich/BPF"] == pytest.approx(0.5)
    assert metrics["BPC"] == pytest.approx(1.25)
    assert metrics["BPF"] == pytest.approx(0.5)
    assert model.forward_batches
    assert model.cell_likelihood_batches
    assert all(batch.metadata["feature_types"] == feature_types for batch in model.forward_batches)
    assert all(
        batch.metadata["feature_types"] == feature_types for batch in model.cell_likelihood_batches
    )
    assert any(
        np.isnan(batch.x_train.cpu().numpy()).any() or np.isnan(batch.x_test.cpu().numpy()).any()
        for batch in model.forward_batches + model.cell_likelihood_batches
    )
    assert any(
        np.isinf(batch.x_train.cpu().numpy()).any() or np.isinf(batch.x_test.cpu().numpy()).any()
        for batch in model.forward_batches + model.cell_likelihood_batches
    )
