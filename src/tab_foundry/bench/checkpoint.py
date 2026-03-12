"""Generic checkpoint-backed classifier helpers for external benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F

from tab_foundry.input_normalization import normalize_train_test_arrays
from tab_foundry.model.factory import build_model_from_spec, model_build_spec_from_mappings
from tab_foundry.types import TaskBatch


def _checkpoint_model_spec(payload: dict[str, Any], cfg: DictConfig | None = None):
    cfg_payload = payload.get("config")
    checkpoint_cfg = cfg_payload if isinstance(cfg_payload, dict) else {}
    task_raw = checkpoint_cfg.get("task", "classification")
    task = str(task_raw).strip().lower()
    if task != "classification":
        raise RuntimeError(f"Checkpoint classifier requires classification checkpoint, got {task!r}")

    fallback_cfg: dict[str, Any] = {}
    if cfg is not None:
        raw_fallback = OmegaConf.to_container(cfg.model, resolve=True)
        if isinstance(raw_fallback, dict):
            fallback_cfg = {str(key): value for key, value in raw_fallback.items()}
    model_cfg = checkpoint_cfg.get("model")
    primary_cfg: dict[str, Any] = {}
    if isinstance(model_cfg, dict):
        primary_cfg = {str(key): value for key, value in model_cfg.items()}
    return model_build_spec_from_mappings(task=task, primary=primary_cfg, fallback=fallback_cfg)


def load_checkpoint_classifier_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
    cfg: DictConfig | None = None,
) -> tuple[torch.nn.Module, Any]:
    """Load one classification checkpoint as an inference-ready model."""

    checkpoint = checkpoint_path.expanduser().resolve()
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise RuntimeError("checkpoint payload must be a mapping")
    spec = _checkpoint_model_spec(payload, cfg=cfg)
    model = build_model_from_spec(spec)
    model.load_state_dict(payload["model"])
    model.to(device)
    model.eval()
    return model, spec


def normalize_train_test(
    x_train: np.ndarray,
    x_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalize features using train-only statistics."""

    mean = x_train.mean(axis=0)
    std = x_train.std(axis=0) + 1.0e-8
    return (x_train - mean) / std, (x_test - mean) / std


class TabFoundryClassifier:
    """Small sklearn-style classifier wrapper around a tab-foundry checkpoint."""

    def __init__(self, checkpoint_path: Path, *, device: str = "cpu") -> None:
        self.checkpoint_path = checkpoint_path.expanduser().resolve()
        self.device = torch.device(device)
        self.model, self.model_spec = load_checkpoint_classifier_model(
            self.checkpoint_path,
            device=self.device,
        )
        self._classes: np.ndarray | None = None
        self._x_train: np.ndarray | None = None
        self._y_train: np.ndarray | None = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "TabFoundryClassifier":
        classes, encoded = np.unique(np.asarray(y_train), return_inverse=True)
        if classes.size < 2:
            raise RuntimeError("benchmark classifier requires at least 2 classes in fit()")
        self._classes = classes
        self._x_train = np.asarray(x_train, dtype=np.float32)
        self._y_train = encoded.astype(np.int64, copy=False)
        return self

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        if self._classes is None or self._x_train is None or self._y_train is None:
            raise RuntimeError("fit() must be called before predict_proba()")

        raw_x_test = np.asarray(x_test, dtype=np.float32)
        if str(getattr(self.model_spec, "input_normalization", "none")).strip().lower() == "none":
            x_train_norm, x_test_norm = normalize_train_test_arrays(
                self._x_train,
                raw_x_test,
                mode="train_zscore",
            )
        else:
            x_train_norm, x_test_norm = self._x_train, raw_x_test
        num_classes = int(self._classes.size)
        batch = TaskBatch(
            x_train=torch.tensor(x_train_norm, dtype=torch.float32, device=self.device),
            y_train=torch.tensor(self._y_train, dtype=torch.int64, device=self.device),
            x_test=torch.tensor(x_test_norm, dtype=torch.float32, device=self.device),
            y_test=torch.zeros((x_test_norm.shape[0],), dtype=torch.int64, device=self.device),
            metadata={"dataset": "external_benchmark"},
            num_classes=num_classes,
        )
        with torch.no_grad():
            output = self.model(batch)
            if output.logits is not None:
                probs = F.softmax(output.logits[:, :num_classes], dim=-1)
            elif output.class_probs is not None:
                probs = output.class_probs[:, :num_classes]
            else:
                raise RuntimeError("checkpoint output does not expose logits or class probabilities")
        return probs.cpu().numpy()

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(x_test)
        classes = self._classes
        if classes is None:
            raise RuntimeError("fit() must be called before predict()")
        return classes[np.asarray(probabilities.argmax(axis=1), dtype=np.int64)]
