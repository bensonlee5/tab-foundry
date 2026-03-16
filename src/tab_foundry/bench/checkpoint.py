"""Generic checkpoint-backed classifier helpers for external benchmarks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F

from tab_foundry.input_normalization import (
    InputNormalizationMode,
    normalize_train_test_arrays,
)
from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.model.spec import (
    ModelBuildSpec,
    checkpoint_model_build_spec_from_mappings,
)
from tab_foundry.model.architectures.tabfoundry_staged.resolved import (
    staged_surface_uses_internal_benchmark_normalization,
)
from tab_foundry.preprocessing import (
    feature_types_include_categorical,
    fit_fitted_preprocessor,
    preprocess_runtime_task_arrays,
)
from tab_foundry.types import TaskBatch


def _checkpoint_model_spec(
    payload: dict[str, Any],
    cfg: DictConfig | None = None,
) -> ModelBuildSpec:
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
    model_state = payload.get("model")
    state_dict = model_state if isinstance(model_state, dict) else None
    return checkpoint_model_build_spec_from_mappings(
        task=task,
        primary=primary_cfg,
        fallback=fallback_cfg,
        state_dict=state_dict,
    )


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


def _normalize_preprocessed_feature_arrays(
    x_train: np.ndarray,
    x_test: np.ndarray,
    *,
    mode: InputNormalizationMode,
    categorical_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if categorical_mask is None or not bool(np.any(categorical_mask)):
        return normalize_train_test_arrays(x_train, x_test, mode=mode)
    numeric_mask = ~np.asarray(categorical_mask, dtype=bool)
    train_out = np.asarray(x_train, dtype=np.float32).copy()
    test_out = np.asarray(x_test, dtype=np.float32).copy()
    if np.any(numeric_mask):
        numeric_train, numeric_test = normalize_train_test_arrays(
            train_out[:, numeric_mask],
            test_out[:, numeric_mask],
            mode=mode,
        )
        train_out[:, numeric_mask] = numeric_train
        test_out[:, numeric_mask] = numeric_test
    return train_out, test_out


def _require_staged_categorical_support(
    *,
    feature_types: Sequence[str] | None,
    model_arch: str,
) -> None:
    if feature_types_include_categorical(feature_types) and model_arch != "tabfoundry_staged":
        raise RuntimeError(
            "categorical feature_types are only supported for tabfoundry_staged checkpoints; "
            f"got arch={model_arch!r}"
        )


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
        self._feature_types: list[str] | None = None

    def fit(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        *,
        feature_types: Sequence[str] | None = None,
    ) -> "TabFoundryClassifier":
        classes, encoded = np.unique(np.asarray(y_train), return_inverse=True)
        if classes.size < 2:
            raise RuntimeError("benchmark classifier requires at least 2 classes in fit()")
        model_arch = str(getattr(self.model_spec, "arch", "tabfoundry")).strip().lower()
        _require_staged_categorical_support(feature_types=feature_types, model_arch=model_arch)
        self._classes = classes
        self._x_train = np.asarray(x_train, dtype=np.float32)
        self._y_train = encoded.astype(np.int64, copy=False)
        state = fit_fitted_preprocessor(
            task="classification",
            x_train=self._x_train,
            y_train=self._y_train,
            feature_types=feature_types,
        )
        self._feature_types = list(state.categorical_feature_policy.feature_types)
        return self

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        if self._classes is None or self._x_train is None or self._y_train is None:
            raise RuntimeError("fit() must be called before predict_proba()")

        raw_x_test = np.asarray(x_test, dtype=np.float32)
        processed = preprocess_runtime_task_arrays(
            task="classification",
            x_train=self._x_train,
            y_train=self._y_train,
            x_test=raw_x_test,
            y_test=None,
            feature_types=self._feature_types,
        )
        x_train_model = np.asarray(processed.x_train, dtype=np.float32)
        x_test_model = np.asarray(processed.x_test, dtype=np.float32)
        model_arch = str(getattr(self.model_spec, "arch", "tabfoundry")).strip().lower()
        normalization_mode = cast(
            InputNormalizationMode,
            str(getattr(self.model_spec, "input_normalization", "none")).strip().lower(),
        )
        internal_normalization = model_arch == "tabfoundry_simple"
        if model_arch == "tabfoundry_staged":
            internal_normalization = staged_surface_uses_internal_benchmark_normalization(
                self.model_spec,
            )
        if internal_normalization or normalization_mode == "none":
            x_train_norm, x_test_norm = x_train_model, x_test_model
        else:
            x_train_norm, x_test_norm = _normalize_preprocessed_feature_arrays(
                x_train_model,
                x_test_model,
                mode=normalization_mode,
                categorical_mask=None
                if processed.feature_state is None
                else processed.feature_state.categorical_mask,
            )
        num_classes = int(self._classes.size)
        batch = TaskBatch(
            x_train=torch.tensor(x_train_norm, dtype=torch.float32, device=self.device),
            y_train=torch.tensor(self._y_train, dtype=torch.int64, device=self.device),
            x_test=torch.tensor(x_test_norm, dtype=torch.float32, device=self.device),
            y_test=torch.zeros((x_test_norm.shape[0],), dtype=torch.int64, device=self.device),
            metadata={"dataset": "external_benchmark"},
            num_classes=num_classes,
            feature_state=None
            if processed.feature_state is None
            else processed.feature_state.to_task_feature_state().to(self.device),
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
