"""Checkpoint-backed prediction helpers for external benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F

from tab_foundry.input_normalization import (
    InputNormalizationMode,
    normalize_train_test_arrays,
)
from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.model.outputs import ClassificationOutput, validate_classification_output_contract
from tab_foundry.model.spec import (
    ModelBuildSpec,
    checkpoint_model_build_spec_from_mappings,
)
from tab_foundry.model.architectures.tabfoundry_staged.resolved import (
    resolve_staged_surface,
    staged_surface_uses_internal_benchmark_normalization,
)
from tab_foundry.preprocessing import (
    FittedPreprocessorState,
    apply_fitted_preprocessor,
    fit_fitted_preprocessor,
    resolve_preprocessing_surface,
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
        raise RuntimeError(
            "checkpoint helper only supports classification checkpoints in this branch, "
            f"got {task!r}"
        )

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


def load_checkpoint_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
    cfg: DictConfig | None = None,
) -> tuple[torch.nn.Module, Any]:
    """Load one checkpoint as an inference-ready model."""

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


def load_checkpoint_classifier_model(
    checkpoint_path: Path,
    *,
    device: torch.device,
    cfg: DictConfig | None = None,
) -> tuple[torch.nn.Module, Any]:
    """Load one classification checkpoint as an inference-ready model."""

    model, spec = load_checkpoint_model(checkpoint_path, device=device, cfg=cfg)
    task = str(getattr(spec, "task", "classification")).strip().lower()
    if task != "classification":
        raise RuntimeError(f"Checkpoint classifier requires classification checkpoint, got {task!r}")
    return model, spec


def _checkpoint_training_surface_record_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.expanduser().resolve().parent.parent / "training_surface_record.json"


def _checkpoint_preprocessing_surface(checkpoint_path: Path) -> Any:
    record_path = _checkpoint_training_surface_record_path(checkpoint_path)
    if not record_path.exists():
        return resolve_preprocessing_surface(None)
    try:
        payload = json.loads(record_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(
            f"failed to load training_surface_record.json for benchmark checkpoint: {record_path}"
        ) from exc
    if not isinstance(payload, Mapping):
        raise RuntimeError(
            "training_surface_record.json must be a JSON object for benchmark checkpoint "
            f"preprocessing: {record_path}"
        )
    raw_preprocessing = payload.get("preprocessing")
    if raw_preprocessing is not None and not isinstance(raw_preprocessing, Mapping):
        raise RuntimeError(
            "training_surface_record.json preprocessing entry must be a JSON object when present: "
            f"{record_path}"
        )
    return resolve_preprocessing_surface(
        None if raw_preprocessing is None else cast(Mapping[str, Any], raw_preprocessing)
    )


def _staged_checkpoint_uses_missingness_token(spec: Any) -> bool:
    if str(getattr(spec, "arch", "")).strip().lower() != "tabfoundry_staged":
        return False
    if not isinstance(spec, ModelBuildSpec):
        raw_overrides = getattr(spec, "module_overrides", None)
        if isinstance(raw_overrides, Mapping):
            tokenizer = raw_overrides.get("tokenizer")
            if isinstance(tokenizer, str) and tokenizer.strip().lower() == "scalar_per_feature_nan_mask":
                return True
        return False
    return resolve_staged_surface(spec).tokenizer == "scalar_per_feature_nan_mask"


class TabFoundryClassifier:
    """Small sklearn-style classifier wrapper around a tab-foundry checkpoint."""

    def __init__(self, checkpoint_path: Path, *, device: str = "cpu") -> None:
        self.checkpoint_path = checkpoint_path.expanduser().resolve()
        self.device = torch.device(device)
        self.model, self.model_spec = load_checkpoint_classifier_model(
            self.checkpoint_path,
            device=self.device,
        )
        self.preprocessing_surface = _checkpoint_preprocessing_surface(self.checkpoint_path)
        self._preserve_non_finite_inputs = _staged_checkpoint_uses_missingness_token(self.model_spec)
        self._classes: np.ndarray | None = None
        self._preprocessor_state: FittedPreprocessorState | None = None
        self._raw_x_train: np.ndarray | None = None
        self._raw_y_train: np.ndarray | None = None

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "TabFoundryClassifier":
        raw_x_train = np.asarray(x_train, dtype=np.float32)
        raw_y_train = np.asarray(y_train, dtype=np.int64)
        classes = np.unique(raw_y_train)
        if classes.size < 2:
            raise RuntimeError("benchmark classifier requires at least 2 classes in fit()")
        self._classes = classes
        self._raw_x_train = raw_x_train
        self._raw_y_train = raw_y_train
        self._preprocessor_state = fit_fitted_preprocessor(
            task="classification",
            x_train=raw_x_train,
            y_train=raw_y_train,
            all_nan_fill=float(self.preprocessing_surface.all_nan_fill),
            label_mapping=str(self.preprocessing_surface.label_mapping),
            unseen_test_label_policy=str(self.preprocessing_surface.unseen_test_label_policy),
        )
        return self

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        if (
            self._classes is None
            or self._preprocessor_state is None
            or self._raw_x_train is None
            or self._raw_y_train is None
        ):
            raise RuntimeError("fit() must be called before predict_proba()")

        raw_x_test = np.asarray(x_test, dtype=np.float32)
        processed = apply_fitted_preprocessor(
            task="classification",
            state=self._preprocessor_state,
            x_train=self._raw_x_train,
            y_train=self._raw_y_train,
            x_test=raw_x_test,
            y_test=None,
            impute_missing=bool(
                self.preprocessing_surface.impute_missing and not self._preserve_non_finite_inputs
            ),
        )
        model_arch = str(getattr(self.model_spec, "arch", "tabfoundry_staged")).strip().lower()
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
            x_train_norm, x_test_norm = processed.x_train, processed.x_test
        else:
            x_train_norm, x_test_norm = normalize_train_test_arrays(
                processed.x_train,
                processed.x_test,
                mode=normalization_mode,
            )
        num_classes = int(processed.num_classes or self._classes.size)
        batch = TaskBatch(
            x_train=torch.tensor(x_train_norm, dtype=torch.float32, device=self.device),
            y_train=torch.tensor(processed.y_train, dtype=torch.int64, device=self.device),
            x_test=torch.tensor(x_test_norm, dtype=torch.float32, device=self.device),
            y_test=torch.zeros((x_test_norm.shape[0],), dtype=torch.int64, device=self.device),
            metadata={"dataset": "external_benchmark"},
            num_classes=num_classes,
        )
        with torch.no_grad():
            output = self.model(batch)
            if not isinstance(output, ClassificationOutput):
                raise RuntimeError("checkpoint output does not expose classification probabilities")
            resolved_num_classes = validate_classification_output_contract(
                output,
                expected_rows=int(batch.x_test.shape[0]),
                expected_num_classes=num_classes,
                context="checkpoint classifier",
            )
            if output.logits is not None:
                probs = F.softmax(output.logits[:, :resolved_num_classes], dim=-1)
            elif output.class_probs is not None:
                probs = output.class_probs
            else:
                raise RuntimeError("checkpoint output does not expose logits or class probabilities")
        return probs.cpu().numpy()

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(x_test)
        classes = self._classes
        if classes is None:
            raise RuntimeError("fit() must be called before predict()")
        return classes[np.asarray(probabilities.argmax(axis=1), dtype=np.int64)]
