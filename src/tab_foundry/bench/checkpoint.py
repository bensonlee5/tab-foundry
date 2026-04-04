"""Checkpoint-backed prediction helpers for external benchmarks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.nn.functional as F

from tab_foundry.bench.openml_benchmark.metrics import (
    BenchmarkClassificationFold,
    BenchmarkClassificationFoldResult,
)
from tab_foundry.input_normalization import (
    InputNormalizationMode,
    normalize_train_test_arrays,
)
from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.model.outputs import ClassificationOutput, validate_classification_output_contract
from tab_foundry.model.spec import (
    ModelBuildSpec,
    SANDWICH_MODEL_ARCH,
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
from tab_foundry.task_batching import collate_task_batch, move_batch
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
        raise RuntimeError(
            f"Checkpoint classifier requires classification checkpoint, got {task!r}"
        )
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


def _checkpoint_preserves_non_finite_benchmark_inputs(spec: Any) -> bool:
    arch = str(getattr(spec, "arch", "")).strip().lower()
    if arch == SANDWICH_MODEL_ARCH:
        return True
    if arch != "tabfoundry_staged":
        return False
    if not isinstance(spec, ModelBuildSpec):
        raw_overrides = getattr(spec, "module_overrides", None)
        if isinstance(raw_overrides, Mapping):
            tokenizer = raw_overrides.get("tokenizer")
            if (
                isinstance(tokenizer, str)
                and tokenizer.strip().lower() == "scalar_per_feature_nan_mask"
            ):
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
        self._preserve_non_finite_inputs = _checkpoint_preserves_non_finite_benchmark_inputs(
            self.model_spec
        )
        self._benchmark_feature_types: list[str] | None = None
        self._classes: np.ndarray | None = None
        self._preprocessor_state: FittedPreprocessorState | None = None
        self._raw_x_train: np.ndarray | None = None
        self._raw_y_train: np.ndarray | None = None
        if str(getattr(self.model_spec, "arch", "")).strip().lower() == SANDWICH_MODEL_ARCH:
            self.evaluate_benchmark_folds_batched = self._evaluate_benchmark_folds_batched_impl

    def set_benchmark_feature_types(self, feature_types: list[str] | None) -> None:
        if feature_types is None:
            self._benchmark_feature_types = None
            return
        if not isinstance(feature_types, list) or not all(
            isinstance(value, str) for value in feature_types
        ):
            raise RuntimeError("benchmark feature_types must be a list of strings")
        self._benchmark_feature_types = list(feature_types)

    def fit(self, x_train: np.ndarray, y_train: np.ndarray) -> "TabFoundryClassifier":
        raw_x_train = np.asarray(x_train, dtype=np.float32)
        raw_y_train = np.asarray(y_train, dtype=np.int64)
        classes = np.unique(raw_y_train)
        if classes.size < 2:
            raise RuntimeError("benchmark classifier requires at least 2 classes in fit()")
        if (
            str(getattr(self.model_spec, "arch", "")).strip().lower() == SANDWICH_MODEL_ARCH
            and self._benchmark_feature_types is None
        ):
            raise RuntimeError(
                "tabfoundry_sandwich benchmark evaluation requires explicit "
                "feature_types for each dataset"
            )
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
            feature_types=self._benchmark_feature_types,
        )
        return self

    def _require_fitted_benchmark_state(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, FittedPreprocessorState]:
        if (
            self._classes is None
            or self._preprocessor_state is None
            or self._raw_x_train is None
            or self._raw_y_train is None
        ):
            raise RuntimeError("fit() must be called before benchmark inference")
        return (
            self._classes,
            self._raw_x_train,
            self._raw_y_train,
            self._preprocessor_state,
        )

    def _preprocess_benchmark_inputs(
        self,
        x_test: np.ndarray,
    ) -> tuple[np.ndarray, FittedPreprocessorState, Any]:
        classes, raw_x_train, raw_y_train, preprocessor_state = self._require_fitted_benchmark_state()
        raw_x_test = np.asarray(x_test, dtype=np.float32)
        processed = apply_fitted_preprocessor(
            task="classification",
            state=preprocessor_state,
            x_train=raw_x_train,
            y_train=raw_y_train,
            x_test=raw_x_test,
            y_test=None,
            impute_missing=bool(
                self.preprocessing_surface.impute_missing and not self._preserve_non_finite_inputs
            ),
        )
        return classes, preprocessor_state, processed

    def _normalize_benchmark_inputs(self, *, processed: Any) -> tuple[np.ndarray, np.ndarray]:
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
        if (
            internal_normalization
            or normalization_mode == "none"
        ):
            return processed.x_train, processed.x_test
        return normalize_train_test_arrays(
            processed.x_train,
            processed.x_test,
            mode=normalization_mode,
            preserve_non_finite=self._preserve_non_finite_inputs,
        )

    def _benchmark_task_batch(
        self,
        *,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        feature_types: list[str],
        num_classes: int,
        device: torch.device | None = None,
    ) -> TaskBatch:
        resolved_device = self.device if device is None else device
        return TaskBatch(
            x_train=torch.tensor(x_train, dtype=torch.float32, device=resolved_device),
            y_train=torch.tensor(y_train, dtype=torch.int64, device=resolved_device),
            x_test=torch.tensor(x_test, dtype=torch.float32, device=resolved_device),
            y_test=torch.zeros((x_test.shape[0],), dtype=torch.int64, device=resolved_device),
            metadata={
                "dataset": "external_benchmark",
                "feature_types": feature_types,
            },
            num_classes=num_classes,
        )

    def _prepare_benchmark_fold(
        self,
        *,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        feature_types: list[str] | None,
    ) -> tuple[np.ndarray, FittedPreprocessorState, Any, np.ndarray, np.ndarray, int]:
        raw_x_train = np.asarray(x_train, dtype=np.float32)
        raw_y_train = np.asarray(y_train, dtype=np.int64)
        classes = np.unique(raw_y_train)
        if classes.size < 2:
            raise RuntimeError("benchmark classifier requires at least 2 classes in fit()")
        if (
            str(getattr(self.model_spec, "arch", "")).strip().lower() == SANDWICH_MODEL_ARCH
            and feature_types is None
        ):
            raise RuntimeError(
                "tabfoundry_sandwich benchmark evaluation requires explicit "
                "feature_types for each dataset"
            )
        preprocessor_state = fit_fitted_preprocessor(
            task="classification",
            x_train=raw_x_train,
            y_train=raw_y_train,
            all_nan_fill=float(self.preprocessing_surface.all_nan_fill),
            label_mapping=str(self.preprocessing_surface.label_mapping),
            unseen_test_label_policy=str(self.preprocessing_surface.unseen_test_label_policy),
            feature_types=feature_types,
        )
        processed = apply_fitted_preprocessor(
            task="classification",
            state=preprocessor_state,
            x_train=raw_x_train,
            y_train=raw_y_train,
            x_test=np.asarray(x_test, dtype=np.float32),
            y_test=None,
            impute_missing=bool(
                self.preprocessing_surface.impute_missing and not self._preserve_non_finite_inputs
            ),
        )
        x_train_norm, x_test_norm = self._normalize_benchmark_inputs(processed=processed)
        num_classes = int(processed.num_classes or classes.size)
        return classes, preprocessor_state, processed, x_train_norm, x_test_norm, num_classes

    def _cell_likelihood_metrics_from_per_cell_bits(
        self,
        per_cell_bits: torch.Tensor,
    ) -> dict[str, float]:
        finite_cell_mask = torch.isfinite(per_cell_bits)
        bpc_cell_count = int(finite_cell_mask.sum().item())
        if bpc_cell_count <= 0:
            raise RuntimeError(
                "checkpoint cell-likelihood output must include at least one finite cell"
            )
        feature_counts = finite_cell_mask.sum(dim=0)
        feature_sums = torch.where(
            finite_cell_mask,
            per_cell_bits,
            torch.zeros_like(per_cell_bits),
        ).sum(dim=0)
        valid_feature_mask = feature_counts > 0
        bpf_feature_count = int(valid_feature_mask.sum().item())
        if bpf_feature_count <= 0:
            raise RuntimeError(
                "checkpoint cell-likelihood output must include at least one valid feature"
            )
        feature_mean_bits = torch.full_like(feature_sums, float("nan"))
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
        return {
            "bpc": float(bpc.detach().item()),
            "bpf": float(bpf.detach().item()),
            "bpc_cell_count": float(bpc_cell_count),
            "bpf_feature_count": float(bpf_feature_count),
        }

    def _evaluate_benchmark_folds_batched_impl(
        self,
        folds: Sequence[BenchmarkClassificationFold],
    ) -> list[BenchmarkClassificationFoldResult]:
        if not folds:
            return []

        prepared: list[tuple[np.ndarray, TaskBatch, int, int]] = []
        for fold in folds:
            classes, preprocessor_state, processed, x_train_norm, x_test_norm, num_classes = (
                self._prepare_benchmark_fold(
                    x_train=fold.x_train,
                    y_train=fold.y_train,
                    x_test=fold.x_test,
                    feature_types=fold.feature_types,
                )
            )
            batch = self._benchmark_task_batch(
                x_train=x_train_norm,
                y_train=processed.y_train,
                x_test=x_test_norm,
                feature_types=list(preprocessor_state.feature_types),
                num_classes=num_classes,
                device=torch.device("cpu"),
            )
            prepared.append((classes, batch, int(batch.x_test.shape[0]), int(batch.x_test.shape[1])))

        batched_batch = collate_task_batch(
            [batch for _classes, batch, _n_test, _n_features in prepared],
            requested_task_batch_size=len(prepared),
        )
        batched_batch.metadata["feature_types"] = list(
            cast(list[str], prepared[0][1].metadata["feature_types"])
        )
        batched_batch = move_batch(batched_batch, self.device)

        with torch.no_grad():
            output = self.model(batched_batch)
            if not isinstance(output, ClassificationOutput):
                raise RuntimeError("checkpoint output does not expose classification probabilities")
            first_num_classes = int(prepared[0][1].num_classes or prepared[0][0].size)
            total_rows = int(batched_batch.y_test.reshape(-1).shape[0])
            resolved_num_classes = validate_classification_output_contract(
                output,
                expected_rows=total_rows,
                expected_num_classes=first_num_classes,
                context="checkpoint classifier batched benchmark",
            )
            if output.logits is not None:
                logits = output.logits[:, :resolved_num_classes]
                batched_probabilities = F.softmax(logits, dim=-1)
            elif output.class_probs is not None:
                batched_probabilities = output.class_probs
            else:
                raise RuntimeError(
                    "checkpoint output does not expose logits or class probabilities"
                )

            forward_cell_likelihood = getattr(self.model, "forward_cell_likelihood", None)
            batched_likelihood_output = (
                None if not callable(forward_cell_likelihood) else forward_cell_likelihood(batched_batch)
            )

        probabilities_by_fold = batched_probabilities.reshape(
            len(prepared),
            prepared[0][2],
            resolved_num_classes,
        )

        results: list[BenchmarkClassificationFoldResult] = []
        for task_index, (classes, batch, _n_test, _n_features) in enumerate(prepared):
            cell_metrics = None
            if batched_likelihood_output is not None:
                task_cell_bits = batched_likelihood_output.per_cell_bits[task_index]
                cell_metrics = self._cell_likelihood_metrics_from_per_cell_bits(task_cell_bits)
                x_all = torch.cat([batch.x_train, batch.x_test], dim=0)
                cell_metrics["excluded_non_finite_cell_count"] = float(
                    (~torch.isfinite(x_all)).sum().item()
                )
            results.append(
                BenchmarkClassificationFoldResult(
                    probabilities=probabilities_by_fold[task_index].cpu().numpy(),
                    classifier_labels=np.asarray(classes, dtype=np.int64),
                    cell_likelihood_metrics=cell_metrics,
                )
            )
        return results

    def predict_proba(self, x_test: np.ndarray) -> np.ndarray:
        classes, preprocessor_state, processed = self._preprocess_benchmark_inputs(x_test)

        x_train_norm, x_test_norm = self._normalize_benchmark_inputs(processed=processed)
        num_classes = int(processed.num_classes or classes.size)
        batch = self._benchmark_task_batch(
            x_train=x_train_norm,
            y_train=processed.y_train,
            x_test=x_test_norm,
            feature_types=list(preprocessor_state.feature_types),
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
                raise RuntimeError(
                    "checkpoint output does not expose logits or class probabilities"
                )
        return probs.cpu().numpy()

    def predict(self, x_test: np.ndarray) -> np.ndarray:
        probabilities = self.predict_proba(x_test)
        classes = self._classes
        if classes is None:
            raise RuntimeError("fit() must be called before predict()")
        return classes[np.asarray(probabilities.argmax(axis=1), dtype=np.int64)]

    def cell_likelihood_metrics(self, x_test: np.ndarray) -> dict[str, float]:
        classes, preprocessor_state, processed = self._preprocess_benchmark_inputs(x_test)
        forward_cell_likelihood = getattr(self.model, "forward_cell_likelihood", None)
        if not callable(forward_cell_likelihood):
            raise RuntimeError("checkpoint model does not expose forward_cell_likelihood()")

        x_train_norm, x_test_norm = self._normalize_benchmark_inputs(processed=processed)
        batch = self._benchmark_task_batch(
            x_train=x_train_norm,
            y_train=processed.y_train,
            x_test=x_test_norm,
            feature_types=list(preprocessor_state.feature_types),
            num_classes=int(processed.num_classes or classes.size),
        )
        with torch.no_grad():
            output = forward_cell_likelihood(batch)
        if output.bpc is None or output.bpf is None:
            raise RuntimeError("checkpoint cell-likelihood output omitted bpc/bpf")
        metrics = {
            "bpc": float(output.bpc.detach().item()),
            "bpf": float(output.bpf.detach().item()),
        }
        if output.aux_metrics is not None:
            for key in (
                "bpc_cell_count",
                "bpf_feature_count",
                "excluded_non_finite_cell_count",
            ):
                raw_value = output.aux_metrics.get(key)
                if raw_value is not None:
                    metrics[key] = float(raw_value)
        return metrics
