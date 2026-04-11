"""Reference bundle loader and executable reference consumer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any

import numpy as np
from safetensors.torch import load_file
import torch
from torch import nn

from tab_foundry.device import resolve_device as _resolve_device_string
from tab_foundry.device import resolve_torch_device
from tab_foundry.feature_types import normalize_feature_types, resolve_feature_types
from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.model.outputs import ClassificationOutput, validate_classification_output_contract
from tab_foundry.preprocessing import preprocess_runtime_task_arrays
from tab_foundry.task_batching import move_batch
from tab_foundry.types import TaskBatch

from .contracts import ExportPreprocessorState, ValidatedBundle
from .exporter import validate_export_bundle

_MATRIX_NDIM = 2


@dataclass(slots=True)
class LoadedExportBundle:
    validated: ValidatedBundle
    model: nn.Module


@dataclass(slots=True)
class ReferenceConsumerOutput:
    task: str
    batch: TaskBatch
    class_probs: np.ndarray | None = None
    quantiles: np.ndarray | None = None
    quantile_levels: np.ndarray | None = None


def _normalize_runtime_device(device: str | torch.device | None) -> torch.device | None:
    if device is None:
        return None
    if isinstance(device, torch.device):
        return device
    return resolve_torch_device(str(device))


def load_export_bundle(
    bundle_dir: Path,
    *,
    device: str | torch.device | None = None,
) -> LoadedExportBundle:
    """Load and validate an exported bundle into a model instance."""

    validated = validate_export_bundle(bundle_dir)
    manifest = validated.manifest

    model_spec = manifest.model.to_build_spec(task=manifest.task)
    model = build_model_from_spec(model_spec)
    if manifest.weights is None:
        raise RuntimeError("v3 bundle is missing embedded weights metadata")
    weights_name = manifest.weights.file
    weights_path = bundle_dir.expanduser().resolve() / weights_name
    state_dict = load_file(str(weights_path))
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Failed to load exported weights strictly: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    resolved_device = _normalize_runtime_device(device)
    if resolved_device is not None:
        model = model.to(resolved_device)
    model.eval()
    return LoadedExportBundle(validated=validated, model=model)


def _require_preprocessor_policy(bundle: LoadedExportBundle) -> ExportPreprocessorState:
    validated_state = bundle.validated.preprocessor_state
    return validated_state


def _dummy_y_test(task: str, *, row_count: int) -> np.ndarray:
    if task == "classification":
        return np.zeros((row_count,), dtype=np.int64)
    raise RuntimeError(f"Unsupported reference-consumer task: {task!r}")


def _non_finite_feature_names(*, x_train: Any, x_test: Any) -> list[str]:
    offenders: list[str] = []
    train = np.asarray(x_train)
    test = np.asarray(x_test)
    if train.size > 0 and bool(np.any(~np.isfinite(train))):
        offenders.append("x_train")
    if test.size > 0 and bool(np.any(~np.isfinite(test))):
        offenders.append("x_test")
    return offenders


def _reference_batch(
    bundle: LoadedExportBundle,
    *,
    x_train: Any,
    y_train: Any,
    x_test: Any,
    feature_types: list[str] | None = None,
    device: torch.device | None = None,
) -> TaskBatch:
    manifest = bundle.validated.manifest
    if manifest.task != "classification":
        raise RuntimeError(
            "Reference consumer only supports classification bundles in this branch; "
            f"got task={manifest.task!r}."
        )
    policy = _require_preprocessor_policy(bundle)
    classification_policy = policy.classification_label_policy
    if classification_policy is None:
        raise RuntimeError("classification reference consumer requires label preprocessing policy")
    processed = preprocess_runtime_task_arrays(
        task=manifest.task,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=None,
        impute_missing=bool(policy.missing_value_policy.impute_missing),
        all_nan_fill=float(policy.missing_value_policy.all_nan_fill),
        label_mapping=str(classification_policy.mapping),
        unseen_test_label_policy=str(classification_policy.unseen_test_label),
    )
    if not bool(policy.missing_value_policy.impute_missing):
        offenders = _non_finite_feature_names(
            x_train=processed.x_train,
            x_test=processed.x_test,
        )
        if offenders:
            joined = ", ".join(offenders)
            raise RuntimeError(
                "reference consumer cannot execute missing-valued inputs when the "
                "embedded preprocessing policy sets impute_missing=false; "
                f"non-finite values remain in {joined}"
            )
    y_train_tensor = torch.from_numpy(np.asarray(processed.y_train, dtype=np.int64))
    y_test_tensor = torch.from_numpy(
        _dummy_y_test("classification", row_count=int(processed.x_test.shape[0]))
    )
    num_classes = processed.num_classes
    model_arch = str(manifest.model.arch).strip().lower()
    if model_arch == "tabfoundry_sandwich":
        if feature_types is None:
            raise RuntimeError(
                "tabfoundry_sandwich reference consumer requires explicit feature_types"
            )
        resolved_feature_types = normalize_feature_types(
            feature_types,
            expected_count=int(processed.x_train.shape[1]),
            context="reference_consumer.feature_types",
        )
    else:
        resolved_feature_types = resolve_feature_types(
            feature_types,
            expected_count=int(processed.x_train.shape[1]),
            context="reference_consumer.feature_types",
        )
    batch = TaskBatch(
        x_train=torch.from_numpy(np.asarray(processed.x_train, dtype=np.float32)),
        y_train=y_train_tensor,
        x_test=torch.from_numpy(np.asarray(processed.x_test, dtype=np.float32)),
        y_test=y_test_tensor,
        metadata={
            "preprocessor_policy": policy.to_dict(),
            "feature_types": resolved_feature_types,
        },
        num_classes=num_classes,
    )
    if device is None:
        return batch
    return move_batch(batch, device)


def _resolve_classification_probs(
    bundle: LoadedExportBundle,
    *,
    batch: TaskBatch,
) -> tuple[torch.Tensor, int]:
    with torch.no_grad():
        output = bundle.model(batch)
    if not isinstance(output, ClassificationOutput):
        raise RuntimeError("classification reference consumer requires ClassificationOutput")
    resolved_num_classes = validate_classification_output_contract(
        output,
        expected_rows=int(batch.x_test.shape[0]),
        expected_num_classes=None if batch.num_classes is None else int(batch.num_classes),
        context="classification reference consumer",
    )

    if output.class_probs is not None:
        probs = output.class_probs
    elif output.logits is not None:
        probs = torch.softmax(output.logits[:, :resolved_num_classes], dim=-1)
    else:
        raise RuntimeError("classification reference consumer did not produce probabilities")
    if not bool(torch.all(torch.isfinite(probs)).item()):
        raise RuntimeError("reference consumer produced non-finite class probabilities")
    return probs, int(resolved_num_classes)


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def run_reference_consumer(
    bundle_dir: Path,
    *,
    x_train: Any,
    y_train: Any,
    x_test: Any,
    feature_types: list[str] | None = None,
    device: str | torch.device | None = None,
) -> ReferenceConsumerOutput:
    """Execute the reference-only inference path for one exported bundle."""

    resolved_device = _normalize_runtime_device(device)
    bundle = (
        load_export_bundle(bundle_dir)
        if resolved_device is None
        else load_export_bundle(bundle_dir, device=resolved_device)
    )
    batch = _reference_batch(
        bundle,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        feature_types=feature_types,
        device=resolved_device,
    )
    probs, _resolved_num_classes = _resolve_classification_probs(bundle, batch=batch)
    return ReferenceConsumerOutput(
        task="classification",
        batch=batch,
        class_probs=probs.detach().cpu().numpy(),
    )


def benchmark_reference_consumer(
    bundle_dir: Path,
    *,
    x_train: Any,
    y_train: Any,
    x_test: Any,
    feature_types: list[str] | None = None,
    device: str | torch.device = "auto",
    warmup_iterations: int = 3,
    measured_iterations: int = 10,
) -> dict[str, Any]:
    """Benchmark reference-consumer latency on one fixed runtime batch."""

    if int(warmup_iterations) < 0:
        raise RuntimeError("warmup_iterations must be >= 0")
    if int(measured_iterations) <= 0:
        raise RuntimeError("measured_iterations must be >= 1")

    resolved_device = _normalize_runtime_device(device)
    if resolved_device is None:
        resolved_device = resolve_torch_device("auto")
    bundle = load_export_bundle(bundle_dir, device=resolved_device)
    batch = _reference_batch(
        bundle,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        feature_types=feature_types,
        device=resolved_device,
    )

    for _ in range(int(warmup_iterations)):
        _synchronize_device(resolved_device)
        _ = _resolve_classification_probs(bundle, batch=batch)
        _synchronize_device(resolved_device)

    elapsed_seconds: list[float] = []
    for _ in range(int(measured_iterations)):
        _synchronize_device(resolved_device)
        started = time.perf_counter()
        _ = _resolve_classification_probs(bundle, batch=batch)
        _synchronize_device(resolved_device)
        elapsed_seconds.append(float(time.perf_counter() - started))

    resolved_device_name = _resolve_device_string(str(resolved_device))
    elapsed_ms = np.asarray(elapsed_seconds, dtype=np.float64) * 1000.0
    return {
        "requested_device": str(device),
        "resolved_device": resolved_device_name,
        "warmup_iterations": int(warmup_iterations),
        "measured_iterations": int(measured_iterations),
        "n_train": (
            int(batch.x_train.shape[0])
            if batch.x_train.ndim == _MATRIX_NDIM
            else int(batch.x_train.shape[1])
        ),
        "n_test": (
            int(batch.x_test.shape[0])
            if batch.x_test.ndim == _MATRIX_NDIM
            else int(batch.x_test.shape[1])
        ),
        "n_features": int(batch.x_train.shape[-1]),
        "num_classes": int(batch.num_classes or 0),
        "mean_ms": float(np.mean(elapsed_ms)),
        "p50_ms": float(np.percentile(elapsed_ms, 50.0)),
        "p95_ms": float(np.percentile(elapsed_ms, 95.0)),
        "max_ms": float(np.max(elapsed_ms)),
        "total_measured_seconds": float(np.sum(elapsed_seconds)),
    }
