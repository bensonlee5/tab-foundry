"""Reference bundle loader and executable reference consumer."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from safetensors.torch import load_file
import torch
from torch import nn

from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.model.outputs import ClassificationOutput, validate_classification_output_contract
from tab_foundry.preprocessing import preprocess_runtime_task_arrays
from tab_foundry.types import TaskBatch

from .contracts import ExportPreprocessorState, SCHEMA_VERSION_V3, ValidatedBundle
from .exporter import validate_export_bundle


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


def load_export_bundle(bundle_dir: Path) -> LoadedExportBundle:
    """Load and validate an exported bundle into a model instance."""

    validated = validate_export_bundle(bundle_dir)
    manifest = validated.manifest

    model_spec = manifest.model.to_build_spec(task=manifest.task)
    model = build_model_from_spec(model_spec)
    if manifest.schema_version == SCHEMA_VERSION_V3:
        if manifest.weights is None:
            raise RuntimeError("v3 bundle is missing embedded weights metadata")
        weights_name = manifest.weights.file
    else:
        if manifest.files is None:
            raise RuntimeError("v2 bundle is missing file metadata")
        weights_name = manifest.files.weights
    weights_path = bundle_dir.expanduser().resolve() / weights_name
    state_dict = load_file(str(weights_path))
    incompatible = model.load_state_dict(state_dict, strict=True)
    if incompatible.missing_keys or incompatible.unexpected_keys:
        raise RuntimeError(
            "Failed to load exported weights strictly: "
            f"missing={incompatible.missing_keys}, unexpected={incompatible.unexpected_keys}"
        )
    model.eval()
    return LoadedExportBundle(validated=validated, model=model)


def _require_preprocessor_policy(bundle: LoadedExportBundle) -> ExportPreprocessorState:
    validated_state = bundle.validated.preprocessor_state
    if bundle.validated.manifest.schema_version != SCHEMA_VERSION_V3:
        raise ValueError(
            "reference consumer only executes tab-foundry-export-v3 bundles; "
            f"got {bundle.validated.manifest.schema_version!r}"
        )
    if not isinstance(validated_state, ExportPreprocessorState):
        raise TypeError("reference consumer requires an embedded preprocessing policy")
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
    return TaskBatch(
        x_train=torch.from_numpy(np.asarray(processed.x_train, dtype=np.float32)),
        y_train=y_train_tensor,
        x_test=torch.from_numpy(np.asarray(processed.x_test, dtype=np.float32)),
        y_test=y_test_tensor,
        metadata={"preprocessor_policy": policy.to_dict()},
        num_classes=num_classes,
    )


def run_reference_consumer(
    bundle_dir: Path,
    *,
    x_train: Any,
    y_train: Any,
    x_test: Any,
) -> ReferenceConsumerOutput:
    """Execute the reference-only inference path for one exported bundle."""

    bundle = load_export_bundle(bundle_dir)
    batch = _reference_batch(
        bundle,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
    )
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
    return ReferenceConsumerOutput(
        task="classification",
        batch=batch,
        class_probs=probs.detach().cpu().numpy(),
    )
