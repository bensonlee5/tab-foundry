"""Developer-facing export inspection helpers."""

from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, cast

import numpy as np

from tab_foundry.model.inspection import model_surface_payload, synthetic_reference_arrays

from .contracts import ExportPreprocessorState, SCHEMA_VERSION_V3, SUPPORTED_SCHEMA_VERSIONS
from .exporter import export_checkpoint, validate_export_bundle
from .loader_ref import run_reference_consumer


def _reference_arrays(
    *,
    manifest_task: str,
    model_payload: Mapping[str, Any],
    preprocessor_state: ExportPreprocessorState | None,
) -> dict[str, Any]:
    if manifest_task != "classification":
        raise RuntimeError(
            "export-check only supports classification bundles in this branch; "
            f"got task={manifest_task!r}"
        )
    build_spec = cast(Mapping[str, Any], model_payload["build_spec"])
    include_missing_inputs = (
        preprocessor_state is not None
        and bool(preprocessor_state.missing_value_policy.impute_missing)
    )
    synthetic = synthetic_reference_arrays(
        model_payload["build_spec_obj"],
        include_missing_inputs=include_missing_inputs,
    )
    return {
        "x_train": synthetic.x_train.cpu().numpy().astype(np.float32, copy=False),
        "y_train": synthetic.y_train.cpu().numpy().astype(np.int64, copy=False),
        "x_test": synthetic.x_test.cpu().numpy().astype(np.float32, copy=False),
        "expected_num_classes": int(synthetic.expected_num_classes),
        "used_missing_inputs": include_missing_inputs,
        "build_spec": dict(build_spec),
    }


def _run_export_check(
    checkpoint_path: Path,
    *,
    bundle_dir: Path,
    artifact_version: str,
) -> dict[str, Any]:
    if artifact_version not in SUPPORTED_SCHEMA_VERSIONS:
        raise RuntimeError(f"unsupported artifact version: {artifact_version!r}")
    if artifact_version != SCHEMA_VERSION_V3:
        raise RuntimeError(
            "export-check requires artifact_version=tab-foundry-export-v3 because the "
            "reference-consumer smoke only executes v3 bundles"
        )

    checkpoint = checkpoint_path.expanduser().resolve()
    started = time.perf_counter()
    result = export_checkpoint(checkpoint, bundle_dir, artifact_version=artifact_version)
    validated = validate_export_bundle(result.bundle_dir)
    model_spec = validated.manifest.model.to_build_spec(task=validated.manifest.task)
    model_payload = model_surface_payload(model_spec)
    model_payload["build_spec_obj"] = model_spec
    preprocessor_state = (
        validated.preprocessor_state
        if isinstance(validated.preprocessor_state, ExportPreprocessorState)
        else None
    )
    arrays = _reference_arrays(
        manifest_task=validated.manifest.task,
        model_payload=model_payload,
        preprocessor_state=preprocessor_state,
    )
    reference_output = run_reference_consumer(
        result.bundle_dir,
        x_train=arrays["x_train"],
        y_train=arrays["y_train"],
        x_test=arrays["x_test"],
    )
    class_probs = reference_output.class_probs
    if class_probs is None:
        raise RuntimeError("reference consumer returned no class probabilities")
    if not bool(np.isfinite(class_probs).all()):
        raise RuntimeError("reference consumer produced non-finite class probabilities")
    expected_shape = (int(arrays["x_test"].shape[0]), int(arrays["expected_num_classes"]))
    if tuple(int(value) for value in class_probs.shape) != expected_shape:
        raise RuntimeError(
            "reference consumer class-probability shape mismatch: "
            f"expected {expected_shape}, got {tuple(class_probs.shape)}"
        )
    elapsed_seconds = float(time.perf_counter() - started)
    model_payload.pop("build_spec_obj", None)
    return {
        "checkpoint": str(checkpoint),
        "bundle_dir": str(result.bundle_dir.resolve()),
        "artifact_version": artifact_version,
        "schema_version": validated.manifest.schema_version,
        "task": validated.manifest.task,
        "model": model_payload,
        "preprocessor": (
            None
            if preprocessor_state is None
            else preprocessor_state.to_dict()
        ),
        "reference_smoke": {
            "used_missing_inputs": bool(arrays["used_missing_inputs"]),
            "output_shape": [int(value) for value in class_probs.shape],
            "output_dtype": str(class_probs.dtype),
            "num_classes": int(reference_output.batch.num_classes or 0),
        },
        "elapsed_seconds": elapsed_seconds,
    }


def export_check(
    checkpoint_path: Path,
    *,
    out_dir: Path | None,
    artifact_version: str,
) -> dict[str, Any]:
    """Export one checkpoint, validate the bundle, and run a reference smoke."""

    if out_dir is None:
        with tempfile.TemporaryDirectory(prefix="tab_foundry_export_check_") as temp_dir:
            bundle_dir = Path(temp_dir) / "bundle"
            payload = _run_export_check(
                checkpoint_path,
                bundle_dir=bundle_dir,
                artifact_version=artifact_version,
            )
        payload["bundle_dir_kept"] = False
        payload["bundle_dir_exists_after"] = bool(Path(str(payload["bundle_dir"])).exists())
        return payload

    payload = _run_export_check(
        checkpoint_path,
        bundle_dir=out_dir.expanduser().resolve(),
        artifact_version=artifact_version,
    )
    payload["bundle_dir_kept"] = True
    payload["bundle_dir_exists_after"] = True
    return payload
