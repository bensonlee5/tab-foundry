"""Validation helpers for missing-valued OpenML benchmark manifests."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tab_foundry.repo_paths import normalize_repo_relative_path

from .dataset_common import unpack_benchmark_dataset
from .datasets import load_benchmark_manifest_datasets


_FEATURE_MATRIX_NDIMS = 2


def _finite_label_mask(values: np.ndarray) -> np.ndarray:
    if np.issubdtype(values.dtype, np.number):
        return np.isfinite(values.astype(np.float64, copy=False))
    flat = values.reshape(-1).astype(object, copy=False)
    return np.asarray(
        [
            value is not None
            and not (isinstance(value, float) and not math.isfinite(value))
            for value in flat
        ],
        dtype=bool,
    ).reshape(values.shape)


def _bundle_selection(surface: Mapping[str, Any]) -> Mapping[str, Any]:
    bundle = surface.get("benchmark_bundle")
    if not isinstance(bundle, Mapping):
        raise RuntimeError("OpenML benchmark manifest omitted benchmark_bundle provenance")
    selection = bundle.get("selection")
    if not isinstance(selection, Mapping):
        raise RuntimeError("OpenML benchmark manifest omitted bundle selection provenance")
    return selection


def validate_openml_missingness_manifest(
    benchmark_manifest_path: Path,
    *,
    summary_out: Path | None = None,
    require_observed_missing: bool = True,
) -> dict[str, Any]:
    """Load a materialized OpenML manifest and validate observed missingness."""

    datasets, task_records, surface = load_benchmark_manifest_datasets(
        benchmark_manifest_path=benchmark_manifest_path,
        allow_missing_values=True,
    )
    if surface.get("allow_missing_values") is not True:
        raise RuntimeError("OpenML benchmark manifest provenance must allow missing values")

    selection = _bundle_selection(surface)
    min_missing_pct = float(selection.get("min_missing_pct", 0.0))
    max_missing_pct = float(selection.get("max_missing_pct", 0.0))
    if min_missing_pct <= 0.0 or max_missing_pct <= 0.0:
        raise RuntimeError(
            "OpenML benchmark bundle selection must require positive missingness: "
            f"min_missing_pct={min_missing_pct:g}, max_missing_pct={max_missing_pct:g}"
        )

    dataset_summaries: list[dict[str, Any]] = []
    total_missing_feature_cells = 0
    total_feature_cells = 0
    total_missing_rows = 0
    for dataset_name, dataset in datasets.items():
        x, y, _feature_types = unpack_benchmark_dataset(dataset_name, dataset)
        missing_mask = np.isnan(np.asarray(x, dtype=np.float32))
        missing_feature_cells = int(missing_mask.sum())
        missing_rows = (
            int(missing_mask.any(axis=1).sum())
            if missing_mask.ndim == _FEATURE_MATRIX_NDIMS
            else 0
        )
        total_cells = int(missing_mask.size)
        labels = np.asarray(y)
        if not bool(_finite_label_mask(labels).all()):
            raise RuntimeError(f"OpenML benchmark dataset labels contain NaN/Inf: {dataset_name}")
        if require_observed_missing and missing_feature_cells <= 0:
            raise RuntimeError(
                "OpenML benchmark dataset has no observed missing feature cells after "
                f"materialization: {dataset_name}"
            )
        total_missing_feature_cells += missing_feature_cells
        total_feature_cells += total_cells
        total_missing_rows += missing_rows
        dataset_summaries.append(
            {
                "dataset_name": str(dataset_name),
                "feature_cells": total_cells,
                "missing_feature_cells": missing_feature_cells,
                "rows": int(x.shape[0]),
                "missing_rows": missing_rows,
            }
        )

    if require_observed_missing and total_missing_feature_cells <= 0:
        raise RuntimeError("OpenML benchmark manifest contains no observed missing feature cells")

    summary = {
        "manifest_path": normalize_repo_relative_path(
            benchmark_manifest_path.expanduser().resolve()
        ),
        "dataset_count": int(len(datasets)),
        "task_record_count": int(len(task_records)),
        "selection": dict(selection),
        "allow_missing_values": True,
        "total_feature_cells": total_feature_cells,
        "total_missing_feature_cells": total_missing_feature_cells,
        "total_missing_rows": total_missing_rows,
        "datasets": dataset_summaries,
    }
    if summary_out is not None:
        resolved_summary_out = summary_out.expanduser().resolve()
        resolved_summary_out.parent.mkdir(parents=True, exist_ok=True)
        resolved_summary_out.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return summary


__all__ = ["validate_openml_missingness_manifest"]
