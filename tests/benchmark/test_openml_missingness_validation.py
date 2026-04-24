from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import tab_foundry.bench.openml_benchmark.missingness_validation as validation_module


def _surface() -> dict[str, object]:
    return {
        "allow_missing_values": True,
        "benchmark_bundle": {
            "name": "openml_classification_missing_wide",
            "version": 1,
            "selection": {
                "task_type": "supervised_classification",
                "new_instances": 200,
                "min_classes": 2,
                "max_classes": 10,
                "max_features": 100,
                "min_missing_pct": 0.5,
                "max_missing_pct": 20.0,
                "min_minority_class_pct": 1.0,
            },
            "allow_missing_values": True,
        },
    }


def test_validate_openml_missingness_manifest_writes_observed_summary(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest.parquet"
    summary_out = tmp_path / "summary.json"

    monkeypatch.setattr(
        validation_module,
        "load_benchmark_manifest_datasets",
        lambda *, benchmark_manifest_path, allow_missing_values: (
            {
                "a": (
                    np.asarray([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32),
                    np.asarray([0, 1], dtype=np.int64),
                ),
                "b": (
                    np.asarray([[np.nan, 1.0], [2.0, 3.0]], dtype=np.float32),
                    np.asarray([1, 0], dtype=np.int64),
                    ["floating", "floating"],
                ),
            },
            [{"dataset_name": "a"}, {"dataset_name": "b"}],
            _surface(),
        ),
    )

    summary = validation_module.validate_openml_missingness_manifest(
        manifest_path,
        summary_out=summary_out,
    )

    assert summary["dataset_count"] == 2
    assert summary["total_missing_feature_cells"] == 2
    assert summary["total_missing_rows"] == 2
    persisted = json.loads(summary_out.read_text(encoding="utf-8"))
    assert persisted["total_missing_feature_cells"] == 2


def test_validate_openml_missingness_manifest_rejects_dataset_without_observed_nan(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        validation_module,
        "load_benchmark_manifest_datasets",
        lambda *, benchmark_manifest_path, allow_missing_values: (
            {
                "clean": (
                    np.asarray([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32),
                    np.asarray([0, 1], dtype=np.int64),
                )
            },
            [{"dataset_name": "clean"}],
            _surface(),
        ),
    )

    with pytest.raises(RuntimeError, match="no observed missing feature cells"):
        validation_module.validate_openml_missingness_manifest(tmp_path / "manifest.parquet")


def test_validate_openml_missingness_manifest_rejects_nonfinite_labels(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        validation_module,
        "load_benchmark_manifest_datasets",
        lambda *, benchmark_manifest_path, allow_missing_values: (
            {
                "bad_labels": (
                    np.asarray([[1.0, np.nan], [2.0, 3.0]], dtype=np.float32),
                    np.asarray([0.0, np.inf], dtype=np.float32),
                )
            },
            [{"dataset_name": "bad_labels"}],
            _surface(),
        ),
    )

    with pytest.raises(RuntimeError, match="labels contain NaN/Inf"):
        validation_module.validate_openml_missingness_manifest(tmp_path / "manifest.parquet")


def test_validate_openml_missingness_manifest_requires_positive_selection_missingness(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    surface = _surface()
    selection = dict(surface["benchmark_bundle"]["selection"])  # type: ignore[index]
    selection["min_missing_pct"] = 0.0
    surface["benchmark_bundle"]["selection"] = selection  # type: ignore[index]
    monkeypatch.setattr(
        validation_module,
        "load_benchmark_manifest_datasets",
        lambda *, benchmark_manifest_path, allow_missing_values: (
            {
                "a": (
                    np.asarray([[1.0, np.nan]], dtype=np.float32),
                    np.asarray([0], dtype=np.int64),
                )
            },
            [{"dataset_name": "a"}],
            surface,
        ),
    )

    with pytest.raises(RuntimeError, match="positive missingness"):
        validation_module.validate_openml_missingness_manifest(tmp_path / "manifest.parquet")
