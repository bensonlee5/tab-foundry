"""Lightweight cached-benchmark dataset helpers shared across helper envs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from tab_foundry.data.validation import assert_no_non_finite_values


class BenchmarkDatasetEvaluationError(RuntimeError):
    """One benchmark dataset failed within a checkpoint evaluation."""

    def __init__(self, dataset_name: str, cause: Exception) -> None:
        self.dataset_name = str(dataset_name)
        self.error_type = type(cause).__name__
        super().__init__(
            f"benchmark evaluation failed for dataset {self.dataset_name!r}: {cause}"
        )


def _assert_finite_benchmark_datasets(
    datasets: Mapping[str, tuple[np.ndarray, np.ndarray]],
    *,
    context: str,
) -> None:
    for dataset_name, (x, y) in datasets.items():
        assert_no_non_finite_values(
            {"x": x, "y": y},
            context=f"{context} dataset={dataset_name!r}",
        )


def save_dataset_cache(path: Path, datasets: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> Path:
    """Persist benchmark datasets for reuse across envs."""

    payload: dict[str, Any] = {"names": np.asarray(list(datasets.keys()), dtype=str)}
    for index, (name, (x, y)) in enumerate(datasets.items()):
        payload[f"x_{index:03d}"] = np.asarray(x, dtype=np.float32)
        payload[f"y_{index:03d}"] = np.asarray(y)
        payload[f"name_{index:03d}"] = np.asarray(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def load_dataset_cache(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load a cached benchmark dataset bundle."""

    cache = np.load(path, allow_pickle=False)
    names = [str(name) for name in cache["names"].tolist()]
    datasets: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for index, name in enumerate(names):
        datasets[name] = (
            np.asarray(cache[f"x_{index:03d}"], dtype=np.float32),
            np.asarray(cache[f"y_{index:03d}"]),
        )
    if not datasets:
        raise RuntimeError(f"dataset cache is empty: {path}")
    return datasets
