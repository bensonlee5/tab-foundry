"""Lightweight cached-benchmark dataset helpers shared across helper envs."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, TypeAlias, cast

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


BenchmarkDataset: TypeAlias = (
    tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, list[str]]
)
_BENCHMARK_DATASET_ARRAY_TUPLE_SIZE = 2
_BENCHMARK_DATASET_FEATURE_TYPES_TUPLE_SIZE = 3


def unpack_benchmark_dataset(
    dataset_name: str,
    dataset: BenchmarkDataset,
) -> tuple[np.ndarray, np.ndarray, list[str] | None]:
    """Normalize one benchmark dataset payload to arrays plus optional feature types."""

    if not isinstance(dataset, tuple):
        raise RuntimeError(
            f"benchmark dataset payload must be a tuple for dataset={dataset_name!r}"
        )
    if len(dataset) == _BENCHMARK_DATASET_ARRAY_TUPLE_SIZE:
        x, y = cast(tuple[np.ndarray, np.ndarray], dataset)
        feature_types = None
    elif len(dataset) == _BENCHMARK_DATASET_FEATURE_TYPES_TUPLE_SIZE:
        x, y, raw_feature_types = cast(tuple[np.ndarray, np.ndarray, list[str]], dataset)
        if not isinstance(raw_feature_types, list) or not all(
            isinstance(value, str) for value in raw_feature_types
        ):
            raise RuntimeError(
                "benchmark dataset feature_types must be a list of strings: "
                f"dataset={dataset_name!r}"
            )
        feature_types = list(raw_feature_types)
    else:
        raise RuntimeError(
            "benchmark dataset payload must be (x, y) or (x, y, feature_types): "
            f"dataset={dataset_name!r}"
        )
    return (
        np.asarray(x, dtype=np.float32),
        np.asarray(y),
        feature_types,
    )


def _assert_finite_benchmark_datasets(
    datasets: Mapping[str, BenchmarkDataset],
    *,
    context: str,
) -> None:
    for dataset_name, dataset in datasets.items():
        x, y, _feature_types = unpack_benchmark_dataset(dataset_name, dataset)
        assert_no_non_finite_values(
            {"x": x, "y": y},
            context=f"{context} dataset={dataset_name!r}",
        )


def save_dataset_cache(path: Path, datasets: Mapping[str, BenchmarkDataset]) -> Path:
    """Persist benchmark datasets for reuse across envs."""

    payload: dict[str, Any] = {"names": np.asarray(list(datasets.keys()), dtype=str)}
    for index, (name, dataset) in enumerate(datasets.items()):
        x, y, feature_types = unpack_benchmark_dataset(name, dataset)
        payload[f"x_{index:03d}"] = x
        payload[f"y_{index:03d}"] = y
        payload[f"name_{index:03d}"] = np.asarray(name)
        if feature_types is not None:
            payload[f"feature_types_{index:03d}"] = np.asarray(feature_types, dtype=str)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)
    return path


def load_dataset_cache(path: Path) -> dict[str, BenchmarkDataset]:
    """Load a cached benchmark dataset bundle."""

    cache = np.load(path, allow_pickle=False)
    names = [str(name) for name in cache["names"].tolist()]
    datasets: dict[str, BenchmarkDataset] = {}
    for index, name in enumerate(names):
        x = np.asarray(cache[f"x_{index:03d}"], dtype=np.float32)
        y = np.asarray(cache[f"y_{index:03d}"])
        feature_types_key = f"feature_types_{index:03d}"
        if feature_types_key in cache.files:
            datasets[name] = (
                x,
                y,
                [str(value) for value in cache[feature_types_key].tolist()],
            )
        else:
            datasets[name] = (x, y)
    if not datasets:
        raise RuntimeError(f"dataset cache is empty: {path}")
    return datasets
