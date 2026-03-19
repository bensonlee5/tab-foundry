from __future__ import annotations

import numpy as np
import pandas as pd

import tab_foundry.bench.nanotabpfn as benchmark_module


class FakeDataset:
    def __init__(self, *, name: str, qualities: dict[str, float], frame: pd.DataFrame, target: pd.Series) -> None:
        self.name = name
        self.qualities = qualities
        self._frame = frame
        self._target = target

    def get_data(self, *, target: str, dataset_format: str) -> tuple[pd.DataFrame, pd.Series, list[bool], list[str]]:
        assert target == "target"
        assert dataset_format == "dataframe"
        return self._frame, self._target, [False] * self._frame.shape[1], list(self._frame.columns)


class FakeTask:
    def __init__(self, dataset: FakeDataset) -> None:
        self.task_type_id = benchmark_module.TaskType.SUPERVISED_CLASSIFICATION
        self.target_name = "target"
        self._dataset = dataset

    def get_dataset(self, *, download_data: bool) -> FakeDataset:
        assert download_data is False
        return self._dataset


def prepared_task(
    *,
    task_id: int,
    dataset_name: str,
    n_rows: int,
    n_features: int,
    n_classes: int,
    raw_feature_count: int | None = None,
    missing_pct: float = 0.0,
    minority_class_pct: float = 25.0,
) -> benchmark_module.PreparedOpenMLBenchmarkTask:
    x = np.arange(n_rows * n_features, dtype=np.float32).reshape(n_rows, n_features)
    y = (np.arange(n_rows, dtype=np.int64) % n_classes).astype(np.int64, copy=False)
    return benchmark_module.PreparedOpenMLBenchmarkTask(
        task_id=task_id,
        dataset_name=dataset_name,
        x=x,
        y=y,
        observed_task={
            "task_id": int(task_id),
            "dataset_name": dataset_name,
            "n_rows": int(n_rows),
            "n_features": int(n_features),
            "n_classes": int(n_classes),
        },
        qualities={
            "NumberOfFeatures": float(n_features if raw_feature_count is None else raw_feature_count),
            "NumberOfClasses": float(n_classes),
            "PercentageOfInstancesWithMissingValues": float(missing_pct),
            "MinorityClassPercentage": float(minority_class_pct),
        },
    )
