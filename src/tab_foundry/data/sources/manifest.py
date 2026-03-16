"""Manifest-backed task source."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from tab_foundry.data.dataset import PackedParquetTaskDataset
from tab_foundry.data.surface import resolve_data_surface
from tab_foundry.preprocessing import resolve_preprocessing_surface


def build_manifest_task_dataset(
    data_cfg: DictConfig,
    *,
    split: str,
    task: str,
    seed: int,
    preprocessing_cfg: DictConfig | None = None,
    enable_categorical_feature_state: bool = False,
) -> PackedParquetTaskDataset:
    """Build one manifest-backed task dataset."""

    data_surface = resolve_data_surface(data_cfg)
    if data_surface.manifest_path is None:
        raise RuntimeError("manifest-backed data surface requires a non-empty manifest path")
    preprocessing_surface = resolve_preprocessing_surface(
        None
        if preprocessing_cfg is None
        else {str(key): value for key, value in preprocessing_cfg.items()}
    )
    return PackedParquetTaskDataset(
        manifest_path=Path(str(data_surface.manifest_path)),
        split=split,
        task=task,
        train_row_cap=data_surface.train_row_cap,
        test_row_cap=data_surface.test_row_cap,
        impute_missing=bool(preprocessing_surface.impute_missing),
        all_nan_fill=float(preprocessing_surface.all_nan_fill),
        label_mapping=str(preprocessing_surface.label_mapping),
        unseen_test_label_policy=str(preprocessing_surface.unseen_test_label_policy),
        allow_missing_values=bool(data_surface.allow_missing_values),
        seed=seed,
        enable_categorical_feature_state=enable_categorical_feature_state,
    )
