"""Manifest-backed task source."""

from __future__ import annotations

from pathlib import Path

from omegaconf import DictConfig

from tab_foundry.data.dataset import PackedParquetTaskDataset


def build_manifest_task_dataset(
    data_cfg: DictConfig,
    *,
    split: str,
    task: str,
    seed: int,
) -> PackedParquetTaskDataset:
    """Build one manifest-backed task dataset."""

    return PackedParquetTaskDataset(
        manifest_path=Path(str(data_cfg.manifest_path)),
        split=split,
        task=task,
        train_row_cap=(int(data_cfg.train_row_cap) if data_cfg.train_row_cap is not None else None),
        test_row_cap=(int(data_cfg.test_row_cap) if data_cfg.test_row_cap is not None else None),
        seed=seed,
    )
