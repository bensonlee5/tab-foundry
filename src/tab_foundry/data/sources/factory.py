"""Source registry for task datasets."""

from __future__ import annotations

from omegaconf import DictConfig
from torch.utils.data import Dataset

from tab_foundry.types import TaskBatch

from .manifest import build_manifest_task_dataset


def build_source_dataset(
    data_cfg: DictConfig,
    *,
    split: str,
    task: str,
    seed: int,
    preprocessing_cfg: DictConfig | None = None,
    enable_categorical_feature_state: bool = False,
) -> Dataset[TaskBatch]:
    """Build a task dataset from one registered source."""

    source = str(getattr(data_cfg, "source", "manifest")).strip().lower()
    if source == "manifest":
        return build_manifest_task_dataset(
            data_cfg,
            split=split,
            task=task,
            seed=seed,
            preprocessing_cfg=preprocessing_cfg,
            enable_categorical_feature_state=enable_categorical_feature_state,
        )
    raise ValueError(f"Unsupported data.source: {source!r}")
