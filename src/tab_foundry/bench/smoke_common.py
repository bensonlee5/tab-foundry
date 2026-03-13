"""Shared helpers for smoke harness config composition and telemetry payloads."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from omegaconf import DictConfig

from tab_foundry.config import compose_config
from tab_foundry.data.manifest import ManifestSummary


def _normalized_stage_dict(stage: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): value for key, value in stage.items()}


def build_manifest_payload(summary: ManifestSummary) -> dict[str, Any]:
    """Serialize manifest summary fields used by smoke telemetry."""

    return {
        "discovered_records": int(summary.discovered_records),
        "excluded_records": int(summary.excluded_records),
        "total_records": int(summary.total_records),
        "train_records": int(summary.train_records),
        "val_records": int(summary.val_records),
        "test_records": int(summary.test_records),
        "filter_policy": str(summary.filter_policy),
        "warnings": list(summary.warnings),
    }


def build_cls_smoke_train_config(
    *,
    manifest_path: Path,
    output_dir: Path,
    history_path: Path,
    device: str,
    checkpoint_every: int,
    schedule_stages: Sequence[Mapping[str, Any]],
    clear_row_caps: bool,
) -> DictConfig:
    """Compose the canonical train config for classification smoke harnesses."""

    cfg = compose_config(["experiment=cls_smoke", "optimizer=adamw", "logging.use_wandb=false"])
    cfg.data.manifest_path = str(manifest_path)
    if clear_row_caps:
        cfg.data.train_row_cap = None
        cfg.data.test_row_cap = None
    cfg.runtime.output_dir = str(output_dir)
    cfg.runtime.device = str(device)
    cfg.runtime.eval_every = 1
    cfg.runtime.checkpoint_every = int(checkpoint_every)
    cfg.runtime.val_batches = 1
    cfg.schedule.stages = [_normalized_stage_dict(stage) for stage in schedule_stages]
    cfg.logging.history_jsonl_path = str(history_path)
    return cfg


def build_cls_smoke_eval_config(
    *,
    manifest_path: Path,
    checkpoint_path: Path,
    device: str,
    clear_row_caps: bool,
) -> DictConfig:
    """Compose the canonical eval config for classification smoke harnesses."""

    cfg = compose_config(["experiment=cls_smoke", "optimizer=adamw", "logging.use_wandb=false"])
    cfg.data.manifest_path = str(manifest_path)
    if clear_row_caps:
        cfg.data.train_row_cap = None
        cfg.data.test_row_cap = None
    cfg.runtime.device = str(device)
    cfg.eval.checkpoint = str(checkpoint_path)
    cfg.eval.split = "test"
    return cfg
