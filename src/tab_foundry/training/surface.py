"""Training-surface resolution and artifact helpers."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping

import pyarrow.parquet as pq

from tab_foundry.data.surface import resolve_data_surface
from tab_foundry.model.architectures.tabfoundry_staged.resolved import resolve_staged_surface
from tab_foundry.model.spec import (
    checkpoint_model_build_spec_from_mappings,
    model_build_spec_from_mappings,
)
from tab_foundry.preprocessing import resolve_preprocessing_surface


TRAINING_SURFACE_SCHEMA = "tab-foundry-training-surface-v1"


def _utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(values: list[int]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "max": None, "mean": None}
    return {
        "count": len(values),
        "min": int(min(values)),
        "max": int(max(values)),
        "mean": float(sum(values) / float(len(values))),
    }


def _manifest_characteristics(manifest_path: Path) -> dict[str, Any]:
    table = pq.read_table(manifest_path)
    rows = table.to_pylist()
    split_counts = Counter(str(row.get("split", "missing")) for row in rows)
    task_counts = Counter(str(row.get("task", "missing")) for row in rows)
    filter_status_counts = Counter(str(row.get("filter_status", "missing")) for row in rows)
    source_root_ids = sorted(
        {
            str(row["source_root_id"])
            for row in rows
            if isinstance(row.get("source_root_id"), str) and row["source_root_id"].strip()
        }
    )
    shard_counts = Counter(
        str(row["source_shard_relpath"])
        for row in rows
        if isinstance(row.get("source_shard_relpath"), str) and row["source_shard_relpath"].strip()
    )
    total_rows = [
        int(row["n_train"]) + int(row["n_test"])
        for row in rows
        if row.get("n_train") is not None and row.get("n_test") is not None
    ]
    n_features = [
        int(row["n_features"])
        for row in rows
        if row.get("n_features") is not None and int(row["n_features"]) >= 0
    ]
    n_classes = [
        int(row["n_classes"])
        for row in rows
        if row.get("n_classes") is not None
    ]
    return {
        "record_count": int(len(rows)),
        "split_counts": dict(sorted(split_counts.items())),
        "task_counts": dict(sorted(task_counts.items())),
        "row_count_distribution": _distribution(total_rows),
        "feature_count_distribution": _distribution(n_features),
        "class_count_distribution": _distribution(n_classes),
        "filter_status_counts": dict(sorted(filter_status_counts.items())),
        "source_root_ids": source_root_ids,
        "source_shard_relpath_summary": {
            "unique_count": int(len(shard_counts)),
            "top_counts": [
                {"relpath": relpath, "count": int(count)}
                for relpath, count in shard_counts.most_common(10)
            ],
        },
    }


def build_training_surface_record(
    *,
    raw_cfg: Mapping[str, Any],
    run_dir: Path,
    state_dict: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build one machine-readable training-surface record."""

    task = str(raw_cfg.get("task", "classification")).strip().lower()
    raw_model_cfg = raw_cfg.get("model")
    raw_data_cfg = raw_cfg.get("data")
    raw_preprocessing_cfg = raw_cfg.get("preprocessing")
    raw_training_cfg = raw_cfg.get("training")
    raw_optimizer_cfg = raw_cfg.get("optimizer")
    raw_schedule_cfg = raw_cfg.get("schedule")
    if not isinstance(raw_model_cfg, Mapping):
        raise RuntimeError("training surface record requires cfg.model to be a mapping")

    model_cfg = {str(key): value for key, value in raw_model_cfg.items()}
    if not isinstance(raw_data_cfg, Mapping):
        data_cfg = {
            "source": "prior_dump",
            "surface_label": "prior_dump",
            "surface_overrides": {},
        }
    else:
        data_cfg = {str(key): value for key, value in raw_data_cfg.items()}
    preprocessing_cfg = (
        None
        if not isinstance(raw_preprocessing_cfg, Mapping)
        else {str(key): value for key, value in raw_preprocessing_cfg.items()}
    )
    training_cfg = (
        None
        if not isinstance(raw_training_cfg, Mapping)
        else {str(key): value for key, value in raw_training_cfg.items()}
    )
    optimizer_cfg = (
        None
        if not isinstance(raw_optimizer_cfg, Mapping)
        else {str(key): value for key, value in raw_optimizer_cfg.items()}
    )
    schedule_cfg = (
        None
        if not isinstance(raw_schedule_cfg, Mapping)
        else {str(key): value for key, value in raw_schedule_cfg.items()}
    )
    if state_dict is None:
        model_spec = model_build_spec_from_mappings(task=task, primary=model_cfg)
    else:
        model_spec = checkpoint_model_build_spec_from_mappings(
            task=task,
            primary=model_cfg,
            state_dict=state_dict,
        )
    data_surface = resolve_data_surface(data_cfg)
    preprocessing_surface = resolve_preprocessing_surface(preprocessing_cfg)
    manifest_payload: dict[str, Any] | None = None
    if data_surface.manifest_path is not None and data_surface.manifest_path.exists():
        manifest_payload = {
            "manifest_path": str(data_surface.manifest_path),
            "manifest_sha256": _sha256_path(data_surface.manifest_path),
        }
        try:
            manifest_payload["characteristics"] = _manifest_characteristics(data_surface.manifest_path)
        except Exception as exc:  # pragma: no cover - defensive compatibility fallback
            manifest_payload["characteristics"] = None
            manifest_payload["characteristics_error"] = str(exc)

    model_payload: dict[str, Any] = {
        "arch": str(model_spec.arch),
        "stage": None if model_spec.stage is None else str(model_spec.stage),
        "stage_label": None if model_spec.stage_label is None else str(model_spec.stage_label),
        "input_normalization": str(model_spec.input_normalization),
        "feature_group_size": int(model_spec.feature_group_size),
        "many_class_base": int(model_spec.many_class_base),
        "build_spec": model_spec.to_dict(),
    }
    model_label = str(model_spec.arch)
    if model_spec.arch == "tabfoundry_staged":
        surface = resolve_staged_surface(model_spec)
        model_payload["benchmark_profile"] = str(surface.benchmark_profile)
        model_payload["module_selection"] = surface.module_selection()
        model_payload["module_hyperparameters"] = surface.component_hyperparameters()
        model_label = str(surface.stage_label)

    data_label = str(data_surface.surface_label)
    preprocessing_label = str(preprocessing_surface.surface_label)
    labels: dict[str, Any] = {
        "model": model_label,
        "data": data_label,
        "preprocessing": preprocessing_label,
    }
    payload = {
        "schema": TRAINING_SURFACE_SCHEMA,
        "generated_at_utc": _utc_now(),
        "run_dir": str(run_dir.expanduser().resolve()),
        "labels": labels,
        "model": model_payload,
        "data": {
            "surface_label": data_label,
            "source": str(data_surface.source),
            "filter_policy": data_surface.filter_policy,
            "manifest": manifest_payload,
            "dagzoo_provenance": data_surface.dagzoo_provenance,
            "train_row_cap": data_surface.train_row_cap,
            "test_row_cap": data_surface.test_row_cap,
            "overrides": data_surface.overrides,
        },
        "preprocessing": {
            "surface_label": preprocessing_label,
            "impute_missing": bool(preprocessing_surface.impute_missing),
            "all_nan_fill": float(preprocessing_surface.all_nan_fill),
            "label_mapping": str(preprocessing_surface.label_mapping),
            "unseen_test_label_policy": str(preprocessing_surface.unseen_test_label_policy),
            "feature_order_policy": str(preprocessing_surface.feature_order_policy),
            "dtype_policy": dict(preprocessing_surface.dtype_policy),
            "overrides": preprocessing_surface.overrides,
        },
    }
    if training_cfg is not None:
        training_label = str(training_cfg.get("surface_label", "training_default"))
        labels["training"] = training_label
        payload["training"] = {
            "surface_label": training_label,
            "apply_schedule": bool(training_cfg.get("apply_schedule", False)),
            "optimizer_name": None
            if optimizer_cfg is None or optimizer_cfg.get("name") is None
            else str(optimizer_cfg["name"]),
            "optimizer_min_lr": None
            if optimizer_cfg is None or optimizer_cfg.get("min_lr") is None
            else float(optimizer_cfg["min_lr"]),
            "schedule_stages": None
            if schedule_cfg is None
            else schedule_cfg.get("stages"),
            "overrides": training_cfg.get("overrides", {}),
        }
    return payload


def write_training_surface_record(
    path: Path,
    *,
    raw_cfg: Mapping[str, Any],
    run_dir: Path,
    state_dict: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one training-surface record and return the payload."""

    record = build_training_surface_record(
        raw_cfg=raw_cfg,
        run_dir=run_dir,
        state_dict=state_dict,
    )
    resolved_path = path.expanduser().resolve()
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record
