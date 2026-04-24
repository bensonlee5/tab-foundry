"""Repo-local OpenML benchmark bundle materialization."""

from __future__ import annotations

from pathlib import Path
import json
import shutil
from typing import Any, Callable, cast

import numpy as np
from tab_realdata_hub.manifest import build_manifest
from tab_realdata_hub.openml import (
    DEFAULT_COMPARATOR_SPLIT_SEED,
    DEFAULT_COMPARATOR_TEST_SIZE,
    OpenMLMaterializationResult,
    _slugify,
    _split_prepared_task_indices,
    _write_packed_shard,
)

from .bundle import (
    benchmark_bundle_allows_missing_values,
    benchmark_bundle_summary,
    load_benchmark_bundle,
)
from .datasets import PreparedOpenMLBenchmarkTask, prepare_openml_benchmark_task


_CLASSIFICATION_TASK_TYPE = "supervised_classification"
_PreparedTaskProvider = Callable[..., PreparedOpenMLBenchmarkTask]


def materialize_benchmark_bundle(
    bundle_path: Path,
    out_root: Path,
    *,
    force: bool = False,
    split_seed: int = DEFAULT_COMPARATOR_SPLIT_SEED,
    test_size: float = DEFAULT_COMPARATOR_TEST_SIZE,
    bundle_source_path_label: str | None = None,
    prepare_task_fn: _PreparedTaskProvider | None = None,
) -> OpenMLMaterializationResult:
    """Materialize one OpenML bundle using the tab-foundry bundle schema."""

    resolved_bundle_path = bundle_path.expanduser().resolve()
    resolved_out_root = out_root.expanduser().resolve()
    if resolved_out_root.exists():
        if not force:
            raise RuntimeError(
                f"output root already exists, rerun with force=True: {resolved_out_root}"
            )
        shutil.rmtree(resolved_out_root)
    data_root = resolved_out_root / "packed_shards"
    manifest_path = resolved_out_root / "manifest.parquet"

    bundle = load_benchmark_bundle(resolved_bundle_path, allow_missing_values=True)
    selection = cast(dict[str, Any], bundle["selection"])
    allow_missing_values = benchmark_bundle_allows_missing_values(bundle)
    task_summaries: list[dict[str, Any]] = []
    provider = prepare_openml_benchmark_task if prepare_task_fn is None else prepare_task_fn
    persisted_bundle_source_path = (
        str(resolved_bundle_path)
        if bundle_source_path_label is None
        else str(bundle_source_path_label)
    )

    data_root.mkdir(parents=True, exist_ok=True)
    for task_order, task_id in enumerate(cast(list[int], bundle["task_ids"]), start=1):
        prepared = provider(
            int(task_id),
            new_instances=int(selection["new_instances"]),
            task_type=str(selection["task_type"]),
        )
        x_train, x_test, y_train, y_test, train_idx, test_idx, split_mode = (
            _split_prepared_task_indices(
                prepared,
                split_seed=split_seed,
                test_size=test_size,
            )
        )
        metadata = {
            "config": {
                "dataset": {
                    "task": (
                        "classification"
                        if str(selection["task_type"]) == _CLASSIFICATION_TASK_TYPE
                        else "regression"
                    )
                }
            },
            "filter": {"mode": "deferred", "status": "not_run"},
            "seed": int(split_seed),
            "n_features": int(prepared.x.shape[1]),
            "n_classes": (
                int(np.unique(prepared.y).size)
                if str(selection["task_type"]) == _CLASSIFICATION_TASK_TYPE
                else None
            ),
            "source_platform": "openml",
            "benchmark_bundle": {
                "name": str(bundle["name"]),
                "version": int(bundle["version"]),
                "source_path": persisted_bundle_source_path,
                "selection": json.loads(json.dumps(selection, sort_keys=True)),
                "task_id": int(task_id),
                "allow_missing_values": bool(allow_missing_values),
            },
            "openml": {
                "task_id": int(task_id),
                "dataset_name": str(prepared.dataset_name),
            },
            "split_policy": {
                "name": "deterministic_holdout",
                "test_size": float(test_size),
                "seed": int(split_seed),
                "mode": split_mode,
            },
        }
        if metadata["n_classes"] is None:
            del metadata["n_classes"]
        shard_dir = data_root / f"shard_{task_order:05d}_{_slugify(prepared.dataset_name)}"
        _write_packed_shard(
            shard_dir,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            train_row_indices=train_idx,
            test_row_indices=test_idx,
            metadata=metadata,
        )
        task_summary = {
            "task_id": int(task_id),
            "dataset_name": str(prepared.dataset_name),
            "n_rows": int(prepared.x.shape[0]),
            "n_features": int(prepared.x.shape[1]),
            "n_train": int(x_train.shape[0]),
            "n_test": int(x_test.shape[0]),
            "split_mode": split_mode,
            "shard_dir": str(shard_dir),
        }
        if str(selection["task_type"]) == _CLASSIFICATION_TASK_TYPE:
            task_summary["n_classes"] = int(np.unique(prepared.y).size)
        task_summaries.append(task_summary)

    build_manifest([data_root], manifest_path)
    return OpenMLMaterializationResult(
        bundle_summary={
            **benchmark_bundle_summary(bundle, source_path=resolved_bundle_path),
            "source_path": persisted_bundle_source_path,
        },
        task_summaries=tuple(task_summaries),
        allow_missing_values=bool(allow_missing_values),
        data_root=data_root,
        manifest_path=manifest_path,
    )


__all__ = ["materialize_benchmark_bundle"]
