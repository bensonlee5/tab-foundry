from __future__ import annotations

from hashlib import sha256
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from tab_foundry.cli import build_parser
from tab_foundry.data.dataset import PackedParquetTaskDataset
from tab_foundry.data.manifest import build_manifest
from tab_foundry.export.exporter import export_checkpoint
from tab_foundry.export.loader_ref import run_reference_consumer
from tab_foundry.model.factory import build_model
from tab_foundry.preprocessing import apply_fitted_preprocessor, fit_fitted_preprocessor


def _manifest_summary_metadata(path: Path) -> dict[str, Any]:
    metadata = pq.ParquetFile(path).schema_arrow.metadata or {}
    raw = metadata[b"tab_foundry_manifest_summary"]
    return json.loads(raw.decode("utf-8"))


def _classification_metadata(
    *,
    n_features: int,
    n_classes: int = 3,
    seed: int = 7,
    filter_status: str | None = "not_run",
    filter_accepted: bool | None = None,
    include_filter: bool = True,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "n_features": n_features,
        "n_classes": n_classes,
        "seed": seed,
        "config": {"dataset": {"task": "classification"}},
    }
    if include_filter:
        filter_payload: dict[str, Any] = {"mode": "deferred"}
        if filter_status is not None:
            filter_payload["status"] = filter_status
        if filter_accepted is not None:
            filter_payload["accepted"] = filter_accepted
        payload["filter"] = filter_payload
    return payload


def _classification_arrays(
    *,
    n_train: int = 16,
    n_test: int = 8,
    n_features: int = 4,
    n_classes: int = 3,
    seed: int = 7,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    x_train = rng.standard_normal((n_train, n_features)).astype(np.float32)
    x_test = rng.standard_normal((n_test, n_features)).astype(np.float32)
    y_train = np.tile(np.arange(n_classes, dtype=np.int64), int(np.ceil(n_train / n_classes)))[:n_train]
    y_test = np.tile(np.arange(n_classes, dtype=np.int64), int(np.ceil(n_test / n_classes)))[:n_test]
    rng.shuffle(y_train)
    rng.shuffle(y_test)
    return x_train, y_train, x_test, y_test


def _build_split_table(rows: list[tuple[int, np.ndarray, np.ndarray]]) -> pa.Table:
    dataset_indices: list[int] = []
    row_indices: list[int] = []
    x_rows: list[list[float]] = []
    y_rows: list[int] = []
    for dataset_index, x, y in rows:
        for row_index in range(int(x.shape[0])):
            dataset_indices.append(int(dataset_index))
            row_indices.append(int(row_index))
            x_rows.append(x[row_index].astype(np.float32, copy=False).tolist())
            y_rows.append(int(y[row_index]))

    return pa.table(
        {
            "dataset_index": pa.array(dataset_indices, type=pa.int64()),
            "row_index": pa.array(row_indices, type=pa.int64()),
            "x": pa.array(x_rows, type=pa.list_(pa.float32())),
            "y": pa.array(y_rows, type=pa.int64()),
        }
    )


def _write_packed_shard(
    shard_dir: Path,
    *,
    datasets: list[dict[str, Any]],
) -> dict[int, tuple[int, int, str]]:
    shard_dir.mkdir(parents=True, exist_ok=True)

    train_rows = [
        (
            int(dataset["dataset_index"]),
            dataset["x_train"],
            dataset["y_train"],
        )
        for dataset in datasets
    ]
    test_rows = [
        (
            int(dataset["dataset_index"]),
            dataset["x_test"],
            dataset["y_test"],
        )
        for dataset in datasets
    ]
    pq.write_table(_build_split_table(train_rows), shard_dir / "train.parquet")
    pq.write_table(_build_split_table(test_rows), shard_dir / "test.parquet")

    offsets: dict[int, tuple[int, int, str]] = {}
    with (shard_dir / "metadata.ndjson").open("wb") as handle:
        for dataset in datasets:
            payload = {
                "dataset_index": int(dataset["dataset_index"]),
                "n_train": int(dataset["x_train"].shape[0]),
                "n_test": int(dataset["x_test"].shape[0]),
                "n_features": int(dataset["x_train"].shape[1]),
                "feature_types": list(dataset["feature_types"]),
                "metadata": dict(dataset["metadata"]),
            }
            raw = (json.dumps(payload, sort_keys=True) + "\n").encode("utf-8")
            offset = int(handle.tell())
            handle.write(raw)
            offsets[int(dataset["dataset_index"])] = (offset, len(raw), sha256(raw).hexdigest())

    return offsets


def _write_dataset(
    shard_dir: Path,
    *,
    dataset_index: int = 0,
    n_train: int = 16,
    n_test: int = 8,
    n_features: int = 4,
    seed: int = 7,
    filter_status: str | None = "not_run",
    filter_accepted: bool | None = None,
    include_filter: bool = True,
) -> dict[int, tuple[int, int, str]]:
    x_train, y_train, x_test, y_test = _classification_arrays(
        n_train=n_train,
        n_test=n_test,
        n_features=n_features,
        seed=seed,
    )
    metadata = _classification_metadata(
        n_features=n_features,
        seed=seed,
        filter_status=filter_status,
        filter_accepted=filter_accepted,
        include_filter=include_filter,
    )
    dataset = {
        "dataset_index": dataset_index,
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "feature_types": ["num"] * n_features,
        "metadata": metadata,
    }
    return _write_packed_shard(shard_dir, datasets=[dataset])


def test_manifest_and_dataset_loading(tmp_path: Path) -> None:
    shard_dir = tmp_path / "run" / "shard_00000"
    _ = _write_dataset(shard_dir)

    manifest_path = tmp_path / "manifest.parquet"
    summary = build_manifest([tmp_path / "run"], manifest_path)
    assert summary.total_records == 1
    assert summary.discovered_records == 1
    assert summary.excluded_records == 0
    assert summary.excluded_for_missing_values == 0
    assert summary.missing_value_policy == "allow_any"
    assert summary.filter_status_counts == {"not_run": 1}
    assert summary.missing_value_status_counts == {"clean": 1}
    assert summary.warnings

    table = pq.read_table(manifest_path)
    row = table.to_pylist()[0]
    persisted_summary = _manifest_summary_metadata(manifest_path)
    split = row["split"]
    assert row["source_shard_relpath"] == "shard_00000"
    assert row["filter_mode"] == "deferred"
    assert row["filter_status"] == "not_run"
    assert row["filter_accepted"] is None
    assert row["missing_value_policy"] == "allow_any"
    assert row["missing_value_status"] == "clean"
    assert persisted_summary["missing_value_policy"] == "allow_any"
    assert persisted_summary["missing_value_status_counts"] == {"clean": 1}

    ds = PackedParquetTaskDataset(manifest_path, split=split, task="classification")
    sample = ds[0]
    assert sample.x_train.ndim == 2
    assert sample.y_test.ndim == 1
    assert sample.metadata["config"]["dataset"]["task"] == "classification"


def test_manifest_include_all_tracks_missing_filter_metadata(tmp_path: Path) -> None:
    accepted_dir = tmp_path / "run" / "accepted" / "shard_00000"
    missing_dir = tmp_path / "run" / "missing" / "shard_00000"
    _ = _write_dataset(accepted_dir, dataset_index=0, filter_status="accepted", filter_accepted=True)
    _ = _write_dataset(missing_dir, dataset_index=1, include_filter=False)

    manifest_path = tmp_path / "manifest.parquet"
    summary = build_manifest([tmp_path / "run"], manifest_path, filter_policy="include_all")

    assert summary.total_records == 2
    assert summary.filter_status_counts == {"accepted": 1, "missing": 1}
    assert any("accepted-only training" in warning for warning in summary.warnings)


def test_manifest_accepted_only_excludes_unaccepted_records(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _ = _write_dataset(root / "accepted" / "shard_00000", dataset_index=0, filter_status="accepted", filter_accepted=True)
    _ = _write_dataset(root / "rejected" / "shard_00000", dataset_index=1, filter_status="rejected", filter_accepted=False)
    _ = _write_dataset(root / "pending" / "shard_00000", dataset_index=2, filter_status="not_run")

    manifest_path = tmp_path / "manifest.parquet"
    summary = build_manifest([root], manifest_path, filter_policy="accepted_only")
    rows = pq.read_table(manifest_path).to_pylist()

    assert summary.discovered_records == 3
    assert summary.total_records == 1
    assert summary.excluded_records == 2
    assert summary.filter_status_counts == {"accepted": 1, "not_run": 1, "rejected": 1}
    assert len(rows) == 1
    assert rows[0]["dataset_index"] == 0
    assert rows[0]["filter_status"] == "accepted"


def test_manifest_forbid_any_excludes_datasets_with_nan_or_inf(tmp_path: Path) -> None:
    root = tmp_path / "run"
    clean_x_train, clean_y_train, clean_x_test, clean_y_test = _classification_arrays(seed=7)
    dirty_x_train, dirty_y_train, dirty_x_test, dirty_y_test = _classification_arrays(seed=11)
    dirty_x_train[0, 0] = np.nan
    dirty_x_test[1, 1] = np.inf
    datasets = [
        {
            "dataset_index": 0,
            "x_train": clean_x_train,
            "y_train": clean_y_train,
            "x_test": clean_x_test,
            "y_test": clean_y_test,
            "feature_types": ["num"] * clean_x_train.shape[1],
            "metadata": _classification_metadata(
                n_features=clean_x_train.shape[1],
                seed=7,
                filter_status="accepted",
                filter_accepted=True,
            ),
        },
        {
            "dataset_index": 1,
            "x_train": dirty_x_train,
            "y_train": dirty_y_train,
            "x_test": dirty_x_test,
            "y_test": dirty_y_test,
            "feature_types": ["num"] * dirty_x_train.shape[1],
            "metadata": _classification_metadata(
                n_features=dirty_x_train.shape[1],
                seed=11,
                filter_status="accepted",
                filter_accepted=True,
            ),
        },
    ]
    _ = _write_packed_shard(root / "shard_00000", datasets=datasets)

    manifest_path = tmp_path / "manifest.parquet"
    summary = build_manifest(
        [root],
        manifest_path,
        filter_policy="accepted_only",
        missing_value_policy="forbid_any",
    )
    rows = pq.read_table(manifest_path).to_pylist()
    persisted_summary = _manifest_summary_metadata(manifest_path)

    assert summary.discovered_records == 2
    assert summary.total_records == 1
    assert summary.excluded_records == 1
    assert summary.excluded_for_missing_values == 1
    assert summary.missing_value_status_counts == {"clean": 1, "contains_nan_or_inf": 1}
    assert rows[0]["dataset_index"] == 0
    assert rows[0]["missing_value_status"] == "clean"
    assert persisted_summary["missing_value_policy"] == "forbid_any"
    assert persisted_summary["excluded_for_missing_values"] == 1


def test_manifest_accepted_only_requires_at_least_one_record(tmp_path: Path) -> None:
    shard_dir = tmp_path / "run" / "shard_00000"
    _ = _write_dataset(shard_dir, filter_status="not_run")

    manifest_path = tmp_path / "manifest.parquet"
    with pytest.raises(RuntimeError, match="no datasets matched filter_policy"):
        _ = build_manifest([tmp_path / "run"], manifest_path, filter_policy="accepted_only")


def test_remap_labels_uses_train_only() -> None:
    y_train = np.array([10, 20, 10], dtype=np.int64)
    y_test = np.array([20, 10], dtype=np.int64)
    state = fit_fitted_preprocessor(
        task="classification",
        x_train=np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        y_train=y_train,
    )
    processed = apply_fitted_preprocessor(
        task="classification",
        state=state,
        x_train=np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        y_train=y_train,
        x_test=np.array([[3.0], [4.0]], dtype=np.float32),
        y_test=y_test,
    )
    assert processed.num_classes == 2
    assert np.array_equal(processed.y_train, np.array([0, 1, 0], dtype=np.int64))
    assert processed.y_test is not None
    assert np.array_equal(processed.y_test, np.array([1, 0], dtype=np.int64))


def test_remap_labels_filters_unseen_test_classes() -> None:
    y_train = np.array([0, 0, 1], dtype=np.int64)
    y_test = np.array([0, 2], dtype=np.int64)
    x_test = np.array([[1], [2]])
    state = fit_fitted_preprocessor(
        task="classification",
        x_train=np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        y_train=y_train,
    )
    processed = apply_fitted_preprocessor(
        task="classification",
        state=state,
        x_train=np.array([[0.0], [1.0], [2.0]], dtype=np.float32),
        y_train=y_train,
        x_test=x_test.astype(np.float32),
        y_test=y_test,
    )
    assert processed.y_test is not None
    assert np.array_equal(processed.y_test, np.array([0], dtype=np.int64))
    assert processed.valid_test_mask is not None
    assert np.array_equal(processed.valid_test_mask, np.array([True, False]))
    assert np.array_equal(x_test[processed.valid_test_mask], np.array([[1]]))


def test_dataset_raises_when_unseen_filter_removes_all_test_rows(tmp_path: Path) -> None:
    shard_dir = tmp_path / "run" / "shard_00000"
    dataset = {
        "dataset_index": 0,
        "x_train": np.array([[0.0, 1.0], [1.0, 2.0], [2.0, 3.0]], dtype=np.float32),
        "y_train": np.array([0, 0, 0], dtype=np.int64),
        "x_test": np.array([[4.0, 6.0], [5.0, 7.0]], dtype=np.float32),
        "y_test": np.array([1, 1], dtype=np.int64),
        "feature_types": ["num", "num"],
        "metadata": _classification_metadata(
            n_features=2,
            n_classes=2,
            seed=7,
            filter_status="accepted",
            filter_accepted=True,
        ),
    }
    _ = _write_packed_shard(shard_dir, datasets=[dataset])

    manifest_path = tmp_path / "manifest.parquet"
    _ = build_manifest([tmp_path / "run"], manifest_path)
    row = pq.read_table(manifest_path).to_pylist()[0]
    ds = PackedParquetTaskDataset(
        manifest_path=manifest_path,
        split=str(row["split"]),
        task="classification",
    )
    with pytest.raises(RuntimeError, match="zero rows after filtering unseen labels"):
        _ = ds[0]


def test_dataset_keeps_nan_features_when_impute_missing_is_false_but_still_remaps_labels(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "run" / "shard_00000"
    dataset = {
        "dataset_index": 0,
        "x_train": np.array([[1.0, np.nan], [2.0, 3.0], [3.0, 4.0]], dtype=np.float32),
        "y_train": np.array([10, 20, 10], dtype=np.int64),
        "x_test": np.array([[np.nan, 5.0], [6.0, np.nan]], dtype=np.float32),
        "y_test": np.array([20, 999], dtype=np.int64),
        "feature_types": ["num", "num"],
        "metadata": _classification_metadata(
            n_features=2,
            n_classes=2,
            seed=7,
            filter_status="accepted",
            filter_accepted=True,
        ),
    }
    _ = _write_packed_shard(shard_dir, datasets=[dataset])

    manifest_path = tmp_path / "manifest.parquet"
    _ = build_manifest([tmp_path / "run"], manifest_path)
    row = pq.read_table(manifest_path).to_pylist()[0]
    ds = PackedParquetTaskDataset(
        manifest_path=manifest_path,
        split=str(row["split"]),
        task="classification",
        impute_missing=False,
        allow_missing_values=True,
    )
    sample = ds[0]

    assert torch.isnan(sample.x_train[0, 1])
    assert torch.isnan(sample.x_test[0, 0])
    assert sample.y_train.tolist() == [0, 1, 0]
    assert sample.y_test.tolist() == [1]
    assert sample.num_classes == 2


def test_dataset_rejects_missing_inputs_by_default(tmp_path: Path) -> None:
    shard_dir = tmp_path / "run" / "shard_00000"
    dataset = {
        "dataset_index": 0,
        "x_train": np.array([[1.0, np.nan], [2.0, 3.0], [3.0, 4.0]], dtype=np.float32),
        "y_train": np.array([0, 1, 0], dtype=np.int64),
        "x_test": np.array([[4.0, 5.0], [6.0, np.inf]], dtype=np.float32),
        "y_test": np.array([1, 0], dtype=np.int64),
        "feature_types": ["num", "num"],
        "metadata": _classification_metadata(
            n_features=2,
            n_classes=2,
            seed=7,
            filter_status="accepted",
            filter_accepted=True,
        ),
    }
    _ = _write_packed_shard(shard_dir, datasets=[dataset])

    manifest_path = tmp_path / "manifest.parquet"
    _ = build_manifest([tmp_path / "run"], manifest_path)
    row = pq.read_table(manifest_path).to_pylist()[0]
    ds = PackedParquetTaskDataset(
        manifest_path=manifest_path,
        split=str(row["split"]),
        task="classification",
    )

    with pytest.raises(RuntimeError, match="contains NaN or Inf"):
        _ = ds[0]


def test_cli_parser_rejects_legacy_flat_and_removed_preprocessor_state_surface() -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["build-preprocessor-state"])

    with pytest.raises(SystemExit):
        parser.parse_args(["build-manifest"])

    with pytest.raises(SystemExit):
        parser.parse_args(["validate-export"])

    with pytest.raises(SystemExit):
        parser.parse_args(["train", "experiment=cls_smoke"])

    with pytest.raises(SystemExit):
        parser.parse_args(["export", "--checkpoint", "checkpoint.pt", "--out-dir", "bundle"])

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "export",
                "bundle",
                "--checkpoint",
                "checkpoint.pt",
                "--out-dir",
                "bundle",
                "--preprocessor-state",
                "state.json",
            ]
        )


def test_dataset_and_reference_consumer_share_runtime_preprocessing_semantics(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "run" / "shard_00000"
    dataset = {
        "dataset_index": 0,
        "x_train": np.array([[1.0, np.nan], [3.0, 5.0], [5.0, 7.0]], dtype=np.float32),
        "y_train": np.array([100, 200, 100], dtype=np.int64),
        "x_test": np.array([[np.nan, 11.0]], dtype=np.float32),
        "y_test": np.array([200], dtype=np.int64),
        "feature_types": ["num", "num"],
        "metadata": _classification_metadata(
            n_features=2,
            n_classes=2,
            seed=13,
            filter_status="accepted",
            filter_accepted=True,
        ),
    }
    _ = _write_packed_shard(shard_dir, datasets=[dataset])

    manifest_path = tmp_path / "manifest.parquet"
    _ = build_manifest([tmp_path / "run"], manifest_path)
    row = pq.read_table(manifest_path).to_pylist()[0]
    ds = PackedParquetTaskDataset(
        manifest_path=manifest_path,
        split=str(row["split"]),
        task="classification",
        allow_missing_values=True,
    )
    sample = ds[0]

    checkpoint = tmp_path / "ckpt.pt"
    cfg = {
        "task": "classification",
        "model": {
            "d_col": 16,
            "d_icl": 32,
            "input_normalization": "none",
            "feature_group_size": 1,
            "many_class_train_mode": "path_nll",
            "max_mixed_radix_digits": 64,
            "tfcol_n_heads": 4,
            "tfcol_n_layers": 1,
            "tfcol_n_inducing": 8,
            "tfrow_n_heads": 4,
            "tfrow_n_layers": 1,
            "tfrow_cls_tokens": 2,
            "tficl_n_heads": 4,
            "tficl_n_layers": 2,
            "tficl_ff_expansion": 2,
            "many_class_base": 10,
            "head_hidden_dim": 32,
            "use_digit_position_embed": True,
        },
    }
    torch.manual_seed(0)
    model = build_model(task="classification", **cfg["model"])
    torch.save({"model": model.state_dict(), "global_step": 1, "config": cfg}, checkpoint)

    bundle_dir = tmp_path / "bundle"
    _ = export_checkpoint(checkpoint, bundle_dir)
    output = run_reference_consumer(
        bundle_dir,
        x_train=dataset["x_train"],
        y_train=dataset["y_train"],
        x_test=dataset["x_test"],
    )

    assert torch.equal(output.batch.y_train, sample.y_train)
    assert output.batch.num_classes == sample.num_classes
    assert torch.allclose(output.batch.x_train, sample.x_train)
    assert torch.allclose(output.batch.x_test, sample.x_test)


def test_manifest_dataset_id_and_split_are_stable_across_root_paths(tmp_path: Path) -> None:
    root_a = tmp_path / "root_a" / "run" / "shard_00000"
    root_b = tmp_path / "root_b" / "run" / "shard_00000"
    _ = _write_dataset(root_a)
    _ = _write_dataset(root_b)

    manifest_a = tmp_path / "manifest_a.parquet"
    manifest_b = tmp_path / "manifest_b.parquet"
    _ = build_manifest([tmp_path / "root_a" / "run"], manifest_a)
    _ = build_manifest([tmp_path / "root_b" / "run"], manifest_b)

    row_a = pq.read_table(manifest_a).to_pylist()[0]
    row_b = pq.read_table(manifest_b).to_pylist()[0]
    assert row_a["dataset_id"] != row_b["dataset_id"]
    assert row_a["source_root_id"] != row_b["source_root_id"]
    assert row_a["dataset_id"].startswith("root_")
    assert row_b["dataset_id"].startswith("root_")


def test_manifest_dataset_id_is_unique_across_nested_runs_with_same_root(tmp_path: Path) -> None:
    root = tmp_path / "root"
    _ = _write_dataset(root / "run_a" / "shard_00000", dataset_index=0)
    _ = _write_dataset(root / "run_b" / "shard_00000", dataset_index=0)

    manifest_path = tmp_path / "manifest.parquet"
    _ = build_manifest([root], manifest_path)
    rows = pq.read_table(manifest_path).to_pylist()
    dataset_ids = [str(row["dataset_id"]) for row in rows]
    shard_relpaths = {str(row["source_shard_relpath"]) for row in rows}

    assert len(dataset_ids) == 2
    assert len(set(dataset_ids)) == 2
    assert shard_relpaths == {"run_a/shard_00000", "run_b/shard_00000"}


def test_dataset_resolves_relative_paths_from_manifest_location(tmp_path: Path) -> None:
    shard_dir = tmp_path / "run" / "shard_00000"
    offsets = _write_dataset(shard_dir, filter_status="accepted", filter_accepted=True)
    metadata_offset, metadata_size, metadata_sha256 = offsets[0]

    manifest_dir = tmp_path / "manifests"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = manifest_dir / "relative_manifest.parquet"
    record = {
        "dataset_id": "root_deadbeef0000/shard_00000/dataset_000000_deadbeef0000",
        "source_root_id": "deadbeef0000",
        "source_shard_relpath": "shard_00000",
        "split": "train",
        "task": "classification",
        "shard_id": 0,
        "dataset_index": 0,
        "train_path": os.path.relpath(shard_dir / "train.parquet", manifest_dir),
        "test_path": os.path.relpath(shard_dir / "test.parquet", manifest_dir),
        "metadata_path": os.path.relpath(shard_dir / "metadata.ndjson", manifest_dir),
        "metadata_offset_bytes": metadata_offset,
        "metadata_size_bytes": metadata_size,
        "metadata_sha256": metadata_sha256,
        "n_train": 16,
        "n_test": 8,
        "n_features": 4,
        "n_classes": 3,
        "seed": 7,
        "filter_mode": "deferred",
        "filter_status": "accepted",
        "filter_accepted": True,
    }
    pq.write_table(pa.Table.from_pylist([record]), manifest_path)

    run_dir = tmp_path / "elsewhere"
    run_dir.mkdir(parents=True, exist_ok=True)
    current = Path.cwd()
    try:
        os.chdir(run_dir)
        ds = PackedParquetTaskDataset(manifest_path, split="train", task="classification")
        sample = ds[0]
    finally:
        os.chdir(current)

    assert sample.x_train.shape[1] == 4
    assert sample.y_train.dtype == torch.int64


def test_manifest_paths_are_relative_to_manifest_dir(tmp_path: Path) -> None:
    shard_dir = tmp_path / "run" / "shard_00000"
    _ = _write_dataset(shard_dir)

    manifest_path = tmp_path / "manifests" / "default.parquet"
    _ = build_manifest([tmp_path / "run"], manifest_path)
    row = pq.read_table(manifest_path).to_pylist()[0]

    assert not Path(str(row["train_path"])).is_absolute()
    assert not Path(str(row["test_path"])).is_absolute()
    assert not Path(str(row["metadata_path"])).is_absolute()
    assert int(row["metadata_offset_bytes"]) >= 0
    assert int(row["metadata_size_bytes"]) > 0
    assert len(str(row["metadata_sha256"])) == 64


def test_manifest_multi_root_order_is_deterministic(tmp_path: Path) -> None:
    root_a = tmp_path / "root_a" / "run" / "shard_00000"
    root_b = tmp_path / "root_b" / "run" / "shard_00000"
    _ = _write_dataset(root_a)
    _ = _write_dataset(root_b)

    manifest_ab = tmp_path / "manifest_ab.parquet"
    manifest_ba = tmp_path / "manifest_ba.parquet"
    _ = build_manifest([tmp_path / "root_a" / "run", tmp_path / "root_b" / "run"], manifest_ab)
    _ = build_manifest([tmp_path / "root_b" / "run", tmp_path / "root_a" / "run"], manifest_ba)

    rows_ab = pq.read_table(manifest_ab).to_pylist()
    rows_ba = pq.read_table(manifest_ba).to_pylist()
    assert rows_ab == rows_ba


def test_manifest_handles_null_n_features_in_metadata(tmp_path: Path) -> None:
    shard_dir = tmp_path / "run" / "shard_00000"
    shard_dir.mkdir(parents=True, exist_ok=True)

    x_train = np.array([[0.0, 1.0]], dtype=np.float32)
    y_train = np.array([0], dtype=np.int64)
    x_test = np.array([[2.0, 3.0]], dtype=np.float32)
    y_test = np.array([0], dtype=np.int64)
    pq.write_table(_build_split_table([(0, x_train, y_train)]), shard_dir / "train.parquet")
    pq.write_table(_build_split_table([(0, x_test, y_test)]), shard_dir / "test.parquet")

    payload = {
        "dataset_index": 0,
        "n_train": 1,
        "n_test": 1,
        "n_features": None,
        "feature_types": ["num", "num"],
        "metadata": {
            "n_features": None,
            "n_classes": 1,
            "seed": 3,
            "filter": {"mode": "deferred", "status": "not_run"},
            "config": {"dataset": {"task": "classification"}},
        },
    }
    with (shard_dir / "metadata.ndjson").open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload) + "\n")

    manifest_path = tmp_path / "manifest.parquet"
    _ = build_manifest([tmp_path / "run"], manifest_path)
    row = pq.read_table(manifest_path).to_pylist()[0]
    assert int(row["n_features"]) == -1


def test_dataset_rejects_metadata_checksum_mismatch(tmp_path: Path) -> None:
    shard_dir = tmp_path / "run" / "shard_00000"
    _ = _write_dataset(shard_dir)

    manifest_path = tmp_path / "manifest.parquet"
    _ = build_manifest([tmp_path / "run"], manifest_path)
    row = pq.read_table(manifest_path).to_pylist()[0]
    metadata_path = shard_dir / "metadata.ndjson"

    offset = int(row["metadata_offset_bytes"])
    with metadata_path.open("r+b") as handle:
        handle.seek(offset + 1)
        original = handle.read(1)
        handle.seek(offset + 1)
        handle.write(b"{" if original != b"{" else b"}")

    ds = PackedParquetTaskDataset(manifest_path, split=str(row["split"]), task="classification")
    with pytest.raises(RuntimeError, match="checksum mismatch"):
        _ = ds[0]
