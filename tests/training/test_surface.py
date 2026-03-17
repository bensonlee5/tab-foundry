from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from tab_foundry.provenance import ProducerInfo
from tab_foundry.training.surface import build_training_surface_record


def _data_cfg(manifest_path: Path, **overrides: object) -> dict[str, object]:
    cfg: dict[str, object] = {
        "source": "manifest",
        "manifest_path": str(manifest_path),
        "surface_label": "anchor_manifest_default",
        "allow_missing_values": False,
    }
    cfg.update(overrides)
    return cfg


def _preprocessing_cfg(**overrides: object) -> dict[str, object]:
    cfg: dict[str, object] = {
        "surface_label": "runtime_default",
        "impute_missing": True,
        "all_nan_fill": 0.0,
        "label_mapping": "train_only_remap",
        "unseen_test_label_policy": "filter",
    }
    cfg.update(overrides)
    return cfg


def _write_manifest(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [
            {
                "dataset_id": "root_a/shard_0/dataset_000001",
                "source_root_id": "root_a",
                "source_shard_relpath": "shard_0",
                "split": "train",
                "task": "classification",
                "dataset_index": 1,
                "train_path": "train.parquet",
                "test_path": "test.parquet",
                "metadata_path": "metadata.ndjson",
                "metadata_offset_bytes": 0,
                "metadata_size_bytes": 16,
                "metadata_sha256": "0" * 64,
                "n_train": 24,
                "n_test": 8,
                "n_features": 6,
                "n_classes": 2,
                "seed": 1,
                "filter_mode": "curated",
                "filter_status": "accepted",
                "filter_accepted": True,
                "missing_value_policy": "forbid_any",
                "missing_value_status": "clean",
            },
            {
                "dataset_id": "root_a/shard_0/dataset_000002",
                "source_root_id": "root_a",
                "source_shard_relpath": "shard_0",
                "split": "val",
                "task": "classification",
                "dataset_index": 2,
                "train_path": "train.parquet",
                "test_path": "test.parquet",
                "metadata_path": "metadata.ndjson",
                "metadata_offset_bytes": 16,
                "metadata_size_bytes": 16,
                "metadata_sha256": "1" * 64,
                "n_train": 30,
                "n_test": 10,
                "n_features": 8,
                "n_classes": 2,
                "seed": 2,
                "filter_mode": "curated",
                "filter_status": "accepted",
                "filter_accepted": True,
                "missing_value_policy": "forbid_any",
                "missing_value_status": "clean",
            },
        ]
    )
    pq.write_table(table, path)
    return path


def _write_legacy_manifest(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [
            {
                "dataset_id": "root_a/shard_0/dataset_000001",
                "source_root_id": "root_a",
                "source_shard_relpath": "shard_0",
                "split": "train",
                "task": "classification",
                "dataset_index": 1,
                "train_path": "train.parquet",
                "test_path": "test.parquet",
                "metadata_path": "metadata.ndjson",
                "metadata_offset_bytes": 0,
                "metadata_size_bytes": 16,
                "metadata_sha256": "0" * 64,
                "n_train": 24,
                "n_test": 8,
                "n_features": 6,
                "n_classes": 2,
                "seed": 1,
                "filter_mode": "curated",
                "filter_status": "accepted",
                "filter_accepted": True,
            }
        ]
    )
    pq.write_table(table, path)
    return path


def test_build_training_surface_record_captures_model_data_and_preprocessing_surfaces(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest.parquet")
    raw_cfg = {
        "task": "classification",
        "model": {
            "arch": "tabfoundry_staged",
            "stage": "nano_exact",
            "stage_label": "delta_row_cls_pool",
            "module_overrides": {"row_pool": "row_cls"},
            "d_icl": 96,
            "input_normalization": "train_zscore_clip",
            "many_class_base": 2,
            "tficl_n_heads": 4,
            "tficl_n_layers": 3,
            "head_hidden_dim": 192,
            "tfrow_n_heads": 2,
            "tfrow_n_layers": 1,
            "tfrow_cls_tokens": 2,
        },
        "data": _data_cfg(
            manifest_path,
            surface_overrides={
                "filter_policy": "accepted_only",
                "dagzoo_provenance": {
                    "commands": ["dagzoo filter --curated-out ..."],
                    "config_refs": ["configs/dagzoo/binary.yaml"],
                },
            },
        ),
        "preprocessing": _preprocessing_cfg(
            surface_label="runtime_no_impute",
            overrides={"impute_missing": False, "all_nan_fill": 1.0},
        ),
    }

    record = build_training_surface_record(
        raw_cfg=raw_cfg,
        run_dir=tmp_path / "run",
    )

    assert record["labels"] == {
        "model": "delta_row_cls_pool",
        "data": "anchor_manifest_default",
        "preprocessing": "runtime_no_impute",
    }
    assert record["model"]["missingness_mode"] == "none"
    assert record["model"]["module_selection"]["row_pool"] == "row_cls"
    assert record["model"]["module_hyperparameters"]["row_pool"]["n_heads"] == 2
    assert record["data"]["manifest"]["characteristics"]["record_count"] == 2
    assert record["data"]["manifest"]["characteristics"]["split_counts"] == {"train": 1, "val": 1}
    assert record["data"]["manifest"]["characteristics"]["missing_value_policy"] == "forbid_any"
    assert record["data"]["manifest"]["characteristics"]["all_records_no_missing"] is True
    assert record["data"]["allow_missing_values"] is False
    assert record["data"]["filter_policy"] == "accepted_only"
    assert record["data"]["dagzoo_provenance"]["commands"] == ["dagzoo filter --curated-out ..."]
    assert record["preprocessing"]["impute_missing"] is False
    assert record["preprocessing"]["all_nan_fill"] == 1.0
    assert "producer" not in record


def test_build_training_surface_record_marks_missing_inputs_when_manifest_is_dirty(
    tmp_path: Path,
) -> None:
    manifest_path = tmp_path / "manifest_dirty.parquet"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(
        [
            {
                "dataset_id": "root_a/shard_0/dataset_000001",
                "source_root_id": "root_a",
                "source_shard_relpath": "shard_0",
                "split": "train",
                "task": "classification",
                "dataset_index": 1,
                "train_path": "train.parquet",
                "test_path": "test.parquet",
                "metadata_path": "metadata.ndjson",
                "metadata_offset_bytes": 0,
                "metadata_size_bytes": 16,
                "metadata_sha256": "0" * 64,
                "n_train": 24,
                "n_test": 8,
                "n_features": 6,
                "n_classes": 2,
                "seed": 1,
                "filter_mode": "curated",
                "filter_status": "accepted",
                "filter_accepted": True,
                "missing_value_policy": "allow_any",
                "missing_value_status": "contains_nan_or_inf",
            }
        ]
    )
    pq.write_table(table, manifest_path)

    record = build_training_surface_record(
        raw_cfg={
            "task": "classification",
            "model": {"arch": "tabfoundry"},
            "data": _data_cfg(manifest_path, allow_missing_values=True),
            "preprocessing": _preprocessing_cfg(),
        },
        run_dir=tmp_path / "run_dirty",
    )

    assert record["data"]["manifest"]["characteristics"]["all_records_no_missing"] is False


def test_build_training_surface_record_captures_post_encoder_norm_component(tmp_path: Path) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest_post_encoder_norm.parquet")

    record = build_training_surface_record(
        raw_cfg={
            "task": "classification",
            "model": {
                "arch": "tabfoundry_staged",
                "stage": "nano_exact",
                "stage_label": "delta_shared_norm_post_ln",
                "module_overrides": {
                    "feature_encoder": "shared",
                    "post_encoder_norm": "layernorm",
                },
                "d_icl": 96,
                "input_normalization": "train_zscore_clip",
                "many_class_base": 2,
                "tficl_n_heads": 4,
                "tficl_n_layers": 3,
                "head_hidden_dim": 192,
            },
            "data": _data_cfg(manifest_path),
            "preprocessing": _preprocessing_cfg(),
        },
        run_dir=tmp_path / "run_post_encoder_norm",
    )

    assert record["model"]["module_selection"]["post_encoder_norm"] == "layernorm"
    assert record["model"]["module_hyperparameters"]["post_encoder_norm"] == {
        "name": "layernorm",
        "norm_type": "layernorm",
    }


def test_build_training_surface_record_marks_legacy_manifest_missingness_as_unknown(
    tmp_path: Path,
) -> None:
    manifest_path = _write_legacy_manifest(tmp_path / "manifest_legacy.parquet")

    record = build_training_surface_record(
        raw_cfg={
            "task": "classification",
            "model": {"arch": "tabfoundry"},
            "data": _data_cfg(manifest_path),
            "preprocessing": _preprocessing_cfg(),
        },
        run_dir=tmp_path / "run_legacy",
    )

    assert record["data"]["manifest"]["characteristics"]["missing_value_status_counts"] == {
        "missing": 1
    }
    assert record["data"]["manifest"]["characteristics"]["all_records_no_missing"] is None


def test_build_training_surface_record_uses_row_cap_overrides_before_top_level_values(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest.parquet")
    overridden_record = build_training_surface_record(
        raw_cfg={
            "task": "classification",
            "model": {"arch": "tabfoundry"},
            "data": _data_cfg(
                manifest_path,
                train_row_cap=10,
                test_row_cap=5,
                surface_overrides={
                    "train_row_cap": 3,
                    "test_row_cap": 2,
                },
            ),
            "preprocessing": _preprocessing_cfg(),
        },
        run_dir=tmp_path / "run_override",
    )
    fallback_record = build_training_surface_record(
        raw_cfg={
            "task": "classification",
            "model": {"arch": "tabfoundry"},
            "data": _data_cfg(
                manifest_path,
                train_row_cap=10,
                test_row_cap=5,
                surface_overrides={},
            ),
            "preprocessing": _preprocessing_cfg(),
        },
        run_dir=tmp_path / "run_top_level",
    )

    assert overridden_record["data"]["train_row_cap"] == 3
    assert overridden_record["data"]["test_row_cap"] == 2
    assert fallback_record["data"]["train_row_cap"] == 10
    assert fallback_record["data"]["test_row_cap"] == 5


def test_build_training_surface_record_includes_optional_training_surface(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest.parquet")
    record = build_training_surface_record(
        raw_cfg={
            "task": "classification",
            "model": {
                "arch": "tabfoundry_staged",
                "stage": "nano_exact",
                "d_icl": 96,
                "input_normalization": "train_zscore_clip",
                "many_class_base": 2,
                "tficl_n_heads": 4,
                "tficl_n_layers": 3,
                "head_hidden_dim": 192,
            },
            "data": _data_cfg(manifest_path),
            "preprocessing": _preprocessing_cfg(),
            "training": {
                "surface_label": "prior_linear_warmup_decay",
                "apply_schedule": True,
                "overrides": {
                    "optimizer": {"min_lr": 4.0e-4},
                },
            },
            "optimizer": {
                "name": "schedulefree_adamw",
                "min_lr": 4.0e-4,
            },
            "schedule": {
                "stages": [
                    {
                        "name": "stage1",
                        "steps": 2500,
                        "lr_max": 4.0e-3,
                        "lr_schedule": "linear",
                        "warmup_ratio": 0.05,
                    }
                ]
            },
        },
        run_dir=tmp_path / "run_training",
    )

    assert record["labels"]["training"] == "prior_linear_warmup_decay"
    assert record["training"]["apply_schedule"] is True
    assert record["training"]["optimizer_name"] == "schedulefree_adamw"
    assert record["training"]["optimizer_min_lr"] == 4.0e-4
    assert record["training"]["schedule_stages"][0]["warmup_ratio"] == 0.05


def test_build_training_surface_record_includes_optional_producer(
    tmp_path: Path,
) -> None:
    manifest_path = _write_manifest(tmp_path / "manifest_with_producer.parquet")

    record = build_training_surface_record(
        raw_cfg={
            "task": "classification",
            "model": {"arch": "tabfoundry"},
            "data": _data_cfg(manifest_path),
            "preprocessing": _preprocessing_cfg(),
        },
        run_dir=tmp_path / "run_with_producer",
        producer=ProducerInfo(
            name="tab-foundry",
            version="0.6.8",
            git_sha="abc123",
            git_dirty=True,
            source_patch_sha256="f" * 64,
            source_patch_path="/tmp/run/source.patch",
        ),
    )

    assert record["producer"] == {
        "name": "tab-foundry",
        "version": "0.6.8",
        "git_sha": "abc123",
        "git_dirty": True,
        "source_patch_sha256": "f" * 64,
        "source_patch_path": "/tmp/run/source.patch",
    }
