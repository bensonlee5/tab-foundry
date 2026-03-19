from __future__ import annotations

from pathlib import Path

from tab_foundry.cli.data_inspect import manifest_inspect_payload
from tab_foundry.data.manifest import build_manifest

from . import manifest_and_dataset_cases as cases


def test_manifest_inspect_reports_summary_and_persisted_metadata(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _ = cases._write_dataset(
        root / "accepted" / "shard_00000",
        dataset_index=0,
        filter_status="accepted",
        filter_accepted=True,
    )
    _ = cases._write_dataset(
        root / "pending" / "shard_00000",
        dataset_index=1,
        filter_status="not_run",
    )
    manifest_path = tmp_path / "manifest.parquet"
    _ = build_manifest([root], manifest_path, filter_policy="include_all")

    payload = manifest_inspect_payload(manifest_path, experiment=None, overrides=[])

    assert payload["total_records"] == 2
    assert payload["task_counts"] == {"classification": 2}
    assert payload["missing_value_status_counts"] == {"clean": 2}
    assert payload["persisted_summary"]["filter_policy"] == "include_all"
    assert payload["compatibility"] is None


def test_manifest_inspect_compatibility_accepts_matching_manifest(tmp_path: Path) -> None:
    shard_dir = tmp_path / "run" / "shard_00000"
    x_train, y_train, x_test, y_test = cases._classification_arrays(n_classes=2, seed=17)
    _ = cases._write_packed_shard(
        shard_dir,
        datasets=[
            {
                "dataset_index": 0,
                "x_train": x_train,
                "y_train": y_train,
                "x_test": x_test,
                "y_test": y_test,
                "feature_types": ["num"] * x_train.shape[1],
                "metadata": cases._classification_metadata(
                    n_features=x_train.shape[1],
                    n_classes=2,
                    seed=17,
                    filter_status="accepted",
                    filter_accepted=True,
                ),
            }
        ],
    )
    manifest_path = tmp_path / "manifest.parquet"
    _ = build_manifest([tmp_path / "run"], manifest_path)

    payload = manifest_inspect_payload(
        manifest_path,
        experiment="cls_smoke",
        overrides=[f"data.manifest_path={manifest_path}"],
    )

    compatibility = payload["compatibility"]
    assert compatibility["verdict"] == "compatible"
    assert compatibility["manifest_path_matches"] is True
    assert compatibility["has_task_rows"] is True
    assert compatibility["has_train_rows"] is True
    assert isinstance(compatibility["has_test_rows"], bool)


def test_manifest_inspect_compatibility_rejects_nonfinite_rows_when_missing_values_disallowed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    clean_x_train, clean_y_train, clean_x_test, clean_y_test = cases._classification_arrays(seed=7)
    dirty_x_train, dirty_y_train, dirty_x_test, dirty_y_test = cases._classification_arrays(seed=11)
    dirty_x_train[0, 0] = float("nan")
    dirty_x_test[0, 1] = float("inf")
    _ = cases._write_packed_shard(
        root / "shard_00000",
        datasets=[
            {
                "dataset_index": 0,
                "x_train": clean_x_train,
                "y_train": clean_y_train,
                "x_test": clean_x_test,
                "y_test": clean_y_test,
                "feature_types": ["num"] * clean_x_train.shape[1],
                "metadata": cases._classification_metadata(
                    n_features=clean_x_train.shape[1],
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
                "metadata": cases._classification_metadata(
                    n_features=dirty_x_train.shape[1],
                    filter_status="accepted",
                    filter_accepted=True,
                ),
            },
        ],
    )
    manifest_path = tmp_path / "manifest.parquet"
    _ = build_manifest([root], manifest_path, filter_policy="accepted_only", missing_value_policy="allow_any")

    payload = manifest_inspect_payload(
        manifest_path,
        experiment="cls_smoke",
        overrides=[f"data.manifest_path={manifest_path}"],
    )

    compatibility = payload["compatibility"]
    assert compatibility["verdict"] == "incompatible"
    assert compatibility["contains_non_finite_rows"] is True
    assert "allow_missing_values=false" in compatibility["summary"]


def test_manifest_inspect_compatibility_rejects_missing_task_rows(tmp_path: Path) -> None:
    shard_dir = tmp_path / "run" / "shard_00000"
    _ = cases._write_dataset(
        shard_dir,
        filter_status="accepted",
        filter_accepted=True,
        metadata_overrides={
            "config": {"dataset": {"task": "regression"}},
            "n_classes": None,
        },
    )
    manifest_path = tmp_path / "manifest.parquet"
    _ = build_manifest([tmp_path / "run"], manifest_path)

    payload = manifest_inspect_payload(
        manifest_path,
        experiment="cls_smoke",
        overrides=[f"data.manifest_path={manifest_path}"],
    )

    compatibility = payload["compatibility"]
    assert compatibility["verdict"] == "incompatible"
    assert "task='classification'" in compatibility["summary"]


def test_manifest_inspect_compatibility_is_not_applicable_for_non_manifest_source(
    tmp_path: Path,
) -> None:
    shard_dir = tmp_path / "run" / "shard_00000"
    _ = cases._write_dataset(
        shard_dir,
        filter_status="accepted",
        filter_accepted=True,
    )
    manifest_path = tmp_path / "manifest.parquet"
    _ = build_manifest([tmp_path / "run"], manifest_path)

    payload = manifest_inspect_payload(
        manifest_path,
        experiment="cls_smoke",
        overrides=["data.source=dagzoo"],
    )

    compatibility = payload["compatibility"]
    assert compatibility["verdict"] == "not_applicable"
    assert compatibility["data_source"] == "dagzoo"
