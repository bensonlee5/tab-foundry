from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq


REPO_ROOT = Path(__file__).resolve().parents[2]


def _load_bundle(stem: str) -> dict[str, object]:
    path = REPO_ROOT / "src" / "tab_foundry" / "bench" / f"{stem}.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _manifest_rows(stem: str) -> list[dict[str, object]]:
    path = REPO_ROOT / "data" / "manifests" / "bench" / stem / "manifest.parquet"
    return pq.read_table(path).to_pylist()


def _manifest_summary(stem: str) -> dict[str, object]:
    path = REPO_ROOT / "data" / "manifests" / "bench" / stem / "manifest.parquet"
    metadata = pq.read_table(path).schema.metadata or {}
    raw = metadata.get(b"tab_foundry_manifest_summary")
    assert raw is not None
    return json.loads(raw.decode("utf-8"))


def _catalog_record(stem: str, manifest_row: dict[str, object]) -> dict[str, object]:
    manifest_root = REPO_ROOT / "data" / "manifests" / "bench" / stem
    catalog_path = manifest_root / Path(str(manifest_row["catalog_path"]))
    catalog = pq.read_table(catalog_path).to_pylist()
    return json.loads(str(catalog[int(manifest_row["catalog_dataset_index"])]["record_json"]))


def _catalogs_present(stem: str, rows: list[dict[str, object]]) -> bool:
    manifest_root = REPO_ROOT / "data" / "manifests" / "bench" / stem
    return all((manifest_root / Path(str(row["catalog_path"]))).exists() for row in rows)


def test_openml_classification_medium_manifest_matches_repo_frozen_bundle() -> None:
    stem = "openml_classification_medium_v1"
    bundle = _load_bundle(stem)
    rows = _manifest_rows(stem)
    summary = _manifest_summary(stem)

    assert bundle["name"] == "openml_classification_medium"
    assert len(rows) == len(bundle["task_ids"]) == 242
    assert summary["total_records"] == 242
    assert summary["missing_value_policy"] == "allow_any"
    assert summary["contract_version"] == 3

    n_classes = {int(row["n_classes"]) for row in rows}
    n_features = [int(row["n_features"]) for row in rows]
    task_names = {str(row["task"]) for row in rows}
    missing_value_policies = {str(row["missing_value_policy"]) for row in rows}
    train_counts = {int(row["n_train"]) for row in rows}
    test_counts = {int(row["n_test"]) for row in rows}

    assert task_names == {"classification"}
    assert missing_value_policies == {"allow_any"}
    assert n_classes == {2, 3, 4, 5, 6, 7, 9, 10}
    assert min(n_features) == 1
    assert max(n_features) == 9
    assert train_counts == {160}
    assert test_counts == {40}
    assert max(n_classes) <= int(bundle["selection"]["max_classes"])
    assert min(n_classes) >= int(bundle["selection"]["min_classes"])
    assert max(n_features) <= int(bundle["selection"]["max_features"])
    assert all(
        int(row["n_train"]) + int(row["n_test"]) == int(bundle["selection"]["new_instances"])
        for row in rows
    )

    if not _catalogs_present(stem, rows):
        return

    task_ids: list[int] = []
    class_counts: set[int] = set()
    bundle_names: set[str] = set()
    source_paths: set[str] = set()
    for row in rows:
        record = _catalog_record(stem, row)
        metadata = record["metadata"]
        benchmark_bundle = metadata["benchmark_bundle"]
        task_ids.append(int(benchmark_bundle["task_id"]))
        class_counts.add(int(metadata["n_classes"]))
        bundle_names.add(str(benchmark_bundle["name"]))
        source_paths.add(str(benchmark_bundle["source_path"]))

    assert sorted(task_ids) == sorted(bundle["task_ids"])
    assert class_counts == {2, 3, 4, 5, 6, 7, 9, 10}
    assert bundle_names == {"openml_classification_medium"}
    assert source_paths == {"src/tab_foundry/bench/openml_classification_medium_v1.json"}


def test_openml_classification_large_manifest_matches_repo_frozen_bundle() -> None:
    stem = "openml_classification_large_v1"
    bundle = _load_bundle(stem)
    rows = _manifest_rows(stem)
    summary = _manifest_summary(stem)

    assert bundle["name"] == "openml_classification_large"
    assert len(rows) == len(bundle["task_ids"]) == 3
    assert summary["total_records"] == 3
    assert summary["missing_value_policy"] == "allow_any"
    assert summary["contract_version"] == 3

    n_classes = {int(row["n_classes"]) for row in rows}
    n_features = [int(row["n_features"]) for row in rows]
    task_names = {str(row["task"]) for row in rows}
    missing_value_policies = {str(row["missing_value_policy"]) for row in rows}
    train_counts = {int(row["n_train"]) for row in rows}
    test_counts = {int(row["n_test"]) for row in rows}

    assert task_names == {"classification"}
    assert missing_value_policies == {"allow_any"}
    assert n_classes == {3}
    assert min(n_features) == 6
    assert max(n_features) == 11
    assert train_counts == {160}
    assert test_counts == {40}
    assert max(n_classes) <= int(bundle["selection"]["max_classes"])
    assert min(n_classes) >= int(bundle["selection"]["min_classes"])
    assert max(n_features) <= int(bundle["selection"]["max_features"])
    assert all(
        int(row["n_train"]) + int(row["n_test"]) == int(bundle["selection"]["new_instances"])
        for row in rows
    )

    if not _catalogs_present(stem, rows):
        return

    task_ids: list[int] = []
    class_counts: set[int] = set()
    bundle_names: set[str] = set()
    source_paths: set[str] = set()
    for row in rows:
        record = _catalog_record(stem, row)
        metadata = record["metadata"]
        benchmark_bundle = metadata["benchmark_bundle"]
        task_ids.append(int(benchmark_bundle["task_id"]))
        class_counts.add(int(metadata["n_classes"]))
        bundle_names.add(str(benchmark_bundle["name"]))
        source_paths.add(str(benchmark_bundle["source_path"]))

    assert sorted(task_ids) == [363685, 363699, 363707]
    assert sorted(task_ids) == sorted(bundle["task_ids"])
    assert class_counts == {3}
    assert bundle_names == {"openml_classification_large"}
    assert source_paths == {"src/tab_foundry/bench/openml_classification_large_v1.json"}
