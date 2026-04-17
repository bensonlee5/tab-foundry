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


def _catalog_record(stem: str, manifest_row: dict[str, object]) -> dict[str, object]:
    manifest_root = REPO_ROOT / "data" / "manifests" / "bench" / stem
    catalog_path = manifest_root / Path(str(manifest_row["catalog_path"]))
    catalog = pq.read_table(catalog_path).to_pylist()
    return json.loads(str(catalog[int(manifest_row["catalog_dataset_index"])]["record_json"]))


def test_openml_classification_medium_manifest_matches_repo_frozen_bundle() -> None:
    stem = "openml_classification_medium_v1"
    bundle = _load_bundle(stem)
    rows = _manifest_rows(stem)

    assert bundle["name"] == "openml_classification_medium"
    assert len(rows) == len(bundle["task_ids"]) == 242

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

    assert bundle["name"] == "openml_classification_large"
    assert len(rows) == len(bundle["task_ids"]) == 3

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
