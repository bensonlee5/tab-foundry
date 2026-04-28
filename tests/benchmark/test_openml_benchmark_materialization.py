from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from tab_foundry.bench.openml_benchmark.materialization import (
    materialize_benchmark_bundle,
)
from tab_foundry.data.dataset import load_manifest_record_metadata
from tests.benchmark.openml_bundle_fakes import prepared_task


def test_materialize_benchmark_bundle_persists_portable_source_path(
    tmp_path: Path,
) -> None:
    bundle_path = (
        tmp_path
        / "foreign-checkout"
        / "src"
        / "tab_foundry"
        / "bench"
        / "openml_classification_missing_wide_v1.json"
    )
    bundle_path.parent.mkdir(parents=True, exist_ok=True)
    bundle_path.write_text(
        json.dumps(
            {
                "name": "openml_classification_missing_wide",
                "version": 1,
                "selection": {
                    "task_type": "supervised_classification",
                    "min_classes": 2,
                    "max_classes": 10,
                    "max_features": 100,
                    "min_missing_pct": 0.5,
                    "max_missing_pct": 20.0,
                    "min_minority_class_pct": 1.0,
                    "new_instances": 20,
                },
                "task_ids": [13],
            }
        ),
        encoding="utf-8",
    )

    result = materialize_benchmark_bundle(
        bundle_path,
        tmp_path / "out",
        split_seed=0,
        test_size=0.20,
        prepare_task_fn=lambda task_id, new_instances, task_type: prepared_task(
            task_id=task_id,
            dataset_name="breast-cancer",
            n_rows=new_instances,
            n_features=4,
            n_classes=2,
            missing_pct=1.0,
        ),
    )

    expected_source_path = (
        "src/tab_foundry/bench/openml_classification_missing_wide_v1.json"
    )
    rows = pq.read_table(result.manifest_path).to_pylist()
    metadata, _feature_types = load_manifest_record_metadata(
        result.manifest_path,
        record=rows[0],
        expected_feature_count=int(rows[0]["n_features"]),
        require_feature_types=False,
    )

    assert result.bundle_summary["source_path"] == expected_source_path
    assert metadata["benchmark_bundle"]["source_path"] == expected_source_path
