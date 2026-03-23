from __future__ import annotations

from importlib.util import module_from_spec, spec_from_file_location
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pyarrow.parquet as pq

from tab_foundry.data.manifest import MANIFEST_SUMMARY_METADATA_KEY, build_manifest


REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "materialize_tf_rd_013_support.py"


def _load_script_module():
    spec = spec_from_file_location("materialize_tf_rd_013_support", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_split_prepared_task_falls_back_when_stratification_is_not_possible() -> None:
    module = _load_script_module()
    prepared = module.PreparedOpenMLBenchmarkTask(
        task_id=1,
        dataset_name="singleton_minority",
        x=np.arange(8, dtype=np.float32).reshape(4, 2),
        y=np.array([0, 0, 0, 1], dtype=np.int64),
        observed_task={"task_id": 1, "dataset_name": "singleton_minority", "n_rows": 4, "n_features": 2},
        qualities={"NumberOfFeatures": 2.0, "PercentageOfInstancesWithMissingValues": 0.0, "NumberOfClasses": 2.0},
    )

    x_train, x_test, y_train, y_test, split_mode = module._split_prepared_task(
        prepared,
        split_seed=0,
        test_size=0.5,
    )

    assert split_mode == "unstratified_fallback"
    assert x_train.shape == (2, 2)
    assert x_test.shape == (2, 2)
    assert y_train.shape == (2,)
    assert y_test.shape == (2,)


def test_materialize_curated_openml_baseline_writes_manifest_backed_shards(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_script_module()
    paths = module._materialization_paths(tmp_path / "outputs" / "staged_ladder_support" / "tf_rd_013")
    bundle_path = tmp_path / "nanotabpfn_openml_binary_large_v1.json"
    bundle_path.write_text("{}", encoding="utf-8")

    fake_bundle = {
        "name": "nanotabpfn_openml_binary_large",
        "version": 1,
        "selection": {
            "new_instances": 10,
            "task_type": "supervised_classification",
            "max_features": 20,
            "max_missing_pct": 5.0,
            "max_classes": 2,
            "min_minority_class_pct": 2.5,
        },
        "task_ids": [101, 102],
        "tasks": [
            {"task_id": 101, "dataset_name": "first_dataset", "n_rows": 10, "n_features": 3, "n_classes": 2},
            {"task_id": 102, "dataset_name": "second_dataset", "n_rows": 10, "n_features": 2, "n_classes": 2},
        ],
    }
    prepared_tasks = {
        101: module.PreparedOpenMLBenchmarkTask(
            task_id=101,
            dataset_name="first_dataset",
            x=np.array(
                [
                    [0.0, 1.0, 2.0],
                    [1.0, 2.0, 3.0],
                    [2.0, 3.0, 4.0],
                    [3.0, 4.0, 5.0],
                    [4.0, 5.0, 6.0],
                    [5.0, 6.0, 7.0],
                    [6.0, 7.0, 8.0],
                    [7.0, 8.0, 9.0],
                    [8.0, 9.0, 10.0],
                    [9.0, 10.0, 11.0],
                ],
                dtype=np.float32,
            ),
            y=np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64),
            observed_task={"task_id": 101, "dataset_name": "first_dataset", "n_rows": 10, "n_features": 3},
            qualities={"NumberOfFeatures": 3.0, "PercentageOfInstancesWithMissingValues": 0.0, "NumberOfClasses": 2.0},
        ),
        102: module.PreparedOpenMLBenchmarkTask(
            task_id=102,
            dataset_name="second_dataset",
            x=np.array(
                [
                    [10.0, 0.0],
                    [11.0, 1.0],
                    [12.0, 2.0],
                    [13.0, 3.0],
                    [14.0, 4.0],
                    [15.0, 5.0],
                    [16.0, 6.0],
                    [17.0, 7.0],
                    [18.0, 8.0],
                    [19.0, 9.0],
                ],
                dtype=np.float32,
            ),
            y=np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1], dtype=np.int64),
            observed_task={"task_id": 102, "dataset_name": "second_dataset", "n_rows": 10, "n_features": 2},
            qualities={"NumberOfFeatures": 2.0, "PercentageOfInstancesWithMissingValues": 0.0, "NumberOfClasses": 2.0},
        ),
    }

    monkeypatch.setattr(
        module,
        "load_benchmark_bundle_for_execution",
        lambda _path: (fake_bundle, True),
    )
    monkeypatch.setattr(
        module,
        "prepare_openml_benchmark_task",
        lambda task_id, **_: prepared_tasks[int(task_id)],
    )

    def fake_run_checked(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        assert "build-manifest" in cmd
        build_manifest([paths.curated_openml_baseline_data_root], paths.curated_openml_baseline_manifest_path)
        return subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setattr(module, "_run_checked", fake_run_checked)

    result = module._materialize_curated_openml_baseline(
        paths=paths,
        bundle_path=bundle_path,
    )

    assert paths.curated_openml_baseline_manifest_path.exists()
    manifest_rows = pq.read_table(paths.curated_openml_baseline_manifest_path).to_pylist()
    assert len(manifest_rows) == 2
    assert result["allow_missing_values"] is True
    assert result["bundle_summary"]["source_path"].endswith("nanotabpfn_openml_binary_large_v1.json")
    assert [entry["task_id"] for entry in result["task_summaries"]] == [101, 102]
    assert all(entry["n_train"] == 8 for entry in result["task_summaries"])
    assert all(entry["n_test"] == 2 for entry in result["task_summaries"])

    metadata_path = paths.curated_openml_baseline_data_root / "shard_00001_first_dataset" / "metadata.ndjson"
    payload = json.loads(metadata_path.read_text(encoding="utf-8").strip())
    assert payload["feature_types"] == ["num", "num", "num"]
    assert payload["metadata"]["filter"] == {"mode": "deferred", "status": "not_run"}
    assert payload["metadata"]["source_platform"] == "openml"
    assert payload["metadata"]["benchmark_bundle"]["source_path"] == module.DEFAULT_BENCHMARK_BUNDLE_REF
    assert payload["metadata"]["openml"]["task_id"] == 101


def test_shape_aware_materializer_merges_multiple_dagzoo_invocations_without_single_handoff_metadata(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module = _load_script_module()

    dagzoo_root = tmp_path / "dagzoo"
    (dagzoo_root / "configs").mkdir(parents=True, exist_ok=True)
    for config_name in (
        "benchmark_cpu.yaml",
        "default.yaml",
        "benchmark_cuda_h100_large_shape.yaml",
    ):
        (dagzoo_root / "configs" / config_name).write_text("seed: 1\n", encoding="utf-8")

    anchor_root = tmp_path / "anchor"
    anchor_data_root = anchor_root / "packed_shards"
    shard_dir = anchor_data_root / "shard_00001_anchor"
    module._write_packed_shard(
        shard_dir,
        x_train=np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32),
        y_train=np.array([0, 1], dtype=np.int64),
        x_test=np.array([[2.0, 3.0]], dtype=np.float32),
        y_test=np.array([1], dtype=np.int64),
        metadata={
            "config": {"dataset": {"task": "classification"}},
            "filter": {"mode": "deferred", "status": "not_run"},
            "n_classes": 2,
            "seed": 0,
        },
    )
    anchor_manifest_path = anchor_root / "manifest.parquet"
    build_manifest([anchor_data_root], anchor_manifest_path)

    fake_bundle = {
        "name": "nanotabpfn_openml_binary_large",
        "version": 1,
        "selection": {
            "new_instances": 10,
            "task_type": "supervised_classification",
            "max_features": 20,
            "max_missing_pct": 5.0,
            "max_classes": 2,
            "min_minority_class_pct": 2.5,
        },
        "task_ids": [101],
        "tasks": [
            {"task_id": 101, "dataset_name": "first_dataset", "n_rows": 10, "n_features": 3, "n_classes": 2},
        ],
    }
    prepared_task = module.PreparedOpenMLBenchmarkTask(
        task_id=101,
        dataset_name="first_dataset",
        x=np.array(
            [
                [0.0, 1.0, 2.0],
                [1.0, 2.0, 3.0],
                [2.0, 3.0, 4.0],
                [3.0, 4.0, 5.0],
                [4.0, 5.0, 6.0],
                [5.0, 6.0, 7.0],
                [6.0, 7.0, 8.0],
                [7.0, 8.0, 9.0],
                [8.0, 9.0, 10.0],
                [9.0, 10.0, 11.0],
            ],
            dtype=np.float32,
        ),
        y=np.array([0, 1, 0, 1, 0, 1, 0, 1, 0, 1], dtype=np.int64),
        observed_task={"task_id": 101, "dataset_name": "first_dataset", "n_rows": 10, "n_features": 3},
        qualities={"NumberOfFeatures": 3.0, "PercentageOfInstancesWithMissingValues": 0.0, "NumberOfClasses": 2.0},
    )

    monkeypatch.setattr(module, "ANCHOR_MANIFEST_PATH", anchor_manifest_path)
    monkeypatch.setattr(module, "SHAPE_AWARE_LOCAL_OUTPUT_ROOT", tmp_path / "outputs" / "shape_aware")
    monkeypatch.setattr(module, "SHAPE_AWARE_SUPPORT_ROOT", tmp_path / "support")
    monkeypatch.setattr(module, "load_benchmark_bundle_for_execution", lambda _path: (fake_bundle, True))
    monkeypatch.setattr(module, "prepare_openml_benchmark_task", lambda *_args, **_kwargs: prepared_task)

    def fake_manifest_inspect(path: Path) -> dict[str, object]:
        metadata = pq.read_metadata(path).metadata or {}
        persisted_summary = None
        raw_summary = metadata.get(MANIFEST_SUMMARY_METADATA_KEY)
        if raw_summary is not None:
            persisted_summary = json.loads(raw_summary.decode("utf-8"))
        table = pq.read_table(path)
        return {
            "manifest_path": module._portable_path(path),
            "persisted_summary": persisted_summary,
            "total_records": int(table.num_rows),
            "unique_dataset_id_count": int(table.num_rows),
            "unique_source_root_count": len({str(value) for value in table.column("source_root_id").to_pylist()}),
        }

    monkeypatch.setattr(module, "_manifest_inspect", fake_manifest_inspect)

    def fake_run_checked(cmd: list[str], *, cwd: Path) -> subprocess.CompletedProcess[str]:
        if "dagzoo" in cmd and "generate" in cmd:
            handoff_root = Path(cmd[cmd.index("--handoff-root") + 1])
            config_path = Path(cmd[cmd.index("--config") + 1])
            generated_dir = handoff_root / "generated" / "shard_00001_generated"
            generate_hex = {
                "benchmark_cpu.yaml": "a" * 32,
                "default.yaml": "b" * 32,
                "benchmark_cuda_h100_large_shape.yaml": "c" * 32,
            }[config_path.name]
            dataset_hex = {
                "benchmark_cpu.yaml": "1" * 32,
                "default.yaml": "2" * 32,
                "benchmark_cuda_h100_large_shape.yaml": "3" * 32,
            }[config_path.name]
            module._write_packed_shard(
                generated_dir,
                x_train=np.array([[0.0, 1.0], [1.0, 2.0]], dtype=np.float32),
                y_train=np.array([0, 1], dtype=np.int64),
                x_test=np.array([[2.0, 3.0]], dtype=np.float32),
                y_test=np.array([1], dtype=np.int64),
                metadata={
                    "config": {"dataset": {"task": "classification"}},
                    "filter": {"mode": "deferred", "status": "not_run"},
                    "dataset_id": dataset_hex,
                    "split_groups": {"request_run": generate_hex},
                    "n_classes": 2,
                    "seed": 1,
                },
            )
            handoff_root.mkdir(parents=True, exist_ok=True)
            (handoff_root / "handoff_manifest.json").write_text(
                json.dumps(
                    {
                        "schema_name": "dagzoo_generate_handoff_manifest",
                        "schema_version": 1,
                        "identity": {
                            "source_family": "dagzoo.fixed_layout_scm",
                            "generate_run_id": generate_hex,
                            "generated_corpus_id": dataset_hex,
                        },
                        "artifacts_relative": {
                            "run_root": ".",
                            "generated_dir": "generated",
                        },
                        "defaults": {
                            "recommended_training_corpus": "generated",
                            "recommended_training_artifact_key": "generated_dir",
                            "curation_policy": "none",
                        },
                        "summary": {"generated_datasets": 1},
                        "throughput": {"generation_stage": {"generated_datasets": 1, "elapsed_seconds": 1.0}},
                        "hardware": {
                            "backend": "cpu",
                            "device_name": "cpu",
                            "hardware_policy": "none",
                            "requested_device": "cpu",
                            "resolved_device": "cpu",
                            "tier": "cpu",
                        },
                        "generate_invocation": {
                            "config_path": module._portable_path(config_path),
                            "overrides": {"num_datasets": 1},
                        },
                    },
                    indent=2,
                    sort_keys=True,
                ),
                encoding="utf-8",
            )
            return subprocess.CompletedProcess(cmd, 0, "", "")
        if "build-manifest" in cmd:
            out_manifest = Path(cmd[cmd.index("--out-manifest") + 1])
            data_root = Path(cmd[cmd.index("--data-root") + 1])
            build_manifest([data_root], out_manifest)
            return subprocess.CompletedProcess(cmd, 0, "", "")
        raise AssertionError(f"unexpected command: {cmd}")

    monkeypatch.setattr(module, "_run_checked", fake_run_checked)

    result = module.materialize_support(
        variant="shape-aware",
        dagzoo_root=dagzoo_root,
        force=True,
    )

    assert result == 0
    support_root = module.SHAPE_AWARE_SUPPORT_ROOT
    materialization_summary = json.loads((support_root / "materialization_summary.json").read_text(encoding="utf-8"))
    manifest_characteristics_summary = json.loads(
        (support_root / "manifest_characteristics_summary.json").read_text(encoding="utf-8")
    )

    dagzoo_surface = materialization_summary["surfaces"]["dagzoo_shape_aware_multi_invocation"]
    dagzoo_provenance = dagzoo_surface["dagzoo_provenance"]
    assert dagzoo_provenance["corpus_variant"] == "dagzoo_shape_aware_multi_invocation"
    assert dagzoo_provenance["config_refs"] == [
        "configs/benchmark_cpu.yaml",
        "configs/default.yaml",
        "configs/benchmark_cuda_h100_large_shape.yaml",
    ]
    assert [entry["invocation_id"] for entry in dagzoo_provenance["invocations"]] == [
        "benchmark_cpu",
        "default_medium",
        "large_shape",
    ]
    assert materialization_summary["assembly"]["invocation_count"] == 3

    combined_manifest_path = Path(dagzoo_surface["manifest_path"])
    if not combined_manifest_path.is_absolute():
        combined_manifest_path = REPO_ROOT / combined_manifest_path
    persisted_summary = json.loads(
        pq.read_metadata(combined_manifest_path).metadata[MANIFEST_SUMMARY_METADATA_KEY].decode("utf-8")
    )
    assert "dagzoo_handoff" not in persisted_summary
    assert persisted_summary["total_records"] == 3

    comparisons = manifest_characteristics_summary["comparisons"]
    assert set(comparisons) == {
        "anchor_vs_curated_realdata_openml_baseline",
        "anchor_vs_dagzoo_shape_aware_multi_invocation",
        "dagzoo_shape_aware_multi_invocation_vs_curated_realdata_openml_baseline",
    }
