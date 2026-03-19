from __future__ import annotations

import pandas as pd
import pytest

import tab_foundry.bench.openml_benchmark_bundle as bundle_module

from tests.benchmark.openml_bundle_fakes import prepared_task


def test_build_openml_benchmark_bundle_discovery_filters_dedupes_and_reports(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def _fake_list_tasks(**kwargs: object) -> pd.DataFrame:
        captured.update(kwargs)
        return pd.DataFrame(
            [
                {
                    "tid": 101,
                    "did": 10,
                    "name": "dup_dataset",
                    "NumberOfInstances": 300,
                    "NumberOfFeatures": 5,
                    "NumberOfClasses": 2,
                    "NumberOfInstancesWithMissingValues": 0,
                    "MinorityClassSize": 75,
                    "estimation_procedure": "holdout",
                },
                {
                    "tid": 102,
                    "did": 10,
                    "name": "dup_dataset",
                    "NumberOfInstances": 300,
                    "NumberOfFeatures": 5,
                    "NumberOfClasses": 2,
                    "NumberOfInstancesWithMissingValues": 0,
                    "MinorityClassSize": 75,
                    "estimation_procedure": "10-fold Crossvalidation",
                },
                {
                    "tid": 103,
                    "did": 11,
                    "name": "kept_dataset",
                    "NumberOfInstances": 400,
                    "NumberOfFeatures": 12,
                    "NumberOfClasses": 2,
                    "NumberOfInstancesWithMissingValues": 0,
                    "MinorityClassSize": 80,
                    "estimation_procedure": "holdout",
                },
                {
                    "tid": 201,
                    "did": 20,
                    "name": "too_wide",
                    "NumberOfInstances": 400,
                    "NumberOfFeatures": 60,
                    "NumberOfClasses": 2,
                    "NumberOfInstancesWithMissingValues": 0,
                    "MinorityClassSize": 80,
                    "estimation_procedure": "10-fold Crossvalidation",
                },
                {
                    "tid": 202,
                    "did": 21,
                    "name": "has_missing",
                    "NumberOfInstances": 400,
                    "NumberOfFeatures": 8,
                    "NumberOfClasses": 2,
                    "NumberOfInstancesWithMissingValues": 1,
                    "MinorityClassSize": 80,
                    "estimation_procedure": "10-fold Crossvalidation",
                },
                {
                    "tid": 203,
                    "did": 22,
                    "name": "too_small",
                    "NumberOfInstances": 150,
                    "NumberOfFeatures": 8,
                    "NumberOfClasses": 2,
                    "NumberOfInstancesWithMissingValues": 0,
                    "MinorityClassSize": 50,
                    "estimation_procedure": "10-fold Crossvalidation",
                },
                {
                    "tid": 204,
                    "did": 23,
                    "name": "too_imbalanced",
                    "NumberOfInstances": 400,
                    "NumberOfFeatures": 8,
                    "NumberOfClasses": 2,
                    "NumberOfInstancesWithMissingValues": 0,
                    "MinorityClassSize": 5,
                    "estimation_procedure": "10-fold Crossvalidation",
                },
                {
                    "tid": 205,
                    "did": 24,
                    "name": "multiclass",
                    "NumberOfInstances": 400,
                    "NumberOfFeatures": 8,
                    "NumberOfClasses": 3,
                    "NumberOfInstancesWithMissingValues": 0,
                    "MinorityClassSize": 100,
                    "estimation_procedure": "10-fold Crossvalidation",
                },
            ]
        )

    monkeypatch.setattr(
        bundle_module.openml.tasks,
        "list_tasks",
        _fake_list_tasks,
    )
    prepared_by_task_id = {
        102: prepared_task(task_id=102, dataset_name="dup_dataset", n_rows=200, n_features=5, n_classes=2),
        103: prepared_task(task_id=103, dataset_name="kept_dataset", n_rows=200, n_features=12, n_classes=2),
    }
    monkeypatch.setattr(
        bundle_module,
        "prepare_openml_benchmark_task",
        lambda task_id, *, new_instances, task_type: prepared_by_task_id[int(task_id)],
    )

    result = bundle_module.build_openml_benchmark_bundle_result(
        bundle_module.OpenMLBenchmarkBundleConfig(
            bundle_name="binary_large_no_missing",
            version=1,
            discover_from_openml=True,
            min_instances=200,
            min_task_count=2,
            max_features=50,
            max_classes=2,
            max_missing_pct=0.0,
            min_minority_class_pct=2.5,
        )
    )

    assert result.bundle["task_ids"] == [102, 103]
    assert [task["dataset_name"] for task in result.bundle["tasks"]] == ["dup_dataset", "kept_dataset"]
    assert captured["number_instances"] == "200.."
    assert captured["number_features"] == "..50"
    assert captured["number_classes"] == 2
    assert captured["number_missing_values"] == 0
    report = bundle_module.render_openml_benchmark_candidate_report(result.report_entries)
    assert "- accepted=2" in report
    assert "- rejected=6" in report
    assert "preferred task_id=102" in report
    assert "number_of_features=60 exceeds max_features=50" in report
    assert "missing_pct=0.25 exceeds max_missing_pct=0" in report
    assert "number_of_instances=150 below min_instances=200" in report
    assert "number_of_classes=3 exceeds max_classes=2" in report


def test_build_openml_benchmark_bundle_discovery_rejects_below_min_task_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        bundle_module.openml.tasks,
        "list_tasks",
        lambda **_kwargs: pd.DataFrame(
            [
                {
                    "tid": 10,
                    "did": 1,
                    "name": "only_one",
                    "NumberOfInstances": 400,
                    "NumberOfFeatures": 4,
                    "NumberOfClasses": 2,
                    "NumberOfInstancesWithMissingValues": 0,
                    "MinorityClassSize": 80,
                    "estimation_procedure": "10-fold Crossvalidation",
                }
            ]
        ),
    )
    monkeypatch.setattr(
        bundle_module,
        "prepare_openml_benchmark_task",
        lambda task_id, *, new_instances, task_type: prepared_task(
            task_id=int(task_id),
            dataset_name="only_one",
            n_rows=200,
            n_features=4,
            n_classes=2,
        ),
    )

    with pytest.raises(RuntimeError, match="below min_task_count"):
        bundle_module.build_openml_benchmark_bundle(
            bundle_module.OpenMLBenchmarkBundleConfig(
                bundle_name="binary_large_no_missing",
                version=1,
                discover_from_openml=True,
                min_instances=200,
                min_task_count=2,
                max_features=50,
                max_classes=2,
                max_missing_pct=0.0,
                min_minority_class_pct=2.5,
            )
        )


def test_build_openml_benchmark_bundle_discovery_falls_back_when_filtered_listing_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def _fake_list_tasks(**kwargs: object) -> pd.DataFrame:
        calls.append(dict(kwargs))
        if len(calls) == 1:
            raise RuntimeError("OpenML task-list filter request failed")
        return pd.DataFrame(
            [
                {
                    "tid": 10,
                    "did": 1,
                    "name": "fallback_ok",
                    "NumberOfInstances": 400,
                    "NumberOfFeatures": 4,
                    "NumberOfClasses": 2,
                    "NumberOfInstancesWithMissingValues": 0,
                    "MinorityClassSize": 80,
                    "estimation_procedure": "10-fold Crossvalidation",
                }
            ]
        )

    monkeypatch.setattr(bundle_module.openml.tasks, "list_tasks", _fake_list_tasks)
    monkeypatch.setattr(
        bundle_module,
        "prepare_openml_benchmark_task",
        lambda task_id, *, new_instances, task_type: prepared_task(
            task_id=int(task_id),
            dataset_name="fallback_ok",
            n_rows=200,
            n_features=4,
            n_classes=2,
        ),
    )

    bundle = bundle_module.build_openml_benchmark_bundle(
        bundle_module.OpenMLBenchmarkBundleConfig(
            bundle_name="binary_large_no_missing",
            version=1,
            discover_from_openml=True,
            min_instances=200,
            min_task_count=1,
            max_features=50,
            max_classes=2,
            max_missing_pct=0.0,
            min_minority_class_pct=2.5,
        )
    )

    assert bundle["task_ids"] == [10]
    assert len(calls) == 2
    assert calls[0]["number_instances"] == "200.."
    assert calls[0]["number_features"] == "..50"
    assert calls[0]["number_classes"] == 2
    assert calls[0]["number_missing_values"] == 0
    assert "number_instances" not in calls[1]
    assert "number_features" not in calls[1]
    assert "number_classes" not in calls[1]
    assert "number_missing_values" not in calls[1]
