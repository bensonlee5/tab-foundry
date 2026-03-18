from __future__ import annotations

import pandas as pd
import pytest

import tab_foundry.bench.openml_benchmark_bundle as bundle_module

from tests.benchmark.openml_bundle_fakes import FakeDataset, FakeTask, prepared_task


def test_build_openml_benchmark_bundle_preserves_binary_filter(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_by_task_id = {
        20: prepared_task(task_id=20, dataset_name="multi", n_rows=200, n_features=4, n_classes=3),
        10: prepared_task(task_id=10, dataset_name="binary", n_rows=200, n_features=3, n_classes=2),
    }
    monkeypatch.setattr(
        bundle_module,
        "prepare_openml_benchmark_task",
        lambda task_id, *, new_instances, task_type: prepared_by_task_id[int(task_id)],
    )
    monkeypatch.setattr(
        bundle_module.openml.tasks,
        "get_task",
        lambda task_id, **_kwargs: FakeTask(
            FakeDataset(
                name=prepared_by_task_id[int(task_id)].dataset_name,
                qualities={
                    "NumberOfFeatures": prepared_by_task_id[int(task_id)].qualities["NumberOfFeatures"],
                    "NumberOfClasses": prepared_by_task_id[int(task_id)].qualities["NumberOfClasses"],
                    "PercentageOfInstancesWithMissingValues": prepared_by_task_id[int(task_id)].qualities[
                        "PercentageOfInstancesWithMissingValues"
                    ],
                    "MinorityClassPercentage": prepared_by_task_id[int(task_id)].qualities[
                        "MinorityClassPercentage"
                    ],
                },
                frame=pd.DataFrame({"placeholder": [0, 1]}),
                target=pd.Series(["no", "yes"]),
            )
        ),
    )

    bundle = bundle_module.build_openml_benchmark_bundle(
        bundle_module.OpenMLBenchmarkBundleConfig(
            bundle_name="binary_small",
            version=1,
            max_classes=2,
            task_ids=(20, 10),
        )
    )

    assert bundle["selection"]["max_classes"] == 2
    assert bundle["task_ids"] == [10]
    assert bundle["tasks"] == [
        {
            "task_id": 10,
            "dataset_name": "binary",
            "n_rows": 200,
            "n_features": 3,
            "n_classes": 2,
        }
    ]


def test_build_openml_benchmark_bundle_auto_max_classes_widens_and_sorts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_by_task_id = {
        20: prepared_task(task_id=20, dataset_name="multi", n_rows=200, n_features=4, n_classes=3),
        10: prepared_task(task_id=10, dataset_name="binary", n_rows=200, n_features=3, n_classes=2),
    }
    monkeypatch.setattr(
        bundle_module,
        "prepare_openml_benchmark_task",
        lambda task_id, *, new_instances, task_type: prepared_by_task_id[int(task_id)],
    )
    monkeypatch.setattr(
        bundle_module.openml.tasks,
        "get_task",
        lambda task_id, **_kwargs: FakeTask(
            FakeDataset(
                name=prepared_by_task_id[int(task_id)].dataset_name,
                qualities={
                    "NumberOfFeatures": prepared_by_task_id[int(task_id)].qualities["NumberOfFeatures"],
                    "NumberOfClasses": prepared_by_task_id[int(task_id)].qualities["NumberOfClasses"],
                    "PercentageOfInstancesWithMissingValues": prepared_by_task_id[int(task_id)].qualities[
                        "PercentageOfInstancesWithMissingValues"
                    ],
                    "MinorityClassPercentage": prepared_by_task_id[int(task_id)].qualities[
                        "MinorityClassPercentage"
                    ],
                },
                frame=pd.DataFrame({"placeholder": [0, 1]}),
                target=pd.Series(["no", "yes"]),
            )
        ),
    )

    bundle = bundle_module.build_openml_benchmark_bundle(
        bundle_module.OpenMLBenchmarkBundleConfig(
            bundle_name="classification_small",
            version=1,
            max_classes=None,
            task_ids=(20, 10),
        )
    )

    assert bundle["selection"]["max_classes"] == 3
    assert bundle["task_ids"] == [10, 20]
    assert [task["task_id"] for task in bundle["tasks"]] == [10, 20]


def test_build_openml_benchmark_bundle_uses_named_task_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_by_task_id = {
        30: prepared_task(task_id=30, dataset_name="source_high", n_rows=200, n_features=4, n_classes=2),
        10: prepared_task(task_id=10, dataset_name="source_low", n_rows=200, n_features=3, n_classes=2),
    }
    monkeypatch.setattr(bundle_module, "task_ids_for_source", lambda task_source: (30, 10))
    monkeypatch.setattr(
        bundle_module,
        "prepare_openml_benchmark_task",
        lambda task_id, *, new_instances, task_type: prepared_by_task_id[int(task_id)],
    )
    monkeypatch.setattr(
        bundle_module.openml.tasks,
        "get_task",
        lambda task_id, **_kwargs: FakeTask(
            FakeDataset(
                name=prepared_by_task_id[int(task_id)].dataset_name,
                qualities={
                    "NumberOfFeatures": prepared_by_task_id[int(task_id)].qualities["NumberOfFeatures"],
                    "NumberOfClasses": prepared_by_task_id[int(task_id)].qualities["NumberOfClasses"],
                    "PercentageOfInstancesWithMissingValues": prepared_by_task_id[int(task_id)].qualities[
                        "PercentageOfInstancesWithMissingValues"
                    ],
                    "MinorityClassPercentage": prepared_by_task_id[int(task_id)].qualities[
                        "MinorityClassPercentage"
                    ],
                },
                frame=pd.DataFrame({"placeholder": [0, 1]}),
                target=pd.Series(["no", "yes"]),
            )
        ),
    )

    bundle = bundle_module.build_openml_benchmark_bundle(
        bundle_module.OpenMLBenchmarkBundleConfig(
            bundle_name="binary_medium",
            version=1,
            task_source="binary_expanded_v1",
        )
    )

    assert bundle["task_ids"] == [10, 30]
    assert [task["dataset_name"] for task in bundle["tasks"]] == ["source_low", "source_high"]


def test_build_openml_benchmark_bundle_rejects_empty_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    prepared_by_task_id = {
        10: prepared_task(
            task_id=10,
            dataset_name="too_wide",
            n_rows=200,
            n_features=3,
            n_classes=2,
            missing_pct=1.0,
        )
    }
    monkeypatch.setattr(
        bundle_module,
        "prepare_openml_benchmark_task",
        lambda task_id, *, new_instances, task_type: prepared_by_task_id[int(task_id)],
    )
    monkeypatch.setattr(
        bundle_module.openml.tasks,
        "get_task",
        lambda task_id, **_kwargs: FakeTask(
            FakeDataset(
                name=prepared_by_task_id[int(task_id)].dataset_name,
                qualities={
                    "NumberOfFeatures": prepared_by_task_id[int(task_id)].qualities["NumberOfFeatures"],
                    "NumberOfClasses": prepared_by_task_id[int(task_id)].qualities["NumberOfClasses"],
                    "PercentageOfInstancesWithMissingValues": prepared_by_task_id[int(task_id)].qualities[
                        "PercentageOfInstancesWithMissingValues"
                    ],
                    "MinorityClassPercentage": prepared_by_task_id[int(task_id)].qualities[
                        "MinorityClassPercentage"
                    ],
                },
                frame=pd.DataFrame({"placeholder": [0, 1]}),
                target=pd.Series(["no", "yes"]),
            )
        ),
    )

    with pytest.raises(RuntimeError, match="produced no eligible tasks"):
        bundle_module.build_openml_benchmark_bundle(
            bundle_module.OpenMLBenchmarkBundleConfig(
                bundle_name="empty",
                version=1,
                task_ids=(10,),
            )
        )
