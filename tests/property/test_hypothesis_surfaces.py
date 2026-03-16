from __future__ import annotations

from pathlib import Path
import string

from hypothesis import given
from hypothesis import strategies as st
import pytest

from tab_foundry.bench.nanotabpfn import normalize_benchmark_bundle
from tab_foundry.data.surface import resolve_data_surface
from tab_foundry.preprocessing.surface import (
    SUPPORTED_LABEL_MAPPINGS,
    SUPPORTED_UNSEEN_TEST_LABEL_POLICIES,
    resolve_preprocessing_surface,
)


_TEXT_ALPHABET = string.ascii_letters + string.digits + "_-/"
_LABEL_TEXT = st.text(alphabet=string.ascii_letters + string.digits + "_-", min_size=1, max_size=24)
_PATH_TEXT = st.text(alphabet=_TEXT_ALPHABET, min_size=1, max_size=32).filter(
    lambda value: not value.startswith("/") and "//" not in value
)
_SELECTION_STRATEGY = st.fixed_dictionaries(
    {
        "new_instances": st.integers(min_value=1, max_value=10_000),
        "task_type": st.just("supervised_classification"),
        "max_features": st.integers(min_value=1, max_value=100),
        "max_classes": st.integers(min_value=1, max_value=10),
        "max_missing_pct": st.floats(min_value=0.0, max_value=100.0, allow_nan=False, allow_infinity=False),
        "min_minority_class_pct": st.floats(
            min_value=0.0,
            max_value=100.0,
            allow_nan=False,
            allow_infinity=False,
        ),
    }
)
_TASK_STRATEGY = st.fixed_dictionaries(
    {
        "task_id": st.integers(min_value=1, max_value=1_000_000),
        "dataset_name": _LABEL_TEXT,
        "n_rows": st.integers(min_value=1, max_value=10_000),
        "n_features": st.integers(min_value=1, max_value=500),
        "n_classes": st.integers(min_value=1, max_value=20),
    }
)
_TASKS_STRATEGY = st.lists(_TASK_STRATEGY, min_size=1, max_size=8, unique_by=lambda task: task["task_id"])


@given(
    name=_LABEL_TEXT,
    version=st.integers(min_value=1, max_value=100),
    selection=_SELECTION_STRATEGY,
    tasks=_TASKS_STRATEGY,
)
def test_normalize_benchmark_bundle_round_trips_valid_payloads(
    name: str,
    version: int,
    selection: dict[str, object],
    tasks: list[dict[str, object]],
) -> None:
    payload = {
        "name": name,
        "version": version,
        "selection": selection,
        "task_ids": [int(task["task_id"]) for task in tasks],
        "tasks": tasks,
    }

    normalized = normalize_benchmark_bundle(payload)

    assert normalized["name"] == name
    assert normalized["version"] == version
    assert normalized["task_ids"] == [int(task["task_id"]) for task in tasks]
    assert [task["task_id"] for task in normalized["tasks"]] == normalized["task_ids"]
    assert normalized["selection"] == {
        "new_instances": int(selection["new_instances"]),
        "task_type": "supervised_classification",
        "max_features": int(selection["max_features"]),
        "max_classes": int(selection["max_classes"]),
        "max_missing_pct": float(selection["max_missing_pct"]),
        "min_minority_class_pct": float(selection["min_minority_class_pct"]),
    }


@given(
    name=_LABEL_TEXT,
    version=st.integers(min_value=1, max_value=100),
    selection=_SELECTION_STRATEGY,
    tasks=st.lists(_TASK_STRATEGY, min_size=2, max_size=8, unique_by=lambda task: task["task_id"]),
)
def test_normalize_benchmark_bundle_rejects_task_id_order_drift(
    name: str,
    version: int,
    selection: dict[str, object],
    tasks: list[dict[str, object]],
) -> None:
    rotated_task_ids = [int(task["task_id"]) for task in tasks[1:]] + [int(tasks[0]["task_id"])]
    payload = {
        "name": name,
        "version": version,
        "selection": selection,
        "task_ids": rotated_task_ids,
        "tasks": tasks,
    }

    with pytest.raises(RuntimeError, match=r"task_ids must match tasks\[\]\.task_id order exactly"):
        _ = normalize_benchmark_bundle(payload)


@given(
    top_source=st.one_of(st.none(), _LABEL_TEXT),
    override_source=st.one_of(st.none(), _LABEL_TEXT),
    top_manifest=st.one_of(st.none(), _PATH_TEXT),
    override_manifest=st.one_of(st.none(), _PATH_TEXT),
    top_filter_policy=st.one_of(st.none(), _LABEL_TEXT),
    override_filter_policy=st.one_of(st.none(), _LABEL_TEXT),
    top_surface_label=st.one_of(st.none(), _LABEL_TEXT),
    override_surface_label=st.one_of(st.none(), _LABEL_TEXT),
    top_allow_missing_values=st.one_of(st.none(), st.booleans()),
    override_allow_missing_values=st.one_of(st.none(), st.booleans()),
    top_train_row_cap=st.one_of(st.none(), st.integers(min_value=1, max_value=50_000)),
    override_train_row_cap=st.one_of(st.none(), st.integers(min_value=1, max_value=50_000)),
    top_test_row_cap=st.one_of(st.none(), st.integers(min_value=1, max_value=50_000)),
    override_test_row_cap=st.one_of(st.none(), st.integers(min_value=1, max_value=50_000)),
    top_provenance=st.one_of(
        st.none(),
        st.dictionaries(_LABEL_TEXT, st.one_of(_LABEL_TEXT, st.integers(min_value=0, max_value=100)), max_size=4),
    ),
    override_provenance=st.one_of(
        st.none(),
        st.dictionaries(_LABEL_TEXT, st.one_of(_LABEL_TEXT, st.integers(min_value=0, max_value=100)), max_size=4),
    ),
)
def test_resolve_data_surface_respects_override_precedence(
    top_source: str | None,
    override_source: str | None,
    top_manifest: str | None,
    override_manifest: str | None,
    top_filter_policy: str | None,
    override_filter_policy: str | None,
    top_surface_label: str | None,
    override_surface_label: str | None,
    top_allow_missing_values: bool | None,
    override_allow_missing_values: bool | None,
    top_train_row_cap: int | None,
    override_train_row_cap: int | None,
    top_test_row_cap: int | None,
    override_test_row_cap: int | None,
    top_provenance: dict[str, object] | None,
    override_provenance: dict[str, object] | None,
) -> None:
    cfg = {
        "source": top_source,
        "manifest_path": top_manifest,
        "filter_policy": top_filter_policy,
        "surface_label": top_surface_label,
        "allow_missing_values": top_allow_missing_values,
        "train_row_cap": top_train_row_cap,
        "test_row_cap": top_test_row_cap,
        "dagzoo_provenance": top_provenance,
        "surface_overrides": {
            key: value
            for key, value in {
                "source": override_source,
                "manifest_path": override_manifest,
                "filter_policy": override_filter_policy,
                "surface_label": override_surface_label,
                "allow_missing_values": override_allow_missing_values,
                "train_row_cap": override_train_row_cap,
                "test_row_cap": override_test_row_cap,
                "dagzoo_provenance": override_provenance,
            }.items()
            if value is not None
        },
    }

    resolved = resolve_data_surface(cfg)
    expected_source = str(override_source if override_source is not None else top_source or "manifest").strip().lower()
    expected_manifest_raw = override_manifest if override_manifest is not None else top_manifest
    expected_filter_policy = (
        None if override_filter_policy is None and top_filter_policy is None else str(override_filter_policy or top_filter_policy).strip()
    )
    expected_surface_label = str(top_surface_label or override_surface_label or expected_source).strip()
    expected_allow_missing_values = (
        override_allow_missing_values
        if override_allow_missing_values is not None
        else top_allow_missing_values
        if top_allow_missing_values is not None
        else False
    )
    expected_train_row_cap = override_train_row_cap if override_train_row_cap is not None else top_train_row_cap
    expected_test_row_cap = override_test_row_cap if override_test_row_cap is not None else top_test_row_cap
    expected_provenance = override_provenance if override_provenance is not None else top_provenance

    assert resolved.source == expected_source
    assert resolved.surface_label == expected_surface_label
    assert resolved.filter_policy == expected_filter_policy
    assert resolved.allow_missing_values is expected_allow_missing_values
    assert resolved.train_row_cap == expected_train_row_cap
    assert resolved.test_row_cap == expected_test_row_cap
    assert resolved.dagzoo_provenance == expected_provenance
    if expected_manifest_raw is None:
        assert resolved.manifest_path is None
        assert resolved.to_dict()["manifest_path"] is None
    else:
        expected_manifest_path = Path(str(expected_manifest_raw)).expanduser().resolve()
        assert resolved.manifest_path == expected_manifest_path
        assert resolved.to_dict()["manifest_path"] == str(expected_manifest_path)


@given(
    top_surface_label=st.one_of(st.none(), _LABEL_TEXT),
    override_surface_label=st.one_of(st.none(), _LABEL_TEXT),
    top_impute_missing=st.one_of(st.none(), st.booleans()),
    override_impute_missing=st.one_of(st.none(), st.booleans()),
    top_all_nan_fill=st.one_of(
        st.none(),
        st.floats(min_value=-1_000.0, max_value=1_000.0, allow_nan=False, allow_infinity=False),
    ),
    override_all_nan_fill=st.one_of(
        st.none(),
        st.floats(min_value=-1_000.0, max_value=1_000.0, allow_nan=False, allow_infinity=False),
    ),
)
def test_resolve_preprocessing_surface_respects_override_precedence(
    top_surface_label: str | None,
    override_surface_label: str | None,
    top_impute_missing: bool | None,
    override_impute_missing: bool | None,
    top_all_nan_fill: float | None,
    override_all_nan_fill: float | None,
) -> None:
    cfg = {
        "surface_label": top_surface_label,
        "impute_missing": top_impute_missing,
        "all_nan_fill": top_all_nan_fill,
        "label_mapping": SUPPORTED_LABEL_MAPPINGS[0],
        "unseen_test_label_policy": SUPPORTED_UNSEEN_TEST_LABEL_POLICIES[0],
        "overrides": {
            key: value
            for key, value in {
                "surface_label": override_surface_label,
                "impute_missing": override_impute_missing,
                "all_nan_fill": override_all_nan_fill,
            }.items()
            if value is not None
        },
    }

    resolved = resolve_preprocessing_surface(cfg)
    expected_surface_label = str(top_surface_label or override_surface_label or "runtime_default").strip()
    expected_impute_missing = (
        override_impute_missing
        if override_impute_missing is not None
        else top_impute_missing
        if top_impute_missing is not None
        else True
    )
    expected_all_nan_fill = float(
        override_all_nan_fill
        if override_all_nan_fill is not None
        else top_all_nan_fill
        if top_all_nan_fill is not None
        else 0.0
    )

    assert resolved.surface_label == expected_surface_label
    assert resolved.impute_missing is expected_impute_missing
    assert resolved.all_nan_fill == expected_all_nan_fill
    assert resolved.label_mapping == SUPPORTED_LABEL_MAPPINGS[0]
    assert resolved.unseen_test_label_policy == SUPPORTED_UNSEEN_TEST_LABEL_POLICIES[0]
