from __future__ import annotations

import json
from pathlib import Path

import pytest

from tab_foundry.control_baseline_registry import (
    REGISTRY_SCHEMA,
    REGISTRY_VERSION,
    default_control_baseline_registry_path,
    load_control_baseline_entry,
    load_control_baseline_registry,
    normalize_registry_path_value,
    resolve_registry_path_value,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _valid_baseline_entry(baseline_id: str = "baseline_v1") -> dict[str, object]:
    return {
        "baseline_id": baseline_id,
        "experiment": "cls_benchmark_linear",
        "config_profile": "cls_benchmark_linear",
        "budget_class": "short-run",
        "manifest_path": "data/manifests/default.parquet",
        "seed_set": [1],
        "run_dir": "outputs/control_baselines/cls_benchmark_linear_v2/train",
        "comparison_summary_path": (
            "outputs/control_baselines/cls_benchmark_linear_v2/benchmark/comparison_summary.json"
        ),
        "benchmark_bundle": {
            "name": "binary_medium",
            "version": 1,
            "source_path": "src/tab_foundry/bench/openml_binary_medium_v1.json",
            "task_count": 10,
            "task_ids": list(range(1, 11)),
        },
        "tab_foundry_metrics": {
            "best_step": 25.0,
            "best_training_time": 1.2,
            "best_roc_auc": 0.81,
            "final_step": 25.0,
            "final_training_time": 1.2,
            "final_roc_auc": 0.8,
        },
    }


def _write_registry(path: Path, *, baselines: dict[str, object]) -> None:
    payload = {
        "schema": REGISTRY_SCHEMA,
        "version": REGISTRY_VERSION,
        "baselines": baselines,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def test_control_baseline_registry_default_path_is_repo_tracked() -> None:
    assert default_control_baseline_registry_path() == (
        REPO_ROOT / "src" / "tab_foundry" / "bench" / "control_baselines_v1.json"
    )


def test_control_baseline_registry_path_values_roundtrip_repo_relative_and_absolute(
    tmp_path: Path,
) -> None:
    repo_local = REPO_ROOT / "reference" / "system_delta_catalog.yaml"
    outside_repo = tmp_path / "outside.json"

    assert normalize_registry_path_value(repo_local) == "reference/system_delta_catalog.yaml"
    assert resolve_registry_path_value("reference/system_delta_catalog.yaml") == repo_local.resolve()
    assert normalize_registry_path_value(outside_repo) == str(outside_repo.resolve())
    assert resolve_registry_path_value(str(outside_repo.resolve())) == outside_repo.resolve()


def test_control_baseline_registry_load_entry_returns_deep_copy(tmp_path: Path) -> None:
    registry_path = tmp_path / "control_baselines_v1.json"
    _write_registry(registry_path, baselines={"baseline_v1": _valid_baseline_entry()})

    loaded = load_control_baseline_entry("baseline_v1", registry_path=registry_path)
    loaded["benchmark_bundle"]["task_ids"].append(999)  # type: ignore[index]
    loaded["tab_foundry_metrics"]["final_roc_auc"] = -1.0  # type: ignore[index]

    reloaded = load_control_baseline_entry("baseline_v1", registry_path=registry_path)

    assert reloaded["benchmark_bundle"]["task_ids"] == list(range(1, 11))
    assert reloaded["tab_foundry_metrics"]["final_roc_auc"] == pytest.approx(0.8)


def test_control_baseline_registry_loads_checked_in_registry() -> None:
    registry = load_control_baseline_registry()

    assert {"cls_benchmark_linear_v1", "cls_benchmark_linear_v2"} <= set(registry["baselines"])


def test_control_baseline_registry_rejects_malformed_payload(tmp_path: Path) -> None:
    registry_path = tmp_path / "control_baselines_v1.json"
    registry_path.write_text(
        json.dumps(
            {
                "schema": REGISTRY_SCHEMA,
                "version": REGISTRY_VERSION,
                "baselines": {"baseline_v1": {"baseline_id": "baseline_v1"}},
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="control baseline entry"):
        _ = load_control_baseline_registry(registry_path)
