from __future__ import annotations

import json
from pathlib import Path
import string
import tempfile

from hypothesis import given, settings
from hypothesis import strategies as st
import pytest

import tab_foundry.bench.control_baseline as control_baseline_module


_REL_PATH = st.text(
    alphabet=string.ascii_letters + string.digits + "_-/.",
    min_size=1,
    max_size=48,
).filter(
    lambda value: not value.startswith("/")
    and "//" not in value
    and ".." not in value
    and value.strip("./") != ""
)


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
            "source_path": "src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
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


def _write_registry(path: Path, *, baselines: dict[str, object], schema: str | None = None, version: int | None = None) -> None:
    payload = {
        "schema": control_baseline_module.REGISTRY_SCHEMA if schema is None else schema,
        "version": control_baseline_module.REGISTRY_VERSION if version is None else version,
        "baselines": baselines,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


@settings(deadline=None, max_examples=35)
@given(rel_path=_REL_PATH)
def test_control_baseline_paths_roundtrip_repo_relative_paths(rel_path: str) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_root = (Path(tmp_dir) / "repo").resolve()
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(control_baseline_module, "project_root", lambda: repo_root)
            absolute_path = (repo_root / rel_path).resolve()

            normalized = control_baseline_module._normalize_path_value(absolute_path)

            assert normalized == str(absolute_path.relative_to(repo_root))
            assert control_baseline_module.resolve_registry_path_value(normalized) == absolute_path


@settings(deadline=None, max_examples=35)
@given(rel_path=_REL_PATH)
def test_control_baseline_paths_roundtrip_absolute_paths_outside_repo(rel_path: str) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        repo_root = (tmp_path / "repo").resolve()
        outside_root = tmp_path / "outside"
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(control_baseline_module, "project_root", lambda: repo_root)
            absolute_path = (outside_root / rel_path).resolve()

            normalized = control_baseline_module._normalize_path_value(absolute_path)

            assert normalized == str(absolute_path)
            assert control_baseline_module.resolve_registry_path_value(normalized) == absolute_path


@settings(deadline=None, max_examples=25)
@given(
    location=st.sampled_from(
        [
            ("entry", "experiment"),
            ("entry", "manifest_path"),
            ("benchmark_bundle", "name"),
            ("tab_foundry_metrics", "best_step"),
        ]
    )
)
def test_validate_baseline_entry_rejects_missing_required_fields(location: tuple[str, str]) -> None:
    scope, key = location
    entry = _valid_baseline_entry()
    if scope == "entry":
        entry.pop(key, None)
    else:
        nested = entry[scope]
        assert isinstance(nested, dict)
        nested.pop(key, None)

    with pytest.raises(RuntimeError, match="control baseline entry"):
        control_baseline_module._validate_baseline_entry(entry, baseline_id="baseline_v1")


@settings(deadline=None, max_examples=25)
@given(other_baseline_id=st.text(alphabet=string.ascii_letters + string.digits + "_-", min_size=1, max_size=16))
def test_validate_baseline_entry_rejects_baseline_id_mismatch(other_baseline_id: str) -> None:
    if other_baseline_id == "baseline_v1":
        other_baseline_id = "baseline_v2"
    entry = _valid_baseline_entry()
    entry["baseline_id"] = other_baseline_id

    with pytest.raises(RuntimeError, match="baseline_id mismatch"):
        control_baseline_module._validate_baseline_entry(entry, baseline_id="baseline_v1")


def test_load_control_baseline_entry_returns_deep_copy() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        registry_path = Path(tmp_dir) / "control_baselines_v1.json"
        _write_registry(
            registry_path,
            baselines={"baseline_v1": _valid_baseline_entry()},
        )

        loaded = control_baseline_module.load_control_baseline_entry(
            "baseline_v1",
            registry_path=registry_path,
        )
        loaded["benchmark_bundle"]["task_ids"].append(999)  # type: ignore[index]
        loaded["tab_foundry_metrics"]["final_roc_auc"] = -1.0  # type: ignore[index]

        reloaded = control_baseline_module.load_control_baseline_entry(
            "baseline_v1",
            registry_path=registry_path,
        )

        assert reloaded["benchmark_bundle"]["task_ids"] == list(range(1, 11))
        assert reloaded["tab_foundry_metrics"]["final_roc_auc"] == pytest.approx(0.8)


@settings(deadline=None, max_examples=25)
@given(
    mutation=st.sampled_from(
        [
            ("schema", "bad-schema"),
            ("version", 999),
            ("drop_top_level", "baselines"),
            ("drop_entry", "run_dir"),
        ]
    )
)
def test_load_control_baseline_registry_rejects_malformed_registry_payloads(
    mutation: tuple[str, object],
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        registry_path = Path(tmp_dir) / "control_baselines_v1.json"
        payload = {
            "schema": control_baseline_module.REGISTRY_SCHEMA,
            "version": control_baseline_module.REGISTRY_VERSION,
            "baselines": {"baseline_v1": _valid_baseline_entry()},
        }
        kind, value = mutation
        if kind == "schema":
            payload["schema"] = value
        elif kind == "version":
            payload["version"] = value
        elif kind == "drop_top_level":
            payload.pop(str(value), None)
        elif kind == "drop_entry":
            entry = payload["baselines"]["baseline_v1"]
            assert isinstance(entry, dict)
            entry.pop(str(value), None)

        registry_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

        with pytest.raises(RuntimeError):
            control_baseline_module.load_control_baseline_registry(registry_path)
