from __future__ import annotations

from pathlib import Path
import string
import tempfile

from hypothesis import given, settings
from hypothesis import strategies as st
import pytest

import tab_foundry.bench.benchmark_run_registry as registry_module


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


def _valid_run_entry(run_id: str = "run_001") -> dict[str, object]:
    return {
        "run_id": run_id,
        "track": "binary_md_v1",
        "experiment": "cls_benchmark_staged_prior",
        "config_profile": "cls_benchmark_staged_prior",
        "budget_class": "short-run",
        "model": {
            "arch": "tabfoundry_staged",
            "stage": "nano_exact",
            "stage_label": "nano_exact",
            "benchmark_profile": "nano_exact",
            "d_icl": 96,
            "tficl_n_heads": 4,
            "tficl_n_layers": 3,
            "head_hidden_dim": 192,
            "input_normalization": "train_zscore_clip",
            "many_class_base": 2,
            "module_selection": {"feature_encoder": "nano"},
            "module_hyperparameters": {"row_pool": {"cls_tokens": 4}},
        },
        "lineage": {
            "parent_run_id": None,
            "anchor_run_id": None,
            "control_baseline_id": "cls_benchmark_linear_v2",
        },
        "manifest_path": "data/manifests/default.parquet",
        "seed_set": [1],
        "benchmark_bundle": {
            "name": "binary_medium",
            "version": 1,
            "source_path": "src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
            "task_count": 1,
            "task_ids": [1],
        },
        "artifacts": {
            "run_dir": "outputs/run_001/prior",
            "benchmark_dir": "outputs/run_001/benchmark",
            "prior_dir": "outputs/run_001/prior",
            "history_path": "outputs/run_001/prior/train_history.jsonl",
            "best_checkpoint_path": "outputs/run_001/prior/checkpoints/best.pt",
            "comparison_summary_path": "outputs/run_001/benchmark/comparison_summary.json",
            "comparison_curve_path": "outputs/run_001/benchmark/comparison_curve.png",
            "benchmark_run_record_path": "outputs/run_001/benchmark/benchmark_run_record.json",
            "training_surface_record_path": "outputs/run_001/benchmark/training_surface_record.json",
        },
        "tab_foundry_metrics": {
            "best_step": 50.0,
            "best_training_time": 100.0,
            "best_roc_auc": 0.81,
            "final_step": 75.0,
            "final_training_time": 120.0,
            "final_roc_auc": 0.8,
        },
        "training_diagnostics": {
            "best_val_loss": 0.2,
            "final_val_loss": 0.25,
            "best_val_step": 50.0,
            "post_warmup_train_loss_var": 0.01,
            "mean_grad_norm": 0.3,
            "max_grad_norm": 0.4,
            "final_grad_norm": 0.35,
            "train_elapsed_seconds": 100.0,
            "wall_elapsed_seconds": 110.0,
        },
        "model_size": {
            "total_params": 10,
            "trainable_params": 10,
        },
        "surface_labels": {
            "model": "nano_exact",
            "data": "manifest",
            "preprocessing": "runtime_default",
        },
        "sweep": {
            "sweep_id": "binary_md_v1",
            "delta_id": "delta_label_token",
            "parent_sweep_id": None,
            "queue_order": 1,
            "run_kind": "primary",
        },
        "comparisons": {
            "vs_parent": None,
            "vs_anchor": None,
        },
        "decision": "defer",
        "conclusion": "Awaiting comparison review.",
        "registered_at_utc": "2026-03-15T00:00:00Z",
    }


@settings(deadline=None, max_examples=35)
@given(rel_path=_REL_PATH)
def test_registry_path_roundtrips_repo_relative_paths(
    rel_path: str,
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        repo_root = (Path(tmp_dir) / "repo").resolve()
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(registry_module, "project_root", lambda: repo_root)

            absolute_path = (repo_root / rel_path).resolve()

            normalized = registry_module._normalize_path_value(absolute_path)

            assert normalized == str(absolute_path.relative_to(repo_root))
            assert registry_module.resolve_registry_path_value(normalized) == absolute_path


@settings(deadline=None, max_examples=35)
@given(rel_path=_REL_PATH)
def test_registry_path_roundtrips_absolute_paths_outside_repo(
    rel_path: str,
) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        repo_root = (tmp_path / "repo").resolve()
        outside_root = tmp_path / "outside"
        with pytest.MonkeyPatch.context() as monkeypatch:
            monkeypatch.setattr(registry_module, "project_root", lambda: repo_root)

            absolute_path = (outside_root / rel_path).resolve()

            normalized = registry_module._normalize_path_value(absolute_path)

            assert normalized == str(absolute_path)
            assert registry_module.resolve_registry_path_value(normalized) == absolute_path


@settings(deadline=None, max_examples=35)
@given(value=st.one_of(st.none(), st.integers(), st.floats(allow_nan=False, allow_infinity=False, width=32)))
def test_optional_finite_number_accepts_finite_numbers_or_none(value: int | float | None) -> None:
    resolved = registry_module._ensure_optional_finite_number(value, context="value")
    if value is None:
        assert resolved is None
    else:
        assert resolved == float(value)


@settings(deadline=None, max_examples=25)
@given(value=st.sampled_from([float("nan"), float("inf"), float("-inf")]))
def test_optional_finite_number_rejects_nan_and_infinity(value: float) -> None:
    with pytest.raises(RuntimeError, match="must be finite when present"):
        _ = registry_module._ensure_optional_finite_number(value, context="value")


@settings(deadline=None, max_examples=25)
@given(
    location=st.sampled_from(
        [
            ("entry", "track"),
            ("entry", "conclusion"),
            ("model", "arch"),
            ("benchmark_bundle", "name"),
            ("artifacts", "run_dir"),
            ("model_size", "total_params"),
        ]
    )
)
def test_validate_run_entry_rejects_missing_required_fields(
    location: tuple[str, str],
) -> None:
    scope, key = location
    entry = _valid_run_entry()
    if scope == "entry":
        entry.pop(key, None)
    else:
        nested = entry[scope]
        assert isinstance(nested, dict)
        nested.pop(key, None)

    with pytest.raises(RuntimeError, match="benchmark run entry run_001 is invalid"):
        registry_module._validate_run_entry(entry, run_id="run_001")


@settings(deadline=None, max_examples=25)
@given(other_run_id=st.text(alphabet=string.ascii_letters + string.digits + "_-", min_size=1, max_size=16))
def test_validate_run_entry_rejects_run_id_mismatch(other_run_id: str) -> None:
    entry = _valid_run_entry()
    if other_run_id == "run_001":
        other_run_id = "run_002"
    entry["run_id"] = other_run_id

    with pytest.raises(RuntimeError, match="benchmark run entry run_id mismatch"):
        registry_module._validate_run_entry(entry, run_id="run_001")
