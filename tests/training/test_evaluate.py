from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

import tab_foundry.training.evaluate as evaluate_module
from tab_foundry.model.spec import model_build_spec_from_mappings


def _checkpoint_model_cfg(*, task: str, **overrides: object) -> dict[str, object]:
    return model_build_spec_from_mappings(task=task, primary=overrides).to_dict()


def _runtime_data_cfg(*, allow_missing_values: bool) -> dict[str, object]:
    return {
        "source": "manifest",
        "manifest_path": "unused.parquet",
        "surface_label": "runtime_manifest",
        "allow_missing_values": allow_missing_values,
        "train_row_cap": None,
        "test_row_cap": None,
    }


def _runtime_preprocessing_cfg(
    *,
    surface_label: str = "runtime_default",
    impute_missing: bool = True,
    all_nan_fill: float = 0.0,
    overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    cfg: dict[str, object] = {
        "surface_label": surface_label,
        "impute_missing": impute_missing,
        "all_nan_fill": all_nan_fill,
        "label_mapping": "train_only_remap",
        "unseen_test_label_policy": "filter",
    }
    if overrides is not None:
        cfg["overrides"] = overrides
    return cfg


def test_evaluate_checkpoint_uses_explicit_weights_only_false(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_load(_path: Path, **kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        return {
            "model": {},
            "config": {
                "task": "classification",
                "model": _checkpoint_model_cfg(task="classification"),
            },
        }

    def _stop_after_load(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("stop_after_load")

    monkeypatch.setattr(evaluate_module.torch, "load", _fake_load)
    monkeypatch.setattr(evaluate_module, "build_model_from_spec", _stop_after_load)

    cfg = OmegaConf.create(
        {
            "eval": {"checkpoint": str(tmp_path / "dummy.pt"), "split": "val", "max_batches": 1},
            "task": "classification",
            "data": _runtime_data_cfg(allow_missing_values=False),
            "preprocessing": _runtime_preprocessing_cfg(),
            "runtime": {"seed": 1, "num_workers": 0, "device": "cpu", "mixed_precision": "no"},
        }
    )

    with pytest.raises(RuntimeError, match="stop_after_load"):
        _ = evaluate_module.evaluate_checkpoint(cfg)

    assert captured["map_location"] == "cpu"
    assert captured["weights_only"] is False


def test_evaluate_checkpoint_uses_checkpoint_model_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_load(_path: Path, **_kwargs: object) -> dict[str, object]:
        return {
            "model": {},
            "config": {
                "task": "regression",
                "model": _checkpoint_model_cfg(
                    task="regression",
                    arch="tabfoundry_simple",
                    d_col=64,
                    d_icl=256,
                    input_normalization="train_zscore",
                    feature_group_size=1,
                    many_class_train_mode="path_nll",
                    max_mixed_radix_digits=32,
                ),
            },
        }

    def _capture_build_model(spec: object) -> None:
        to_dict = getattr(spec, "to_dict", None)
        if callable(to_dict):
            payload = to_dict()
            if isinstance(payload, dict):
                captured.update(payload)
        raise RuntimeError("stop_after_build")

    monkeypatch.setattr(evaluate_module.torch, "load", _fake_load)
    monkeypatch.setattr(evaluate_module, "build_model_from_spec", _capture_build_model)

    cfg = OmegaConf.create(
        {
            "eval": {"checkpoint": str(tmp_path / "dummy.pt"), "split": "val", "max_batches": 1},
            "task": "classification",
            "data": _runtime_data_cfg(allow_missing_values=False),
            "preprocessing": _runtime_preprocessing_cfg(),
            "runtime": {"seed": 1, "num_workers": 0, "device": "cpu", "mixed_precision": "no"},
        }
    )

    with pytest.raises(RuntimeError, match="stop_after_build"):
        _ = evaluate_module.evaluate_checkpoint(cfg)

    assert captured["task"] == "regression"
    assert captured["arch"] == "tabfoundry_simple"
    assert captured["d_col"] == 64
    assert captured["d_icl"] == 256
    assert captured["input_normalization"] == "train_zscore"
    assert captured["feature_group_size"] == 1


def test_evaluate_checkpoint_prefers_checkpoint_preprocessing_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class _DummyModel:
        def load_state_dict(self, _state: object) -> None:
            return None

    def _fake_load(_path: Path, **_kwargs: object) -> dict[str, object]:
        return {
            "model": {},
            "config": {
                "task": "classification",
                "model": _checkpoint_model_cfg(task="classification"),
                "preprocessing": {
                    "surface_label": "runtime_no_impute",
                    "impute_missing": True,
                    "all_nan_fill": 0.0,
                    "label_mapping": "train_only_remap",
                    "unseen_test_label_policy": "filter",
                    "overrides": {"impute_missing": False},
                },
            },
        }

    def _capture_dataset(*_args: object, preprocessing_cfg: object = None, **_kwargs: object) -> None:
        captured["preprocessing_cfg"] = preprocessing_cfg
        raise RuntimeError("stop_after_dataset")

    monkeypatch.setattr(evaluate_module.torch, "load", _fake_load)
    monkeypatch.setattr(evaluate_module, "build_model_from_spec", lambda _spec: _DummyModel())
    monkeypatch.setattr(evaluate_module, "build_task_dataset", _capture_dataset)

    cfg = OmegaConf.create(
        {
            "eval": {"checkpoint": str(tmp_path / "dummy.pt"), "split": "val", "max_batches": 1},
            "task": "classification",
            "data": _runtime_data_cfg(allow_missing_values=False),
            "preprocessing": _runtime_preprocessing_cfg(
                overrides={"impute_missing": True},
            ),
            "runtime": {"seed": 1, "num_workers": 0, "device": "cpu", "mixed_precision": "no"},
        }
    )

    with pytest.raises(RuntimeError, match="stop_after_dataset"):
        _ = evaluate_module.evaluate_checkpoint(cfg)

    preprocessing_cfg = captured["preprocessing_cfg"]
    assert OmegaConf.to_container(preprocessing_cfg, resolve=True) == {
        "surface_label": "runtime_no_impute",
        "impute_missing": True,
        "all_nan_fill": 0.0,
        "label_mapping": "train_only_remap",
        "unseen_test_label_policy": "filter",
        "overrides": {"impute_missing": False},
    }


def test_evaluate_checkpoint_falls_back_to_runtime_preprocessing_config(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    class _DummyModel:
        def load_state_dict(self, _state: object) -> None:
            return None

    def _fake_load(_path: Path, **_kwargs: object) -> dict[str, object]:
        return {
            "model": {},
            "config": {
                "task": "classification",
                "model": _checkpoint_model_cfg(task="classification"),
            },
        }

    def _capture_dataset(*_args: object, preprocessing_cfg: object = None, **_kwargs: object) -> None:
        captured["preprocessing_cfg"] = preprocessing_cfg
        raise RuntimeError("stop_after_dataset")

    monkeypatch.setattr(evaluate_module.torch, "load", _fake_load)
    monkeypatch.setattr(evaluate_module, "build_model_from_spec", lambda _spec: _DummyModel())
    monkeypatch.setattr(evaluate_module, "build_task_dataset", _capture_dataset)

    cfg = OmegaConf.create(
        {
            "eval": {"checkpoint": str(tmp_path / "dummy.pt"), "split": "val", "max_batches": 1},
            "task": "classification",
            "data": _runtime_data_cfg(allow_missing_values=False),
            "preprocessing": _runtime_preprocessing_cfg(
                surface_label="runtime_all_nan_fill_one",
                overrides={"all_nan_fill": 1.0},
            ),
            "runtime": {"seed": 1, "num_workers": 0, "device": "cpu", "mixed_precision": "no"},
        }
    )

    with pytest.raises(RuntimeError, match="stop_after_dataset"):
        _ = evaluate_module.evaluate_checkpoint(cfg)

    preprocessing_cfg = captured["preprocessing_cfg"]
    assert OmegaConf.to_container(preprocessing_cfg, resolve=True) == {
        "surface_label": "runtime_all_nan_fill_one",
        "impute_missing": True,
        "all_nan_fill": 0.0,
        "label_mapping": "train_only_remap",
        "unseen_test_label_policy": "filter",
        "overrides": {"all_nan_fill": 1.0},
    }


def test_checkpoint_model_settings_rejects_legacy_grouped_weights_without_override() -> None:
    checkpoint_model_cfg = _checkpoint_model_cfg(task="classification", missingness_mode="none")
    checkpoint_model_cfg.pop("feature_group_size")
    payload = {
        "model": {
            "group_linear.weight": torch.zeros((128, 96)),
        },
        "config": {
            "task": "classification",
            "model": checkpoint_model_cfg,
        }
    }
    cfg = OmegaConf.create({"eval": {"model_overrides": None}})

    with pytest.raises(ValueError, match="ambiguous across multiple tabfoundry layouts"):
        _ = evaluate_module._checkpoint_model_settings(payload, cfg)


def test_checkpoint_model_settings_supports_explicit_override_for_legacy_weights() -> None:
    checkpoint_model_cfg = _checkpoint_model_cfg(task="classification", missingness_mode="none")
    checkpoint_model_cfg.pop("feature_group_size")
    payload = {
        "model": {
            "group_linear.weight": torch.zeros((128, 96)),
        },
        "config": {
            "task": "classification",
            "model": checkpoint_model_cfg,
        },
    }
    cfg = OmegaConf.create(
        {
            "eval": {
                "model_overrides": {
                    "feature_group_size": 32,
                }
            },
        }
    )

    spec = evaluate_module._checkpoint_model_settings(payload, cfg)

    assert spec.feature_group_size == 32


def test_evaluate_checkpoint_rejects_missingness_mode_without_allow_missing_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fake_load(_path: Path, **_kwargs: object) -> dict[str, object]:
        return {
            "model": {"group_linear.weight": torch.zeros((128, 6))},
            "config": {
                "task": "classification",
                "model": _checkpoint_model_cfg(
                    task="classification",
                    missingness_mode="feature_mask",
                    feature_group_size=1,
                ),
                "preprocessing": _runtime_preprocessing_cfg(
                    impute_missing=False,
                    overrides={"impute_missing": False},
                ),
            },
        }

    monkeypatch.setattr(evaluate_module.torch, "load", _fake_load)

    cfg = OmegaConf.create(
        {
            "eval": {"checkpoint": str(tmp_path / "dummy.pt"), "split": "val", "max_batches": 1},
            "task": "classification",
            "data": _runtime_data_cfg(allow_missing_values=False),
            "preprocessing": _runtime_preprocessing_cfg(
                impute_missing=False,
                overrides={"impute_missing": False},
            ),
            "runtime": {"seed": 1, "num_workers": 0, "device": "cpu", "mixed_precision": "no"},
        }
    )

    with pytest.raises(ValueError, match="allow_missing_values"):
        _ = evaluate_module.evaluate_checkpoint(cfg)


def test_evaluate_checkpoint_rejects_missingness_mode_with_imputation_enabled(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fake_load(_path: Path, **_kwargs: object) -> dict[str, object]:
        return {
            "model": {"group_linear.weight": torch.zeros((128, 6))},
            "config": {
                "task": "classification",
                "model": _checkpoint_model_cfg(
                    task="classification",
                    missingness_mode="feature_mask",
                    feature_group_size=1,
                ),
            },
        }

    monkeypatch.setattr(evaluate_module.torch, "load", _fake_load)

    cfg = OmegaConf.create(
        {
            "eval": {"checkpoint": str(tmp_path / "dummy.pt"), "split": "val", "max_batches": 1},
            "task": "classification",
            "data": _runtime_data_cfg(allow_missing_values=True),
            "preprocessing": _runtime_preprocessing_cfg(
                overrides={"impute_missing": True},
            ),
            "runtime": {"seed": 1, "num_workers": 0, "device": "cpu", "mixed_precision": "no"},
        }
    )

    with pytest.raises(ValueError, match="impute_missing"):
        _ = evaluate_module.evaluate_checkpoint(cfg)
