from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
import pytest
import torch

import tab_foundry.training.evaluate as evaluate_module


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
                "model": {
                    "d_col": 128,
                    "d_icl": 512,
                    "feature_group_size": 1,
                    "many_class_train_mode": "path_nll",
                    "max_mixed_radix_digits": 64,
                },
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
            "model": {
                "d_col": 128,
                "d_icl": 512,
                "feature_group_size": 1,
                "many_class_train_mode": "path_nll",
                "max_mixed_radix_digits": 64,
            },
            "data": {"manifest_path": "unused.parquet", "train_row_cap": None, "test_row_cap": None},
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
                "model": {
                    "arch": "tabfoundry_simple",
                    "d_col": 64,
                    "d_icl": 256,
                    "input_normalization": "train_zscore",
                    "feature_group_size": 1,
                    "many_class_train_mode": "path_nll",
                    "max_mixed_radix_digits": 32,
                },
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
            "model": {
                "d_col": 128,
                "d_icl": 512,
                "feature_group_size": 1,
                "many_class_train_mode": "full_probs",
                "max_mixed_radix_digits": 64,
            },
            "data": {"manifest_path": "unused.parquet", "train_row_cap": None, "test_row_cap": None},
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
                "model": {
                    "d_col": 128,
                    "d_icl": 512,
                    "feature_group_size": 1,
                    "many_class_train_mode": "path_nll",
                    "max_mixed_radix_digits": 64,
                },
                "preprocessing": {
                    "surface_label": "runtime_no_impute",
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
            "model": {
                "d_col": 128,
                "d_icl": 512,
                "feature_group_size": 1,
                "many_class_train_mode": "path_nll",
                "max_mixed_radix_digits": 64,
            },
            "data": {"manifest_path": "unused.parquet", "train_row_cap": None, "test_row_cap": None},
            "preprocessing": {
                "surface_label": "runtime_default",
                "overrides": {"impute_missing": True},
            },
            "runtime": {"seed": 1, "num_workers": 0, "device": "cpu", "mixed_precision": "no"},
        }
    )

    with pytest.raises(RuntimeError, match="stop_after_dataset"):
        _ = evaluate_module.evaluate_checkpoint(cfg)

    preprocessing_cfg = captured["preprocessing_cfg"]
    assert OmegaConf.to_container(preprocessing_cfg, resolve=True) == {
        "surface_label": "runtime_no_impute",
        "overrides": {"impute_missing": False},
    }


def test_evaluate_checkpoint_prefers_checkpoint_data_config(
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
                "model": {
                    "d_col": 128,
                    "d_icl": 512,
                    "feature_group_size": 1,
                    "many_class_train_mode": "path_nll",
                    "max_mixed_radix_digits": 64,
                },
                "data": {
                    "manifest_path": "checkpoint_manifest.parquet",
                    "train_row_cap": 16,
                    "test_row_cap": 8,
                },
            },
        }

    def _capture_dataset(
        data_cfg: object,
        *_args: object,
        preprocessing_cfg: object = None,
        **_kwargs: object,
    ) -> None:
        del preprocessing_cfg
        captured["data_cfg"] = data_cfg
        raise RuntimeError("stop_after_dataset")

    monkeypatch.setattr(evaluate_module.torch, "load", _fake_load)
    monkeypatch.setattr(evaluate_module, "build_model_from_spec", lambda _spec: _DummyModel())
    monkeypatch.setattr(evaluate_module, "build_task_dataset", _capture_dataset)

    cfg = OmegaConf.create(
        {
            "eval": {"checkpoint": str(tmp_path / "dummy.pt"), "split": "val", "max_batches": 1},
            "task": "classification",
            "model": {
                "d_col": 128,
                "d_icl": 512,
                "feature_group_size": 1,
                "many_class_train_mode": "path_nll",
                "max_mixed_radix_digits": 64,
            },
            "data": {
                "manifest_path": "runtime_manifest.parquet",
                "train_row_cap": 64,
                "test_row_cap": 32,
            },
            "runtime": {"seed": 1, "num_workers": 0, "device": "cpu", "mixed_precision": "no"},
        }
    )

    with pytest.raises(RuntimeError, match="stop_after_dataset"):
        _ = evaluate_module.evaluate_checkpoint(cfg)

    data_cfg = captured["data_cfg"]
    assert OmegaConf.to_container(data_cfg, resolve=True) == {
        "manifest_path": "checkpoint_manifest.parquet",
        "train_row_cap": 16,
        "test_row_cap": 8,
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
                "model": {
                    "d_col": 128,
                    "d_icl": 512,
                    "feature_group_size": 1,
                    "many_class_train_mode": "path_nll",
                    "max_mixed_radix_digits": 64,
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
            "model": {
                "d_col": 128,
                "d_icl": 512,
                "feature_group_size": 1,
                "many_class_train_mode": "path_nll",
                "max_mixed_radix_digits": 64,
            },
            "data": {"manifest_path": "unused.parquet", "train_row_cap": None, "test_row_cap": None},
            "preprocessing": {
                "surface_label": "runtime_all_nan_fill_one",
                "overrides": {"all_nan_fill": 1.0},
            },
            "runtime": {"seed": 1, "num_workers": 0, "device": "cpu", "mixed_precision": "no"},
        }
    )

    with pytest.raises(RuntimeError, match="stop_after_dataset"):
        _ = evaluate_module.evaluate_checkpoint(cfg)

    preprocessing_cfg = captured["preprocessing_cfg"]
    assert OmegaConf.to_container(preprocessing_cfg, resolve=True) == {
        "surface_label": "runtime_all_nan_fill_one",
        "overrides": {"all_nan_fill": 1.0},
    }


def test_evaluate_checkpoint_prefers_checkpoint_runtime_seed_for_dataset_and_loader(
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
                "model": {
                    "d_col": 128,
                    "d_icl": 512,
                    "feature_group_size": 1,
                    "many_class_train_mode": "path_nll",
                    "max_mixed_radix_digits": 64,
                },
                "runtime": {"seed": 77},
            },
        }

    def _capture_dataset(*_args: object, seed: int, **_kwargs: object) -> object:
        captured["dataset_seed"] = seed
        return object()

    def _capture_loader(
        _dataset: object,
        *,
        shuffle: bool,
        num_workers: int,
        seed: int,
    ) -> None:
        captured["shuffle"] = shuffle
        captured["num_workers"] = num_workers
        captured["loader_seed"] = seed
        raise RuntimeError("stop_after_loader")

    monkeypatch.setattr(evaluate_module.torch, "load", _fake_load)
    monkeypatch.setattr(evaluate_module, "build_model_from_spec", lambda _spec: _DummyModel())
    monkeypatch.setattr(evaluate_module, "build_task_dataset", _capture_dataset)
    monkeypatch.setattr(evaluate_module, "build_task_loader", _capture_loader)

    cfg = OmegaConf.create(
        {
            "eval": {"checkpoint": str(tmp_path / "dummy.pt"), "split": "val", "max_batches": 1},
            "task": "classification",
            "model": {
                "d_col": 128,
                "d_icl": 512,
                "feature_group_size": 1,
                "many_class_train_mode": "path_nll",
                "max_mixed_radix_digits": 64,
            },
            "data": {"manifest_path": "unused.parquet", "train_row_cap": None, "test_row_cap": None},
            "runtime": {"seed": 1, "num_workers": 3, "device": "cpu", "mixed_precision": "no"},
        }
    )

    with pytest.raises(RuntimeError, match="stop_after_loader"):
        _ = evaluate_module.evaluate_checkpoint(cfg)

    assert captured["dataset_seed"] == 77
    assert captured["loader_seed"] == 77
    assert captured["shuffle"] is False
    assert captured["num_workers"] == 3


def test_checkpoint_model_settings_rejects_legacy_grouped_weights_without_override() -> None:
    payload = {
        "model": {
            "group_linear.weight": torch.zeros((128, 96)),
        },
        "config": {
            "task": "classification",
            "model": {
                "d_col": 128,
                "d_icl": 512,
                "many_class_train_mode": "path_nll",
                "max_mixed_radix_digits": 64,
            },
        }
    }
    cfg = OmegaConf.create(
        {
            "task": "classification",
            "model": {
                "d_col": 128,
                "d_icl": 512,
                "feature_group_size": 1,
                "many_class_train_mode": "path_nll",
                "max_mixed_radix_digits": 64,
            },
        }
    )

    with pytest.raises(ValueError, match="Resolved feature_group_size=1 is incompatible"):
        _ = evaluate_module._checkpoint_model_settings(payload, cfg)


def test_checkpoint_model_settings_supports_explicit_override_for_legacy_weights() -> None:
    payload = {
        "model": {
            "group_linear.weight": torch.zeros((128, 96)),
        },
        "config": {
            "task": "classification",
            "model": {
                "d_col": 128,
                "d_icl": 512,
                "many_class_train_mode": "path_nll",
                "max_mixed_radix_digits": 64,
            },
        },
    }
    cfg = OmegaConf.create(
        {
            "task": "classification",
            "model": {
                "d_col": 128,
                "d_icl": 512,
                "feature_group_size": 32,
                "many_class_train_mode": "path_nll",
                "max_mixed_radix_digits": 64,
            },
        }
    )

    spec = evaluate_module._checkpoint_model_settings(payload, cfg)

    assert spec.feature_group_size == 32
