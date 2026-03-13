from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf
import pytest

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
                    "feature_group_size": 32,
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
                "feature_group_size": 32,
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
                    "d_col": 64,
                    "d_icl": 256,
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
                "feature_group_size": 32,
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
    assert captured["d_col"] == 64
    assert captured["d_icl"] == 256
    assert captured["feature_group_size"] == 1


def test_checkpoint_model_settings_defaults_feature_group_size_to_one() -> None:
    payload = {
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
                "many_class_train_mode": "path_nll",
                "max_mixed_radix_digits": 64,
            },
        }
    )

    spec = evaluate_module._checkpoint_model_settings(payload, cfg)

    assert spec.feature_group_size == 1
