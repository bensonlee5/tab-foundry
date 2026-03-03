from __future__ import annotations

import json
from pathlib import Path
import typing

import pytest
import torch

import tab_foundry.export.exporter as exporter_module
from tab_foundry.export.contracts import SCHEMA_VERSION_V1
from tab_foundry.export.exporter import export_checkpoint, validate_export_bundle
from tab_foundry.export.loader_ref import load_export_bundle
from tab_foundry.model.factory import build_model
from tab_foundry.types import TaskBatch


def _make_config(task: str) -> dict[str, object]:
    return {
        "task": task,
        "model": {
            "d_col": 128,
            "d_icl": 512,
            "feature_group_size": 32,
            "many_class_train_mode": "path_nll",
            "max_mixed_radix_digits": 64,
        },
    }


def _write_checkpoint(path: Path, *, task: str) -> torch.nn.Module:
    cfg = _make_config(task)
    model = build_model(
        task=task,
        d_col=128,
        d_icl=512,
        feature_group_size=32,
        many_class_train_mode="path_nll",
        max_mixed_radix_digits=64,
    )
    payload = {
        "model": model.state_dict(),
        "global_step": 3,
        "config": cfg,
    }
    torch.save(payload, path)
    return model


def test_export_bundle_writes_expected_files_and_schema(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    _ = _write_checkpoint(checkpoint, task="classification")
    out_dir = tmp_path / "export_cls"

    result = export_checkpoint(checkpoint, out_dir)
    assert result.schema_version == SCHEMA_VERSION_V1

    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists()
    assert (out_dir / "weights.safetensors").exists()
    assert (out_dir / "inference_config.json").exists()
    assert (out_dir / "preprocessor_state.json").exists()

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["schema_version"] == SCHEMA_VERSION_V1
    assert manifest["task"] == "classification"
    assert manifest["model"]["arch"] == "tabiclv2"


def test_validate_export_detects_checksum_tamper(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    _ = _write_checkpoint(checkpoint, task="classification")
    out_dir = tmp_path / "export_cls"
    _ = export_checkpoint(checkpoint, out_dir)

    inference_cfg = out_dir / "inference_config.json"
    with inference_cfg.open("a", encoding="utf-8") as handle:
        handle.write("\n")

    with pytest.raises(ValueError, match="checksum mismatch"):
        _ = validate_export_bundle(out_dir)


def test_validate_export_rejects_unsupported_schema(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    _ = _write_checkpoint(checkpoint, task="classification")
    out_dir = tmp_path / "export_cls"
    _ = export_checkpoint(checkpoint, out_dir)

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["schema_version"] = "tab-foundry-export-v2"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    with pytest.raises(ValueError, match="Unsupported schema version"):
        _ = validate_export_bundle(out_dir)


def test_export_checkpoint_uses_explicit_weights_only_false(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    _ = _write_checkpoint(checkpoint, task="classification")
    out_dir = tmp_path / "export_cls"

    captured: list[dict[str, object]] = []
    orig_load = exporter_module.torch.load

    def _recording_load(*args: typing.Any, **kwargs: typing.Any) -> typing.Any:
        captured.append(dict(kwargs))
        return orig_load(*args, **kwargs)

    monkeypatch.setattr(exporter_module.torch, "load", _recording_load)
    _ = export_checkpoint(checkpoint, out_dir)

    assert captured
    assert captured[0]["map_location"] == "cpu"
    assert captured[0]["weights_only"] is False


def test_reference_loader_round_trip_classification_logits(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    source_model = _write_checkpoint(checkpoint, task="classification")
    source_model.eval()
    out_dir = tmp_path / "export_cls"
    _ = export_checkpoint(checkpoint, out_dir)

    loaded = load_export_bundle(out_dir)

    batch = TaskBatch(
        x_train=torch.randn(12, 7),
        y_train=torch.randint(0, 4, (12,)),
        x_test=torch.randn(5, 7),
        y_test=torch.randint(0, 4, (5,)),
        metadata={},
        num_classes=4,
    )
    with torch.no_grad():
        src_out = source_model(batch)
        dst_out = loaded.model(batch)

    assert src_out.logits is not None
    assert dst_out.logits is not None
    assert torch.allclose(src_out.logits, dst_out.logits)


def test_reference_loader_round_trip_regression_quantiles(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt_reg.pt"
    source_model = _write_checkpoint(checkpoint, task="regression")
    source_model.eval()
    out_dir = tmp_path / "export_reg"
    _ = export_checkpoint(checkpoint, out_dir)

    loaded = load_export_bundle(out_dir)
    with (out_dir / "inference_config.json").open("r", encoding="utf-8") as handle:
        inference_cfg = json.load(handle)
    assert "quantile_levels" in inference_cfg
    assert len(inference_cfg["quantile_levels"]) == 999

    batch = TaskBatch(
        x_train=torch.randn(10, 6),
        y_train=torch.randn(10),
        x_test=torch.randn(4, 6),
        y_test=torch.randn(4),
        metadata={},
        num_classes=None,
    )
    with torch.no_grad():
        src_out = source_model(batch)
        dst_out = loaded.model(batch)

    assert torch.allclose(src_out.quantiles, dst_out.quantiles)
