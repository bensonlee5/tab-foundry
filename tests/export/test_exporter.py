from __future__ import annotations

import json
from pathlib import Path
import typing

from omegaconf import OmegaConf
import pytest
import torch

import tab_foundry.export.exporter as exporter_module
import tab_foundry.training.evaluate as evaluate_module
from tab_foundry.export.checksums import sha256_file
from tab_foundry.export.contracts import SCHEMA_VERSION_V2
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
            "tfcol_n_heads": 8,
            "tfcol_n_layers": 3,
            "tfcol_n_inducing": 128,
            "tfrow_n_heads": 8,
            "tfrow_n_layers": 3,
            "tfrow_cls_tokens": 4,
            "tficl_n_heads": 8,
            "tficl_n_layers": 12,
            "tficl_ff_expansion": 2,
            "many_class_base": 10,
            "head_hidden_dim": 1024,
            "use_digit_position_embed": True,
        },
    }


def _write_checkpoint(path: Path, *, task: str) -> torch.nn.Module:
    cfg = _make_config(task)
    model_cfg = cfg["model"]
    assert isinstance(model_cfg, dict)
    model = build_model(
        task=task,
        **model_cfg,
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
    assert result.schema_version == SCHEMA_VERSION_V2

    manifest_path = out_dir / "manifest.json"
    assert manifest_path.exists()
    assert (out_dir / "weights.safetensors").exists()
    assert (out_dir / "inference_config.json").exists()
    assert (out_dir / "preprocessor_state.json").exists()

    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    assert manifest["schema_version"] == SCHEMA_VERSION_V2
    assert manifest["task"] == "classification"
    assert manifest["model"]["arch"] == "tabfoundry"
    assert manifest["model"]["tfcol_n_heads"] == 8
    assert manifest["model"]["tficl_n_layers"] == 12
    assert manifest["model"]["many_class_base"] == 10
    assert manifest["model"]["head_hidden_dim"] == 1024
    assert manifest["model"]["use_digit_position_embed"] is True

    with (out_dir / "inference_config.json").open("r", encoding="utf-8") as handle:
        inference_cfg = json.load(handle)
    assert inference_cfg["model_arch"] == "tabfoundry"


def test_export_bundle_defaults_feature_group_size_to_one_when_omitted(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "ckpt_default_group.pt"
    cfg = _make_config("classification")
    model_cfg = cfg["model"]
    assert isinstance(model_cfg, dict)
    model_cfg.pop("feature_group_size", None)
    model = build_model(task="classification", **model_cfg)
    torch.save(
        {
            "model": model.state_dict(),
            "global_step": 1,
            "config": cfg,
        },
        checkpoint,
    )

    out_dir = tmp_path / "export_default_group"
    _ = export_checkpoint(checkpoint, out_dir)

    with (out_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with (out_dir / "inference_config.json").open("r", encoding="utf-8") as handle:
        inference_cfg = json.load(handle)

    assert manifest["model"]["feature_group_size"] == 1
    assert inference_cfg["feature_group_size"] == 1


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
    manifest["schema_version"] = "tab-foundry-export-v1"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle)

    with pytest.raises(ValueError, match="Unsupported schema version"):
        _ = validate_export_bundle(out_dir)


def test_validate_export_rejects_old_manifest_arch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    _ = _write_checkpoint(checkpoint, task="classification")
    out_dir = tmp_path / "export_cls"
    _ = export_checkpoint(checkpoint, out_dir)

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["model"]["arch"] = "tabiclv2"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with pytest.raises(ValueError, match="Unsupported model arch"):
        _ = validate_export_bundle(out_dir)


def test_validate_export_rejects_old_inference_model_arch(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    _ = _write_checkpoint(checkpoint, task="classification")
    out_dir = tmp_path / "export_cls"
    _ = export_checkpoint(checkpoint, out_dir)

    inference_path = out_dir / "inference_config.json"
    with inference_path.open("r", encoding="utf-8") as handle:
        inference_cfg = json.load(handle)
    inference_cfg["model_arch"] = "tabiclv2"
    with inference_path.open("w", encoding="utf-8") as handle:
        json.dump(inference_cfg, handle, indent=2, sort_keys=True)
        handle.write("\n")

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["checksums"]["inference_config"] = sha256_file(inference_path)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with pytest.raises(ValueError, match="Unsupported inference model_arch"):
        _ = validate_export_bundle(out_dir)


def test_validate_export_requires_quantile_levels_for_regression(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    _ = _write_checkpoint(checkpoint, task="regression")
    out_dir = tmp_path / "export_reg"
    _ = export_checkpoint(checkpoint, out_dir)

    inference_path = out_dir / "inference_config.json"
    with inference_path.open("r", encoding="utf-8") as handle:
        inference_cfg = json.load(handle)
    inference_cfg.pop("quantile_levels", None)
    with inference_path.open("w", encoding="utf-8") as handle:
        json.dump(inference_cfg, handle, indent=2, sort_keys=True)
        handle.write("\n")

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["checksums"]["inference_config"] = sha256_file(inference_path)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with pytest.raises(ValueError, match="quantile_levels"):
        _ = validate_export_bundle(out_dir)


def test_validate_export_rejects_quantile_levels_for_classification(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    _ = _write_checkpoint(checkpoint, task="classification")
    out_dir = tmp_path / "export_cls"
    _ = export_checkpoint(checkpoint, out_dir)

    inference_path = out_dir / "inference_config.json"
    with inference_path.open("r", encoding="utf-8") as handle:
        inference_cfg = json.load(handle)
    inference_cfg["quantile_levels"] = [0.25, 0.5, 0.75]
    with inference_path.open("w", encoding="utf-8") as handle:
        json.dump(inference_cfg, handle, indent=2, sort_keys=True)
        handle.write("\n")

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["checksums"]["inference_config"] = sha256_file(inference_path)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with pytest.raises(ValueError, match="only valid for regression"):
        _ = validate_export_bundle(out_dir)


def test_export_preprocessor_policy_matches_runtime_filtering(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    _ = _write_checkpoint(checkpoint, task="classification")
    out_dir = tmp_path / "export_cls"
    _ = export_checkpoint(checkpoint, out_dir)

    preproc_path = out_dir / "preprocessor_state.json"
    with preproc_path.open("r", encoding="utf-8") as handle:
        preproc = json.load(handle)
    assert preproc["classification_label_policy"]["mapping"] == "train_only_remap"
    assert preproc["classification_label_policy"]["unseen_test_label"] == "filter"


def test_validate_export_rejects_fixed_inference_contract_drift(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    _ = _write_checkpoint(checkpoint, task="classification")
    out_dir = tmp_path / "export_cls"
    _ = export_checkpoint(checkpoint, out_dir)

    inference_path = out_dir / "inference_config.json"
    with inference_path.open("r", encoding="utf-8") as handle:
        inference_cfg = json.load(handle)
    inference_cfg["group_shifts"] = [99]
    inference_cfg["many_class_threshold"] = 123
    with inference_path.open("w", encoding="utf-8") as handle:
        json.dump(inference_cfg, handle, indent=2, sort_keys=True)
        handle.write("\n")

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["checksums"]["inference_config"] = sha256_file(inference_path)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with pytest.raises(ValueError, match="group_shifts"):
        _ = validate_export_bundle(out_dir)


def test_validate_export_rejects_fixed_preprocessor_contract_drift(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    _ = _write_checkpoint(checkpoint, task="classification")
    out_dir = tmp_path / "export_cls"
    _ = export_checkpoint(checkpoint, out_dir)

    preproc_path = out_dir / "preprocessor_state.json"
    with preproc_path.open("r", encoding="utf-8") as handle:
        preproc = json.load(handle)
    preproc["feature_order_policy"] = "reverse_columns"
    preproc["missing_value_policy"] = {"strategy": "zero_fill", "all_nan_fill": 1.0}
    preproc["dtype_policy"] = {
        "features": "float64",
        "classification_labels": "int32",
        "regression_targets": "float16",
    }
    with preproc_path.open("w", encoding="utf-8") as handle:
        json.dump(preproc, handle, indent=2, sort_keys=True)
        handle.write("\n")

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["checksums"]["preprocessor_state"] = sha256_file(preproc_path)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with pytest.raises(ValueError, match="feature_order_policy"):
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

    assert getattr(loaded.model, "many_class_train_mode") == "path_nll"
    assert loaded.validated.inference_config.many_class_inference_mode == "full_probs"
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


def test_model_config_round_trip_across_eval_export_and_loader(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt_custom.pt"
    cfg = _make_config("classification")
    model_cfg = cfg["model"]
    assert isinstance(model_cfg, dict)
    model_cfg.update(
        {
            "d_col": 64,
            "d_icl": 256,
            "feature_group_size": 1,
            "many_class_train_mode": "full_probs",
            "max_mixed_radix_digits": 32,
            "tfcol_n_layers": 2,
            "tfrow_n_layers": 2,
            "tficl_n_layers": 4,
            "many_class_base": 12,
            "head_hidden_dim": 384,
            "use_digit_position_embed": False,
        }
    )
    model = build_model(task="classification", **model_cfg)
    torch.save({"model": model.state_dict(), "global_step": 1, "config": cfg}, checkpoint)

    out_dir = tmp_path / "export_custom"
    _ = export_checkpoint(checkpoint, out_dir)
    loaded = load_export_bundle(out_dir)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert isinstance(payload, dict)
    fallback_cfg = OmegaConf.create(
        {
            "task": "classification",
            "model": {
                "d_col": 128,
                "d_icl": 512,
                "feature_group_size": 1,
                "many_class_train_mode": "path_nll",
                "max_mixed_radix_digits": 64,
                "tfcol_n_layers": 3,
                "tfrow_n_layers": 3,
                "tficl_n_layers": 12,
                "many_class_base": 10,
                "head_hidden_dim": 1024,
                "use_digit_position_embed": True,
            },
        }
    )
    eval_spec = evaluate_module._checkpoint_model_settings(payload, fallback_cfg)

    assert eval_spec.task == loaded.validated.manifest.task
    assert eval_spec.many_class_train_mode == loaded.validated.manifest.model.many_class_train_mode
    assert eval_spec.tfcol_n_layers == loaded.validated.manifest.model.tfcol_n_layers
    assert eval_spec.tfrow_n_layers == loaded.validated.manifest.model.tfrow_n_layers
    assert eval_spec.tficl_n_layers == loaded.validated.manifest.model.tficl_n_layers
    assert eval_spec.many_class_base == loaded.validated.manifest.model.many_class_base
    assert eval_spec.head_hidden_dim == loaded.validated.manifest.model.head_hidden_dim
    assert eval_spec.use_digit_position_embed == loaded.validated.manifest.model.use_digit_position_embed
    assert getattr(loaded.model, "many_class_train_mode") == eval_spec.many_class_train_mode
    assert getattr(loaded.model, "many_class_base") == eval_spec.many_class_base
    assert getattr(loaded.model, "head_hidden_dim") == eval_spec.head_hidden_dim
    assert bool(getattr(loaded.model, "use_digit_position_embed")) is eval_spec.use_digit_position_embed
