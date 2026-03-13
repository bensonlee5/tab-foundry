from __future__ import annotations

import json
from pathlib import Path
import typing

import numpy as np
from omegaconf import OmegaConf
import pytest
import torch

import tab_foundry.export.exporter as exporter_module
import tab_foundry.training.evaluate as evaluate_module
from tab_foundry.export.contracts import (
    ExportPreprocessorState,
    SCHEMA_VERSION_V2,
    SCHEMA_VERSION_V3,
)
from tab_foundry.export.exporter import export_checkpoint, validate_export_bundle
from tab_foundry.export.loader_ref import load_export_bundle, run_reference_consumer
from tab_foundry.model.factory import build_model
from tab_foundry.types import TaskBatch


def _classification_reference_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(
            [
                [1.0, np.nan, 0.0],
                [3.0, 5.0, 2.0],
                [5.0, 7.0, 4.0],
                [7.0, 9.0, 6.0],
            ],
            dtype=np.float32,
        ),
        np.asarray([10, 20, 10, 20], dtype=np.int64),
        np.asarray(
            [
                [np.nan, 11.0, 8.0],
                [4.0, np.nan, 3.0],
            ],
            dtype=np.float32,
        ),
    )


def _regression_reference_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.asarray(
            [
                [1.0, np.nan],
                [3.0, 4.0],
                [5.0, 8.0],
            ],
            dtype=np.float32,
        ),
        np.asarray([0.25, 0.75, 1.25], dtype=np.float32),
        np.asarray([[np.nan, 6.0]], dtype=np.float32),
    )


def _export_v3_checkpoint(checkpoint: Path, out_dir: Path) -> object:
    return export_checkpoint(checkpoint, out_dir)


def _make_config(
    task: str,
    *,
    input_normalization: str = "none",
    model_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    model_cfg: dict[str, object] = {
        "d_col": 128,
        "d_icl": 512,
        "input_normalization": input_normalization,
        "feature_group_size": 1,
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
    }
    if model_overrides is not None:
        model_cfg.update(model_overrides)
    return {
        "task": task,
        "model": model_cfg,
    }


def _write_checkpoint(
    path: Path,
    *,
    task: str,
    input_normalization: str = "none",
    model_overrides: dict[str, object] | None = None,
    seed: int = 0,
) -> torch.nn.Module:
    cfg = _make_config(
        task,
        input_normalization=input_normalization,
        model_overrides=model_overrides,
    )
    model_cfg = cfg["model"]
    assert isinstance(model_cfg, dict)
    torch.manual_seed(seed)
    model = build_model(
        task=task,
        **model_cfg,
    )
    payload: dict[str, object] = {
        "model": model.state_dict(),
        "global_step": 3,
        "config": cfg,
    }
    torch.save(payload, path)
    return model


def _load_fixture(name: str) -> dict[str, object]:
    fixture = Path(__file__).resolve().parent / "fixtures" / name
    with fixture.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert isinstance(payload, dict)
    return payload


def test_export_bundle_defaults_to_v3_and_embeds_single_manifest(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    _ = _write_checkpoint(
        checkpoint,
        task="classification",
        input_normalization="train_zscore",
    )
    out_dir = tmp_path / "export_cls"

    result = _export_v3_checkpoint(checkpoint, out_dir)
    assert result.schema_version == SCHEMA_VERSION_V3

    with (out_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    assert manifest["schema_version"] == SCHEMA_VERSION_V3
    assert manifest["model"]["arch"] == "tabfoundry"
    assert manifest["model"]["input_normalization"] == "train_zscore"
    assert isinstance(manifest["manifest_sha256"], str)
    assert len(manifest["manifest_sha256"]) == 64
    assert manifest["inference"]["model_arch"] == "tabfoundry"
    assert manifest["inference"]["many_class_inference_mode"] == "full_probs"
    assert manifest["preprocessor"]["feature_order_policy"] == "positional_feature_ids"
    assert manifest["preprocessor"]["classification_label_policy"]["mapping"] == "train_only_remap"
    assert manifest["weights"]["file"] == "weights.safetensors"
    assert (out_dir / "weights.safetensors").exists()
    assert not (out_dir / "inference_config.json").exists()
    assert not (out_dir / "preprocessor_state.json").exists()

    validated = validate_export_bundle(out_dir)
    assert isinstance(validated.preprocessor_state, ExportPreprocessorState)
    assert validated.manifest.model.input_normalization == "train_zscore"


def test_export_bundle_supports_explicit_v2(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    _ = _write_checkpoint(checkpoint, task="classification")
    out_dir = tmp_path / "export_cls_v2"

    result = export_checkpoint(checkpoint, out_dir, artifact_version=SCHEMA_VERSION_V2)
    assert result.schema_version == SCHEMA_VERSION_V2

    with (out_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    with (out_dir / "preprocessor_state.json").open("r", encoding="utf-8") as handle:
        preprocessor_state = json.load(handle)

    assert manifest["schema_version"] == SCHEMA_VERSION_V2
    assert "feature_ids" not in preprocessor_state
    assert preprocessor_state["classification_label_policy"]["mapping"] == "train_only_remap"


def test_export_bundle_defaults_omitted_feature_group_size_to_one_when_weights_match(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "ckpt_default_group.pt"
    cfg = _make_config("classification")
    model_cfg = cfg["model"]
    assert isinstance(model_cfg, dict)
    default_model_cfg = dict(model_cfg)
    model_cfg.pop("feature_group_size", None)
    default_model_cfg.pop("feature_group_size", None)
    torch.manual_seed(0)
    model = build_model(task="classification", feature_group_size=1, **default_model_cfg)
    torch.save(
        {
            "model": model.state_dict(),
            "global_step": 1,
            "config": cfg,
        },
        checkpoint,
    )

    out_dir = tmp_path / "export_default_group"
    _ = _export_v3_checkpoint(checkpoint, out_dir)

    with (out_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    loaded = load_export_bundle(out_dir)

    assert manifest["model"]["feature_group_size"] == 1
    assert manifest["inference"]["feature_group_size"] == 1
    assert loaded.validated.manifest.model.feature_group_size == 1


def test_export_bundle_rejects_legacy_grouped_weights_when_feature_group_size_is_omitted(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "ckpt_legacy_group.pt"
    cfg = _make_config("classification")
    model_cfg = cfg["model"]
    assert isinstance(model_cfg, dict)
    legacy_model_cfg = dict(model_cfg)
    model_cfg.pop("feature_group_size", None)
    legacy_model_cfg.pop("feature_group_size", None)
    torch.manual_seed(0)
    model = build_model(task="classification", feature_group_size=32, **legacy_model_cfg)
    torch.save(
        {
            "model": model.state_dict(),
            "global_step": 1,
            "config": cfg,
        },
        checkpoint,
    )

    with pytest.raises(ValueError, match="omitted feature_group_size"):
        _ = _export_v3_checkpoint(checkpoint, tmp_path / "export_legacy_group")


def test_export_bundle_supports_explicit_nondefault_feature_group_size(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "ckpt_group_32.pt"
    _ = _write_checkpoint(
        checkpoint,
        task="classification",
        model_overrides={"feature_group_size": 32},
    )

    out_dir = tmp_path / "export_group_32"
    _ = _export_v3_checkpoint(checkpoint, out_dir)

    with (out_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    assert manifest["model"]["feature_group_size"] == 32
    assert manifest["inference"]["feature_group_size"] == 32


def test_validate_export_detects_weights_checksum_tamper(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    _ = _write_checkpoint(checkpoint, task="classification")
    out_dir = tmp_path / "export_cls"
    _ = _export_v3_checkpoint(checkpoint, out_dir)

    with (out_dir / "weights.safetensors").open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(ValueError, match="checksum mismatch for weights"):
        _ = validate_export_bundle(out_dir)


def test_validate_export_rejects_unsupported_schema(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    _ = _write_checkpoint(checkpoint, task="classification")
    out_dir = tmp_path / "export_cls"
    _ = _export_v3_checkpoint(checkpoint, out_dir)

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
    _ = _export_v3_checkpoint(checkpoint, out_dir)

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
    _ = _export_v3_checkpoint(checkpoint, out_dir)

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["inference"]["model_arch"] = "tabiclv2"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with pytest.raises(ValueError, match="Unsupported inference model_arch"):
        _ = validate_export_bundle(out_dir)


def test_validate_export_rejects_invalid_input_normalization(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    _ = _write_checkpoint(checkpoint, task="classification")
    out_dir = tmp_path / "export_cls"
    _ = _export_v3_checkpoint(checkpoint, out_dir)

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["model"]["input_normalization"] = "bogus"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with pytest.raises(ValueError, match="input_normalization"):
        _ = validate_export_bundle(out_dir)


def test_validate_export_rejects_manifest_model_tamper_with_stale_manifest_sha256(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    _ = _write_checkpoint(checkpoint, task="classification", input_normalization="train_zscore")
    out_dir = tmp_path / "export_cls"
    _ = _export_v3_checkpoint(checkpoint, out_dir)

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["model"]["input_normalization"] = "none"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with pytest.raises(ValueError, match="manifest.manifest_sha256 mismatch"):
        _ = validate_export_bundle(out_dir)


def test_validate_export_requires_quantile_levels_for_regression(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    _ = _write_checkpoint(checkpoint, task="regression")
    out_dir = tmp_path / "export_reg"
    _ = _export_v3_checkpoint(checkpoint, out_dir)

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["inference"].pop("quantile_levels", None)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with pytest.raises(ValueError, match="quantile_levels"):
        _ = validate_export_bundle(out_dir)


def test_validate_export_rejects_manifest_inference_tamper_with_stale_manifest_sha256(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    _ = _write_checkpoint(checkpoint, task="regression")
    out_dir = tmp_path / "export_reg"
    _ = _export_v3_checkpoint(checkpoint, out_dir)

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    quantile_levels = manifest["inference"]["quantile_levels"]
    assert isinstance(quantile_levels, list)
    manifest["inference"]["quantile_levels"] = list(reversed(quantile_levels))
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with pytest.raises(ValueError, match="manifest.manifest_sha256 mismatch"):
        _ = validate_export_bundle(out_dir)


def test_validate_export_rejects_quantile_levels_for_classification(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    _ = _write_checkpoint(checkpoint, task="classification")
    out_dir = tmp_path / "export_cls"
    _ = _export_v3_checkpoint(checkpoint, out_dir)

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["inference"]["quantile_levels"] = [0.25, 0.5, 0.75]
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with pytest.raises(ValueError, match="only valid for regression"):
        _ = validate_export_bundle(out_dir)


def test_export_manifest_embeds_policy_only_preprocessor(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    _ = _write_checkpoint(checkpoint, task="classification")
    out_dir = tmp_path / "export_cls"
    _ = _export_v3_checkpoint(checkpoint, out_dir)

    with (out_dir / "manifest.json").open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    preproc = manifest["preprocessor"]

    assert preproc["classification_label_policy"]["mapping"] == "train_only_remap"
    assert preproc["classification_label_policy"]["unseen_test_label"] == "filter"
    assert "feature_ids" not in preproc
    assert "fill_values" not in preproc["missing_value_policy"]
    assert "label_values" not in preproc["classification_label_policy"]


def test_validate_export_rejects_fixed_inference_contract_drift(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    _ = _write_checkpoint(checkpoint, task="classification")
    out_dir = tmp_path / "export_cls"
    _ = _export_v3_checkpoint(checkpoint, out_dir)

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["inference"]["group_shifts"] = [99]
    manifest["inference"]["many_class_threshold"] = 123
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with pytest.raises(ValueError, match="group_shifts"):
        _ = validate_export_bundle(out_dir)


def test_validate_export_rejects_fixed_preprocessor_contract_drift(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    _ = _write_checkpoint(checkpoint, task="classification")
    out_dir = tmp_path / "export_cls"
    _ = _export_v3_checkpoint(checkpoint, out_dir)

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["preprocessor"]["feature_order_policy"] = "reverse_columns"
    manifest["preprocessor"]["dtype_policy"] = {
        "features": "float64",
        "classification_labels": "int32",
        "regression_targets": "float16",
    }
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with pytest.raises(ValueError, match="feature_order_policy"):
        _ = validate_export_bundle(out_dir)


def test_validate_export_rejects_manifest_preprocessor_tamper_with_stale_manifest_sha256(
    tmp_path: Path,
) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    _ = _write_checkpoint(checkpoint, task="classification")
    out_dir = tmp_path / "export_cls"
    _ = _export_v3_checkpoint(checkpoint, out_dir)

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["preprocessor"]["missing_value_policy"]["all_nan_fill"] = 0
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with pytest.raises(ValueError, match="manifest.manifest_sha256 mismatch"):
        _ = validate_export_bundle(out_dir)


def test_validate_export_requires_manifest_sha256_for_v3(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    _ = _write_checkpoint(checkpoint, task="classification")
    out_dir = tmp_path / "export_cls"
    _ = _export_v3_checkpoint(checkpoint, out_dir)

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest.pop("manifest_sha256", None)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with pytest.raises(ValueError, match="older v3 bundles must be regenerated"):
        _ = validate_export_bundle(out_dir)


def test_validate_export_rejects_malformed_manifest_sha256_for_v3(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    _ = _write_checkpoint(checkpoint, task="classification")
    out_dir = tmp_path / "export_cls"
    _ = _export_v3_checkpoint(checkpoint, out_dir)

    manifest_path = out_dir / "manifest.json"
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    manifest["manifest_sha256"] = "bad"
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")

    with pytest.raises(ValueError, match="manifest.manifest_sha256"):
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
    _ = _export_v3_checkpoint(checkpoint, out_dir)

    assert captured
    assert captured[0]["map_location"] == "cpu"
    assert captured[0]["weights_only"] is False


def test_reference_loader_round_trip_classification_logits(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt.pt"
    source_model = _write_checkpoint(checkpoint, task="classification")
    source_model.eval()
    out_dir = tmp_path / "export_cls"
    _ = _export_v3_checkpoint(checkpoint, out_dir)

    loaded = load_export_bundle(out_dir)

    torch.manual_seed(123)
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
    _ = _export_v3_checkpoint(checkpoint, out_dir)

    loaded = load_export_bundle(out_dir)
    assert loaded.validated.inference_config.quantile_levels is not None
    assert len(loaded.validated.inference_config.quantile_levels) == 999

    torch.manual_seed(456)
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
    cfg = _make_config(
        "classification",
        input_normalization="train_zscore_clip",
        model_overrides={
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
        },
    )
    model_cfg = cfg["model"]
    assert isinstance(model_cfg, dict)
    torch.manual_seed(0)
    model = build_model(task="classification", **model_cfg)
    torch.save(
        {
            "model": model.state_dict(),
            "global_step": 1,
            "config": cfg,
        },
        checkpoint,
    )

    out_dir = tmp_path / "export_custom"
    _ = _export_v3_checkpoint(checkpoint, out_dir)
    loaded = load_export_bundle(out_dir)

    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert isinstance(payload, dict)
    fallback_cfg = OmegaConf.create(
        {
            "task": "classification",
            "model": {
                "d_col": 128,
                "d_icl": 512,
                "input_normalization": "none",
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
    assert eval_spec.input_normalization == loaded.validated.manifest.model.input_normalization
    assert eval_spec.many_class_train_mode == loaded.validated.manifest.model.many_class_train_mode
    assert eval_spec.tfcol_n_layers == loaded.validated.manifest.model.tfcol_n_layers
    assert eval_spec.tfrow_n_layers == loaded.validated.manifest.model.tfrow_n_layers
    assert eval_spec.tficl_n_layers == loaded.validated.manifest.model.tficl_n_layers
    assert eval_spec.many_class_base == loaded.validated.manifest.model.many_class_base
    assert eval_spec.head_hidden_dim == loaded.validated.manifest.model.head_hidden_dim
    assert eval_spec.use_digit_position_embed == loaded.validated.manifest.model.use_digit_position_embed
    assert getattr(loaded.model, "input_normalization") == eval_spec.input_normalization
    assert getattr(loaded.model, "many_class_train_mode") == eval_spec.many_class_train_mode
    assert getattr(loaded.model, "many_class_base") == eval_spec.many_class_base
    assert getattr(loaded.model, "head_hidden_dim") == eval_spec.head_hidden_dim
    assert bool(getattr(loaded.model, "use_digit_position_embed")) is eval_spec.use_digit_position_embed


def test_reference_consumer_classification_matches_golden_fixture(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt_reference_cls.pt"
    _ = _write_checkpoint(
        checkpoint,
        task="classification",
        input_normalization="train_zscore",
        model_overrides={
            "d_col": 16,
            "d_icl": 32,
            "tfcol_n_heads": 4,
            "tfcol_n_layers": 1,
            "tfcol_n_inducing": 8,
            "tfrow_n_heads": 4,
            "tfrow_n_layers": 1,
            "tfrow_cls_tokens": 2,
            "tficl_n_heads": 4,
            "tficl_n_layers": 2,
            "tficl_ff_expansion": 2,
            "head_hidden_dim": 32,
        },
        seed=1234,
    )
    out_dir = tmp_path / "export_reference_cls"
    _ = _export_v3_checkpoint(checkpoint, out_dir)

    fixture = _load_fixture("reference_consumer_classification_v3.json")
    output = run_reference_consumer(
        out_dir,
        x_train=fixture["x_train"],
        y_train=fixture["y_train"],
        x_test=fixture["x_test"],
    )

    assert output.class_probs is not None
    np.testing.assert_allclose(
        output.class_probs,
        np.asarray(fixture["expected_class_probs"], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )


def test_reference_consumer_regression_matches_golden_fixture(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt_reference_reg.pt"
    _ = _write_checkpoint(
        checkpoint,
        task="regression",
        input_normalization="train_zscore_clip",
        model_overrides={
            "d_col": 16,
            "d_icl": 32,
            "tfcol_n_heads": 4,
            "tfcol_n_layers": 1,
            "tfcol_n_inducing": 8,
            "tfrow_n_heads": 4,
            "tfrow_n_layers": 1,
            "tfrow_cls_tokens": 2,
            "tficl_n_heads": 4,
            "tficl_n_layers": 2,
            "tficl_ff_expansion": 2,
            "head_hidden_dim": 32,
        },
        seed=5678,
    )
    out_dir = tmp_path / "export_reference_reg"
    _ = _export_v3_checkpoint(checkpoint, out_dir)

    fixture = _load_fixture("reference_consumer_regression_v3.json")
    output = run_reference_consumer(
        out_dir,
        x_train=fixture["x_train"],
        y_train=fixture["y_train"],
        x_test=fixture["x_test"],
    )

    assert output.quantiles is not None
    assert output.quantile_levels is not None
    np.testing.assert_allclose(
        output.quantiles[:, fixture["expected_quantile_indices"]],
        np.asarray([fixture["expected_quantiles_at_indices"]], dtype=np.float32),
        rtol=1e-6,
        atol=1e-6,
    )
    np.testing.assert_allclose(
        output.quantile_levels[fixture["expected_quantile_indices"]],
        np.asarray(fixture["expected_quantile_levels_at_indices"], dtype=np.float32),
        rtol=1e-7,
        atol=1e-7,
    )


def test_reference_consumer_rejects_v2_bundle(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt_v2.pt"
    _ = _write_checkpoint(checkpoint, task="classification")
    out_dir = tmp_path / "export_v2"
    _ = export_checkpoint(checkpoint, out_dir, artifact_version=SCHEMA_VERSION_V2)
    x_train, y_train, x_test = _classification_reference_arrays()

    with pytest.raises(ValueError, match="only executes tab-foundry-export-v3 bundles"):
        _ = run_reference_consumer(out_dir, x_train=x_train, y_train=y_train, x_test=x_test)


def test_reference_consumer_derives_preprocessing_from_runtime_support_set(tmp_path: Path) -> None:
    checkpoint = tmp_path / "ckpt_runtime_preproc.pt"
    _ = _write_checkpoint(checkpoint, task="classification", input_normalization="none", seed=7)
    out_dir = tmp_path / "export_runtime_preproc"
    _ = _export_v3_checkpoint(checkpoint, out_dir)

    output = run_reference_consumer(
        out_dir,
        x_train=np.asarray(
            [
                [1.0, np.nan],
                [3.0, 5.0],
                [5.0, 7.0],
            ],
            dtype=np.float32,
        ),
        y_train=np.asarray([100, 200, 100], dtype=np.int64),
        x_test=np.asarray([[np.nan, 11.0]], dtype=np.float32),
    )

    assert output.batch.y_train.tolist() == [0, 1, 0]
    assert output.batch.num_classes == 2
    assert torch.allclose(
        output.batch.x_train,
        torch.tensor([[1.0, 6.0], [3.0, 5.0], [5.0, 7.0]], dtype=torch.float32),
    )
    assert torch.allclose(
        output.batch.x_test,
        torch.tensor([[3.0, 11.0]], dtype=torch.float32),
    )
