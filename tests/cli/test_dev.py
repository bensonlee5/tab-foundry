from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import tab_foundry.cli.dev as dev_module
import tab_foundry.device as device_module
from tab_foundry.cli.dev import diff_config_payloads, forward_check, resolve_config_payload
from tab_foundry.export.contracts import SCHEMA_VERSION_V2, SCHEMA_VERSION_V3
from tab_foundry.export.inspection import export_check

from tests.export import exporter_cases


_SMALL_STAGED_OVERRIDES = [
    "experiment=cls_smoke",
    "logging.use_wandb=false",
    "model.stage=row_cls_pool",
    "model.stage_label=row_cls_pool_dev_test",
    "model.d_icl=32",
    "model.many_class_base=4",
    "model.tficl_n_heads=4",
    "model.tficl_n_layers=1",
    "model.head_hidden_dim=64",
    "model.tfrow_n_heads=2",
    "model.tfrow_n_layers=1",
    "model.tfrow_cls_tokens=2",
    "model.tfcol_n_heads=2",
    "model.tfcol_n_layers=1",
    "model.tfcol_n_inducing=8",
]


def test_resolve_config_payload_reports_resolved_surfaces() -> None:
    payload = resolve_config_payload(_SMALL_STAGED_OVERRIDES)

    assert payload["experiment"] == "cls_smoke"
    assert payload["task"] == "classification"
    assert payload["model"]["stage_label"] == "row_cls_pool_dev_test"
    assert payload["model"]["module_selection"]["row_pool"] == "row_cls"
    assert payload["training"]["surface_label"] == "training_default"
    assert payload["model"]["parameter_counts"]["total_params"] > 0


def test_resolve_config_payload_keeps_unmaterialized_corpus_ref_for_inspection() -> None:
    payload = resolve_config_payload(
        [
            *_SMALL_STAGED_OVERRIDES,
            "data.surface_overrides.corpus_ref=tf_rd_013_current_corpus_default_v1",
        ]
    )

    assert payload["data"]["source"] == "manifest"
    assert payload["data"]["corpus_ref"] == "tf_rd_013_current_corpus_default_v1"
    assert payload["data"]["recipe_id"] == "tf_rd_013_current_corpus_default_v1"
    assert payload["data"]["corpus_id"] is None
    assert payload["data"]["corpus_record_path"] is None
    assert payload["data"]["manifest_path"] is None


def test_forward_check_passes_for_direct_head_surface() -> None:
    payload = forward_check(
        _SMALL_STAGED_OVERRIDES,
        requested_device="cpu",
        seed=7,
    )

    assert payload["output_kind"] == "logits"
    assert payload["surface_label"] == "row_cls_pool_dev_test"
    assert payload["output_shape"][0] == 2
    assert payload["batched_output"] is not None


def test_forward_check_uses_shared_device_resolver(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: dict[str, str] = {}

    def fake_resolve_torch_device(requested: str) -> torch.device:
        captured["requested"] = requested
        return torch.device("cpu")

    monkeypatch.setattr(dev_module, "resolve_torch_device", fake_resolve_torch_device)

    payload = dev_module.forward_check(
        _SMALL_STAGED_OVERRIDES,
        requested_device="cpu",
        seed=7,
    )

    assert captured == {"requested": "cpu"}
    assert payload["output_kind"] == "logits"


def test_forward_check_rejects_unavailable_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)

    with pytest.raises(RuntimeError, match="requested --device cuda, but CUDA is not available"):
        _ = dev_module.forward_check(
            _SMALL_STAGED_OVERRIDES,
            requested_device="cuda",
            seed=7,
        )


def test_forward_check_passes_for_many_class_surface() -> None:
    payload = forward_check(
        [
            "experiment=cls_smoke",
            "logging.use_wandb=false",
            "model.stage=many_class",
            "model.stage_label=many_class_dev_test",
            "model.d_icl=32",
            "model.many_class_base=4",
            "model.tficl_n_heads=4",
            "model.tficl_n_layers=1",
            "model.head_hidden_dim=64",
            "model.tfrow_n_heads=2",
            "model.tfrow_n_layers=1",
            "model.tfrow_cls_tokens=2",
            "model.tfcol_n_heads=2",
            "model.tfcol_n_layers=1",
            "model.tfcol_n_inducing=8",
        ],
        requested_device="cpu",
        seed=11,
    )

    assert payload["output_kind"] == "class_probs"
    assert payload["surface_label"] == "many_class_dev_test"
    assert payload["output_shape"] == [2, 5]
    assert payload["batched_output"] is None


def test_resolve_device_auto_prefers_mps_when_cuda_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(
        torch.backends,
        "mps",
        SimpleNamespace(is_available=lambda: True),
        raising=False,
    )

    assert device_module.resolve_device("auto") == "mps"


def test_resolve_torch_device_rejects_unavailable_mps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        torch.backends,
        "mps",
        SimpleNamespace(is_available=lambda: False),
        raising=False,
    )

    with pytest.raises(RuntimeError, match="requested --device mps, but MPS is not available"):
        _ = device_module.resolve_torch_device("mps")


def test_resolve_torch_device_accepts_explicit_cpu() -> None:
    assert device_module.resolve_torch_device("cpu") == torch.device("cpu")


def test_resolve_torch_device_rejects_unknown_device() -> None:
    with pytest.raises(RuntimeError, match="unsupported device: 'tpu'"):
        _ = device_module.resolve_torch_device("tpu")


def test_diff_config_payloads_only_reports_effective_differences() -> None:
    payload = diff_config_payloads(
        [
            "experiment=cls_smoke",
            "runtime.output_dir=/tmp/left",
            "model.stage=row_cls_pool",
            "model.stage_label=row_cls_pool_dev_test",
        ],
        [
            "experiment=cls_smoke",
            "runtime.output_dir=/tmp/right",
            "model.stage=qass_context",
            "model.stage_label=qass_context_dev_test",
        ],
    )

    paths = {str(difference["path"]) for difference in payload["differences"]}
    assert "model.stage" in paths
    assert "model.stage_label" in paths
    assert "runtime.output_dir" not in paths


def test_export_check_uses_temporary_bundle_by_default(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _ = exporter_cases._write_checkpoint(checkpoint, task="classification", seed=21)

    payload = export_check(
        checkpoint,
        out_dir=None,
        artifact_version=SCHEMA_VERSION_V3,
    )

    assert payload["schema_version"] == SCHEMA_VERSION_V3
    assert payload["bundle_dir_kept"] is False
    assert payload["bundle_dir_exists_after"] is False
    assert Path(str(payload["bundle_dir"])).exists() is False
    assert payload["reference_smoke"]["output_shape"][0] == 2


def test_export_check_keeps_explicit_bundle_dir(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _ = exporter_cases._write_checkpoint(
        checkpoint,
        task="classification",
        preprocessing_cfg={
            "surface_label": "runtime_no_impute",
            "overrides": {"impute_missing": False, "all_nan_fill": 1.0},
        },
        seed=23,
    )
    out_dir = tmp_path / "bundle"

    payload = export_check(
        checkpoint,
        out_dir=out_dir,
        artifact_version=SCHEMA_VERSION_V3,
    )

    assert payload["bundle_dir_kept"] is True
    assert payload["bundle_dir_exists_after"] is True
    assert out_dir.exists()
    assert payload["preprocessor"]["missing_value_policy"]["impute_missing"] is False
    assert payload["reference_smoke"]["used_missing_inputs"] is False


def test_export_check_rejects_non_v3_artifact_version(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    _ = exporter_cases._write_checkpoint(checkpoint, task="classification", seed=29)

    try:
        _ = export_check(
            checkpoint,
            out_dir=tmp_path / "bundle_v2",
            artifact_version=SCHEMA_VERSION_V2,
        )
    except RuntimeError as exc:
        assert "requires artifact_version=tab-foundry-export-v3" in str(exc)
    else:  # pragma: no cover - defensive assertion
        raise AssertionError("expected export_check() to reject v2 artifact versions")
