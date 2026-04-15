from __future__ import annotations

from omegaconf import OmegaConf
import pytest

import tab_foundry.training.trainer_runtime_config as runtime_config_module
from tab_foundry.training.trainer_runtime_config import (
    _resolve_compile_shape_dispatch_max_families,
    _resolve_compile_shape_dispatch_mode,
    _resolve_cuda_graph_capture_mode,
    _resolve_cuda_graph_max_families,
    _resolve_loader_task_batch_cache_mode,
    _resolve_signature_family_optimizer_step_block_length,
    _resolve_signature_family_run_length,
    default_loader_num_workers,
    default_loader_prefetch_factor,
    resolve_loader_overlap_runtime_settings,
)


def test_default_loader_num_workers_uses_cpu_heuristic() -> None:
    assert default_loader_num_workers(cpu_count=1) == 1
    assert default_loader_num_workers(cpu_count=4) == 1
    assert default_loader_num_workers(cpu_count=8) == 2
    assert default_loader_num_workers(cpu_count=16) == 3
    assert default_loader_num_workers(cpu_count=40) == 8


def test_default_loader_prefetch_factor_tracks_worker_bands() -> None:
    assert default_loader_prefetch_factor(num_workers=0) is None
    assert default_loader_prefetch_factor(num_workers=1) == 2
    assert default_loader_prefetch_factor(num_workers=2) == 2
    assert default_loader_prefetch_factor(num_workers=3) == 4
    assert default_loader_prefetch_factor(num_workers=8) == 4


def test_resolve_loader_overlap_runtime_settings_fills_auto_from_cpu_heuristic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runtime_config_module.os, "cpu_count", lambda: 40)
    runtime_cfg = OmegaConf.create({"num_workers": "auto", "loader_prefetch_factor": "auto"})

    resolved = resolve_loader_overlap_runtime_settings(runtime_cfg)

    assert resolved.num_workers == 8
    assert resolved.prefetch_factor == 4
    assert resolved.num_workers_is_auto is True
    assert resolved.prefetch_factor_is_auto is True


def test_resolve_loader_overlap_runtime_settings_preserves_explicit_values() -> None:
    runtime_cfg = OmegaConf.create({"num_workers": 3, "loader_prefetch_factor": 2})

    resolved = resolve_loader_overlap_runtime_settings(runtime_cfg)

    assert resolved.num_workers == 3
    assert resolved.prefetch_factor == 2
    assert resolved.num_workers_is_auto is False
    assert resolved.prefetch_factor_is_auto is False


def test_resolve_loader_task_batch_cache_mode_preserves_legacy_precedence() -> None:
    assert (
        _resolve_loader_task_batch_cache_mode(
            OmegaConf.create(
                {
                    "loader_task_batch_cache_mode": "bounded_streaming",
                    "loader_task_batch_cache": True,
                }
            )
        )
        == "bounded_streaming"
    )
    assert (
        _resolve_loader_task_batch_cache_mode(
            OmegaConf.create({"loader_task_batch_cache": True})
        )
        == "eager_full"
    )
    assert (
        _resolve_loader_task_batch_cache_mode(
            OmegaConf.create({"loader_task_batch_cache": False})
        )
        == "off"
    )


def test_resolve_compile_shape_dispatch_controls() -> None:
    runtime_cfg = OmegaConf.create(
        {
            "compile_shape_dispatch_mode": "signature_family",
            "compile_shape_dispatch_max_families": 8,
            "cuda_graph_capture_mode": "signature_family",
            "cuda_graph_max_families": 6,
            "signature_family_run_length": 4,
            "signature_family_optimizer_step_block_length": 2,
        }
    )

    assert _resolve_compile_shape_dispatch_mode(runtime_cfg) == "signature_family"
    assert _resolve_compile_shape_dispatch_max_families(runtime_cfg) == 8
    assert _resolve_cuda_graph_capture_mode(runtime_cfg) == "signature_family"
    assert _resolve_cuda_graph_max_families(runtime_cfg) == 6
    assert _resolve_signature_family_run_length(runtime_cfg) == 4
    assert _resolve_signature_family_optimizer_step_block_length(runtime_cfg) == 2
