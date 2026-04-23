from __future__ import annotations

import types

import pytest

import tab_foundry.hardware_profiles as hardware_profiles


def test_normalize_gpu_class_maps_known_accelerators() -> None:
    assert hardware_profiles.normalize_gpu_class("NVIDIA A100-SXM4-80GB", device_type="cuda") == "a100"
    assert hardware_profiles.normalize_gpu_class("NVIDIA H100 PCIe", device_type="cuda") == "h100"
    assert hardware_profiles.normalize_gpu_class("Quadro RTX 8000", device_type="cuda") == "rtx8000"
    assert hardware_profiles.normalize_gpu_class("cpu", device_type="cpu") == "cpu"


def test_build_hardware_summary_reports_cuda_device_identity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake_cuda = types.SimpleNamespace(
        is_available=lambda: True,
        current_device=lambda: 0,
        get_device_properties=lambda index: types.SimpleNamespace(
            name="NVIDIA A100-SXM4-80GB",
            total_memory=80 * 1024**3,
        ),
    )
    fake_torch = types.SimpleNamespace(cuda=fake_cuda)
    monkeypatch.setattr(hardware_profiles, "_load_torch", lambda: fake_torch)

    summary = hardware_profiles.build_hardware_summary("cuda")

    assert summary == {
        "device_type": "cuda",
        "raw_device_name": "NVIDIA A100-SXM4-80GB",
        "gpu_class": "a100",
        "total_device_vram_bytes": 80 * 1024**3,
        "vram_class_gb": 80,
        "hardware_profile_id": "a100_80gb",
    }


def test_build_hardware_summary_reports_cpu_profile() -> None:
    summary = hardware_profiles.build_hardware_summary("cpu")

    assert summary == {
        "device_type": "cpu",
        "raw_device_name": "cpu",
        "gpu_class": "cpu",
        "total_device_vram_bytes": None,
        "vram_class_gb": None,
        "hardware_profile_id": "cpu",
    }


def test_resolve_gpu_utilization_capability_reports_supported_bf16_gpu() -> None:
    capability = hardware_profiles.resolve_gpu_utilization_capability(
        gpu_class="a100",
        mixed_precision="bf16",
    )

    assert capability == {
        "theoretical_peak_tflops_per_second": 312.0,
        "theoretical_hbm_bandwidth_gbps": 2039.0,
        "roofline_knee_flops_per_byte": pytest.approx(153.0161844031388),
        "peak_compute_basis": "tensorcore_bf16_dense",
    }


def test_resolve_gpu_utilization_capability_rejects_unknown_or_unsupported_pairs() -> None:
    assert hardware_profiles.resolve_gpu_utilization_capability(
        gpu_class="cpu",
        mixed_precision="bf16",
    ) is None
    assert hardware_profiles.resolve_gpu_utilization_capability(
        gpu_class="a100",
        mixed_precision="no",
    ) is None
    assert hardware_profiles.resolve_gpu_utilization_capability(
        gpu_class="mystery_gpu",
        mixed_precision="bf16",
    ) is None
