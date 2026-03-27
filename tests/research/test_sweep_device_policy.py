from __future__ import annotations

from pathlib import Path

import pytest

import tab_foundry.research.sweep.curve_reuse as curve_reuse_module
import tab_foundry.research.sweep.device_policy as device_policy_module


def test_resolve_sweep_execution_device_prefers_cuda_for_auto() -> None:
    resolved = device_policy_module.resolve_sweep_execution_device(
        'auto',
        auto_resolve_fn=lambda _device: 'cuda',
        strict_resolve_torch_device_fn=lambda _device: 'cuda',
    )

    assert resolved == 'cuda'


def test_resolve_sweep_execution_device_falls_back_to_cpu_for_auto() -> None:
    resolved = device_policy_module.resolve_sweep_execution_device(
        'auto',
        auto_resolve_fn=lambda _device: 'cpu',
        strict_resolve_torch_device_fn=lambda _device: 'cpu',
    )

    assert resolved == 'cpu'


def test_resolve_sweep_execution_device_never_uses_mps_for_auto() -> None:
    resolved = device_policy_module.resolve_sweep_execution_device(
        'auto',
        auto_resolve_fn=lambda _device: 'mps',
        strict_resolve_torch_device_fn=lambda _device: 'cpu',
    )

    assert resolved == 'cpu'


def test_resolve_sweep_execution_device_validates_explicit_cuda() -> None:
    def _raise_unavailable(_device: str) -> None:
        raise RuntimeError('requested --device cuda, but CUDA is not available')

    with pytest.raises(RuntimeError, match='requested --device cuda, but CUDA is not available'):
        _ = device_policy_module.resolve_sweep_execution_device(
            'cuda',
            strict_resolve_torch_device_fn=_raise_unavailable,
        )


def test_resolved_nanotabpfn_signature_remaps_auto_mps_to_cpu(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle_path = tmp_path / 'bundle.json'
    bundle_path.write_text('{}\n', encoding='utf-8')
    nanotabpfn_root = tmp_path / 'nano'
    prior_dump = nanotabpfn_root / '300k_150x5_2.h5'
    prior_dump.parent.mkdir(parents=True, exist_ok=True)
    prior_dump.write_bytes(b'prior')

    monkeypatch.setattr(curve_reuse_module, 'resolve_device', lambda _device: 'mps')
    monkeypatch.setattr(
        curve_reuse_module,
        'benchmark_host_fingerprint',
        lambda: 'runner-host',
    )

    signature = curve_reuse_module._resolved_nanotabpfn_signature(
        benchmark_bundle_path=bundle_path,
        control_baseline_id='cls_benchmark_linear_v2',
        nanotabpfn_root=nanotabpfn_root,
        prior_dump=prior_dump,
        requested_device='auto',
    )
    metadata = curve_reuse_module._signature_metadata(signature)

    assert signature['device'] == 'auto'
    assert signature['resolved_device'] == 'cpu'
    assert metadata['device'] == 'auto'
    assert metadata['resolved_device'] == 'cpu'
