from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

from omegaconf import OmegaConf
import pytest

from tab_foundry.research.system_delta_promote import PromotionPaths, promote_anchor, resolve_run_id_for_order
import tab_foundry.research.system_delta_promote as promote_module


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / 'src' / 'tab_foundry' / 'bench' / 'benchmark_run_registry_v1.json'


def _copy_reference_workspace(tmp_path: Path) -> tuple[Path, Path]:
    reference_root = tmp_path / 'reference'
    sweeps_root = reference_root / 'system_delta_sweeps'
    source_sweeps_root = REPO_ROOT / 'reference' / 'system_delta_sweeps'
    sweeps_root.mkdir(parents=True, exist_ok=True)
    (reference_root / 'system_delta_catalog.yaml').write_text(
        (REPO_ROOT / 'reference' / 'system_delta_catalog.yaml').read_text(encoding='utf-8'),
        encoding='utf-8',
    )
    (sweeps_root / 'index.yaml').write_text(
        (source_sweeps_root / 'index.yaml').read_text(encoding='utf-8'),
        encoding='utf-8',
    )
    for source_dir in sorted(source_sweeps_root.iterdir()):
        if source_dir.name == 'index.yaml' or not source_dir.is_dir():
            continue
        shutil.copytree(source_dir, sweeps_root / source_dir.name)
    return reference_root, sweeps_root


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _build_paths(tmp_path: Path, sweeps_root: Path, reference_root: Path) -> PromotionPaths:
    program_path = tmp_path / 'program.md'
    program_path.write_text((REPO_ROOT / 'program.md').read_text(encoding='utf-8'), encoding='utf-8')
    return PromotionPaths(
        index_path=sweeps_root / 'index.yaml',
        catalog_path=reference_root / 'system_delta_catalog.yaml',
        sweeps_root=sweeps_root,
        registry_path=REGISTRY_PATH,
        program_path=program_path,
    )


def test_resolve_run_id_for_order_uses_queue_run_id(tmp_path: Path) -> None:
    reference_root, sweeps_root = _copy_reference_workspace(tmp_path)
    paths = _build_paths(tmp_path, sweeps_root, reference_root)

    run_id = resolve_run_id_for_order(sweep_id='input_norm_followup', order=7, paths=paths)

    assert run_id == 'sd_input_norm_followup_07_dpnb_input_norm_anchor_replay_batch64_sqrt_v1'


def test_promote_anchor_updates_sweep_and_index_without_touching_program_for_inactive_sweep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reference_root, sweeps_root = _copy_reference_workspace(tmp_path)
    paths = _build_paths(tmp_path, sweeps_root, reference_root)
    rendered: list[str] = []
    synced: list[str] = []

    monkeypatch.setattr(promote_module, '_render_sweep_matrix', lambda **kwargs: rendered.append(kwargs['sweep_id']))
    monkeypatch.setattr(
        promote_module.system_delta,
        'sync_active_sweep_aliases',
        lambda **kwargs: synced.append(kwargs['sweep_id']) or {},
    )

    _ = promote_anchor(
        sweep_id='input_norm_followup',
        anchor_run_id='sd_input_norm_followup_09_dpnb_input_norm_zscore_tanh_batch64_sqrt_v1',
        paths=paths,
    )

    sweep = _load_yaml(sweeps_root / 'input_norm_followup' / 'sweep.yaml')
    index = _load_yaml(sweeps_root / 'index.yaml')
    program_text = paths.program_path.read_text(encoding='utf-8')

    assert sweep['anchor_run_id'] == 'sd_input_norm_followup_09_dpnb_input_norm_zscore_tanh_batch64_sqrt_v1'
    assert sweep['anchor_context']['run_id'] == 'sd_input_norm_followup_09_dpnb_input_norm_zscore_tanh_batch64_sqrt_v1'
    assert index['sweeps']['input_norm_followup']['anchor_run_id'] == 'sd_input_norm_followup_09_dpnb_input_norm_zscore_tanh_batch64_sqrt_v1'
    assert rendered == ['input_norm_followup']
    assert synced == []
    assert 'sd_input_norm_followup_07_dpnb_input_norm_anchor_replay_batch64_sqrt_v1' in program_text


def test_promote_anchor_updates_program_for_active_sweep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reference_root, sweeps_root = _copy_reference_workspace(tmp_path)
    paths = _build_paths(tmp_path, sweeps_root, reference_root)
    rendered: list[str] = []
    synced: list[str] = []

    monkeypatch.setattr(promote_module, '_render_sweep_matrix', lambda **kwargs: rendered.append(kwargs['sweep_id']))
    monkeypatch.setattr(
        promote_module.system_delta,
        'sync_active_sweep_aliases',
        lambda **kwargs: synced.append(kwargs['sweep_id']) or {},
    )

    _ = promote_anchor(
        sweep_id='input_norm_none_followup',
        anchor_run_id='sd_input_norm_none_followup_01_dpnb_input_norm_none_batch64_sqrt_v1',
        paths=paths,
    )

    program_text = paths.program_path.read_text(encoding='utf-8')

    assert rendered == ['input_norm_none_followup']
    assert synced == ['input_norm_none_followup']
    assert 'sd_input_norm_none_followup_01_dpnb_input_norm_none_batch64_sqrt_v1' in program_text
    assert 'outputs/staged_ladder/research/input_norm_none_followup/dpnb_input_norm_none_batch64_sqrt/sd_input_norm_none_followup_01_dpnb_input_norm_none_batch64_sqrt_v1/train' in program_text
    assert '- active sweep id: `input_norm_none_followup`' in program_text
    assert '- canonical sweep queue: `reference/system_delta_sweeps/input_norm_none_followup/queue.yaml`' in program_text
