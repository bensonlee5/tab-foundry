from __future__ import annotations

from pathlib import Path
import shutil
from typing import Any

from omegaconf import OmegaConf
import pytest

from tab_foundry.benchmark_registry import default_benchmark_run_registry_path
from tab_foundry.research.sweep.promote import PromotionPaths, promote_anchor, resolve_run_id_for_order
import tab_foundry.research.sweep.promote as promote_module


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = default_benchmark_run_registry_path()


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

    assert run_id == 'sd_input_norm_followup_07_dpnb_input_norm_anchor_replay_batch64_sqrt_v2'


def test_promote_anchor_updates_sweep_and_index_without_touching_program_for_inactive_sweep(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reference_root, sweeps_root = _copy_reference_workspace(tmp_path)
    paths = _build_paths(tmp_path, sweeps_root, reference_root)
    rendered: list[str] = []

    monkeypatch.setattr(promote_module, '_render_sweep_matrix', lambda **kwargs: rendered.append(kwargs['sweep_id']))

    _ = promote_anchor(
        sweep_id='input_norm_followup',
        anchor_run_id='sd_input_norm_followup_09_dpnb_input_norm_zscore_tanh_batch64_sqrt_v1',
        paths=paths,
    )

    sweep = _load_yaml(sweeps_root / 'input_norm_followup' / 'sweep.yaml')
    index = _load_yaml(sweeps_root / 'index.yaml')

    assert sweep['anchor_run_id'] == 'sd_input_norm_followup_09_dpnb_input_norm_zscore_tanh_batch64_sqrt_v1'
    assert sweep['anchor_context']['run_id'] == 'sd_input_norm_followup_09_dpnb_input_norm_zscore_tanh_batch64_sqrt_v1'
    assert index['sweeps']['input_norm_followup']['anchor_run_id'] == 'sd_input_norm_followup_09_dpnb_input_norm_zscore_tanh_batch64_sqrt_v1'
    assert rendered == ['input_norm_followup']


def test_promote_anchor_updates_selected_sweep_without_global_active_side_effects(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    reference_root, sweeps_root = _copy_reference_workspace(tmp_path)
    paths = _build_paths(tmp_path, sweeps_root, reference_root)
    rendered: list[str] = []

    monkeypatch.setattr(promote_module, '_render_sweep_matrix', lambda **kwargs: rendered.append(kwargs['sweep_id']))

    _ = promote_anchor(
        sweep_id='tf_rd_021b_sandwich_feature_removal_v1',
        anchor_run_id='tf_rd_021b_hybrid_full_cell_compact_prior_v1',
        paths=paths,
    )

    sweep = _load_yaml(sweeps_root / 'tf_rd_021b_sandwich_feature_removal_v1' / 'sweep.yaml')
    index = _load_yaml(sweeps_root / 'index.yaml')

    assert rendered == ['tf_rd_021b_sandwich_feature_removal_v1']
    assert index['schema'] == 'tab-foundry-system-delta-sweep-index-v2'
    assert 'active_sweep_id' not in index
    assert sweep['anchor_run_id'] == 'tf_rd_021b_hybrid_full_cell_compact_prior_v1'
