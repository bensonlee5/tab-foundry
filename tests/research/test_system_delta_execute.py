from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

from omegaconf import OmegaConf
import pytest

from tab_foundry.research.system_delta import create_sweep
from tab_foundry.research.system_delta_execute import (
    ExecutionPaths,
    _queue_metrics,
    _result_card_text,
    execute_sweep,
    select_queue_rows,
)
import tab_foundry.research.system_delta_execute as execute_module


REPO_ROOT = Path(__file__).resolve().parents[2]
REGISTRY_PATH = REPO_ROOT / 'src' / 'tab_foundry' / 'bench' / 'benchmark_run_registry_v1.json'
ANCHOR_RUN_ID = 'sd_input_norm_followup_07_dpnb_input_norm_anchor_replay_batch64_sqrt_v2'


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


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(OmegaConf.to_yaml(OmegaConf.create(payload), resolve=True), encoding='utf-8')


def _build_paths(tmp_path: Path, sweeps_root: Path, reference_root: Path) -> ExecutionPaths:
    return ExecutionPaths(
        repo_root=tmp_path,
        index_path=sweeps_root / 'index.yaml',
        catalog_path=reference_root / 'system_delta_catalog.yaml',
        sweeps_root=sweeps_root,
        registry_path=REGISTRY_PATH,
        program_path=tmp_path / 'program.md',
        control_baseline_registry_path=REPO_ROOT / 'src' / 'tab_foundry' / 'bench' / 'control_baselines_v1.json',
    )


def _make_exec_sweep(tmp_path: Path) -> tuple[str, ExecutionPaths, Path]:
    reference_root, sweeps_root = _copy_reference_workspace(tmp_path)
    sweep_id = 'exec_test_sweep'
    _ = create_sweep(
        sweep_id=sweep_id,
        anchor_run_id=ANCHOR_RUN_ID,
        parent_sweep_id='input_norm_followup',
        complexity_level='binary_md',
        benchmark_bundle_path='src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json',
        control_baseline_id='cls_benchmark_linear_v2',
        delta_refs=['delta_anchor_activation_trace_baseline', 'delta_shared_feature_norm'],
        index_path=sweeps_root / 'index.yaml',
        catalog_path=reference_root / 'system_delta_catalog.yaml',
        registry_path=REGISTRY_PATH,
        sweeps_root=sweeps_root,
    )
    program_path = tmp_path / 'program.md'
    program_path.write_text((REPO_ROOT / 'program.md').read_text(encoding='utf-8'), encoding='utf-8')
    paths = _build_paths(tmp_path, sweeps_root, reference_root)
    queue_path = sweeps_root / sweep_id / 'queue.yaml'
    return sweep_id, paths, queue_path


def test_select_queue_rows_defaults_to_ready_rows() -> None:
    queue = {
        'rows': [
            {'order': 1, 'status': 'ready'},
            {'order': 2, 'status': 'completed'},
            {'order': 3, 'status': 'ready'},
        ]
    }

    selected = select_queue_rows(queue)

    assert [int(row['order']) for row in selected] == [1, 3]


def test_select_queue_rows_requires_include_completed_for_explicit_completed_rows() -> None:
    queue = {
        'rows': [
            {'order': 1, 'status': 'completed'},
            {'order': 2, 'status': 'ready'},
        ]
    }

    with pytest.raises(RuntimeError, match='include-completed'):
        _ = select_queue_rows(queue, orders=[1])


def test_execute_sweep_defaults_to_active_sweep_and_ready_rows(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sweep_id, paths, queue_path = _make_exec_sweep(tmp_path)
    index = _load_yaml(paths.index_path)
    index['active_sweep_id'] = sweep_id
    _write_yaml(paths.index_path, index)
    queue = _load_yaml(queue_path)
    queue['rows'][0]['status'] = 'ready'
    queue['rows'][1]['status'] = 'completed'
    queue['rows'][1]['run_id'] = 'historical_run_v1'
    _write_yaml(queue_path, queue)

    calls: list[dict[str, Any]] = []

    def fake_run_row(**kwargs: Any) -> str:
        calls.append({'order': int(kwargs['queue_row']['order']), 'sweep_id': kwargs['sweep_id']})
        return f"run_{kwargs['queue_row']['order']}"

    monkeypatch.setattr(execute_module, '_run_row', fake_run_row)
    monkeypatch.setattr(execute_module, '_sync_sweep_matrix', lambda **_: None)
    monkeypatch.setattr(execute_module, '_sync_active_aliases_if_active', lambda **_: None)

    executed = execute_sweep(
        sweep_id=None,
        prior_dump=Path('/tmp/prior.h5'),
        nanotabpfn_root=Path('/tmp/nanotabpfn'),
        device='cuda',
        fallback_python=REPO_ROOT / '.venv' / 'bin' / 'python',
        paths=paths,
    )

    assert executed == ['run_1']
    assert calls == [{'order': 1, 'sweep_id': sweep_id}]


def test_execute_sweep_applies_overrides_and_promotes_first_row(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sweep_id, paths, queue_path = _make_exec_sweep(tmp_path)
    queue = _load_yaml(queue_path)
    queue['rows'][0]['status'] = 'ready'
    queue['rows'][1]['status'] = 'ready'
    _write_yaml(queue_path, queue)

    calls: list[dict[str, Any]] = []
    promotions: list[dict[str, Any]] = []

    def fake_run_row(**kwargs: Any) -> str:
        order = int(kwargs['queue_row']['order'])
        run_id = 'promoted_anchor_v2' if order == 1 else 'row_2_v1'
        calls.append(
            {
                'order': order,
                'decision': kwargs['decision'],
                'conclusion': kwargs['conclusion'],
                'anchor_run_id': kwargs['anchor_run_id'],
                'parent_run_id': kwargs['parent_run_id'],
                'run_id': run_id,
            }
        )
        return run_id

    def fake_promote_anchor(**kwargs: Any) -> dict[str, str]:
        promotions.append({'sweep_id': kwargs['sweep_id'], 'anchor_run_id': kwargs['anchor_run_id']})
        return {'sweep_id': kwargs['sweep_id'], 'anchor_run_id': kwargs['anchor_run_id']}

    monkeypatch.setattr(execute_module, '_run_row', fake_run_row)
    monkeypatch.setattr(execute_module, 'promote_anchor', fake_promote_anchor)
    monkeypatch.setattr(execute_module, '_sync_sweep_matrix', lambda **_: None)
    monkeypatch.setattr(execute_module, '_sync_active_aliases_if_active', lambda **_: None)

    executed = execute_sweep(
        sweep_id=sweep_id,
        prior_dump=Path('/tmp/prior.h5'),
        nanotabpfn_root=Path('/tmp/nanotabpfn'),
        device='cuda',
        fallback_python=REPO_ROOT / '.venv' / 'bin' / 'python',
        decision_default='defer',
        conclusion_default='default conclusion',
        decision_overrides={2: 'reject'},
        conclusion_overrides={2: 'explicit conclusion'},
        promote_first_executed_row_to_anchor=True,
        paths=paths,
    )

    assert executed == ['promoted_anchor_v2', 'row_2_v1']
    assert promotions == [{'sweep_id': sweep_id, 'anchor_run_id': 'promoted_anchor_v2'}]
    assert calls[0] == {
        'order': 1,
        'decision': 'defer',
        'conclusion': 'default conclusion',
        'anchor_run_id': None,
        'parent_run_id': None,
        'run_id': 'promoted_anchor_v2',
    }
    assert calls[1] == {
        'order': 2,
        'decision': 'reject',
        'conclusion': 'explicit conclusion',
        'anchor_run_id': 'promoted_anchor_v2',
        'parent_run_id': 'promoted_anchor_v2',
        'run_id': 'row_2_v1',
    }


def test_queue_metrics_capture_log_loss_and_anchor_deltas(tmp_path: Path) -> None:
    (tmp_path / 'gradient_history.jsonl').write_text(
        ''.join(
            [
                json.dumps({'step': 1, 'grad_clip_triggered': False}) + '\n',
                json.dumps({'step': 2, 'grad_clip_triggered': True}) + '\n',
            ]
        ),
        encoding='utf-8',
    )

    summary = {
        'tab_foundry': {
            'best_step': 75.0,
            'best_log_loss': 0.41,
            'final_log_loss': 0.43,
            'best_to_final_log_loss_delta': 0.02,
            'best_brier_score': 0.11,
            'final_brier_score': 0.13,
            'best_to_final_brier_score_delta': 0.02,
            'best_roc_auc': 0.82,
            'final_roc_auc': 0.80,
            'best_to_final_roc_auc_delta': -0.02,
            'training_diagnostics': {'max_grad_norm': 7.5},
        },
        'nanotabpfn': {
            'best_log_loss': 0.46,
            'final_log_loss': 0.48,
            'best_brier_score': 0.14,
            'final_brier_score': 0.15,
            'best_roc_auc': 0.81,
            'final_roc_auc': 0.79,
        },
    }
    run_entry = {
        'comparisons': {
            'vs_anchor': {
                'final_log_loss_delta': -0.03,
                'final_brier_score_delta': 0.01,
                'final_roc_auc_delta': -0.02,
            }
        }
    }

    queue_metrics = _queue_metrics(summary, run_dir=tmp_path, run_entry=run_entry)

    assert queue_metrics['best_step'] == 75
    assert queue_metrics['best_log_loss'] == pytest.approx(0.41)
    assert queue_metrics['final_log_loss'] == pytest.approx(0.43)
    assert queue_metrics['final_minus_best_log_loss'] == pytest.approx(0.02)
    assert queue_metrics['final_brier_score'] == pytest.approx(0.13)
    assert queue_metrics['best_roc_auc'] == pytest.approx(0.82)
    assert queue_metrics['final_roc_auc'] == pytest.approx(0.80)
    assert queue_metrics['delta_final_log_loss'] == pytest.approx(-0.03)
    assert queue_metrics['delta_final_brier_score'] == pytest.approx(0.01)
    assert queue_metrics['delta_final_roc_auc'] == pytest.approx(-0.02)
    assert queue_metrics['nanotabpfn_final_log_loss'] == pytest.approx(0.48)
    assert queue_metrics['clipped_step_fraction'] == pytest.approx(0.5)


def test_result_card_text_reports_log_loss_before_roc() -> None:
    text = _result_card_text(
        row={
            'delta_id': 'delta',
            'description': 'Use the refreshed benchmark surface.',
            'anchor_delta': 'anchor-only comparison.',
        },
        run_id='sd_test_v1',
        anchor_run_id='anchor_v1',
        summary={'tab_foundry': {}, 'nanotabpfn': {}},
        queue_metrics={
            'best_step': 125,
            'best_log_loss': 0.401,
            'final_log_loss': 0.409,
            'final_minus_best_log_loss': 0.008,
            'delta_final_log_loss': -0.011,
            'final_brier_score': 0.118,
            'final_minus_best_brier_score': 0.006,
            'delta_final_brier_score': -0.004,
            'best_roc_auc': 0.812,
            'final_roc_auc': 0.804,
            'final_minus_best_roc_auc': -0.008,
            'delta_final_roc_auc': -0.006,
            'max_grad_norm': 3.2,
            'clipped_step_fraction': 0.125,
        },
        decision='defer',
        conclusion='Monitor log-loss deltas before promotion.',
    )

    assert '- Best log loss: `0.4010` at step `125`' in text
    assert '- Delta final log loss vs anchor: `-0.0110`' in text
    assert '- Final ROC AUC: `0.8040`' in text
    assert text.index('Best log loss') < text.index('Best ROC AUC')
