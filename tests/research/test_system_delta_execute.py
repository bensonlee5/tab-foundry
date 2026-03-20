from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from types import SimpleNamespace
from typing import Any

from omegaconf import OmegaConf
import pytest

from tab_foundry.research.system_delta import create_sweep
from tab_foundry.research.system_delta_execute import (
    ExecutionPaths,
    _compose_cfg,
    _queue_metrics,
    _result_card_text,
    execute_sweep,
    select_queue_rows,
)
import tab_foundry.research.system_delta_execute as execute_module
import tab_foundry.research.sweep.runner as runner_module
from tab_foundry.research.sweep.artifacts import read_yaml as read_artifact_yaml, write_research_package
from tab_foundry.research.sweep.screening import pick_screen_winner, screen_metrics as load_screen_metrics


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


def _write_python_probe_stub(path: Path, *, torch_import_exit_code: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"$1\" = \"-c\" ] && [ \"$2\" = \"import torch\" ]; then\n"
        f"  exit {torch_import_exit_code}\n"
        "fi\n"
        "exit 0\n",
        encoding='utf-8',
    )
    path.chmod(0o755)


def _write_compare_summary(
    path: Path,
    *,
    bundle_path: Path,
    curve_path: Path | None,
    nanotab_root: Path,
    nanotab_python: Path,
    control_baseline_id: str | None,
    device: str | None,
    resolved_device: str | None = None,
    host_fingerprint: str | None = None,
    prior_dump_path: Path | None,
    steps: int | None = None,
    eval_every: int | None = None,
    seeds: int | None = None,
    batch_size: int | None = None,
    lr: float | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        'benchmark_bundle': {
            'name': 'bundle',
            'source_path': str(bundle_path.resolve()),
        },
        'nanotabpfn': {
            'root': str(nanotab_root.resolve()),
            'python': str(nanotab_python.resolve()),
            'num_seeds': 2 if seeds is None else int(seeds),
        },
        'artifacts': {},
    }
    nanotabpfn = payload['nanotabpfn']
    if device is not None:
        nanotabpfn['device'] = str(device)
    if device is not None and resolved_device is None:
        nanotabpfn['resolved_device'] = runner_module.resolve_device(str(device))
    if resolved_device is not None:
        nanotabpfn['resolved_device'] = str(resolved_device)
    if device is not None and host_fingerprint is None:
        nanotabpfn['benchmark_host_fingerprint'] = runner_module.benchmark_host_fingerprint()
    if host_fingerprint is not None:
        nanotabpfn['benchmark_host_fingerprint'] = str(host_fingerprint)
    if prior_dump_path is not None:
        nanotabpfn['prior_dump_path'] = str(prior_dump_path.resolve())
    if steps is not None:
        nanotabpfn['steps'] = int(steps)
    if eval_every is not None:
        nanotabpfn['eval_every'] = int(eval_every)
    if seeds is not None:
        nanotabpfn['num_seeds'] = int(seeds)
    if batch_size is not None:
        nanotabpfn['batch_size'] = int(batch_size)
    if lr is not None:
        nanotabpfn['lr'] = float(lr)
    if curve_path is not None:
        payload['artifacts']['nanotabpfn_curve_jsonl'] = str(curve_path.resolve())
    if control_baseline_id is not None:
        payload['control_baseline'] = {'baseline_id': str(control_baseline_id)}
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')


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


def test_ensure_nanotabpfn_python_rewrites_existing_interpreter_without_torch(tmp_path: Path) -> None:
    nanotabpfn_root = tmp_path / 'nanoTabPFN'
    nanotab_python = nanotabpfn_root / '.venv' / 'bin' / 'python'
    fallback_python = tmp_path / 'fallback' / 'bin' / 'python'

    _write_python_probe_stub(nanotab_python, torch_import_exit_code=1)
    _write_python_probe_stub(fallback_python, torch_import_exit_code=0)

    observed = runner_module.ensure_nanotabpfn_python(
        nanotabpfn_root=nanotabpfn_root,
        fallback_python=fallback_python,
    )

    assert observed == nanotab_python
    assert str(fallback_python) in observed.read_text(encoding='utf-8')
    assert subprocess.run(
        [str(observed), '-c', 'import torch'],
        check=False,
    ).returncode == 0


def test_ensure_nanotabpfn_python_keeps_existing_usable_interpreter(tmp_path: Path) -> None:
    nanotabpfn_root = tmp_path / 'nanoTabPFN'
    nanotab_python = nanotabpfn_root / '.venv' / 'bin' / 'python'
    fallback_python = tmp_path / 'fallback' / 'bin' / 'python'

    _write_python_probe_stub(nanotab_python, torch_import_exit_code=0)
    _write_python_probe_stub(fallback_python, torch_import_exit_code=1)
    original_contents = nanotab_python.read_text(encoding='utf-8')

    observed = runner_module.ensure_nanotabpfn_python(
        nanotabpfn_root=nanotabpfn_root,
        fallback_python=fallback_python,
    )

    assert observed == nanotab_python
    assert observed.read_text(encoding='utf-8') == original_contents


def test_ensure_nanotabpfn_python_preserves_fallback_symlink_path(tmp_path: Path) -> None:
    nanotabpfn_root = tmp_path / 'nanoTabPFN'
    nanotab_python = nanotabpfn_root / '.venv' / 'bin' / 'python'
    fallback_python = tmp_path / 'fallback' / '.venv' / 'bin' / 'python'
    resolved_target = tmp_path / 'pyenv' / 'versions' / '3.14.3' / 'bin' / 'python3.14'

    _write_python_probe_stub(nanotab_python, torch_import_exit_code=1)
    resolved_target.parent.mkdir(parents=True, exist_ok=True)
    resolved_target.write_text(
        "#!/usr/bin/env bash\n"
        "if [ \"$1\" = \"-c\" ] && [ \"$2\" = \"import torch\" ]; then\n"
        "  case \"$0\" in\n"
        "    */.venv/bin/python) exit 0 ;;\n"
        "    *) exit 1 ;;\n"
        "  esac\n"
        "fi\n"
        "exit 0\n",
        encoding='utf-8',
    )
    resolved_target.chmod(0o755)
    fallback_python.parent.mkdir(parents=True, exist_ok=True)
    fallback_python.symlink_to(resolved_target)

    observed = runner_module.ensure_nanotabpfn_python(
        nanotabpfn_root=nanotabpfn_root,
        fallback_python=fallback_python,
    )

    shim = observed.read_text(encoding='utf-8')
    assert str(fallback_python) in shim
    assert str(resolved_target) not in shim
    assert subprocess.run(
        [str(observed), '-c', 'import torch'],
        check=False,
    ).returncode == 0


def test_main_preserves_tab_foundry_python_symlink_path(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    prior_dump = tmp_path / 'prior.h5'
    nanotabpfn_root = tmp_path / 'nanoTabPFN'
    resolved_target = tmp_path / 'pyenv' / 'versions' / '3.14.3' / 'bin' / 'python3.14'
    fallback_python = tmp_path / '.venv' / 'bin' / 'python'

    prior_dump.write_text('stub', encoding='utf-8')
    nanotabpfn_root.mkdir(parents=True, exist_ok=True)
    resolved_target.parent.mkdir(parents=True, exist_ok=True)
    resolved_target.write_text('#!/usr/bin/env bash\nexit 0\n', encoding='utf-8')
    resolved_target.chmod(0o755)
    fallback_python.parent.mkdir(parents=True, exist_ok=True)
    fallback_python.symlink_to(resolved_target)

    captured: dict[str, Any] = {}

    def fake_execute_sweep(**kwargs: Any) -> list[str]:
        captured.update(kwargs)
        return []

    monkeypatch.setattr(execute_module, 'execute_sweep', fake_execute_sweep)

    exit_code = execute_module.main(
        [
            '--sweep-id',
            'shared_surface_bridge_v1',
            '--prior-dump',
            str(prior_dump),
            '--nanotabpfn-root',
            str(nanotabpfn_root),
            '--tab-foundry-python',
            str(fallback_python),
            '--device',
            'cpu',
        ]
    )

    assert exit_code == 0
    assert captured['fallback_python'] == fallback_python


def test_select_queue_rows_requires_include_completed_for_explicit_screened_rows() -> None:
    queue = {
        'rows': [
            {'order': 1, 'status': 'screened'},
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


def test_execute_sweep_uses_completed_parent_delta_ref(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sweep_id, paths, queue_path = _make_exec_sweep(tmp_path)
    queue = _load_yaml(queue_path)
    queue['rows'][0]['status'] = 'completed'
    queue['rows'][0]['run_id'] = 'completed_parent_v1'
    queue['rows'][1]['status'] = 'ready'
    queue['rows'][1]['parent_delta_ref'] = 'delta_anchor_activation_trace_baseline'
    _write_yaml(queue_path, queue)

    calls: list[dict[str, Any]] = []

    def fake_run_row(**kwargs: Any) -> str:
        calls.append(
            {
                'order': int(kwargs['queue_row']['order']),
                'anchor_run_id': kwargs['anchor_run_id'],
                'parent_run_id': kwargs['parent_run_id'],
            }
        )
        return 'row_2_v1'

    monkeypatch.setattr(execute_module, '_run_row', fake_run_row)
    monkeypatch.setattr(execute_module, '_sync_sweep_matrix', lambda **_: None)
    monkeypatch.setattr(execute_module, '_sync_active_aliases_if_active', lambda **_: None)

    executed = execute_sweep(
        sweep_id=sweep_id,
        prior_dump=Path('/tmp/prior.h5'),
        nanotabpfn_root=Path('/tmp/nanotabpfn'),
        device='cuda',
        fallback_python=REPO_ROOT / '.venv' / 'bin' / 'python',
        paths=paths,
    )

    assert executed == ['row_2_v1']
    assert calls == [
        {
            'order': 2,
            'anchor_run_id': ANCHOR_RUN_ID,
            'parent_run_id': 'completed_parent_v1',
        }
    ]


def test_execute_sweep_uses_same_invocation_parent_delta_ref(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sweep_id, paths, queue_path = _make_exec_sweep(tmp_path)
    queue = _load_yaml(queue_path)
    queue['rows'][0]['status'] = 'ready'
    queue['rows'][1]['status'] = 'ready'
    queue['rows'][1]['parent_delta_ref'] = 'delta_anchor_activation_trace_baseline'
    _write_yaml(queue_path, queue)

    calls: list[dict[str, Any]] = []

    def fake_run_row(**kwargs: Any) -> str:
        order = int(kwargs['queue_row']['order'])
        run_id = f'row_{order}_v1'
        kwargs['queue_row']['run_id'] = run_id
        calls.append(
            {
                'order': order,
                'anchor_run_id': kwargs['anchor_run_id'],
                'parent_run_id': kwargs['parent_run_id'],
            }
        )
        return run_id

    monkeypatch.setattr(execute_module, '_run_row', fake_run_row)
    monkeypatch.setattr(execute_module, '_sync_sweep_matrix', lambda **_: None)
    monkeypatch.setattr(execute_module, '_sync_active_aliases_if_active', lambda **_: None)

    executed = execute_sweep(
        sweep_id=sweep_id,
        prior_dump=Path('/tmp/prior.h5'),
        nanotabpfn_root=Path('/tmp/nanotabpfn'),
        device='cuda',
        fallback_python=REPO_ROOT / '.venv' / 'bin' / 'python',
        paths=paths,
    )

    assert executed == ['row_1_v1', 'row_2_v1']
    assert calls == [
        {
            'order': 1,
            'anchor_run_id': ANCHOR_RUN_ID,
            'parent_run_id': ANCHOR_RUN_ID,
        },
        {
            'order': 2,
            'anchor_run_id': ANCHOR_RUN_ID,
            'parent_run_id': 'row_1_v1',
        },
    ]


def test_resolve_parent_run_id_defaults_to_active_anchor() -> None:
    queue_rows = [
        {'order': 1, 'delta_ref': 'delta_anchor_activation_trace_baseline'},
        {'order': 2, 'delta_ref': 'delta_shared_feature_norm'},
    ]

    observed = runner_module._resolve_parent_run_id(
        queue_row=queue_rows[1],
        queue_rows=queue_rows,
        active_anchor='anchor_v1',
    )

    assert observed == 'anchor_v1'


def test_resolve_parent_run_id_prefers_latest_earlier_matching_row() -> None:
    queue_rows = [
        {'order': 1, 'delta_ref': 'delta_dup', 'run_id': 'dup_v1'},
        {'order': 2, 'delta_ref': 'delta_mid'},
        {'order': 3, 'delta_ref': 'delta_dup', 'run_id': 'dup_v3'},
        {'order': 4, 'delta_ref': 'delta_target', 'parent_delta_ref': 'delta_dup'},
    ]

    observed = runner_module._resolve_parent_run_id(
        queue_row=queue_rows[3],
        queue_rows=queue_rows,
        active_anchor='anchor_v1',
    )

    assert observed == 'dup_v3'


def test_resolve_parent_run_id_rejects_missing_parent_delta_ref_target() -> None:
    queue_rows = [
        {'order': 1, 'delta_ref': 'delta_anchor_activation_trace_baseline'},
        {'order': 2, 'delta_ref': 'delta_shared_feature_norm', 'parent_delta_ref': 'delta_missing'},
    ]

    with pytest.raises(RuntimeError, match='parent_delta_ref'):
        _ = runner_module._resolve_parent_run_id(
            queue_row=queue_rows[1],
            queue_rows=queue_rows,
            active_anchor='anchor_v1',
        )


def test_resolve_parent_run_id_rejects_self_reference() -> None:
    queue_rows = [
        {'order': 1, 'delta_ref': 'delta_anchor_activation_trace_baseline', 'parent_delta_ref': 'delta_anchor_activation_trace_baseline'},
    ]

    with pytest.raises(RuntimeError, match='not itself'):
        _ = runner_module._resolve_parent_run_id(
            queue_row=queue_rows[0],
            queue_rows=queue_rows,
            active_anchor='anchor_v1',
        )


def test_resolve_parent_run_id_rejects_forward_reference() -> None:
    queue_rows = [
        {'order': 1, 'delta_ref': 'delta_target', 'parent_delta_ref': 'delta_later'},
        {'order': 2, 'delta_ref': 'delta_later', 'run_id': 'later_v1'},
    ]

    with pytest.raises(RuntimeError, match='must reference an earlier row'):
        _ = runner_module._resolve_parent_run_id(
            queue_row=queue_rows[0],
            queue_rows=queue_rows,
            active_anchor='anchor_v1',
        )


def test_resolve_parent_run_id_rejects_parent_without_run_id() -> None:
    queue_rows = [
        {'order': 1, 'delta_ref': 'delta_anchor_activation_trace_baseline'},
        {'order': 2, 'delta_ref': 'delta_shared_feature_norm', 'parent_delta_ref': 'delta_anchor_activation_trace_baseline'},
    ]

    with pytest.raises(RuntimeError, match='does not have a completed run_id'):
        _ = runner_module._resolve_parent_run_id(
            queue_row=queue_rows[1],
            queue_rows=queue_rows,
            active_anchor='anchor_v1',
        )


def test_compose_cfg_sets_queue_aware_wandb_run_name(tmp_path: Path) -> None:
    run_dir = (
        tmp_path
        / 'outputs'
        / 'staged_ladder'
        / 'research'
        / 'cuda_capacity_pilot'
        / 'dpnb_cuda_large_anchor'
        / 'sd_cuda_capacity_pilot_01_dpnb_cuda_large_anchor_v1'
        / 'train'
    )

    cfg = _compose_cfg(
        row={'model': {'stage_label': 'dpnb_cuda_large_anchor'}},
        run_dir=run_dir,
        device='cuda',
    )

    assert str(cfg.runtime.output_dir) == str(run_dir.resolve())
    assert str(cfg.logging.run_name) == 'sd_cuda_capacity_pilot_01_dpnb_cuda_large_anchor_v1'
    assert str(cfg.model.stage_label) == 'dpnb_cuda_large_anchor'


def test_compose_cfg_replaces_module_overrides_to_allow_post_encoder_norm(tmp_path: Path) -> None:
    run_dir = tmp_path / 'outputs' / 'staged_ladder' / 'research' / 'cuda_stability_followup' / 'train'

    cfg = _compose_cfg(
        row={
            'model': {
                'module_overrides': {
                    'table_block_style': 'prenorm',
                    'allow_test_self_attention': False,
                    'row_pool': 'row_cls',
                    'post_encoder_norm': 'rmsnorm',
                }
            }
        },
        run_dir=run_dir,
        device='cuda',
    )

    assert cfg.model.module_overrides == {
        'table_block_style': 'prenorm',
        'allow_test_self_attention': False,
        'row_pool': 'row_cls',
        'post_encoder_norm': 'rmsnorm',
    }


def test_compose_cfg_uses_requested_training_experiment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, Any] = {}

    def fake_compose_config(args: list[str]) -> Any:
        captured['args'] = list(args)
        return OmegaConf.create(
            {
                'runtime': {'output_dir': '', 'device': ''},
                'logging': {'run_name': ''},
                'model': {'stage_label': 'base', 'module_overrides': {}},
            }
        )

    monkeypatch.setattr(runner_module, 'compose_config', fake_compose_config)

    _ = _compose_cfg(
        row={'model': {'stage_label': 'dpnb_architecture_screen_probe'}},
        run_dir=tmp_path / 'train',
        device='cuda',
        training_experiment='cls_benchmark_staged',
    )

    assert captured['args'] == ['experiment=cls_benchmark_staged']


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
    (tmp_path / 'telemetry.json').write_text(
        json.dumps(
            {
                'diagnostics': {
                    'stage_local_gradients': {
                        'modules': {
                            'column_encoder': {
                                'windows': {'final_10pct': {'mean_grad_norm': 0.42}}
                            },
                            'row_pool': {
                                'windows': {'final_10pct': {'mean_grad_norm': 0.55}}
                            },
                            'context_encoder': {
                                'windows': {'final_10pct': {'mean_grad_norm': 0.73}}
                            },
                        }
                    },
                    'activation_windows': {
                        'tracked_activations': {
                            'post_column_encoder': {
                                'early_to_final_mean_delta': 0.12,
                                'windows': {'final_10pct': {'mean': 1.3}},
                            },
                            'post_row_pool': {
                                'early_to_final_mean_delta': 0.18,
                                'windows': {'final_10pct': {'mean': 1.7}},
                            },
                            'post_context_encoder': {
                                'early_to_final_mean_delta': 0.24,
                                'windows': {'final_10pct': {'mean': 2.1}},
                            },
                        }
                    },
                }
            }
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
    assert queue_metrics['column_encoder_final_window_mean_grad_norm'] == pytest.approx(0.42)
    assert queue_metrics['row_pool_final_window_mean_grad_norm'] == pytest.approx(0.55)
    assert queue_metrics['context_encoder_final_window_mean_grad_norm'] == pytest.approx(0.73)
    assert queue_metrics['column_activation_early_to_final_mean_delta'] == pytest.approx(0.12)
    assert queue_metrics['context_activation_final_window_mean'] == pytest.approx(2.1)


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
            'column_encoder_final_window_mean_grad_norm': 0.42,
            'row_pool_final_window_mean_grad_norm': 0.55,
            'context_encoder_final_window_mean_grad_norm': 0.73,
            'column_activation_early_to_final_mean_delta': 0.12,
            'row_activation_early_to_final_mean_delta': 0.18,
            'context_activation_early_to_final_mean_delta': 0.24,
        },
        decision='defer',
        conclusion='Monitor log-loss deltas before promotion.',
    )

    assert '- Best log loss: `0.4010` at step `125`' in text
    assert '- Delta final log loss vs anchor: `-0.0110`' in text
    assert '- Final ROC AUC: `0.8040`' in text
    assert '## Stage-local stability' in text
    assert '- Column stage: final-window mean grad norm `0.4200`, activation early-to-final mean delta `+0.1200`' in text
    assert '- Context stage: final-window mean grad norm `0.7300`, activation early-to-final mean delta `+0.2400`' in text
    assert text.index('Best log loss') < text.index('Best ROC AUC')


def test_screen_metrics_reads_upper_block_summary_and_final_train_loss_ema(tmp_path: Path) -> None:
    (tmp_path / 'telemetry.json').write_text(
        json.dumps(
            {
                'diagnostics': {
                    'grad_clip': {'clipped_step_fraction': 0.125},
                    'activation_windows': {
                        'upper_transformer_blocks': {
                            'block_names': [
                                'post_transformer_block_8',
                                'post_transformer_block_9',
                                'post_transformer_block_10',
                                'post_transformer_block_11',
                            ],
                            'aggregate': {
                                'final_window_mean': 12.5,
                                'post_warmup_mean_slope': 0.03125,
                            },
                            'blocks': {
                                'post_transformer_block_8': {
                                    'final_window_mean': 10.0,
                                    'post_warmup_slope': 0.02,
                                },
                            },
                        }
                    },
                }
            }
        ),
        encoding='utf-8',
    )
    (tmp_path / 'train_history.jsonl').write_text(
        ''.join(
            [
                json.dumps({'step': 1, 'train_loss_ema': 0.7}) + '\n',
                json.dumps({'step': 2, 'train_loss_ema': 0.5}) + '\n',
            ]
        ),
        encoding='utf-8',
    )

    metrics = load_screen_metrics(run_dir=tmp_path)

    assert metrics['upper_block_names'] == [
        'post_transformer_block_8',
        'post_transformer_block_9',
        'post_transformer_block_10',
        'post_transformer_block_11',
    ]
    assert metrics['upper_block_final_window_mean'] == pytest.approx(12.5)
    assert metrics['upper_block_post_warmup_mean_slope'] == pytest.approx(0.03125)
    assert metrics['clipped_step_fraction'] == pytest.approx(0.125)
    assert metrics['final_train_loss_ema'] == pytest.approx(0.5)


def test_pick_screen_winner_prefers_rmsnorm_on_full_tie() -> None:
    resolution = pick_screen_winner(
        candidates=[
            {
                'order': 2,
                'value': 'rmsnorm',
                'screen_metrics': {
                    'upper_block_final_window_mean': 100.0,
                    'upper_block_post_warmup_mean_slope': 0.02,
                    'clipped_step_fraction': 0.1,
                    'final_train_loss_ema': 0.6,
                },
            },
            {
                'order': 3,
                'value': 'layernorm',
                'screen_metrics': {
                    'upper_block_final_window_mean': 100.0,
                    'upper_block_post_warmup_mean_slope': 0.02,
                    'clipped_step_fraction': 0.1,
                    'final_train_loss_ema': 0.6,
                },
            },
        ],
        tie_break_preference='rmsnorm',
    )

    assert resolution == {
        'winning_order': 2,
        'winning_value': 'rmsnorm',
        'reason': 'tie-break preference after upper-block, slope, clip-rate, and loss-ema ties',
    }


def test_run_row_screen_only_updates_queue_without_benchmark(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sweep_id = 'cuda_stack_scale_followup'
    delta_ref = 'dpnb_cuda_stack_scale_control'
    run_id = f'sd_{sweep_id}_01_{delta_ref}_v1'
    train_dir = (
        tmp_path
        / 'outputs'
        / 'staged_ladder'
        / 'research'
        / sweep_id
        / delta_ref
        / run_id
        / 'train'
    )
    (train_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (train_dir / 'train_history.jsonl').write_text('', encoding='utf-8')
    (train_dir / 'gradient_history.jsonl').write_text('', encoding='utf-8')
    (train_dir / 'telemetry.json').write_text('{}', encoding='utf-8')
    (train_dir / 'training_surface_record.json').write_text('{}', encoding='utf-8')
    (train_dir / 'checkpoints' / 'latest.pt').write_text('stub', encoding='utf-8')

    queue_row = {
        'order': 1,
        'delta_ref': delta_ref,
        'model': {'stage_label': delta_ref},
        'training': {},
        'execution_policy': 'screen_only',
        'notes': [],
    }
    materialized_row = {
        'delta_id': delta_ref,
        'dimension_family': 'training',
        'family': 'screening',
        'description': 'Train-only screen.',
        'anchor_delta': 'Keep the batch32 replay surface fixed.',
        'parameter_adequacy_plan': [],
        'adequacy_knobs': [],
        'model': {'stage_label': delta_ref},
        'notes': [],
    }
    queue = {'rows': [queue_row]}
    paths = ExecutionPaths(
        repo_root=tmp_path,
        index_path=tmp_path / 'reference' / 'system_delta_sweeps' / 'index.yaml',
        catalog_path=tmp_path / 'reference' / 'system_delta_catalog.yaml',
        sweeps_root=tmp_path / 'reference' / 'system_delta_sweeps',
        registry_path=REGISTRY_PATH,
        program_path=tmp_path / 'program.md',
        control_baseline_registry_path=REPO_ROOT / 'src' / 'tab_foundry' / 'bench' / 'control_baselines_v1.json',
    )

    captured_research_package: dict[str, Any] = {}

    def fake_write_research_package(**kwargs: Any) -> None:
        captured_research_package.update(kwargs)

    monkeypatch.setattr(runner_module, 'write_research_package', fake_write_research_package)
    monkeypatch.setattr(
        runner_module,
        'screen_metrics',
        lambda **_: {
            'upper_block_final_window_mean': 42.0,
            'upper_block_post_warmup_mean_slope': 0.01,
            'clipped_step_fraction': 0.0,
            'final_train_loss_ema': 0.5,
        },
    )
    monkeypatch.setattr(runner_module, 'run_nanotabpfn_benchmark', lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('benchmark should be skipped')))
    monkeypatch.setattr(runner_module, 'register_benchmark_run', lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('registry should be skipped')))

    observed_run_id = runner_module.run_row(
        sweep_id=sweep_id,
        sweep_meta={
            'control_baseline_id': 'cls_benchmark_linear_v2',
            'benchmark_bundle_path': 'bundle.json',
            'training_experiment': 'cls_benchmark_staged',
            'training_config_profile': 'cls_benchmark_staged',
            'surface_role': 'architecture_screen',
        },
        queue_row=queue_row,
        materialized_row=materialized_row,
        anchor_run_id='anchor_v1',
        parent_run_id='anchor_v1',
        queue=queue,
        prior_dump=Path('/tmp/prior.h5'),
        nanotabpfn_root=Path('/tmp/nanotabpfn'),
        device='cuda',
        fallback_python=REPO_ROOT / '.venv' / 'bin' / 'python',
        decision='defer',
        conclusion='Keep as a train-only screen.',
        paths=paths,
    )

    assert observed_run_id == run_id
    assert captured_research_package['training_experiment'] == 'cls_benchmark_staged'
    assert captured_research_package['training_config_profile'] == 'cls_benchmark_staged'
    assert captured_research_package['surface_role'] == 'architecture_screen'
    assert queue_row['status'] == 'screened'
    assert queue_row['interpretation_status'] == 'screened'
    assert queue_row['decision'] == 'defer'
    assert queue_row['benchmark_metrics'] is None
    assert queue_row['screen_metrics']['upper_block_final_window_mean'] == pytest.approx(42.0)


def test_completed_train_artifacts_exist_accepts_stage_scoped_latest_checkpoint(
    tmp_path: Path,
) -> None:
    run_dir = tmp_path / 'completed_train_run'
    (run_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (run_dir / 'train_history.jsonl').write_text('{}\n', encoding='utf-8')
    (run_dir / 'gradient_history.jsonl').write_text('{}\n', encoding='utf-8')
    (run_dir / 'telemetry.json').write_text('{}', encoding='utf-8')
    (run_dir / 'training_surface_record.json').write_text('{}', encoding='utf-8')
    (run_dir / 'checkpoints' / 'latest_stage1.pt').write_text('stub', encoding='utf-8')

    assert runner_module.completed_train_artifacts_exist(run_dir) is True


def test_run_row_legacy_sweep_meta_ignores_synthetic_anchor_context_experiment(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sweep_id = 'legacy_sweep'
    delta_ref = 'legacy_delta'
    run_id = f'sd_{sweep_id}_01_{delta_ref}_v1'
    train_dir = (
        tmp_path
        / 'outputs'
        / 'staged_ladder'
        / 'research'
        / sweep_id
        / delta_ref
        / run_id
        / 'train'
    )

    queue_row = {
        'order': 1,
        'delta_ref': delta_ref,
        'model': {},
        'training': {},
        'execution_policy': 'screen_only',
        'notes': [],
    }
    materialized_row = {
        'delta_id': delta_ref,
        'dimension_family': 'training',
        'family': 'schedule',
        'description': 'Legacy sweep execution should keep the hybrid default.',
        'anchor_delta': 'Retain the old implicit hybrid surface.',
        'parameter_adequacy_plan': [],
        'adequacy_knobs': [],
        'model': {},
        'notes': [],
    }
    queue = {'rows': [queue_row]}
    paths = ExecutionPaths(
        repo_root=tmp_path,
        index_path=tmp_path / 'reference' / 'system_delta_sweeps' / 'index.yaml',
        catalog_path=tmp_path / 'reference' / 'system_delta_catalog.yaml',
        sweeps_root=tmp_path / 'reference' / 'system_delta_sweeps',
        registry_path=REGISTRY_PATH,
        program_path=tmp_path / 'program.md',
        control_baseline_registry_path=REPO_ROOT / 'src' / 'tab_foundry' / 'bench' / 'control_baselines_v1.json',
    )

    captured_research_package: dict[str, Any] = {}
    captured_compose_cfg: dict[str, Any] = {}

    monkeypatch.setattr(runner_module, 'write_research_package', lambda **kwargs: captured_research_package.update(kwargs))

    def fake_compose_cfg(**kwargs: Any) -> Any:
        captured_compose_cfg.update(kwargs)
        return SimpleNamespace()

    monkeypatch.setattr(runner_module, 'compose_cfg', fake_compose_cfg)
    monkeypatch.setattr(
        runner_module,
        'train_tabfoundry_simple_prior',
        lambda *_args, **_kwargs: SimpleNamespace(output_dir=str(train_dir)),
    )
    monkeypatch.setattr(
        runner_module,
        'screen_metrics',
        lambda **_: {
            'upper_block_final_window_mean': 42.0,
            'upper_block_post_warmup_mean_slope': 0.01,
            'clipped_step_fraction': 0.0,
            'final_train_loss_ema': 0.5,
        },
    )
    monkeypatch.setattr(
        runner_module,
        'run_nanotabpfn_benchmark',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('benchmark should be skipped')),
    )
    monkeypatch.setattr(
        runner_module,
        'register_benchmark_run',
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError('registry should be skipped')),
    )
    monkeypatch.setattr(
        runner_module,
        'posthoc_update_wandb_summary',
        lambda **_kwargs: (_ for _ in ()).throw(AssertionError('wandb posthoc sync should be skipped')),
    )

    observed_run_id = runner_module.run_row(
        sweep_id=sweep_id,
        sweep_meta={
            'control_baseline_id': 'cls_benchmark_linear_v2',
            'benchmark_bundle_path': 'bundle.json',
            'anchor_context': {
                'experiment': 'stability_followup',
                'config_profile': 'stability_followup',
            },
        },
        queue_row=queue_row,
        materialized_row=materialized_row,
        anchor_run_id='anchor_v1',
        parent_run_id='anchor_v1',
        queue=queue,
        prior_dump=Path('/tmp/prior.h5'),
        nanotabpfn_root=Path('/tmp/nanotabpfn'),
        device='cuda',
        fallback_python=REPO_ROOT / '.venv' / 'bin' / 'python',
        decision='defer',
        conclusion='Legacy sweep should stay on the hybrid default.',
        paths=paths,
    )

    assert observed_run_id == run_id
    assert captured_compose_cfg['training_experiment'] == 'cls_benchmark_staged_prior'
    assert captured_research_package['training_experiment'] == 'cls_benchmark_staged_prior'
    assert captured_research_package['training_config_profile'] == 'cls_benchmark_staged_prior'
    assert captured_research_package['surface_role'] == 'hybrid_diagnostic'


def test_run_row_benchmark_full_uses_sweep_training_contract_for_registration(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sweep_id = 'architecture_screen_probe'
    delta_ref = 'dpnb_architecture_screen_probe'
    run_id = f'sd_{sweep_id}_01_{delta_ref}_v1'
    train_dir = (
        tmp_path
        / 'outputs'
        / 'staged_ladder'
        / 'research'
        / sweep_id
        / delta_ref
        / run_id
        / 'train'
    )
    benchmark_dir = train_dir.parent / 'benchmark'
    (train_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (train_dir / 'train_history.jsonl').write_text('', encoding='utf-8')
    (train_dir / 'gradient_history.jsonl').write_text('', encoding='utf-8')
    (train_dir / 'telemetry.json').write_text('{}', encoding='utf-8')
    (train_dir / 'training_surface_record.json').write_text('{}', encoding='utf-8')
    (train_dir / 'checkpoints' / 'latest.pt').write_text('stub', encoding='utf-8')
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    queue_row = {
        'order': 1,
        'delta_ref': delta_ref,
        'model': {'stage_label': delta_ref},
        'training': {},
        'execution_policy': 'benchmark_full',
        'notes': [],
    }
    materialized_row = {
        'delta_id': delta_ref,
        'dimension_family': 'model',
        'family': 'screening',
        'description': 'Benchmark the architecture-screen row.',
        'anchor_delta': 'Switch to the architecture-screen experiment surface.',
        'parameter_adequacy_plan': [],
        'adequacy_knobs': [],
        'model': {'stage_label': delta_ref},
        'notes': [],
    }
    queue = {'rows': [queue_row]}
    paths = ExecutionPaths(
        repo_root=tmp_path,
        index_path=tmp_path / 'reference' / 'system_delta_sweeps' / 'index.yaml',
        catalog_path=tmp_path / 'reference' / 'system_delta_catalog.yaml',
        sweeps_root=tmp_path / 'reference' / 'system_delta_sweeps',
        registry_path=REGISTRY_PATH,
        program_path=tmp_path / 'program.md',
        control_baseline_registry_path=REPO_ROOT / 'src' / 'tab_foundry' / 'bench' / 'control_baselines_v1.json',
    )

    captured_registration: dict[str, Any] = {}
    captured_posthoc: dict[str, Any] = {}

    monkeypatch.setattr(runner_module, 'write_research_package', lambda **_: None)
    monkeypatch.setattr(runner_module, 'ensure_nanotabpfn_python', lambda **_: tmp_path / 'python')
    monkeypatch.setattr(
        runner_module,
        'run_nanotabpfn_benchmark',
        lambda *_args, **_kwargs: {
            'tab_foundry': {
                'best_step': 100.0,
                'best_log_loss': 0.41,
                'final_log_loss': 0.40,
                'best_to_final_log_loss_delta': -0.01,
                'best_brier_score': 0.12,
                'final_brier_score': 0.11,
                'best_to_final_brier_score_delta': -0.01,
                'best_roc_auc': 0.80,
                'final_roc_auc': 0.81,
                'best_to_final_roc_auc_delta': 0.01,
                'training_diagnostics': {'max_grad_norm': 7.5},
            },
            'nanotabpfn': {
                'best_log_loss': 0.46,
                'final_log_loss': 0.45,
                'best_brier_score': 0.14,
                'final_brier_score': 0.13,
                'best_roc_auc': 0.78,
                'final_roc_auc': 0.77,
            },
        },
    )
    monkeypatch.setattr(
        runner_module,
        'queue_metrics',
        lambda *_args, **_kwargs: {
            'best_step': 100,
            'final_log_loss': 0.40,
            'delta_final_log_loss': -0.02,
            'column_encoder_final_window_mean_grad_norm': 0.42,
            'row_pool_final_window_mean_grad_norm': 0.55,
            'context_encoder_final_window_mean_grad_norm': 0.73,
            'column_activation_early_to_final_mean_delta': 0.12,
            'row_activation_early_to_final_mean_delta': 0.08,
            'context_activation_early_to_final_mean_delta': 0.15,
            'column_activation_final_window_mean': 1.8,
            'row_activation_final_window_mean': 1.9,
            'context_activation_final_window_mean': 2.1,
        },
    )
    monkeypatch.setattr(
        runner_module,
        'posthoc_update_wandb_summary',
        lambda *, telemetry_path, payload: captured_posthoc.update(
            {'telemetry_path': telemetry_path, 'payload': payload}
        )
        or True,
    )

    def fake_register_benchmark_run(**kwargs: Any) -> dict[str, Any]:
        captured_registration.update(kwargs)
        return {
            'run': {
                'sweep': {
                    'sweep_id': sweep_id,
                    'delta_id': delta_ref,
                    'parent_sweep_id': 'input_norm_followup',
                    'queue_order': 1,
                    'run_kind': 'primary',
                },
                'comparisons': {
                    'vs_anchor': {
                        'reference_run_id': 'anchor_v1',
                        'final_log_loss_delta': -0.02,
                        'final_brier_score_delta': -0.01,
                        'final_roc_auc_delta': 0.01,
                    },
                    'vs_parent': {
                        'reference_run_id': 'anchor_v1',
                        'final_log_loss_delta': -0.02,
                    }
                }
            }
        }

    monkeypatch.setattr(runner_module, 'register_benchmark_run', fake_register_benchmark_run)

    observed_run_id = runner_module.run_row(
        sweep_id=sweep_id,
        sweep_meta={
            'control_baseline_id': 'cls_benchmark_linear_v2',
            'benchmark_bundle_path': 'bundle.json',
            'training_experiment': 'cls_benchmark_staged',
            'training_config_profile': 'cls_benchmark_staged',
            'surface_role': 'architecture_screen',
            'parent_sweep_id': 'input_norm_followup',
        },
        queue_row=queue_row,
        materialized_row=materialized_row,
        anchor_run_id='anchor_v1',
        parent_run_id='anchor_v1',
        queue=queue,
        prior_dump=Path('/tmp/prior.h5'),
        nanotabpfn_root=Path('/tmp/nanotabpfn'),
        device='cuda',
        fallback_python=REPO_ROOT / '.venv' / 'bin' / 'python',
        decision='defer',
        conclusion='Use the architecture-screen surface for benchmark-facing evidence.',
        paths=paths,
    )

    assert observed_run_id == run_id
    assert captured_registration['experiment'] == 'cls_benchmark_staged'
    assert captured_registration['config_profile'] == 'cls_benchmark_staged'
    assert captured_registration['sweep_id'] == sweep_id
    assert captured_registration['parent_sweep_id'] == 'input_norm_followup'
    assert captured_posthoc['telemetry_path'] == train_dir / 'telemetry.json'
    assert captured_posthoc['payload']['sweep'] == {
        'sweep_id': sweep_id,
        'delta_id': delta_ref,
        'parent_sweep_id': 'input_norm_followup',
        'queue_order': 1,
        'run_kind': 'primary',
    }
    assert captured_posthoc['payload']['comparison']['vs_anchor']['final_log_loss_delta'] == pytest.approx(-0.02)
    assert captured_posthoc['payload']['comparison']['vs_parent']['reference_run_id'] == 'anchor_v1'
    assert captured_posthoc['payload']['comparison']['stage_local_stability']['column'] == {
        'final_window_mean_grad_norm': 0.42,
        'activation_early_to_final_mean_delta': 0.12,
        'activation_final_window_mean': 1.8,
    }
    assert captured_posthoc['payload']['comparison']['stage_local_stability']['context']['activation_final_window_mean'] == pytest.approx(2.1)
    assert queue_row['status'] == 'completed'
    assert queue_row['interpretation_status'] == 'completed'


def test_run_row_benchmark_full_reuses_anchor_curve_without_bootstrapping_nanotabpfn_env(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sweep_id = 'tokenization_migration_v1'
    delta_ref = 'delta_architecture_screen_grouped_tokens'
    run_id = f'sd_{sweep_id}_01_{delta_ref}_v1'
    train_dir = (
        tmp_path
        / 'outputs'
        / 'staged_ladder'
        / 'research'
        / sweep_id
        / delta_ref
        / run_id
        / 'train'
    )
    benchmark_dir = train_dir.parent / 'benchmark'
    (train_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (train_dir / 'train_history.jsonl').write_text('', encoding='utf-8')
    (train_dir / 'gradient_history.jsonl').write_text('{}\n', encoding='utf-8')
    (train_dir / 'telemetry.json').write_text('{}', encoding='utf-8')
    (train_dir / 'training_surface_record.json').write_text('{}', encoding='utf-8')
    (train_dir / 'checkpoints' / 'latest.pt').write_text('stub', encoding='utf-8')
    benchmark_dir.mkdir(parents=True, exist_ok=True)

    bundle_path = tmp_path / 'bundle.json'
    bundle_path.write_text('{}\n', encoding='utf-8')
    nanotab_root = tmp_path / 'nano'
    nanotab_python = nanotab_root / '.venv' / 'bin' / 'python'
    prior_dump = nanotab_root / '300k_150x5_2.h5'
    monkeypatch.setattr(runner_module, 'resolve_device', lambda _device: 'cuda')
    monkeypatch.setattr(
        runner_module,
        'benchmark_host_fingerprint',
        lambda: 'runner-host',
    )

    anchor_curve_path = tmp_path / 'anchor_curve.jsonl'
    anchor_curve_path.write_text('{}\n', encoding='utf-8')
    anchor_summary_path = tmp_path / 'anchor_summary.json'
    _write_compare_summary(
        anchor_summary_path,
        bundle_path=bundle_path,
        curve_path=anchor_curve_path,
        nanotab_root=nanotab_root,
        nanotab_python=nanotab_python,
        control_baseline_id='cls_benchmark_linear_v2',
        device='auto',
        resolved_device='cuda',
        host_fingerprint='runner-host',
        prior_dump_path=prior_dump,
        steps=runner_module.DEFAULT_NANOTABPFN_STEPS,
        eval_every=runner_module.DEFAULT_NANOTABPFN_EVAL_EVERY,
        seeds=runner_module.DEFAULT_NANOTABPFN_SEEDS,
        batch_size=runner_module.DEFAULT_NANOTABPFN_BATCH_SIZE,
        lr=runner_module.DEFAULT_NANOTABPFN_LR,
    )
    registry_path = tmp_path / 'benchmark_run_registry.json'
    registry_path.write_text(
        json.dumps(
            {
                'runs': {
                    'anchor_v1': {
                        'artifacts': {
                            'comparison_summary_path': str(anchor_summary_path.resolve()),
                        }
                    }
                }
            },
            indent=2,
            sort_keys=True,
        )
        + '\n',
        encoding='utf-8',
    )
    control_baseline_registry_path = tmp_path / 'control_baselines.json'
    control_baseline_registry_path.write_text(
        json.dumps({'baselines': {}}, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )

    queue_row = {
        'order': 1,
        'delta_ref': delta_ref,
        'model': {'stage_label': delta_ref},
        'training': {},
        'execution_policy': 'benchmark_full',
        'notes': [],
    }
    materialized_row = {
        'delta_id': delta_ref,
        'dimension_family': 'model',
        'family': 'tokenization',
        'description': 'Benchmark the grouped-token row.',
        'anchor_delta': 'Switch only the tokenizer stage on the shared surface.',
        'parameter_adequacy_plan': [],
        'adequacy_knobs': [],
        'model': {'stage_label': delta_ref},
        'notes': [],
    }
    queue = {'rows': [queue_row]}
    paths = ExecutionPaths(
        repo_root=tmp_path,
        index_path=tmp_path / 'reference' / 'system_delta_sweeps' / 'index.yaml',
        catalog_path=tmp_path / 'reference' / 'system_delta_catalog.yaml',
        sweeps_root=tmp_path / 'reference' / 'system_delta_sweeps',
        registry_path=registry_path,
        program_path=tmp_path / 'program.md',
        control_baseline_registry_path=control_baseline_registry_path,
    )

    captured_benchmark_config: dict[str, Any] = {}

    monkeypatch.setattr(runner_module, 'write_research_package', lambda **_: None)
    monkeypatch.setattr(
        runner_module,
        'ensure_nanotabpfn_python',
        lambda **_: (_ for _ in ()).throw(AssertionError('unexpected nanotabpfn bootstrap')),
    )

    def fake_benchmark(config: Any) -> dict[str, Any]:
        captured_benchmark_config['reuse_nanotabpfn_curve_path'] = config.reuse_nanotabpfn_curve_path
        captured_benchmark_config['reuse_nanotabpfn_metadata'] = config.reuse_nanotabpfn_metadata
        return {
            'tab_foundry': {
                'best_step': 100.0,
                'best_log_loss': 0.41,
                'final_log_loss': 0.40,
                'best_to_final_log_loss_delta': -0.01,
                'best_brier_score': 0.12,
                'final_brier_score': 0.11,
                'best_to_final_brier_score_delta': -0.01,
                'best_roc_auc': 0.80,
                'final_roc_auc': 0.81,
                'best_to_final_roc_auc_delta': 0.01,
                'training_diagnostics': {'max_grad_norm': 7.5},
            },
            'nanotabpfn': {
                'best_log_loss': 0.46,
                'final_log_loss': 0.45,
                'best_brier_score': 0.14,
                'final_brier_score': 0.13,
                'best_roc_auc': 0.78,
                'final_roc_auc': 0.77,
                'curve_source_mode': 'reused',
                'reused_curve_path': str(anchor_curve_path.resolve()),
            },
        }

    monkeypatch.setattr(runner_module, 'run_nanotabpfn_benchmark', fake_benchmark)
    monkeypatch.setattr(
        runner_module,
        'register_benchmark_run',
        lambda **_kwargs: {
            'run': {
                'comparisons': {
                    'vs_anchor': {
                        'final_log_loss_delta': -0.02,
                        'final_brier_score_delta': -0.01,
                        'final_roc_auc_delta': 0.01,
                    }
                }
            }
        },
    )

    observed_run_id = runner_module.run_row(
        sweep_id=sweep_id,
        sweep_meta={
            'control_baseline_id': 'cls_benchmark_linear_v2',
            'benchmark_bundle_path': str(bundle_path.resolve()),
            'training_experiment': 'cls_benchmark_staged',
            'training_config_profile': 'cls_benchmark_staged',
            'surface_role': 'architecture_screen',
        },
        queue_row=queue_row,
        materialized_row=materialized_row,
        anchor_run_id='anchor_v1',
        parent_run_id='anchor_v1',
        queue=queue,
        prior_dump=prior_dump,
        nanotabpfn_root=nanotab_root,
        device='cuda',
        fallback_python=REPO_ROOT / '.venv' / 'bin' / 'python',
        decision='defer',
        conclusion='Reuse the locked anchor control curve for the Tier-1 check.',
        paths=paths,
    )

    assert observed_run_id == run_id
    assert captured_benchmark_config['reuse_nanotabpfn_curve_path'] == anchor_curve_path.resolve()
    assert captured_benchmark_config['reuse_nanotabpfn_metadata'] == {
        'root': str(nanotab_root.resolve()),
        'python': str(nanotab_python.resolve()),
        'device': 'auto',
        'resolved_device': 'cuda',
        'benchmark_host_fingerprint': 'runner-host',
        'prior_dump_path': str(prior_dump.resolve()),
        'num_seeds': runner_module.DEFAULT_NANOTABPFN_SEEDS,
        'steps': runner_module.DEFAULT_NANOTABPFN_STEPS,
        'eval_every': runner_module.DEFAULT_NANOTABPFN_EVAL_EVERY,
        'batch_size': runner_module.DEFAULT_NANOTABPFN_BATCH_SIZE,
        'lr': runner_module.DEFAULT_NANOTABPFN_LR,
    }
    result_card = (
        tmp_path
        / 'outputs'
        / 'staged_ladder'
        / 'research'
        / sweep_id
        / delta_ref
        / 'result_card.md'
    ).read_text(encoding='utf-8')
    assert "nanoTabPFN curve source: `reused`" in result_card
    assert str(anchor_curve_path.resolve()) in result_card


def test_resolve_reusable_nanotabpfn_curve_falls_back_to_control_baseline_when_anchor_is_unavailable(
    tmp_path: Path,
) -> None:
    bundle_path = tmp_path / 'bundle.json'
    bundle_path.write_text('{}\n', encoding='utf-8')
    nanotab_root = tmp_path / 'nano'
    nanotab_python = nanotab_root / '.venv' / 'bin' / 'python'
    nanotab_python.parent.mkdir(parents=True, exist_ok=True)
    nanotab_python.write_text('#!/usr/bin/env bash\nexit 0\n', encoding='utf-8')
    nanotab_python.chmod(0o755)
    prior_dump = nanotab_root / '300k_150x5_2.h5'
    prior_dump.write_bytes(b'prior')

    registry_path = tmp_path / 'benchmark_run_registry.json'
    registry_path.write_text(
        json.dumps(
            {
                'runs': {
                    'anchor_v1': {
                        'artifacts': {
                            'comparison_summary_path': str((tmp_path / 'missing_anchor_summary.json').resolve()),
                        }
                    }
                }
            },
            indent=2,
            sort_keys=True,
        )
        + '\n',
        encoding='utf-8',
    )

    baseline_curve_path = tmp_path / 'baseline_curve.jsonl'
    baseline_curve_path.write_text('{}\n', encoding='utf-8')
    baseline_summary_path = tmp_path / 'baseline_summary.json'
    _write_compare_summary(
        baseline_summary_path,
        bundle_path=bundle_path,
        curve_path=baseline_curve_path,
        nanotab_root=nanotab_root,
        nanotab_python=nanotab_python,
        control_baseline_id=None,
        device='cuda',
        resolved_device='cuda',
        host_fingerprint=runner_module.benchmark_host_fingerprint(),
        prior_dump_path=prior_dump,
        steps=runner_module.DEFAULT_NANOTABPFN_STEPS,
        eval_every=runner_module.DEFAULT_NANOTABPFN_EVAL_EVERY,
        seeds=runner_module.DEFAULT_NANOTABPFN_SEEDS,
        batch_size=runner_module.DEFAULT_NANOTABPFN_BATCH_SIZE,
        lr=runner_module.DEFAULT_NANOTABPFN_LR,
    )
    control_baseline_registry_path = tmp_path / 'control_baselines.json'
    control_baseline_registry_path.write_text(
        json.dumps(
            {
                'baselines': {
                    'cls_benchmark_linear_v2': {
                        'comparison_summary_path': str(baseline_summary_path.resolve()),
                    }
                }
            },
            indent=2,
            sort_keys=True,
        )
        + '\n',
        encoding='utf-8',
    )

    paths = ExecutionPaths(
        repo_root=tmp_path,
        index_path=tmp_path / 'reference' / 'system_delta_sweeps' / 'index.yaml',
        catalog_path=tmp_path / 'reference' / 'system_delta_catalog.yaml',
        sweeps_root=tmp_path / 'reference' / 'system_delta_sweeps',
        registry_path=registry_path,
        program_path=tmp_path / 'program.md',
        control_baseline_registry_path=control_baseline_registry_path,
    )

    selection = runner_module.resolve_reusable_nanotabpfn_curve(
        sweep_meta={
            'control_baseline_id': 'cls_benchmark_linear_v2',
            'benchmark_bundle_path': str(bundle_path.resolve()),
        },
        anchor_run_id='anchor_v1',
        nanotabpfn_root=nanotab_root,
        prior_dump=prior_dump,
        requested_device='cuda',
        paths=paths,
    )

    assert selection is not None
    assert selection.curve_path == baseline_curve_path.resolve()
    assert selection.source_label == 'control baseline'


def test_resolve_reusable_nanotabpfn_curve_requires_resolved_device_match_even_when_requested_device_matches(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle_path = tmp_path / 'bundle.json'
    bundle_path.write_text('{}\n', encoding='utf-8')
    nanotab_root = tmp_path / 'nano'
    nanotab_python = nanotab_root / '.venv' / 'bin' / 'python'
    nanotab_python.parent.mkdir(parents=True, exist_ok=True)
    nanotab_python.write_text('#!/usr/bin/env bash\nexit 0\n', encoding='utf-8')
    nanotab_python.chmod(0o755)
    prior_dump = nanotab_root / '300k_150x5_2.h5'
    prior_dump.write_bytes(b'prior')

    monkeypatch.setattr(runner_module, 'resolve_device', lambda _device: 'cuda')
    monkeypatch.setattr(
        runner_module,
        'benchmark_host_fingerprint',
        lambda: 'runner-host',
    )
    anchor_curve_path = tmp_path / 'anchor_curve.jsonl'
    anchor_curve_path.write_text('{}\n', encoding='utf-8')
    anchor_summary_path = tmp_path / 'anchor_summary.json'
    _write_compare_summary(
        anchor_summary_path,
        bundle_path=bundle_path,
        curve_path=anchor_curve_path,
        nanotab_root=nanotab_root,
        nanotab_python=nanotab_python,
        control_baseline_id='cls_benchmark_linear_v2',
        device='auto',
        resolved_device='cpu',
        host_fingerprint='runner-host',
        prior_dump_path=prior_dump,
        steps=runner_module.DEFAULT_NANOTABPFN_STEPS,
        eval_every=runner_module.DEFAULT_NANOTABPFN_EVAL_EVERY,
        seeds=runner_module.DEFAULT_NANOTABPFN_SEEDS,
        batch_size=runner_module.DEFAULT_NANOTABPFN_BATCH_SIZE,
        lr=runner_module.DEFAULT_NANOTABPFN_LR,
    )
    registry_path = tmp_path / 'benchmark_run_registry.json'
    registry_path.write_text(
        json.dumps(
            {
                'runs': {
                    'anchor_v1': {
                        'artifacts': {
                            'comparison_summary_path': str(anchor_summary_path.resolve()),
                        }
                    }
                }
            },
            indent=2,
            sort_keys=True,
        )
        + '\n',
        encoding='utf-8',
    )
    control_baseline_registry_path = tmp_path / 'control_baselines.json'
    control_baseline_registry_path.write_text(
        json.dumps({'baselines': {}}, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    paths = ExecutionPaths(
        repo_root=tmp_path,
        index_path=tmp_path / 'reference' / 'system_delta_sweeps' / 'index.yaml',
        catalog_path=tmp_path / 'reference' / 'system_delta_catalog.yaml',
        sweeps_root=tmp_path / 'reference' / 'system_delta_sweeps',
        registry_path=registry_path,
        program_path=tmp_path / 'program.md',
        control_baseline_registry_path=control_baseline_registry_path,
    )

    selection = runner_module.resolve_reusable_nanotabpfn_curve(
        sweep_meta={
            'control_baseline_id': 'cls_benchmark_linear_v2',
            'benchmark_bundle_path': str(bundle_path.resolve()),
        },
        anchor_run_id='anchor_v1',
        nanotabpfn_root=nanotab_root,
        prior_dump=prior_dump,
        requested_device='auto',
        paths=paths,
    )

    assert selection is None


def test_resolve_reusable_nanotabpfn_curve_requires_host_fingerprint_match(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle_path = tmp_path / 'bundle.json'
    bundle_path.write_text('{}\n', encoding='utf-8')
    nanotab_root = tmp_path / 'nano'
    nanotab_python = nanotab_root / '.venv' / 'bin' / 'python'
    nanotab_python.parent.mkdir(parents=True, exist_ok=True)
    nanotab_python.write_text('#!/usr/bin/env bash\nexit 0\n', encoding='utf-8')
    nanotab_python.chmod(0o755)
    prior_dump = nanotab_root / '300k_150x5_2.h5'
    prior_dump.write_bytes(b'prior')
    monkeypatch.setattr(runner_module, 'resolve_device', lambda _device: 'cuda')
    monkeypatch.setattr(
        runner_module,
        'benchmark_host_fingerprint',
        lambda: 'runner-host',
    )

    anchor_curve_path = tmp_path / 'anchor_curve.jsonl'
    anchor_curve_path.write_text('{}\n', encoding='utf-8')
    anchor_summary_path = tmp_path / 'anchor_summary.json'
    _write_compare_summary(
        anchor_summary_path,
        bundle_path=bundle_path,
        curve_path=anchor_curve_path,
        nanotab_root=nanotab_root,
        nanotab_python=nanotab_python,
        control_baseline_id='cls_benchmark_linear_v2',
        device='cuda',
        resolved_device='cuda',
        host_fingerprint='other-host',
        prior_dump_path=prior_dump,
        steps=runner_module.DEFAULT_NANOTABPFN_STEPS,
        eval_every=runner_module.DEFAULT_NANOTABPFN_EVAL_EVERY,
        seeds=runner_module.DEFAULT_NANOTABPFN_SEEDS,
        batch_size=runner_module.DEFAULT_NANOTABPFN_BATCH_SIZE,
        lr=runner_module.DEFAULT_NANOTABPFN_LR,
    )
    registry_path = tmp_path / 'benchmark_run_registry.json'
    registry_path.write_text(
        json.dumps(
            {
                'runs': {
                    'anchor_v1': {
                        'artifacts': {
                            'comparison_summary_path': str(anchor_summary_path.resolve()),
                        }
                    }
                }
            },
            indent=2,
            sort_keys=True,
        )
        + '\n',
        encoding='utf-8',
    )
    control_baseline_registry_path = tmp_path / 'control_baselines.json'
    control_baseline_registry_path.write_text(
        json.dumps({'baselines': {}}, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    paths = ExecutionPaths(
        repo_root=tmp_path,
        index_path=tmp_path / 'reference' / 'system_delta_sweeps' / 'index.yaml',
        catalog_path=tmp_path / 'reference' / 'system_delta_catalog.yaml',
        sweeps_root=tmp_path / 'reference' / 'system_delta_sweeps',
        registry_path=registry_path,
        program_path=tmp_path / 'program.md',
        control_baseline_registry_path=control_baseline_registry_path,
    )

    selection = runner_module.resolve_reusable_nanotabpfn_curve(
        sweep_meta={
            'control_baseline_id': 'cls_benchmark_linear_v2',
            'benchmark_bundle_path': str(bundle_path.resolve()),
        },
        anchor_run_id='anchor_v1',
        nanotabpfn_root=nanotab_root,
        prior_dump=prior_dump,
        requested_device='cuda',
        paths=paths,
    )

    assert selection is None


def test_resolve_reusable_nanotabpfn_curve_rejects_legacy_summary_without_timing_metadata(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    bundle_path = tmp_path / 'bundle.json'
    bundle_path.write_text('{}\n', encoding='utf-8')
    nanotab_root = tmp_path / 'nano'
    nanotab_python = nanotab_root / '.venv' / 'bin' / 'python'
    nanotab_python.parent.mkdir(parents=True, exist_ok=True)
    nanotab_python.write_text('#!/usr/bin/env bash\nexit 0\n', encoding='utf-8')
    nanotab_python.chmod(0o755)
    prior_dump = nanotab_root / '300k_150x5_2.h5'
    prior_dump.write_bytes(b'prior')
    monkeypatch.setattr(runner_module, 'resolve_device', lambda _device: 'cuda')
    monkeypatch.setattr(
        runner_module,
        'benchmark_host_fingerprint',
        lambda: 'runner-host',
    )

    anchor_curve_path = tmp_path / 'anchor_curve.jsonl'
    anchor_curve_path.write_text('{}\n', encoding='utf-8')
    anchor_summary_path = tmp_path / 'anchor_summary.json'
    _write_compare_summary(
        anchor_summary_path,
        bundle_path=bundle_path,
        curve_path=anchor_curve_path,
        nanotab_root=nanotab_root,
        nanotab_python=nanotab_python,
        control_baseline_id='cls_benchmark_linear_v2',
        device='cuda',
        prior_dump_path=prior_dump,
        steps=runner_module.DEFAULT_NANOTABPFN_STEPS,
        eval_every=runner_module.DEFAULT_NANOTABPFN_EVAL_EVERY,
        seeds=runner_module.DEFAULT_NANOTABPFN_SEEDS,
        batch_size=runner_module.DEFAULT_NANOTABPFN_BATCH_SIZE,
        lr=runner_module.DEFAULT_NANOTABPFN_LR,
    )
    payload = json.loads(anchor_summary_path.read_text(encoding='utf-8'))
    nanotabpfn = payload['nanotabpfn']
    nanotabpfn.pop('resolved_device', None)
    nanotabpfn.pop('benchmark_host_fingerprint', None)
    anchor_summary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + '\n', encoding='utf-8')

    registry_path = tmp_path / 'benchmark_run_registry.json'
    registry_path.write_text(
        json.dumps(
            {
                'runs': {
                    'anchor_v1': {
                        'artifacts': {
                            'comparison_summary_path': str(anchor_summary_path.resolve()),
                        }
                    }
                }
            },
            indent=2,
            sort_keys=True,
        )
        + '\n',
        encoding='utf-8',
    )
    control_baseline_registry_path = tmp_path / 'control_baselines.json'
    control_baseline_registry_path.write_text(
        json.dumps({'baselines': {}}, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    paths = ExecutionPaths(
        repo_root=tmp_path,
        index_path=tmp_path / 'reference' / 'system_delta_sweeps' / 'index.yaml',
        catalog_path=tmp_path / 'reference' / 'system_delta_catalog.yaml',
        sweeps_root=tmp_path / 'reference' / 'system_delta_sweeps',
        registry_path=registry_path,
        program_path=tmp_path / 'program.md',
        control_baseline_registry_path=control_baseline_registry_path,
    )

    selection = runner_module.resolve_reusable_nanotabpfn_curve(
        sweep_meta={
            'control_baseline_id': 'cls_benchmark_linear_v2',
            'benchmark_bundle_path': str(bundle_path.resolve()),
        },
        anchor_run_id='anchor_v1',
        nanotabpfn_root=nanotab_root,
        prior_dump=prior_dump,
        requested_device='cuda',
        paths=paths,
    )

    assert selection is None


def test_resolve_reusable_nanotabpfn_curve_skips_missing_summary_or_curve(
    tmp_path: Path,
) -> None:
    bundle_path = tmp_path / 'bundle.json'
    bundle_path.write_text('{}\n', encoding='utf-8')
    nanotab_root = tmp_path / 'nano'
    nanotab_python = nanotab_root / '.venv' / 'bin' / 'python'
    nanotab_python.parent.mkdir(parents=True, exist_ok=True)
    nanotab_python.write_text('#!/usr/bin/env bash\nexit 0\n', encoding='utf-8')
    nanotab_python.chmod(0o755)
    prior_dump = nanotab_root / '300k_150x5_2.h5'
    prior_dump.write_bytes(b'prior')

    anchor_summary_path = tmp_path / 'anchor_summary.json'
    _write_compare_summary(
        anchor_summary_path,
        bundle_path=bundle_path,
        curve_path=None,
        nanotab_root=nanotab_root,
        nanotab_python=nanotab_python,
        control_baseline_id='cls_benchmark_linear_v2',
        device='cuda',
        resolved_device='cuda',
        host_fingerprint=runner_module.benchmark_host_fingerprint(),
        prior_dump_path=prior_dump,
        steps=runner_module.DEFAULT_NANOTABPFN_STEPS,
        eval_every=runner_module.DEFAULT_NANOTABPFN_EVAL_EVERY,
        seeds=runner_module.DEFAULT_NANOTABPFN_SEEDS,
        batch_size=runner_module.DEFAULT_NANOTABPFN_BATCH_SIZE,
        lr=runner_module.DEFAULT_NANOTABPFN_LR,
    )
    registry_path = tmp_path / 'benchmark_run_registry.json'
    registry_path.write_text(
        json.dumps(
            {
                'runs': {
                    'anchor_v1': {
                        'artifacts': {
                            'comparison_summary_path': str(anchor_summary_path.resolve()),
                        }
                    }
                }
            },
            indent=2,
            sort_keys=True,
        )
        + '\n',
        encoding='utf-8',
    )
    control_baseline_registry_path = tmp_path / 'control_baselines.json'
    control_baseline_registry_path.write_text(
        json.dumps({'baselines': {}}, indent=2, sort_keys=True) + '\n',
        encoding='utf-8',
    )
    paths = ExecutionPaths(
        repo_root=tmp_path,
        index_path=tmp_path / 'reference' / 'system_delta_sweeps' / 'index.yaml',
        catalog_path=tmp_path / 'reference' / 'system_delta_catalog.yaml',
        sweeps_root=tmp_path / 'reference' / 'system_delta_sweeps',
        registry_path=registry_path,
        program_path=tmp_path / 'program.md',
        control_baseline_registry_path=control_baseline_registry_path,
    )

    selection = runner_module.resolve_reusable_nanotabpfn_curve(
        sweep_meta={
            'control_baseline_id': 'cls_benchmark_linear_v2',
            'benchmark_bundle_path': str(bundle_path.resolve()),
        },
        anchor_run_id='anchor_v1',
        nanotabpfn_root=nanotab_root,
        prior_dump=prior_dump,
        requested_device='cuda',
        paths=paths,
    )

    assert selection is None


def test_write_research_package_uses_resolved_lane_contract_fields(tmp_path: Path) -> None:
    delta_root = tmp_path / "outputs" / "staged_ladder" / "research" / "legacy_sweep" / "delta_example"

    write_research_package(
        delta_root=delta_root,
        materialized_row={
            "delta_id": "delta_example",
            "dimension_family": "training",
            "family": "schedule",
            "description": "Exercise the resolved research-package contract.",
            "anchor_delta": "Keep the existing anchor unchanged.",
            "parameter_adequacy_plan": [],
            "rationale": "Legacy sweep metadata should still write a complete package.",
            "hypothesis": "The fallback lane contract should be written explicitly.",
        },
        queue_row={"model": {}, "data": {}, "preprocessing": {}, "training": {}},
        sweep_meta={
            "control_baseline_id": "cls_benchmark_linear_v2",
            "benchmark_bundle_path": "src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
        },
        sweep_id="legacy_sweep",
        anchor_run_id="anchor_v1",
        device="cuda",
        training_experiment="cls_benchmark_staged_prior",
        training_config_profile="cls_benchmark_staged_prior",
        surface_role="hybrid_diagnostic",
    )

    research_card = (delta_root / "research_card.md").read_text(encoding="utf-8")
    campaign = read_artifact_yaml(delta_root / "campaign.yaml")

    assert "`training_experiment`: `cls_benchmark_staged_prior`" in research_card
    assert "`training_config_profile`: `cls_benchmark_staged_prior`" in research_card
    assert "`surface_role`: `hybrid_diagnostic`" in research_card
    assert campaign["training_experiment"] == "cls_benchmark_staged_prior"
    assert campaign["training_config_profile"] == "cls_benchmark_staged_prior"
    assert campaign["surface_role"] == "hybrid_diagnostic"


def test_run_row_resolves_dynamic_post_stack_norm_from_screened_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sweep_id = 'cuda_stack_scale_followup'
    delta_ref = 'dpnb_cuda_stack_scale_depth_scaled_plus_norm_winner'
    run_id = f'sd_{sweep_id}_05_{delta_ref}_v1'
    train_dir = (
        tmp_path
        / 'outputs'
        / 'staged_ladder'
        / 'research'
        / sweep_id
        / delta_ref
        / run_id
        / 'train'
    )
    (train_dir / 'checkpoints').mkdir(parents=True, exist_ok=True)
    (train_dir / 'train_history.jsonl').write_text('', encoding='utf-8')
    (train_dir / 'gradient_history.jsonl').write_text('', encoding='utf-8')
    (train_dir / 'telemetry.json').write_text('{}', encoding='utf-8')
    (train_dir / 'training_surface_record.json').write_text('{}', encoding='utf-8')
    (train_dir / 'checkpoints' / 'latest.pt').write_text('stub', encoding='utf-8')

    row2 = {
        'order': 2,
        'status': 'screened',
        'screen_metrics': {
            'upper_block_final_window_mean': 90.0,
            'upper_block_post_warmup_mean_slope': 0.01,
            'clipped_step_fraction': 0.1,
            'final_train_loss_ema': 0.5,
        },
    }
    row3 = {
        'order': 3,
        'status': 'screened',
        'screen_metrics': {
            'upper_block_final_window_mean': 100.0,
            'upper_block_post_warmup_mean_slope': 0.02,
            'clipped_step_fraction': 0.2,
            'final_train_loss_ema': 0.6,
        },
    }
    queue_row = {
        'order': 5,
        'delta_ref': delta_ref,
        'model': {
            'stage_label': delta_ref,
            'module_overrides': {
                'table_block_style': 'prenorm',
                'table_block_residual_scale': 'depth_scaled',
            },
        },
        'training': {},
        'execution_policy': 'screen_only',
        'dynamic_model_overrides': {
            'post_stack_norm': {
                'kind': 'screen_winner',
                'compare_orders': [
                    {'order': 2, 'value': 'rmsnorm'},
                    {'order': 3, 'value': 'layernorm'},
                ],
                'tie_break_preference': 'rmsnorm',
            }
        },
        'notes': [],
    }
    queue = {'rows': [row2, row3, queue_row]}
    materialized_row = {
        'delta_id': delta_ref,
        'dimension_family': 'model',
        'family': 'normalization',
        'description': 'Combine depth scaling with the winning post-stack norm.',
        'anchor_delta': 'Keep the batch32 replay surface fixed.',
        'parameter_adequacy_plan': [],
        'adequacy_knobs': [],
        'model': {
            'stage_label': delta_ref,
            'module_overrides': {
                'table_block_style': 'prenorm',
                'table_block_residual_scale': 'depth_scaled',
            },
        },
        'notes': [],
    }
    paths = ExecutionPaths(
        repo_root=tmp_path,
        index_path=tmp_path / 'reference' / 'system_delta_sweeps' / 'index.yaml',
        catalog_path=tmp_path / 'reference' / 'system_delta_catalog.yaml',
        sweeps_root=tmp_path / 'reference' / 'system_delta_sweeps',
        registry_path=REGISTRY_PATH,
        program_path=tmp_path / 'program.md',
        control_baseline_registry_path=REPO_ROOT / 'src' / 'tab_foundry' / 'bench' / 'control_baselines_v1.json',
    )

    monkeypatch.setattr(runner_module, 'write_research_package', lambda **_: None)
    monkeypatch.setattr(
        runner_module,
        'screen_metrics',
        lambda **_: {
            'upper_block_final_window_mean': 80.0,
            'upper_block_post_warmup_mean_slope': 0.009,
            'clipped_step_fraction': 0.0,
            'final_train_loss_ema': 0.4,
        },
    )

    _ = runner_module.run_row(
        sweep_id=sweep_id,
        sweep_meta={'control_baseline_id': 'cls_benchmark_linear_v2', 'benchmark_bundle_path': 'bundle.json'},
        queue_row=queue_row,
        materialized_row=materialized_row,
        anchor_run_id='anchor_v1',
        parent_run_id='anchor_v1',
        queue=queue,
        prior_dump=Path('/tmp/prior.h5'),
        nanotabpfn_root=Path('/tmp/nanotabpfn'),
        device='cuda',
        fallback_python=REPO_ROOT / '.venv' / 'bin' / 'python',
        decision='defer',
        conclusion='Carry the winning norm into the combined row.',
        paths=paths,
    )

    assert queue_row['model']['module_overrides']['post_stack_norm'] == 'rmsnorm'
    assert materialized_row['model']['module_overrides']['post_stack_norm'] == 'rmsnorm'
    assert queue_row['dynamic_model_overrides']['post_stack_norm']['resolved_value'] == 'rmsnorm'
    assert queue_row['dynamic_model_overrides']['post_stack_norm']['resolved_from_order'] == 2
