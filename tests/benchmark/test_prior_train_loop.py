from __future__ import annotations

import functools
import inspect

from . import prior_train_cases as cases


def _export(names: list[str]) -> None:
    for name in names:
        case = getattr(cases, name)

        def _wrapped(*args, __case=case, **kwargs):
            return __case(*args, **kwargs)

        functools.update_wrapper(_wrapped, case)
        _wrapped.__name__ = name
        _wrapped.__qualname__ = name
        _wrapped.__signature__ = inspect.signature(case)
        globals()[name] = _wrapped


_export(
    [
        "test_train_tabfoundry_simple_prior_averages_task_loss_and_steps_per_batch",
        "test_train_tabfoundry_simple_prior_saves_checkpoints_in_eval_mode",
        "test_train_tabfoundry_simple_prior_matches_nanotabpfn_loss_for_one_batch",
        "test_train_tabfoundry_staged_prior_writes_staged_gradient_keys",
        "test_train_tabfoundry_staged_prior_writes_activation_norms_when_enabled",
        "test_train_tabfoundry_staged_prior_writes_context_gradient_keys_when_active",
        "test_train_tabfoundry_simple_prior_skip_policy_preserves_successful_step_budget",
        "test_train_tabfoundry_simple_prior_accepts_plain_adamw",
        "test_evaluate_tab_foundry_run_supports_runs_without_best_checkpoint",
        "test_tabfoundry_staged_nano_exact_matches_simple_prior_batch_loss",
    ]
)
