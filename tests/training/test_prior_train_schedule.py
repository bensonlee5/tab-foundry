from __future__ import annotations

# ruff: noqa: F401

from tests.support.prior_train_cases import (
    test_train_tabfoundry_simple_prior_applies_linear_decay_schedule,
    test_train_tabfoundry_simple_prior_applies_linear_warmup_decay_schedule,
    test_train_tabfoundry_simple_prior_keeps_constant_lr_when_schedule_is_disabled,
    test_train_tabfoundry_simple_prior_rejects_mismatched_schedule_steps,
    test_train_tabfoundry_simple_prior_scales_lr_with_prior_dump_batch_size,
)
