from __future__ import annotations

# ruff: noqa: F401

from tests.support.prior_train_cases import (
    test_resolve_prior_training_device_name_rejects_auto_when_it_resolves_to_mps,
    test_resolve_prior_training_device_name_rejects_explicit_mps,
    test_train_tabfoundry_simple_prior_rejects_mps_runtime_device,
)
