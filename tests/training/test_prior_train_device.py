from __future__ import annotations

# ruff: noqa: F401

from tests.support.prior_train_cases import (
    test_resolve_prior_training_device_name_falls_back_for_multilayer_row_cls_on_mps,
    test_resolve_prior_training_device_name_keeps_mps_for_single_layer_row_cls,
    test_resolve_prior_training_device_name_keeps_mps_for_target_column,
    test_train_tabfoundry_staged_prior_falls_back_to_cpu_for_multilayer_row_cls_on_mps,
)
