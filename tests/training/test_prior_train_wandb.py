from __future__ import annotations

# ruff: noqa: F401

from tests.support.prior_train_cases import (
    test_resolve_prior_wandb_run_name_prefers_queue_run_id_from_output_dir,
    test_train_tabfoundry_simple_prior_closes_wandb_for_setup_failures,
    test_train_tabfoundry_simple_prior_logs_wandb_failure_summary,
    test_train_tabfoundry_simple_prior_logs_wandb_metrics_and_summary,
)
