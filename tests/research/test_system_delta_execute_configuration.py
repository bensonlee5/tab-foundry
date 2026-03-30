from __future__ import annotations

# ruff: noqa: F401

from tests.support_research.system_delta_execute import (
    test_compose_cfg_keeps_explicit_allow_missing_values_with_corpus_ref,
    test_compose_cfg_replaces_module_overrides_to_allow_post_encoder_norm,
    test_compose_cfg_resolves_sweep_local_corpus_from_nondefault_sweeps_root,
    test_compose_cfg_routes_data_corpus_ref_into_surface_overrides,
    test_compose_cfg_sets_queue_aware_wandb_run_name_and_sweep_group,
    test_compose_cfg_uses_requested_training_experiment,
    test_resolve_parent_run_id_defaults_to_active_anchor,
    test_resolve_parent_run_id_prefers_latest_earlier_matching_row,
    test_resolve_parent_run_id_rejects_forward_reference,
    test_resolve_parent_run_id_rejects_missing_parent_delta_ref_target,
    test_resolve_parent_run_id_rejects_parent_without_run_id,
    test_resolve_parent_run_id_rejects_self_reference,
)
