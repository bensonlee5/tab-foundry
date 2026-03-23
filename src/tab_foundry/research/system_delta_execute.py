"""Compatibility facade for the sweep-native execute entrypoint."""

from __future__ import annotations

import tab_foundry.research.sweep.execute as _execute


DEFAULT_BUDGET_CLASS = _execute.DEFAULT_BUDGET_CLASS
DEFAULT_CONFIG_PROFILE = _execute.DEFAULT_CONFIG_PROFILE
DEFAULT_CONCLUSION = _execute.DEFAULT_CONCLUSION
DEFAULT_DECISION = _execute.DEFAULT_DECISION
DEFAULT_DEVICE = _execute.DEFAULT_DEVICE
DEFAULT_EXPERIMENT = _execute.DEFAULT_EXPERIMENT
DEFAULT_NANOTABPFN_ROOT = _execute.DEFAULT_NANOTABPFN_ROOT
DEFAULT_PRIOR_DUMP = _execute.DEFAULT_PRIOR_DUMP
DEFAULT_TRACK = _execute.DEFAULT_TRACK
ExecutionPaths = _execute.ExecutionPaths
_ALLOWED_DECISIONS = _execute._ALLOWED_DECISIONS
_absolute_path_without_resolving_symlinks = _execute._absolute_path_without_resolving_symlinks
_apply_mapping = _execute._apply_mapping
_compose_cfg = _execute._compose_cfg
_completed_train_artifacts_exist = _execute._completed_train_artifacts_exist
_ensure_nanotabpfn_python = _execute._ensure_nanotabpfn_python
_materialized_row_map = _execute._materialized_row_map
_optional_metric = _execute._optional_metric
_queue_metrics = _execute._queue_metrics
_read_yaml = _execute._read_yaml
_result_card_text = _execute._result_card_text
_row_id_for_order = _execute._row_id_for_order
_run_row = _execute._run_row
_sync_active_aliases_if_active = _execute._sync_active_aliases_if_active
_sync_sweep_matrix = _execute._sync_sweep_matrix
_update_queue_row = _execute._update_queue_row
_write_research_package = _execute._write_research_package
_write_yaml = _execute._write_yaml
build_parser = _execute.build_parser
configure_parser = _execute.configure_parser
execute_sweep = _execute.execute_sweep
main = _execute.main
parse_order_overrides = _execute.parse_order_overrides
promote_anchor = _execute.promote_anchor
run_from_args = _execute.run_from_args
select_queue_rows = _execute.select_queue_rows
