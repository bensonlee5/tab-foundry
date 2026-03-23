"""Sweep-aware library helpers for the anchor-only system-delta workflow."""

from __future__ import annotations

from . import anchor as _anchor
from . import catalog as _catalog
from . import manage as _manage
from . import materialize as _materialize
from . import matrix as _matrix
from . import paths_io as _paths_io
from . import validation as _validation


LEGACY_PRIOR_CONSTANT_LR_LABEL = _anchor.LEGACY_PRIOR_CONSTANT_LR_LABEL
UNAVAILABLE_TRAINING_LABEL = _anchor.UNAVAILABLE_TRAINING_LABEL
CATALOG_SCHEMA = _catalog.CATALOG_SCHEMA
SWEEP_INDEX_SCHEMA = _catalog.SWEEP_INDEX_SCHEMA
SWEEP_QUEUE_SCHEMA = _catalog.SWEEP_QUEUE_SCHEMA
SWEEP_SCHEMA = _catalog.SWEEP_SCHEMA
DEFAULT_SWEEP_STATUS = _manage.DEFAULT_SWEEP_STATUS
MATERIALIZED_QUEUE_SCHEMA = _materialize.MATERIALIZED_QUEUE_SCHEMA

load_system_delta_catalog = _catalog.load_system_delta_catalog
load_system_delta_index = _catalog.load_system_delta_index
load_system_delta_queue_instance = _catalog.load_system_delta_queue_instance
load_system_delta_sweep = _catalog.load_system_delta_sweep
resolve_selected_sweep_id = _catalog.resolve_selected_sweep_id
create_sweep = _manage.create_sweep
instantiate_queue_row = _manage.instantiate_queue_row
list_sweeps = _manage.list_sweeps
set_active_sweep = _manage.set_active_sweep
sync_active_sweep_aliases = _manage.sync_active_sweep_aliases
evaluate_applicability_guard = _materialize.evaluate_applicability_guard
guarded_initial_state = _materialize.guarded_initial_state
load_system_delta_queue = _materialize.load_system_delta_queue
materialize_row = _materialize.materialize_row
materialize_system_delta_queue = _materialize.materialize_system_delta_queue
next_ready_row = _materialize.next_ready_row
ordered_rows = _materialize.ordered_rows
metric_summary = _matrix.metric_summary
render_and_write_system_delta_matrix = _matrix.render_and_write_system_delta_matrix
render_model_change_payload = _matrix.render_model_change_payload
render_system_delta_matrix = _matrix.render_system_delta_matrix
result_card_path = _matrix.result_card_path
validate_system_delta_queue = _matrix.validate_system_delta_queue
default_catalog_path = _paths_io.default_catalog_path
default_matrix_path = _paths_io.default_matrix_path
default_queue_path = _paths_io.default_queue_path
default_registry_path = _paths_io.default_registry_path
default_sweep_index_path = _paths_io.default_sweep_index_path
default_sweeps_root = _paths_io.default_sweeps_root
repo_root = _paths_io.repo_root
sweep_dir = _paths_io.sweep_dir
sweep_matrix_path = _paths_io.sweep_matrix_path
sweep_metadata_path = _paths_io.sweep_metadata_path
sweep_queue_path = _paths_io.sweep_queue_path
ensure_mapping = _validation.ensure_mapping
ensure_non_empty_string = _validation.ensure_non_empty_string
ensure_rows = _validation.ensure_rows
ensure_string_list = _validation.ensure_string_list
validate_prose_fields = _validation.validate_prose_fields

_copy_jsonable = _paths_io._copy_jsonable
_load_yaml_mapping = _paths_io._load_yaml_mapping
_render_path = _paths_io._render_path
_write_text = _paths_io._write_text
_write_yaml = _paths_io._write_yaml
_ensure_non_empty_string = _validation.ensure_non_empty_string
_ensure_mapping = _validation.ensure_mapping
_ensure_rows = _validation.ensure_rows
_ensure_string_list = _validation.ensure_string_list
_validate_prose_fields = _validation.validate_prose_fields
_resolve_selected_sweep_id = _catalog.resolve_selected_sweep_id
_staged_module_selection_from_run_model = _anchor.staged_module_selection_from_run_model
_anchor_context_from_registry_run = _anchor.anchor_context_from_registry_run
_surface_label_from_anchor_context = _anchor.surface_label_from_anchor_context
_anchor_training_surface_label = _anchor.anchor_training_surface_label
_anchor_module_selection = _anchor.anchor_module_selection
_module_choice = _anchor.module_choice
_describe_feature_encoder = _anchor.describe_feature_encoder
_describe_target_conditioner = _anchor.describe_target_conditioner
_describe_table_block = _anchor.describe_table_block
_describe_tokenizer = _anchor.describe_tokenizer
_describe_column_encoder = _anchor.describe_column_encoder
_describe_row_pool = _anchor.describe_row_pool
_describe_context_encoder = _anchor.describe_context_encoder
_describe_head = _anchor.describe_head
_build_anchor_surface = _anchor.build_anchor_surface
_evaluate_applicability_guard = _materialize.evaluate_applicability_guard
_guarded_initial_state = _materialize.guarded_initial_state
_materialize_row = _materialize.materialize_row
_render_model_change_payload = _matrix.render_model_change_payload
_metric_summary = _matrix.metric_summary
_result_card_path = _matrix.result_card_path
_instantiate_queue_row = _manage.instantiate_queue_row
