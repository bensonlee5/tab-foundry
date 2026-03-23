"""Compatibility facade for the sweep-native promote entrypoint."""

from __future__ import annotations

import tab_foundry.research.sweep.promote as _promote

PromotionPaths = _promote.PromotionPaths
_OBJECTIVE_RE = _promote._OBJECTIVE_RE
_read_yaml = _promote._read_yaml
_render_sweep_matrix = _promote._render_sweep_matrix
_replace_prefixed_line = _promote._replace_prefixed_line
_update_program_contract = _promote._update_program_contract
build_parser = _promote.build_parser
configure_parser = _promote.configure_parser
main = _promote.main
promote_anchor = _promote.promote_anchor
resolve_run_id_for_order = _promote.resolve_run_id_for_order
run_from_args = _promote.run_from_args
system_delta = _promote.sweep_core
