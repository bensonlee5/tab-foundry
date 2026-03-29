"""Compatibility wrapper for dagzoo handoff helpers."""

from __future__ import annotations

from tab_realdata_hub.dagzoo_handoff import (
    DAGZOO_HANDOFF_SCHEMA_NAME,
    DAGZOO_HANDOFF_SCHEMA_VERSION,
    DagzooGeneratedIdentityAccumulator,
    DagzooHandoffInfo,
    is_canonical_dagzoo_id,
    load_dagzoo_handoff_info,
    stable_dagzoo_generated_corpus_id,
    verify_dagzoo_handoff_matches_generated_corpus,
)

__all__ = [
    "DAGZOO_HANDOFF_SCHEMA_NAME",
    "DAGZOO_HANDOFF_SCHEMA_VERSION",
    "DagzooGeneratedIdentityAccumulator",
    "DagzooHandoffInfo",
    "is_canonical_dagzoo_id",
    "load_dagzoo_handoff_info",
    "stable_dagzoo_generated_corpus_id",
    "verify_dagzoo_handoff_matches_generated_corpus",
]
