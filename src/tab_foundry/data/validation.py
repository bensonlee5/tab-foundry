"""Compatibility wrapper for shared numeric dataset validation helpers."""

from __future__ import annotations

from tab_realdata_hub.validation import (
    MISSING_VALUE_STATUS_CLEAN,
    MISSING_VALUE_STATUS_CONTAINS_NAN_OR_INF,
    SUPPORTED_MISSING_VALUE_POLICIES,
    assert_no_non_finite_values,
    contains_non_finite_values,
    missing_value_status,
)

__all__ = [
    "MISSING_VALUE_STATUS_CLEAN",
    "MISSING_VALUE_STATUS_CONTAINS_NAN_OR_INF",
    "SUPPORTED_MISSING_VALUE_POLICIES",
    "assert_no_non_finite_values",
    "contains_non_finite_values",
    "missing_value_status",
]
