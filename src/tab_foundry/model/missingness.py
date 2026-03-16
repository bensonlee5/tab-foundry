"""Shared missingness-mode helpers."""

from __future__ import annotations

from typing import Any


SUPPORTED_MISSINGNESS_MODES = ("none", "explicit_token", "feature_mask")


def normalize_missingness_mode(value: Any, *, context: str) -> str:
    """Normalize one public missingness mode."""

    normalized = str(value).strip().lower()
    if normalized not in SUPPORTED_MISSINGNESS_MODES:
        raise ValueError(
            f"{context} must be one of {SUPPORTED_MISSINGNESS_MODES}, got {value!r}"
        )
    return normalized


def missingness_mode_requires_raw_missing_values(missingness_mode: Any) -> bool:
    """Whether the model must receive raw missing values for this mode."""

    return normalize_missingness_mode(
        missingness_mode,
        context="missingness_mode",
    ) != "none"


def validate_missingness_runtime_policy(
    *,
    missingness_mode: Any,
    allow_missing_values: bool | None = None,
    impute_missing: bool | None = None,
    context: str,
) -> None:
    """Validate that runtime preprocessing preserves raw missingness when required."""

    normalized_mode = normalize_missingness_mode(
        missingness_mode,
        context=f"{context}.missingness_mode",
    )
    if normalized_mode == "none":
        return

    if allow_missing_values is not None and not bool(allow_missing_values):
        raise ValueError(
            f"{context} requires data.allow_missing_values=True when "
            f"model.missingness_mode={normalized_mode!r}"
        )
    if impute_missing is not None and bool(impute_missing):
        raise ValueError(
            f"{context} requires preprocessing.impute_missing=False when "
            f"model.missingness_mode={normalized_mode!r}"
        )
