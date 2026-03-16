"""Resolved preprocessing surface settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

from .state import (
    CLASSIFICATION_LABEL_MAPPING_TRAIN_ONLY_REMAP,
    DTYPE_POLICY,
    FEATURE_ORDER_POLICY_POSITIONAL,
    UNSEEN_TEST_LABEL_POLICY_FILTER,
)


SUPPORTED_LABEL_MAPPINGS = (CLASSIFICATION_LABEL_MAPPING_TRAIN_ONLY_REMAP,)
SUPPORTED_UNSEEN_TEST_LABEL_POLICIES = (UNSEEN_TEST_LABEL_POLICY_FILTER,)


def _mapping_from_any(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"preprocessing.overrides must be a mapping or null, got {value!r}")
    return {str(key): item for key, item in value.items()}


@dataclass(slots=True, frozen=True)
class PreprocessingSurfaceConfig:
    surface_label: str
    impute_missing: bool
    all_nan_fill: float
    label_mapping: str
    unseen_test_label_policy: str
    feature_order_policy: str
    dtype_policy: dict[str, str]
    overrides: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


def resolve_preprocessing_surface(
    preprocessing_cfg: Mapping[str, Any] | None,
) -> PreprocessingSurfaceConfig:
    """Resolve one runtime preprocessing surface with additive overrides."""

    if preprocessing_cfg is None:
        raise ValueError("preprocessing config is required")
    cfg = {str(key): value for key, value in preprocessing_cfg.items()}
    overrides = _mapping_from_any(cfg.get("overrides"))
    raw_impute_missing = overrides.get("impute_missing")
    if raw_impute_missing is None:
        raw_impute_missing = cfg.get("impute_missing")
    if raw_impute_missing is None:
        raise ValueError("preprocessing.impute_missing must be explicitly configured")
    impute_missing = bool(raw_impute_missing)
    raw_all_nan_fill = overrides.get("all_nan_fill")
    if raw_all_nan_fill is None:
        raw_all_nan_fill = cfg.get("all_nan_fill")
    if raw_all_nan_fill is None:
        raise ValueError("preprocessing.all_nan_fill must be explicitly configured")
    all_nan_fill = float(raw_all_nan_fill)
    label_mapping_raw = overrides.get("label_mapping")
    if label_mapping_raw is None:
        label_mapping_raw = cfg.get("label_mapping")
    if label_mapping_raw is None:
        raise ValueError("preprocessing.label_mapping must be explicitly configured")
    label_mapping = str(label_mapping_raw).strip()
    if label_mapping not in SUPPORTED_LABEL_MAPPINGS:
        raise ValueError(
            f"preprocessing.label_mapping must be one of {SUPPORTED_LABEL_MAPPINGS}, got {label_mapping!r}"
        )
    unseen_test_label_policy_raw = overrides.get("unseen_test_label_policy")
    if unseen_test_label_policy_raw is None:
        unseen_test_label_policy_raw = cfg.get("unseen_test_label_policy")
    if unseen_test_label_policy_raw is None:
        raise ValueError("preprocessing.unseen_test_label_policy must be explicitly configured")
    unseen_test_label_policy = str(unseen_test_label_policy_raw).strip()
    if unseen_test_label_policy not in SUPPORTED_UNSEEN_TEST_LABEL_POLICIES:
        raise ValueError(
            "preprocessing.unseen_test_label_policy must be one of "
            f"{SUPPORTED_UNSEEN_TEST_LABEL_POLICIES}, got {unseen_test_label_policy!r}"
        )
    surface_label_raw = cfg.get("surface_label")
    if surface_label_raw is None:
        surface_label_raw = overrides.get("surface_label")
    if surface_label_raw is None or not str(surface_label_raw).strip():
        raise ValueError("preprocessing.surface_label must be explicitly configured")
    surface_label = str(surface_label_raw).strip()
    return PreprocessingSurfaceConfig(
        surface_label=surface_label,
        impute_missing=impute_missing,
        all_nan_fill=all_nan_fill,
        label_mapping=label_mapping,
        unseen_test_label_policy=unseen_test_label_policy,
        feature_order_policy=FEATURE_ORDER_POLICY_POSITIONAL,
        dtype_policy=dict(DTYPE_POLICY),
        overrides=overrides,
    )
