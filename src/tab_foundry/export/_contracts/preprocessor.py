"""Preprocessor-state validation for export bundles."""

from __future__ import annotations

from typing import Any

from tab_foundry.preprocessing import (
    CLASSIFICATION_LABEL_MAPPING_TRAIN_ONLY_REMAP,
    DTYPE_POLICY,
    FEATURE_ORDER_POLICY_POSITIONAL,
    MISSING_VALUE_STRATEGY_TRAIN_MEAN,
    UNSEEN_TEST_LABEL_POLICY_FILTER,
)

from .common import _as_float, _as_str, _require_keys
from .models import (
    EXPECTED_MISSING_VALUE_ALL_NAN_FILL,
    EXPECTED_V2_FEATURE_ORDER_POLICY,
    SCHEMA_VERSION_V2,
    SCHEMA_VERSION_V3,
    ExportClassificationLabelPolicy,
    ExportMissingValuePolicy,
    ExportPreprocessorState,
    LegacyPreprocessorState,
)


def _validate_dtype_policy(dtype_policy: Any) -> dict[str, str]:
    if not isinstance(dtype_policy, dict):
        raise ValueError("preprocessor_state.dtype_policy must be object")
    _require_keys(
        dtype_policy,
        keys=set(DTYPE_POLICY.keys()),
        context="preprocessor_state.dtype_policy",
    )
    normalized: dict[str, str] = {}
    for key, expected in DTYPE_POLICY.items():
        actual = _as_str(dtype_policy[key], context=f"preprocessor_state.dtype_policy.{key}")
        if actual != expected:
            raise ValueError(
                f"preprocessor_state.dtype_policy.{key} must equal {expected!r}, got {actual!r}"
            )
        normalized[key] = actual
    return normalized


def _validate_v2_preprocessor_state(payload: dict[str, Any]) -> LegacyPreprocessorState:
    _require_keys(
        payload,
        keys={
            "feature_order_policy",
            "missing_value_policy",
            "classification_label_policy",
            "dtype_policy",
        },
        context="preprocessor_state",
    )

    feature_order_policy = _as_str(
        payload["feature_order_policy"],
        context="preprocessor_state.feature_order_policy",
    )
    if feature_order_policy != EXPECTED_V2_FEATURE_ORDER_POLICY:
        raise ValueError(
            "preprocessor_state.feature_order_policy must equal "
            f"{EXPECTED_V2_FEATURE_ORDER_POLICY!r}"
        )

    missing_value_policy = payload["missing_value_policy"]
    if not isinstance(missing_value_policy, dict):
        raise ValueError("preprocessor_state.missing_value_policy must be object")
    _require_keys(
        missing_value_policy,
        keys={"strategy", "all_nan_fill"},
        context="preprocessor_state.missing_value_policy",
    )
    strategy = _as_str(
        missing_value_policy["strategy"],
        context="preprocessor_state.missing_value_policy.strategy",
    )
    if strategy != MISSING_VALUE_STRATEGY_TRAIN_MEAN:
        raise ValueError(
            "preprocessor_state.missing_value_policy.strategy must equal "
            f"{MISSING_VALUE_STRATEGY_TRAIN_MEAN!r}"
        )
    all_nan_fill = _as_float(
        missing_value_policy["all_nan_fill"],
        context="preprocessor_state.missing_value_policy.all_nan_fill",
    )
    if all_nan_fill != EXPECTED_MISSING_VALUE_ALL_NAN_FILL:
        raise ValueError(
            "preprocessor_state.missing_value_policy.all_nan_fill must equal "
            f"{EXPECTED_MISSING_VALUE_ALL_NAN_FILL}"
        )

    classification_label_policy = payload["classification_label_policy"]
    if not isinstance(classification_label_policy, dict):
        raise ValueError("preprocessor_state.classification_label_policy must be object")
    _require_keys(
        classification_label_policy,
        keys={"mapping", "unseen_test_label"},
        context="preprocessor_state.classification_label_policy",
    )
    mapping = _as_str(
        classification_label_policy["mapping"],
        context="preprocessor_state.classification_label_policy.mapping",
    )
    if mapping != CLASSIFICATION_LABEL_MAPPING_TRAIN_ONLY_REMAP:
        raise ValueError(
            "preprocessor_state.classification_label_policy.mapping must be "
            f"{CLASSIFICATION_LABEL_MAPPING_TRAIN_ONLY_REMAP!r}"
        )
    unseen_test_label = _as_str(
        classification_label_policy["unseen_test_label"],
        context="preprocessor_state.classification_label_policy.unseen_test_label",
    )
    if unseen_test_label != UNSEEN_TEST_LABEL_POLICY_FILTER:
        raise ValueError(
            "preprocessor_state.classification_label_policy.unseen_test_label must equal "
            f"{UNSEEN_TEST_LABEL_POLICY_FILTER!r}"
        )

    return LegacyPreprocessorState(
        feature_order_policy=feature_order_policy,
        missing_value_policy={
            "strategy": strategy,
            "all_nan_fill": all_nan_fill,
        },
        classification_label_policy={
            "mapping": mapping,
            "unseen_test_label": unseen_test_label,
        },
        dtype_policy=_validate_dtype_policy(payload["dtype_policy"]),
    )


def _validate_v3_missing_value_policy(payload: Any) -> ExportMissingValuePolicy:
    if not isinstance(payload, dict):
        raise ValueError("preprocessor_state.missing_value_policy must be object")
    _require_keys(
        payload,
        keys={"strategy", "all_nan_fill"},
        context="preprocessor_state.missing_value_policy",
    )
    strategy = _as_str(
        payload["strategy"],
        context="preprocessor_state.missing_value_policy.strategy",
    )
    if strategy != MISSING_VALUE_STRATEGY_TRAIN_MEAN:
        raise ValueError(
            "preprocessor_state.missing_value_policy.strategy must equal "
            f"{MISSING_VALUE_STRATEGY_TRAIN_MEAN!r}"
        )
    all_nan_fill = _as_float(
        payload["all_nan_fill"],
        context="preprocessor_state.missing_value_policy.all_nan_fill",
    )
    if all_nan_fill != EXPECTED_MISSING_VALUE_ALL_NAN_FILL:
        raise ValueError(
            "preprocessor_state.missing_value_policy.all_nan_fill must equal "
            f"{EXPECTED_MISSING_VALUE_ALL_NAN_FILL}"
        )
    return ExportMissingValuePolicy(strategy=strategy, all_nan_fill=all_nan_fill)


def _validate_v3_classification_label_policy(
    payload: Any,
    *,
    task: str,
) -> ExportClassificationLabelPolicy | None:
    if task != "classification":
        raise ValueError(f"Unsupported preprocessor_state task: {task!r}")
    if not isinstance(payload, dict):
        raise ValueError("preprocessor_state.classification_label_policy must be object for classification")
    _require_keys(
        payload,
        keys={"mapping", "unseen_test_label"},
        context="preprocessor_state.classification_label_policy",
    )
    mapping = _as_str(
        payload["mapping"],
        context="preprocessor_state.classification_label_policy.mapping",
    )
    if mapping != CLASSIFICATION_LABEL_MAPPING_TRAIN_ONLY_REMAP:
        raise ValueError(
            "preprocessor_state.classification_label_policy.mapping must equal "
            f"{CLASSIFICATION_LABEL_MAPPING_TRAIN_ONLY_REMAP!r}"
        )
    unseen_test_label = _as_str(
        payload["unseen_test_label"],
        context="preprocessor_state.classification_label_policy.unseen_test_label",
    )
    if unseen_test_label != UNSEEN_TEST_LABEL_POLICY_FILTER:
        raise ValueError(
            "preprocessor_state.classification_label_policy.unseen_test_label must equal "
            f"{UNSEEN_TEST_LABEL_POLICY_FILTER!r}"
        )
    return ExportClassificationLabelPolicy(
        mapping=mapping,
        unseen_test_label=unseen_test_label,
    )


def _validate_v3_preprocessor_state(
    payload: dict[str, Any],
    *,
    task: str,
) -> ExportPreprocessorState:
    _require_keys(
        payload,
        keys={
            "feature_order_policy",
            "missing_value_policy",
            "classification_label_policy",
            "dtype_policy",
        },
        context="preprocessor_state",
    )

    feature_order_policy = _as_str(
        payload["feature_order_policy"],
        context="preprocessor_state.feature_order_policy",
    )
    if feature_order_policy != FEATURE_ORDER_POLICY_POSITIONAL:
        raise ValueError(
            "preprocessor_state.feature_order_policy must equal "
            f"{FEATURE_ORDER_POLICY_POSITIONAL!r}"
        )
    return ExportPreprocessorState(
        feature_order_policy=feature_order_policy,
        missing_value_policy=_validate_v3_missing_value_policy(payload["missing_value_policy"]),
        classification_label_policy=_validate_v3_classification_label_policy(
            payload["classification_label_policy"],
            task=task,
        ),
        dtype_policy=_validate_dtype_policy(payload["dtype_policy"]),
    )


def validate_preprocessor_state_dict(
    payload: dict[str, Any],
    *,
    schema_version: str = SCHEMA_VERSION_V2,
    task: str = "classification",
) -> LegacyPreprocessorState | ExportPreprocessorState:
    if schema_version == SCHEMA_VERSION_V2:
        return _validate_v2_preprocessor_state(payload)
    if schema_version == SCHEMA_VERSION_V3:
        return _validate_v3_preprocessor_state(payload, task=task)
    raise ValueError(f"Unsupported schema version: {schema_version!r}")
