"""Shared fitted preprocessing state for task-level tabular inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


FEATURE_ORDER_POLICY_POSITIONAL = "positional_feature_ids"
MISSING_VALUE_STRATEGY_TRAIN_MEAN = "train_mean"
CLASSIFICATION_LABEL_MAPPING_TRAIN_ONLY_REMAP = "train_only_remap"
UNSEEN_TEST_LABEL_POLICY_FILTER = "filter"
NUMERIC_FEATURE_TYPE = "num"
CATEGORICAL_FEATURE_TYPE = "cat"
CATEGORICAL_VALUE_POLICY_OOV_BUCKET = "oov_bucket"
DTYPE_POLICY = {
    "features": "float32",
    "classification_labels": "int64",
    "regression_targets": "float32",
}


@dataclass(slots=True, frozen=True)
class MissingValuePolicyState:
    """Fitted feature-imputation state."""

    strategy: str
    all_nan_fill: float
    fill_values: list[float]

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(slots=True, frozen=True)
class ClassificationLabelPolicyState:
    """Fitted classification-label remapping state."""

    mapping: str
    unseen_test_label: str
    label_values: list[int]

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(slots=True, frozen=True)
class CategoricalFeaturePolicyState:
    """Fitted categorical-column remapping state."""

    feature_types: list[str]
    train_value_vocab: list[list[float]]
    cardinalities: list[int]
    value_policy: str

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(slots=True, frozen=True)
class FittedPreprocessorState:
    """Serializable fitted preprocessing state."""

    feature_order_policy: str
    feature_ids: list[int]
    missing_value_policy: MissingValuePolicyState
    categorical_feature_policy: CategoricalFeaturePolicyState
    classification_label_policy: ClassificationLabelPolicyState | None
    dtype_policy: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))
