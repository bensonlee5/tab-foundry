"""Shared fitted preprocessing state for task-level tabular inputs."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence


FEATURE_ORDER_POLICY_POSITIONAL = "positional_feature_ids"
MISSING_VALUE_STRATEGY_TRAIN_MEAN = "train_mean"
CLASSIFICATION_LABEL_MAPPING_TRAIN_ONLY_REMAP = "train_only_remap"
UNSEEN_TEST_LABEL_POLICY_FILTER = "filter"
NUMERIC_FEATURE_TYPE = "num"
CATEGORICAL_FEATURE_TYPE = "cat"
CATEGORICAL_VALUE_POLICY_OOV_BUCKET = "oov_bucket"
_FEATURE_TYPE_ALIASES = {
    NUMERIC_FEATURE_TYPE: NUMERIC_FEATURE_TYPE,
    CATEGORICAL_FEATURE_TYPE: CATEGORICAL_FEATURE_TYPE,
    "categorical": CATEGORICAL_FEATURE_TYPE,
}
DTYPE_POLICY = {
    "features": "float32",
    "classification_labels": "int64",
    "regression_targets": "float32",
}
NumericScalar = int | float


def normalize_feature_types(
    feature_types: Sequence[str] | None,
    *,
    width: int | None = None,
) -> list[str]:
    """Normalize feature-type aliases to canonical tokens."""

    if feature_types is None:
        return [] if width is None else [NUMERIC_FEATURE_TYPE] * width
    raw_types = list(feature_types)
    if width is not None and len(raw_types) != width:
        raise RuntimeError(
            "feature_types length must match feature count: "
            f"expected={width}, got={len(raw_types)}"
        )
    normalized: list[str] = []
    for index, value in enumerate(raw_types):
        token = str(value).strip().lower()
        normalized_token = _FEATURE_TYPE_ALIASES.get(token)
        if normalized_token is None:
            raise RuntimeError(
                "feature_types entries must be one of "
                f"{tuple(sorted(_FEATURE_TYPE_ALIASES))}, got {value!r} at index={index}"
            )
        normalized.append(normalized_token)
    return normalized


def feature_types_include_categorical(feature_types: Sequence[str] | None) -> bool:
    """Whether a declared feature schema includes at least one categorical column."""

    return CATEGORICAL_FEATURE_TYPE in normalize_feature_types(feature_types)


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
    train_value_vocab: list[list[NumericScalar]]
    value_dtypes: list[str | None]
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
