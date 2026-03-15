"""Shared preprocessing surfaces."""

from .state import (
    CLASSIFICATION_LABEL_MAPPING_TRAIN_ONLY_REMAP,
    DTYPE_POLICY,
    FEATURE_ORDER_POLICY_POSITIONAL,
    MISSING_VALUE_STRATEGY_TRAIN_MEAN,
    UNSEEN_TEST_LABEL_POLICY_FILTER,
    ClassificationLabelPolicyState,
    FittedPreprocessorState,
    MissingValuePolicyState,
)
from .surface import (
    SUPPORTED_LABEL_MAPPINGS,
    SUPPORTED_UNSEEN_TEST_LABEL_POLICIES,
    PreprocessingSurfaceConfig,
    resolve_preprocessing_surface,
)
from .tabular import (
    PreprocessedTaskArrays,
    apply_fitted_preprocessor,
    fit_fitted_preprocessor,
    preprocess_runtime_task_arrays,
)

__all__ = [
    "CLASSIFICATION_LABEL_MAPPING_TRAIN_ONLY_REMAP",
    "DTYPE_POLICY",
    "FEATURE_ORDER_POLICY_POSITIONAL",
    "MISSING_VALUE_STRATEGY_TRAIN_MEAN",
    "UNSEEN_TEST_LABEL_POLICY_FILTER",
    "ClassificationLabelPolicyState",
    "FittedPreprocessorState",
    "MissingValuePolicyState",
    "PreprocessedTaskArrays",
    "PreprocessingSurfaceConfig",
    "SUPPORTED_LABEL_MAPPINGS",
    "SUPPORTED_UNSEEN_TEST_LABEL_POLICIES",
    "apply_fitted_preprocessor",
    "fit_fitted_preprocessor",
    "preprocess_runtime_task_arrays",
    "resolve_preprocessing_surface",
]
