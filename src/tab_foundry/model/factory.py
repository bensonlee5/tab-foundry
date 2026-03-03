"""Model factory."""

from __future__ import annotations

from torch import nn

from .tabiclv2 import TabICLv2Classifier, TabICLv2Regressor


def build_model(
    task: str,
    *,
    d_col: int = 128,
    d_icl: int = 512,
    feature_group_size: int = 32,
    many_class_train_mode: str = "path_nll",
    max_mixed_radix_digits: int = 64,
) -> nn.Module:
    """Instantiate model for task."""

    if task == "classification":
        return TabICLv2Classifier(
            d_col=d_col,
            d_icl=d_icl,
            feature_group_size=feature_group_size,
            many_class_train_mode=many_class_train_mode,
            max_mixed_radix_digits=max_mixed_radix_digits,
        )
    if task == "regression":
        return TabICLv2Regressor(
            d_col=d_col,
            d_icl=d_icl,
            feature_group_size=feature_group_size,
        )
    raise ValueError(f"Unsupported task: {task}")
