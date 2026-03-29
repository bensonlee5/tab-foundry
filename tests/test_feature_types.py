from __future__ import annotations

import pytest

from tab_foundry.feature_types import normalize_feature_types


def test_normalize_feature_types_accepts_dagzoo_aliases() -> None:
    assert normalize_feature_types(
        ["num", "cat", "floating", "integer"],
        expected_count=4,
        context="feature_types",
    ) == ["floating", "unknown", "floating", "integer"]


def test_normalize_feature_types_rejects_unknown_aliases() -> None:
    with pytest.raises(ValueError, match="feature_types\\[0\\]"):
        normalize_feature_types(
            ["categorical"],
            expected_count=1,
            context="feature_types",
        )
