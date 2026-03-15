from __future__ import annotations

import string

from hypothesis import given
from hypothesis import strategies as st
import pytest

from tab_foundry.data.manifest import (
    _coerce_optional_int,
    _dataset_id,
    _is_record_selected,
    _stable_split,
)


_TEXT = st.text(alphabet=string.ascii_letters + string.digits + "_-", min_size=1, max_size=24)
_SHARD_RELPATH = st.text(
    alphabet=string.ascii_letters + string.digits + "_-/",
    min_size=1,
    max_size=32,
).filter(lambda value: not value.startswith("/") and "//" not in value and value.strip("/") != "")


@st.composite
def _split_ratios(draw: st.DrawFn) -> tuple[float, float]:
    train_bp = draw(st.integers(min_value=1, max_value=9_998))
    val_bp = draw(st.integers(min_value=0, max_value=9_999 - train_bp))
    return train_bp / 10_000.0, val_bp / 10_000.0


@given(key=_TEXT, ratios=_split_ratios())
def test_stable_split_is_deterministic_and_returns_supported_bucket(
    key: str,
    ratios: tuple[float, float],
) -> None:
    train_ratio, val_ratio = ratios

    first = _stable_split(key, train_ratio, val_ratio)
    second = _stable_split(key, train_ratio, val_ratio)

    assert first == second
    assert first in {"train", "val", "test"}


@given(root_id=_TEXT, shard_relpath=_SHARD_RELPATH, dataset_index=st.integers(min_value=1, max_value=1_000_000))
def test_dataset_id_is_deterministic_and_changes_when_inputs_change(
    root_id: str,
    shard_relpath: str,
    dataset_index: int,
) -> None:
    baseline = _dataset_id(
        root_id=root_id,
        shard_relpath=shard_relpath,
        dataset_index=dataset_index,
    )

    assert baseline == _dataset_id(
        root_id=root_id,
        shard_relpath=shard_relpath,
        dataset_index=dataset_index,
    )
    assert baseline != _dataset_id(
        root_id=f"{root_id}_alt",
        shard_relpath=shard_relpath,
        dataset_index=dataset_index,
    )
    assert baseline != _dataset_id(
        root_id=root_id,
        shard_relpath=f"{shard_relpath.strip('/')}/alt",
        dataset_index=dataset_index,
    )
    assert baseline != _dataset_id(
        root_id=root_id,
        shard_relpath=shard_relpath,
        dataset_index=dataset_index + 1,
    )


@given(default=st.integers(min_value=-10_000, max_value=10_000))
def test_coerce_optional_int_uses_default_for_none(default: int) -> None:
    assert _coerce_optional_int(None, default=default, context="value") == default


@given(value=st.integers(min_value=-10_000, max_value=10_000))
def test_coerce_optional_int_accepts_int_compatible_values(value: int) -> None:
    assert _coerce_optional_int(value, default=0, context="value") == value
    assert _coerce_optional_int(str(value), default=0, context="value") == value


@given(value=st.text(alphabet=string.ascii_letters, min_size=1, max_size=16))
def test_coerce_optional_int_rejects_non_int_compatible_values(value: str) -> None:
    with pytest.raises(RuntimeError, match="must be int-compatible or null"):
        _ = _coerce_optional_int(value, default=0, context="value")


@given(
    filter_status=st.one_of(st.none(), _TEXT),
    filter_accepted=st.one_of(st.none(), st.booleans()),
)
def test_record_selection_policy_behavior(
    filter_status: str | None,
    filter_accepted: bool | None,
) -> None:
    assert _is_record_selected(
        filter_policy="include_all",
        filter_status=filter_status,
        filter_accepted=filter_accepted,
    ) is True
    assert _is_record_selected(
        filter_policy="accepted_only",
        filter_status=filter_status,
        filter_accepted=filter_accepted,
    ) is (filter_status == "accepted" or filter_accepted is True)
