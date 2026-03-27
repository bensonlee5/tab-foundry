from __future__ import annotations

import pytest

from tab_foundry.external_benchmarks import (
    ALLOWED_EXTERNAL_BENCHMARKS,
    DEFAULT_CLI_EXTERNAL_BENCHMARKS,
    DEFAULT_EXTERNAL_BENCHMARKS,
    EXTERNAL_BENCHMARK_LABELS,
    EXTERNAL_BENCHMARK_NANOTABPFN,
    EXTERNAL_BENCHMARK_TABICLV2,
    normalize_external_benchmarks,
)


def test_external_benchmark_contracts_expose_expected_defaults() -> None:
    assert ALLOWED_EXTERNAL_BENCHMARKS == (
        EXTERNAL_BENCHMARK_TABICLV2,
        EXTERNAL_BENCHMARK_NANOTABPFN,
    )
    assert DEFAULT_EXTERNAL_BENCHMARKS == (EXTERNAL_BENCHMARK_NANOTABPFN,)
    assert DEFAULT_CLI_EXTERNAL_BENCHMARKS == (EXTERNAL_BENCHMARK_TABICLV2,)
    assert EXTERNAL_BENCHMARK_LABELS == {
        EXTERNAL_BENCHMARK_NANOTABPFN: "nanoTabPFN",
        EXTERNAL_BENCHMARK_TABICLV2: "TabICLv2",
    }


def test_normalize_external_benchmarks_uses_default_when_missing_or_empty() -> None:
    assert normalize_external_benchmarks(None) == DEFAULT_EXTERNAL_BENCHMARKS
    assert normalize_external_benchmarks([]) == DEFAULT_EXTERNAL_BENCHMARKS


def test_normalize_external_benchmarks_allows_explicit_empty_when_requested() -> None:
    assert normalize_external_benchmarks([], allow_empty=True) == ()


def test_normalize_external_benchmarks_normalizes_case_and_whitespace() -> None:
    assert normalize_external_benchmarks([" NanoTabPFN ", "TabICLv2"]) == (
        EXTERNAL_BENCHMARK_NANOTABPFN,
        EXTERNAL_BENCHMARK_TABICLV2,
    )


def test_normalize_external_benchmarks_rejects_duplicates() -> None:
    with pytest.raises(RuntimeError, match="must not contain duplicates"):
        _ = normalize_external_benchmarks(["tabiclv2", "tabiclv2"])


def test_normalize_external_benchmarks_rejects_unknown_values() -> None:
    with pytest.raises(RuntimeError, match="must be one of"):
        _ = normalize_external_benchmarks(["unknown"])
