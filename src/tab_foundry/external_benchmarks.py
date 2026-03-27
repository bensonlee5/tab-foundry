"""Dependency-light external benchmark ids, defaults, labels, and normalization."""

from __future__ import annotations

from typing import Sequence


EXTERNAL_BENCHMARK_NANOTABPFN = "nanotabpfn"
EXTERNAL_BENCHMARK_TABICLV2 = "tabiclv2"
ALLOWED_EXTERNAL_BENCHMARKS = (
    EXTERNAL_BENCHMARK_TABICLV2,
    EXTERNAL_BENCHMARK_NANOTABPFN,
)
DEFAULT_EXTERNAL_BENCHMARKS = (EXTERNAL_BENCHMARK_NANOTABPFN,)
DEFAULT_CLI_EXTERNAL_BENCHMARKS = (EXTERNAL_BENCHMARK_TABICLV2,)
EXTERNAL_BENCHMARK_LABELS = {
    EXTERNAL_BENCHMARK_NANOTABPFN: "nanoTabPFN",
    EXTERNAL_BENCHMARK_TABICLV2: "TabICLv2",
}


def normalize_external_benchmarks(
    values: Sequence[str] | None,
    *,
    default: Sequence[str] = DEFAULT_EXTERNAL_BENCHMARKS,
    context: str = "external_benchmarks",
    allow_empty: bool = False,
) -> tuple[str, ...]:
    """Normalize and validate one external-benchmark selection list."""

    requested = default if values is None or not values else values
    if values is not None and not values and allow_empty:
        return ()
    normalized: list[str] = []
    for index, raw_value in enumerate(requested):
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise RuntimeError(f"{context}[{index}] must be a non-empty string")
        value = str(raw_value).strip().lower()
        if value not in ALLOWED_EXTERNAL_BENCHMARKS:
            raise RuntimeError(
                f"{context}[{index}] must be one of {sorted(ALLOWED_EXTERNAL_BENCHMARKS)!r}, got {raw_value!r}"
            )
        if value in normalized:
            raise RuntimeError(f"{context} must not contain duplicates: {value!r}")
        normalized.append(value)
    if not normalized and not allow_empty:
        raise RuntimeError(f"{context} must contain at least one comparator")
    return tuple(normalized)
