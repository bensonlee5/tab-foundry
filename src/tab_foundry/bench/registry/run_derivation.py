"""Compatibility re-exports for benchmark run derivation helpers."""

from tab_foundry.bench.run_registration import (
    comparison_delta,
    derive_benchmark_run_entry,
    derive_benchmark_run_record,
    empty_registry,
    load_registry_payload,
    sweep_payload,
    validate_record_payload,
    validate_run_entry,
)

__all__ = [
    "comparison_delta",
    "derive_benchmark_run_entry",
    "derive_benchmark_run_record",
    "empty_registry",
    "load_registry_payload",
    "sweep_payload",
    "validate_record_payload",
    "validate_run_entry",
]
