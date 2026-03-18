"""Named pinned OpenML task-id pools for benchmark bundle generation."""

from __future__ import annotations


DEFAULT_OPENML_TASK_SOURCE = "tabarena_v0_1"

TABARENA_V0_1_TASK_IDS: tuple[int, ...] = (
    363612,
    363613,
    363614,
    363615,
    363616,
    363618,
    363619,
    363620,
    363621,
    363623,
    363624,
    363625,
    363626,
    363627,
    363628,
    363629,
    363630,
    363631,
    363632,
    363671,
    363672,
    363673,
    363674,
    363675,
    363676,
    363677,
    363678,
    363679,
    363681,
    363682,
    363683,
    363684,
    363685,
    363686,
    363689,
    363691,
    363693,
    363694,
    363696,
    363697,
    363698,
    363699,
    363700,
    363702,
    363704,
    363705,
    363706,
    363707,
    363708,
    363711,
    363712,
)

# Keep this source pool pinned and small enough to benchmark repeatedly while still
# expanding the canonical binary surface beyond the legacy 3-task TabArena subset.
BINARY_EXPANDED_V1_TASK_IDS: tuple[int, ...] = (
    42,
    3777,
    10091,
    10093,
    3638,
    9958,
    146230,
    363613,
    363621,
    363629,
)

# Reviewed 64-task no-missing binary benchmark surface discovered from the
# global OpenML classification task listing and then validated through
# prepare_openml_benchmark_task().
BINARY_LARGE_NO_MISSING_V1_TASK_IDS: tuple[int, ...] = (
    3,
    31,
    37,
    42,
    49,
    52,
    57,
    134,
    135,
    139,
    146,
    147,
    148,
    150,
    152,
    154,
    157,
    206,
    208,
    209,
    211,
    212,
    215,
    219,
    220,
    221,
    229,
    230,
    2137,
    2142,
    2147,
    2148,
    2253,
    2255,
    2257,
    2262,
    2264,
    3484,
    3492,
    3493,
    3494,
    3495,
    3496,
    3539,
    3542,
    3555,
    3581,
    3583,
    3586,
    3587,
    3588,
    3589,
    3590,
    3591,
    3593,
    3594,
    3596,
    3599,
    3600,
    3601,
    3603,
    3606,
    3607,
    3609,
)

OPENML_TASK_SOURCE_REGISTRY: dict[str, tuple[int, ...]] = {
    "tabarena_v0_1": TABARENA_V0_1_TASK_IDS,
    "binary_expanded_v1": BINARY_EXPANDED_V1_TASK_IDS,
    "binary_large_no_missing_v1": BINARY_LARGE_NO_MISSING_V1_TASK_IDS,
}


def task_source_names() -> tuple[str, ...]:
    """Return stable CLI-facing task-source names."""

    return tuple(OPENML_TASK_SOURCE_REGISTRY.keys())


def task_ids_for_source(task_source: str) -> tuple[int, ...]:
    """Resolve one named task source into its pinned OpenML task ids."""

    try:
        return OPENML_TASK_SOURCE_REGISTRY[str(task_source)]
    except KeyError as exc:
        choices = ", ".join(repr(name) for name in task_source_names())
        raise ValueError(f"unknown OpenML task source {task_source!r}; expected one of: {choices}") from exc
