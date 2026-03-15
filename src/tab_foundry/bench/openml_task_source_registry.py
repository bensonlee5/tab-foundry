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

OPENML_TASK_SOURCE_REGISTRY: dict[str, tuple[int, ...]] = {
    "tabarena_v0_1": TABARENA_V0_1_TASK_IDS,
    "binary_expanded_v1": BINARY_EXPANDED_V1_TASK_IDS,
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
