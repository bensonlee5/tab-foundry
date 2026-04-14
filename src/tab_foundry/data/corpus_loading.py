"""Corpus recipe/index loading and shared corpus path helpers."""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, cast

import yaml

from tab_foundry.hashing import sha256_path, sha256_text
from tab_foundry.repo_paths import repo_root as shared_repo_root
from tab_foundry.timestamps import utc_now


CORPUS_RECIPE_SCHEMA = "tab-foundry-corpus-recipe-v1"
CORPUS_RECIPE_INDEX_SCHEMA = "tab-foundry-corpus-recipe-index-v1"
CORPUS_RECORD_SCHEMA = "tab-foundry-corpus-record-v1"
CORPUS_LATEST_SCHEMA = "tab-foundry-corpus-latest-v1"
RECIPE_KIND_DAGZOO_SINGLE = "dagzoo_single_invocation"
RECIPE_KIND_DAGZOO_MULTI = "dagzoo_multi_invocation_manifest"
RECIPE_KIND_DAGZOO_PYTHON_GENERATED = "dagzoo_python_generated"
_VALID_RECIPE_KINDS = {
    RECIPE_KIND_DAGZOO_SINGLE,
    RECIPE_KIND_DAGZOO_MULTI,
    RECIPE_KIND_DAGZOO_PYTHON_GENERATED,
}


def _repo_root() -> Path:
    return shared_repo_root()


def corpus_recipes_root(*, repo_root: Path | None = None) -> Path:
    resolved_repo_root = (repo_root or _repo_root()).expanduser().resolve()
    return resolved_repo_root / "reference" / "corpus_recipes"


def corpus_recipe_index_path(*, repo_root: Path | None = None) -> Path:
    return corpus_recipes_root(repo_root=repo_root) / "index.yaml"


def sweep_corpus_recipes_root(
    sweep_id: str,
    *,
    repo_root: Path | None = None,
    sweeps_root: Path | None = None,
) -> Path:
    resolved_repo_root = (repo_root or _repo_root()).expanduser().resolve()
    resolved_sweeps_root = (
        sweeps_root.expanduser().resolve()
        if sweeps_root is not None
        else resolved_repo_root / "reference" / "system_delta_sweeps"
    )
    return resolved_sweeps_root / str(sweep_id) / "corpus_recipes"


def sweep_corpus_recipe_index_path(
    sweep_id: str,
    *,
    repo_root: Path | None = None,
    sweeps_root: Path | None = None,
) -> Path:
    return sweep_corpus_recipes_root(
        sweep_id,
        repo_root=repo_root,
        sweeps_root=sweeps_root,
    ) / "index.yaml"


def corpus_outputs_root(*, repo_root: Path | None = None) -> Path:
    resolved_repo_root = (repo_root or _repo_root()).expanduser().resolve()
    return resolved_repo_root / "outputs" / "corpora"


def _read_json_mapping(path: Path, *, context: str) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} must decode to a JSON object: {path.expanduser().resolve()}")
    return cast(dict[str, Any], payload)


def _load_yaml_mapping(path: Path, *, context: str) -> dict[str, Any]:
    payload = yaml.safe_load(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} must decode to a mapping: {path.expanduser().resolve()}")
    return cast(dict[str, Any], payload)


def _copy_jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _drop_none_values(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in payload.items()
        if value is not None
    }


def _resolve_from_root(root: Path, raw_path: Path) -> Path:
    expanded = raw_path.expanduser()
    return expanded.resolve() if expanded.is_absolute() else (root / expanded).resolve()


def _ensure_non_empty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{context} must be a non-empty string")
    return str(value)


def _ensure_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _optional_string(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return str(value)


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _optional_float(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_mapping(value: Any) -> dict[str, Any] | None:
    if not isinstance(value, Mapping):
        return None
    return {str(key): item for key, item in value.items()}


def _optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _optional_string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [normalized for item in value if (normalized := _optional_string(item)) is not None]


def _string_candidates(*values: Any) -> list[str]:
    candidates: set[str] = set()
    for value in values:
        if isinstance(value, list):
            candidates.update(_optional_string_list(value))
            continue
        normalized = _optional_string(value)
        if normalized is not None:
            candidates.add(normalized)
    return sorted(candidates)


def _bool_candidates(*values: Any) -> set[bool]:
    candidates: set[bool] = set()
    for value in values:
        if isinstance(value, list):
            candidates.update(
                normalized
                for item in value
                if (normalized := _optional_bool(item)) is not None
            )
            continue
        normalized = _optional_bool(value)
        if normalized is not None:
            candidates.add(normalized)
    return candidates


def _float_candidates(*values: Any) -> list[float]:
    candidates: set[float] = set()
    for value in values:
        if isinstance(value, list):
            candidates.update(
                normalized
                for item in value
                if (normalized := _optional_float(item)) is not None
            )
            continue
        normalized = _optional_float(value)
        if normalized is not None:
            candidates.add(normalized)
    return sorted(candidates)


def _first_non_empty_group[T](groups: list[T]) -> T | None:
    for group in groups:
        if group:
            return group
    return None


def _normalized_range_from_mapping(value: Any) -> dict[str, Any] | None:
    mapping = _optional_mapping(value)
    if mapping is None:
        return None
    result: dict[str, Any] = {}
    minimum = _optional_int(mapping.get("min"))
    if minimum is not None:
        result["min"] = minimum
    if "max" in mapping:
        result["max"] = None if mapping.get("max") is None else _optional_int(mapping.get("max"))
    elif (maximum := _optional_int(mapping.get("max"))) is not None:
        result["max"] = maximum
    return result or None


def _aggregate_range_bounds(bounds: list[dict[str, Any]]) -> dict[str, Any] | None:
    minima: list[int] = []
    maxima: list[int] = []
    saw_explicit_none_max = False
    for bound in bounds:
        minimum = _optional_int(bound.get("min"))
        if minimum is not None:
            minima.append(minimum)
        if "max" in bound:
            if bound.get("max") is None:
                saw_explicit_none_max = True
            elif (maximum := _optional_int(bound.get("max"))) is not None:
                maxima.append(maximum)
    if not minima and not maxima and not saw_explicit_none_max:
        return None
    result: dict[str, Any] = {}
    if minima:
        result["min"] = min(minima)
    if maxima:
        result["max"] = max(maxima)
    elif saw_explicit_none_max:
        result["max"] = None
    return result or None


def _normalized_float_range_from_mapping(value: Any) -> dict[str, float] | None:
    mapping = _optional_mapping(value)
    if mapping is None:
        return None
    result: dict[str, float] = {}
    minimum = _optional_float(mapping.get("min"))
    maximum = _optional_float(mapping.get("max"))
    if minimum is not None:
        result["min"] = float(minimum)
    if maximum is not None:
        result["max"] = float(maximum)
    return result or None


def _aggregate_float_range_bounds(bounds: list[dict[str, float]]) -> dict[str, float] | None:
    minima = [float(bound["min"]) for bound in bounds if "min" in bound]
    maxima = [float(bound["max"]) for bound in bounds if "max" in bound]
    if not minima and not maxima:
        return None
    result: dict[str, float] = {}
    if minima:
        result["min"] = float(min(minima))
    if maxima:
        result["max"] = float(max(maxima))
    return result or None


def _invocation_config_overrides(invocation: Any) -> dict[str, Any]:
    if isinstance(invocation, Mapping):
        return _optional_mapping(invocation.get("config_overrides")) or {}
    return _optional_mapping(getattr(invocation, "config_overrides", None)) or {}


def _recipe_invocation_override_sections(recipe: Any, section_name: str) -> list[dict[str, Any]]:
    sections: list[dict[str, Any]] = []
    for invocation in _recipe_like_invocations(recipe):
        overrides = _invocation_config_overrides(invocation)
        section = _optional_mapping(overrides.get(section_name))
        if section is not None:
            sections.append(section)
    return sections


def _recipe_like_value(recipe: Any, field_name: str) -> Any:
    if isinstance(recipe, Mapping):
        return recipe.get(field_name)
    return getattr(recipe, field_name, None)


def _recipe_like_invocations(recipe: Any) -> list[Any]:
    raw_invocations = _recipe_like_value(recipe, "invocations")
    if isinstance(raw_invocations, tuple):
        return list(raw_invocations)
    if isinstance(raw_invocations, list):
        return raw_invocations
    return []


def _invocation_config_ref(invocation: Any) -> str | None:
    if isinstance(invocation, Mapping):
        return _optional_string(invocation.get("config_ref")) or _optional_string(
            invocation.get("base_config_ref")
        )
    return _optional_string(getattr(invocation, "config_ref", None)) or _optional_string(
        getattr(invocation, "base_config_ref", None)
    )


def _string_list(value: Any) -> list[str]:
    if not isinstance(value, list):
        return []
    return [normalized for item in value if (normalized := _optional_string(item)) is not None]


def build_dagzoo_provenance_summary(
    *,
    recipe: Any,
    corpus_ref: str,
    corpus_id: str,
    provenance: Mapping[str, Any] | None = None,
    surface_label: str | None = None,
) -> dict[str, Any]:
    raw_provenance = {} if not isinstance(provenance, Mapping) else {str(key): item for key, item in provenance.items()}
    provenance_labels = _optional_mapping(_recipe_like_value(recipe, "provenance_labels")) or {}
    generator = _optional_mapping(_recipe_like_value(recipe, "generator")) or {}
    review_summary = _optional_mapping(_recipe_like_value(recipe, "review_summary")) or {}
    resolved_recipe_id = (
        _optional_string(_recipe_like_value(recipe, "recipe_id"))
        or _optional_string(raw_provenance.get("recipe_id"))
        or _ensure_non_empty_string(corpus_ref.partition("/")[0], context="corpus_ref recipe_id")
    )
    resolved_surface_label = (
        _optional_string(surface_label)
        or _optional_string(raw_provenance.get("surface_label"))
        or _optional_string(_recipe_like_value(recipe, "surface_label"))
    )
    config_refs = _string_list(raw_provenance.get("config_refs"))
    if not config_refs:
        config_refs = _string_list(review_summary.get("config_refs"))
    if not config_refs:
        config_refs = sorted(
            {
                config_ref
                for invocation in _recipe_like_invocations(recipe)
                if (config_ref := _invocation_config_ref(invocation)) is not None
            }
        )
    invocation_count = _optional_int(review_summary.get("invocation_count"))
    if invocation_count is None and isinstance(raw_provenance.get("invocations"), list):
        invocation_count = len(raw_provenance["invocations"])
    manifest_record_count = _optional_int(review_summary.get("manifest_record_count"))
    recipe_dataset_overrides = _recipe_invocation_override_sections(recipe, "dataset")
    recipe_diagnostics_overrides = _recipe_invocation_override_sections(recipe, "diagnostics")
    invocation_handoff_provenances = [
        handoff_provenance
        for invocation in cast(list[Any], raw_provenance.get("invocations", []))
        for handoff_provenance in (
            _optional_mapping(
                invocation.get("handoff_provenance")
                if isinstance(invocation, Mapping)
                else None
            ),
        )
        if handoff_provenance is not None
    ]
    invocation_filter_payloads = [
        filter_payload
        for invocation in cast(list[Any], raw_provenance.get("invocations", []))
        for filter_payload in (
            _optional_mapping(
                invocation.get("filter")
                if isinstance(invocation, Mapping)
                else None
            ),
        )
        if filter_payload is not None
    ]
    recipe_manifest = _optional_mapping(_recipe_like_value(recipe, "manifest")) or {}
    explicit_filter_policies = _string_candidates(raw_provenance.get("filter_policy"))
    invocation_filter_policies = _string_candidates(
        *[item.get("filter_policy") for item in invocation_filter_payloads]
    )
    recipe_filter_policies = _string_candidates(recipe_manifest.get("filter_policy"))
    filter_policies = _first_non_empty_group(
        [
            explicit_filter_policies,
            invocation_filter_policies,
            recipe_filter_policies,
        ]
    ) or []
    resolved_filter_policy = (
        filter_policies[0]
        if len(filter_policies) == 1
        else None
    )
    if (
        resolved_filter_policy == "include_all"
        and not invocation_filter_payloads
        and not explicit_filter_policies
    ):
        resolved_filter_policy = None
    invocation_accepted_dataset_counts = [
        int(value)
        for value in (_optional_int(item.get("accepted_datasets")) for item in invocation_filter_payloads)
        if value is not None
    ]
    invocation_rejected_dataset_counts = [
        int(value)
        for value in (_optional_int(item.get("rejected_datasets")) for item in invocation_filter_payloads)
        if value is not None
    ]
    invocation_curated_accepted_dataset_counts = [
        int(value)
        for value in (
            _optional_int(item.get("curated_accepted_datasets"))
            for item in invocation_filter_payloads
        )
        if value is not None
    ]
    accepted_datasets = _optional_int(raw_provenance.get("accepted_datasets"))
    if accepted_datasets is None and invocation_filter_payloads:
        accepted_datasets = sum(invocation_accepted_dataset_counts)
    rejected_datasets = _optional_int(raw_provenance.get("rejected_datasets"))
    if rejected_datasets is None and invocation_filter_payloads:
        rejected_datasets = sum(invocation_rejected_dataset_counts)
    curated_accepted_datasets = _optional_int(raw_provenance.get("curated_accepted_datasets"))
    if curated_accepted_datasets is None and invocation_filter_payloads:
        curated_accepted_datasets = sum(invocation_curated_accepted_dataset_counts)
    acceptance_rate = _optional_float(raw_provenance.get("acceptance_rate"))
    if (
        acceptance_rate is None
        and accepted_datasets is not None
        and rejected_datasets is not None
        and (accepted_datasets + rejected_datasets) > 0
    ):
        acceptance_rate = float(accepted_datasets) / float(accepted_datasets + rejected_datasets)
    top_level_materialization_timing = (
        _optional_mapping(raw_provenance.get("materialization_timing")) or {}
    )
    invocation_materialization_timings = [
        timing_payload
        for invocation in cast(list[Any], raw_provenance.get("invocations", []))
        for timing_payload in (
            _optional_mapping(
                invocation.get("materialization_timing")
                if isinstance(invocation, Mapping)
                else None
            ),
        )
        if timing_payload is not None
    ]

    def _sum_optional_float(items: Sequence[Mapping[str, Any]], key: str) -> float | None:
        resolved = [
            value
            for value in (_optional_float(item.get(key)) for item in items)
            if value is not None
        ]
        if not resolved:
            return None
        return float(sum(resolved))

    def _sum_optional_int(items: Sequence[Mapping[str, Any]], key: str) -> int | None:
        resolved = [
            value
            for value in (_optional_int(item.get(key)) for item in items)
            if value is not None
        ]
        if not resolved:
            return None
        return int(sum(resolved))

    materialization_timing = _drop_none_values(
        {
            "recipe_elapsed_seconds": _optional_float(
                top_level_materialization_timing.get("recipe_elapsed_seconds")
            ),
            "invocation_fanout_elapsed_seconds": _optional_float(
                top_level_materialization_timing.get("invocation_fanout_elapsed_seconds")
            ),
            "staged_compaction_elapsed_seconds": _optional_float(
                top_level_materialization_timing.get("staged_compaction_elapsed_seconds")
            ),
            "staged_compaction_status": (
                str(top_level_materialization_timing["staged_compaction_status"])
                if isinstance(top_level_materialization_timing.get("staged_compaction_status"), str)
                and str(top_level_materialization_timing["staged_compaction_status"]).strip()
                else None
            ),
            "manifest_build_elapsed_seconds": _optional_float(
                top_level_materialization_timing.get("manifest_build_elapsed_seconds")
            ),
            "manifest_workers": _optional_int(
                top_level_materialization_timing.get("manifest_workers")
            ),
            "promotion_elapsed_seconds": _optional_float(
                top_level_materialization_timing.get("promotion_elapsed_seconds")
            ),
            "timed_invocation_count": len(invocation_materialization_timings)
            if invocation_materialization_timings
            else None,
            "cumulative_round_count": _sum_optional_int(
                invocation_materialization_timings,
                "round_count",
            ),
            "cumulative_generated_datasets": _sum_optional_int(
                invocation_materialization_timings,
                "generated_datasets",
            ),
            "cumulative_generate_elapsed_seconds": _sum_optional_float(
                invocation_materialization_timings,
                "generate_elapsed_seconds",
            ),
            "cumulative_filter_elapsed_seconds": _sum_optional_float(
                invocation_materialization_timings,
                "filter_elapsed_seconds",
            ),
            "cumulative_copy_elapsed_seconds": _sum_optional_float(
                invocation_materialization_timings,
                "copy_elapsed_seconds",
            ),
            "cumulative_upstream_elapsed_seconds": _sum_optional_float(
                invocation_materialization_timings,
                "upstream_elapsed_seconds",
            ),
            "cumulative_local_overhead_elapsed_seconds": _sum_optional_float(
                invocation_materialization_timings,
                "local_overhead_elapsed_seconds",
            ),
            "cumulative_invocation_elapsed_seconds": _sum_optional_float(
                invocation_materialization_timings,
                "invocation_elapsed_seconds",
            ),
        }
    )
    posterior_predictive_factorizations = _first_non_empty_group(
        [
            _string_candidates(
                raw_provenance.get("posterior_predictive_factorization"),
                raw_provenance.get("posterior_predictive_factorizations"),
            ),
            _string_candidates(
                *[
                    item.get("posterior_predictive_factorization")
                    for item in invocation_handoff_provenances
                ]
            ),
            _string_candidates(
                provenance_labels.get("posterior_predictive_factorization"),
                provenance_labels.get("posterior_predictive_factorizations"),
            ),
            _string_candidates(
                review_summary.get("posterior_predictive_factorization"),
                review_summary.get("posterior_predictive_factorizations"),
            ),
        ]
    )
    if posterior_predictive_factorizations is None:
        posterior_predictive_factorizations = []
    teacher_conditional_export_values = _first_non_empty_group(
        [
            _bool_candidates(raw_provenance.get("teacher_conditional_export")),
            _bool_candidates(
                *[
                    item.get("teacher_conditional_export")
                    for item in invocation_handoff_provenances
                ]
            ),
            _bool_candidates(provenance_labels.get("teacher_conditional_export")),
            _bool_candidates(review_summary.get("teacher_conditional_export")),
            _bool_candidates(
                *[
                    item.get("teacher_conditional_export")
                    for item in recipe_diagnostics_overrides
                ]
            ),
        ]
    )
    if teacher_conditional_export_values is None:
        teacher_conditional_export_values = set()
    teacher_conditional_export = (
        next(iter(teacher_conditional_export_values))
        if len(teacher_conditional_export_values) == 1
        else None
    )
    teacher_conditional_metric_definitions = _first_non_empty_group(
        [
            _string_candidates(
                raw_provenance.get("teacher_conditional_metric_definition"),
                raw_provenance.get("metric_definition"),
            ),
            _string_candidates(
                *[
                    item.get("teacher_conditional_metric_definition")
                    or item.get("metric_definition")
                    for item in invocation_handoff_provenances
                ]
            ),
            _string_candidates(
                provenance_labels.get("teacher_conditional_metric_definition"),
                provenance_labels.get("metric_definition"),
            ),
            _string_candidates(
                review_summary.get("teacher_conditional_metric_definition"),
                review_summary.get("metric_definition"),
            ),
        ]
    )
    if teacher_conditional_metric_definitions is None:
        teacher_conditional_metric_definitions = []
    target_parent_priors = _first_non_empty_group(
        [
            _string_candidates(raw_provenance.get("target_parent_prior")),
            _string_candidates(
                *[
                    item.get("target_parent_prior")
                    for item in invocation_handoff_provenances
                ]
            ),
            _string_candidates(provenance_labels.get("target_parent_prior")),
            _string_candidates(review_summary.get("target_parent_prior")),
            _string_candidates(
                *[
                    item.get("target_parent_prior")
                    for item in recipe_dataset_overrides
                ]
            ),
        ]
    )
    if target_parent_priors is None:
        target_parent_priors = []
    target_parent_modes = _first_non_empty_group(
        [
            _string_candidates(raw_provenance.get("target_parent_mode")),
            _string_candidates(
                *[
                    item.get("target_parent_mode")
                    for item in invocation_handoff_provenances
                ]
            ),
            _string_candidates(provenance_labels.get("target_parent_mode")),
            _string_candidates(review_summary.get("target_parent_mode")),
        ]
    )
    if target_parent_modes is None:
        target_parent_modes = []
    target_parent_near_max_band_min_fractions = _first_non_empty_group(
        [
            _float_candidates(raw_provenance.get("target_parent_near_max_band_min_fraction")),
            _float_candidates(
                *[
                    item.get("target_parent_near_max_band_min_fraction")
                    for item in invocation_handoff_provenances
                ]
            ),
            _float_candidates(provenance_labels.get("target_parent_near_max_band_min_fraction")),
            _float_candidates(review_summary.get("target_parent_near_max_band_min_fraction")),
            _float_candidates(
                *[
                    item.get("target_parent_near_max_band_min_fraction")
                    for item in recipe_dataset_overrides
                ]
            ),
        ]
    )
    if target_parent_near_max_band_min_fractions is None:
        target_parent_near_max_band_min_fractions = []
    target_parent_below_sqrt_probs = _first_non_empty_group(
        [
            _float_candidates(raw_provenance.get("target_parent_below_sqrt_prob")),
            _float_candidates(
                *[
                    item.get("target_parent_below_sqrt_prob")
                    for item in invocation_handoff_provenances
                ]
            ),
            _float_candidates(provenance_labels.get("target_parent_below_sqrt_prob")),
            _float_candidates(review_summary.get("target_parent_below_sqrt_prob")),
            _float_candidates(
                *[
                    item.get("target_parent_below_sqrt_prob")
                    for item in recipe_dataset_overrides
                ]
            ),
        ]
    )
    if target_parent_below_sqrt_probs is None:
        target_parent_below_sqrt_probs = []
    target_parent_midrange_probs = _first_non_empty_group(
        [
            _float_candidates(raw_provenance.get("target_parent_midrange_prob")),
            _float_candidates(
                *[
                    item.get("target_parent_midrange_prob")
                    for item in invocation_handoff_provenances
                ]
            ),
            _float_candidates(provenance_labels.get("target_parent_midrange_prob")),
            _float_candidates(review_summary.get("target_parent_midrange_prob")),
            _float_candidates(
                *[
                    item.get("target_parent_midrange_prob")
                    for item in recipe_dataset_overrides
                ]
            ),
        ]
    )
    if target_parent_midrange_probs is None:
        target_parent_midrange_probs = []
    target_parent_regimes_present = sorted(
        {
            regime
            for regime in (
                *(
                    str(item).strip()
                    for item in cast(list[Any], raw_provenance.get("target_parent_regimes_present", []))
                    if isinstance(item, str) and str(item).strip()
                ),
                *[
                    str(item).strip()
                    for invocation_provenance in invocation_handoff_provenances
                    for item in cast(list[Any], invocation_provenance.get("target_parent_regimes_present", []))
                    if isinstance(item, str) and str(item).strip()
                ],
            )
            if regime
        }
    )
    target_parent_count_range = _first_non_empty_group(
        [
            _normalized_range_from_mapping(raw_provenance.get("target_parent_count_range")),
            _aggregate_range_bounds(
                [
                    bound
                    for bound in (
                        _normalized_range_from_mapping(item.get("target_parent_count_range"))
                        for item in invocation_handoff_provenances
                    )
                    if bound is not None
                ]
            ),
            _normalized_range_from_mapping(provenance_labels.get("target_parent_count_range")),
            _normalized_range_from_mapping(review_summary.get("target_parent_count_range")),
            _aggregate_range_bounds(
                [
                    bound
                    for bound in (
                        {
                            "min": item.get("target_parent_count_min"),
                            "max": item.get("target_parent_count_max"),
                        }
                        for item in recipe_dataset_overrides
                        if "target_parent_count_min" in item or "target_parent_count_max" in item
                    )
                    if bound
                ]
            ),
        ]
    )
    target_parent_fraction_bounds = [
        bound
        for bound in (
            _optional_mapping(raw_provenance.get("target_parent_fraction_range")),
            *[
                _optional_mapping(item.get("target_parent_fraction_range"))
                for item in invocation_handoff_provenances
            ],
        )
        if bound is not None
    ]
    target_parent_fraction_minima = [
        fraction
        for bound in target_parent_fraction_bounds
        for fraction in (_optional_float(bound.get("min")),)
        if fraction is not None
    ]
    target_parent_fraction_maxima = [
        fraction
        for bound in target_parent_fraction_bounds
        for fraction in (_optional_float(bound.get("max")),)
        if fraction is not None
    ]
    target_derivations = _first_non_empty_group(
        [
            _string_candidates(raw_provenance.get("target_derivation")),
            _string_candidates(
                *[
                    item.get("target_derivation")
                    for item in invocation_handoff_provenances
                ]
            ),
            _string_candidates(provenance_labels.get("target_derivation")),
            _string_candidates(review_summary.get("target_derivation")),
        ]
    )
    if target_derivations is None:
        target_derivations = []
    target_relevant_feature_count_range = _first_non_empty_group(
        [
            _normalized_range_from_mapping(raw_provenance.get("target_relevant_feature_count_range")),
            _aggregate_range_bounds(
                [
                    bound
                    for bound in (
                        _normalized_range_from_mapping(item.get("target_relevant_feature_count_range"))
                        for item in invocation_handoff_provenances
                    )
                    if bound is not None
                ]
            ),
            _normalized_range_from_mapping(provenance_labels.get("target_relevant_feature_count_range")),
            _normalized_range_from_mapping(review_summary.get("target_relevant_feature_count_range")),
        ]
    )
    target_relevant_feature_fraction_range = _first_non_empty_group(
        [
            _normalized_float_range_from_mapping(raw_provenance.get("target_relevant_feature_fraction_range")),
            _aggregate_float_range_bounds(
                [
                    bound
                    for bound in (
                        _normalized_float_range_from_mapping(item.get("target_relevant_feature_fraction_range"))
                        for item in invocation_handoff_provenances
                    )
                    if bound is not None
                ]
            ),
            _normalized_float_range_from_mapping(provenance_labels.get("target_relevant_feature_fraction_range")),
            _normalized_float_range_from_mapping(review_summary.get("target_relevant_feature_fraction_range")),
        ]
    )
    return _drop_none_values(
        {
            "corpus_ref": _ensure_non_empty_string(corpus_ref, context="corpus_ref"),
            "recipe_id": resolved_recipe_id,
            "corpus_id": _ensure_non_empty_string(corpus_id, context="corpus_id"),
            "recipe_kind": (
                _optional_string(raw_provenance.get("recipe_kind"))
                or _optional_string(_recipe_like_value(recipe, "kind"))
            ),
            "surface_label": resolved_surface_label,
            "corpus_variant": (
                _optional_string(raw_provenance.get("corpus_variant"))
                or _optional_string(provenance_labels.get("corpus_variant"))
                or resolved_surface_label
            ),
            "comparator_role": (
                _optional_string(raw_provenance.get("comparator_role"))
                or _optional_string(provenance_labels.get("comparator_role"))
            ),
            "config_refs": config_refs,
            "provenance_labels": _copy_jsonable(provenance_labels) if provenance_labels else None,
            "generator_fingerprint": _optional_string(generator.get("fingerprint")),
            "invocation_count": invocation_count,
            "manifest_record_count": manifest_record_count,
            "filter_policy": resolved_filter_policy,
            "accepted_datasets": accepted_datasets,
            "rejected_datasets": rejected_datasets,
            "curated_accepted_datasets": curated_accepted_datasets,
            "acceptance_rate": acceptance_rate,
            "materialization_timing": (
                _copy_jsonable(materialization_timing) if materialization_timing else None
            ),
            "posterior_predictive_factorization": (
                posterior_predictive_factorizations[0]
                if len(posterior_predictive_factorizations) == 1
                else None
            ),
            "posterior_predictive_factorizations": (
                posterior_predictive_factorizations
                if len(posterior_predictive_factorizations) > 1
                else None
            ),
            "teacher_conditional_export": teacher_conditional_export,
            "teacher_conditional_metric_definition": (
                teacher_conditional_metric_definitions[0]
                if len(teacher_conditional_metric_definitions) == 1
                else (
                    "label-target log loss per test cell"
                    if teacher_conditional_export
                    else None
                )
            ),
            "target_parent_prior": (
                target_parent_priors[0] if len(target_parent_priors) == 1 else None
            ),
            "target_parent_mode": (
                target_parent_modes[0] if len(target_parent_modes) == 1 else None
            ),
            "target_parent_regimes_present": (
                target_parent_regimes_present if target_parent_regimes_present else None
            ),
            "target_parent_count_range": target_parent_count_range,
            "target_parent_fraction_range": (
                {
                    "min": min(target_parent_fraction_minima),
                    "max": max(target_parent_fraction_maxima),
                }
                if target_parent_fraction_minima and target_parent_fraction_maxima
                else None
            ),
            "target_parent_near_max_band_min_fraction": (
                target_parent_near_max_band_min_fractions[0]
                if len(target_parent_near_max_band_min_fractions) == 1
                else None
            ),
            "target_parent_below_sqrt_prob": (
                target_parent_below_sqrt_probs[0]
                if len(target_parent_below_sqrt_probs) == 1
                else None
            ),
            "target_parent_midrange_prob": (
                target_parent_midrange_probs[0]
                if len(target_parent_midrange_probs) == 1
                else None
            ),
            "target_derivation": (
                target_derivations[0] if len(target_derivations) == 1 else None
            ),
            "target_relevant_feature_count_range": target_relevant_feature_count_range,
            "target_relevant_feature_fraction_range": target_relevant_feature_fraction_range,
            "review_summary": _copy_jsonable(review_summary) if review_summary else None,
        }
    )


def _coerce_int(value: Any, *, context: str) -> int:
    if value is None or isinstance(value, bool):
        raise RuntimeError(f"{context} must be an integer-compatible value")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{context} must be an integer-compatible value") from exc


def _deep_merge_payload(base: Any, overrides: Any) -> Any:
    if isinstance(base, Mapping) and isinstance(overrides, Mapping):
        merged = {
            str(key): _copy_jsonable(value)
            for key, value in base.items()
        }
        for key, value in overrides.items():
            key_str = str(key)
            if key_str in merged:
                merged[key_str] = _deep_merge_payload(merged[key_str], value)
            else:
                merged[key_str] = _copy_jsonable(value)
        return merged
    return _copy_jsonable(overrides)


def _recipe_path_from_index_entry(
    recipe_id: str,
    entry: Mapping[str, Any],
    *,
    root: Path,
) -> Path:
    raw_path = _ensure_non_empty_string(entry.get("path"), context=f"recipe index entry {recipe_id!r}.path")
    candidate = Path(raw_path)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


@dataclass(slots=True, frozen=True)
class CorpusManifestPolicy:
    train_ratio: float
    val_ratio: float
    filter_policy: str
    missing_value_policy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_ratio": float(self.train_ratio),
            "val_ratio": float(self.val_ratio),
            "filter_policy": str(self.filter_policy),
            "missing_value_policy": str(self.missing_value_policy),
        }


@dataclass(slots=True, frozen=True)
class DagzooInvocationRecipe:
    invocation_id: str
    config_ref: str | None
    base_config_ref: str | None
    config_overrides: dict[str, Any]
    num_datasets: int
    seed: int | None
    rows: str | None
    device: str | None
    hardware_policy: str
    diagnostics: bool
    diagnostics_out_dir: str | None
    missing_rate: float | None
    missing_mechanism: str | None
    missing_mar_observed_fraction: float | None
    missing_mar_logit_scale: float | None
    missing_mnar_logit_scale: float | None

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "invocation_id": str(self.invocation_id),
            "num_datasets": int(self.num_datasets),
            "seed": None if self.seed is None else int(self.seed),
            "rows": self.rows,
            "device": self.device,
            "hardware_policy": str(self.hardware_policy),
            "diagnostics": bool(self.diagnostics),
            "diagnostics_out_dir": self.diagnostics_out_dir,
            "missing_rate": self.missing_rate,
            "missing_mechanism": self.missing_mechanism,
            "missing_mar_observed_fraction": self.missing_mar_observed_fraction,
            "missing_mar_logit_scale": self.missing_mar_logit_scale,
            "missing_mnar_logit_scale": self.missing_mnar_logit_scale,
        }
        if self.config_ref is not None:
            payload["config_ref"] = str(self.config_ref)
        if self.base_config_ref is not None:
            payload["base_config_ref"] = str(self.base_config_ref)
            payload["config_overrides"] = _copy_jsonable(self.config_overrides)
        return payload


@dataclass(slots=True, frozen=True)
class CorpusRecipe:
    recipe_id: str
    kind: str
    description: str
    surface_label: str
    manifest_policy: CorpusManifestPolicy
    invocations: tuple[DagzooInvocationRecipe, ...]
    provenance_labels: dict[str, Any]
    generator: dict[str, Any] | None
    review_summary: dict[str, Any] | None
    recipe_path: Path

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "schema": CORPUS_RECIPE_SCHEMA,
            "recipe_id": str(self.recipe_id),
            "kind": str(self.kind),
            "description": str(self.description),
            "surface_label": str(self.surface_label),
            "manifest": self.manifest_policy.to_dict(),
            "provenance_labels": dict(self.provenance_labels),
            "invocations": [invocation.to_dict() for invocation in self.invocations],
            "recipe_path": str(self.recipe_path),
        }
        if self.generator is not None:
            payload["generator"] = _copy_jsonable(self.generator)
        if self.review_summary is not None:
            payload["review_summary"] = _copy_jsonable(self.review_summary)
        return payload


@dataclass(slots=True, frozen=True)
class CorpusRecipeStorageContext:
    recipe_identity: str
    recipe_relative_path: str | None
    uses_scoped_identity: bool


def _manifest_policy_from_payload(payload: Mapping[str, Any]) -> CorpusManifestPolicy:
    manifest = _ensure_mapping(payload.get("manifest"), context="recipe.manifest")
    return CorpusManifestPolicy(
        train_ratio=float(manifest.get("train_ratio", 0.90)),
        val_ratio=float(manifest.get("val_ratio", 0.05)),
        filter_policy=str(manifest.get("filter_policy", "include_all")),
        missing_value_policy=str(manifest.get("missing_value_policy", "allow_any")),
    )


def _invocation_from_payload(
    payload: Mapping[str, Any],
    *,
    default_invocation_id: str,
) -> DagzooInvocationRecipe:
    raw_num_datasets = payload.get("num_datasets")
    raw_seed = payload.get("seed")
    config_ref = _optional_string(payload.get("config_ref"))
    base_config_ref = _optional_string(payload.get("base_config_ref"))
    has_config_overrides = "config_overrides" in payload
    if config_ref is not None:
        if base_config_ref is not None or has_config_overrides:
            raise RuntimeError(
                "recipe invocation must define either config_ref or "
                "base_config_ref + config_overrides"
            )
        config_overrides: dict[str, Any] = {}
    else:
        if base_config_ref is None:
            raise RuntimeError(
                "recipe invocation must define either config_ref or "
                "base_config_ref + config_overrides"
            )
        if not has_config_overrides:
            raise RuntimeError(
                "recipe invocation config_overrides must be provided when base_config_ref is set"
            )
        config_overrides = _ensure_mapping(
            payload.get("config_overrides"),
            context="recipe invocation config_overrides",
        )
    return DagzooInvocationRecipe(
        invocation_id=_optional_string(payload.get("invocation_id")) or default_invocation_id,
        config_ref=config_ref,
        base_config_ref=base_config_ref,
        config_overrides=config_overrides,
        num_datasets=_coerce_int(raw_num_datasets, context="recipe invocation num_datasets"),
        seed=None if raw_seed is None else _coerce_int(raw_seed, context="recipe invocation seed"),
        rows=None if payload.get("rows") is None else str(payload["rows"]),
        device=None if payload.get("device") is None else str(payload["device"]),
        hardware_policy=str(payload.get("hardware_policy", "none")),
        diagnostics=bool(payload.get("diagnostics", False)),
        diagnostics_out_dir=(
            None if payload.get("diagnostics_out_dir") is None else str(payload["diagnostics_out_dir"])
        ),
        missing_rate=None if payload.get("missing_rate") is None else float(payload["missing_rate"]),
        missing_mechanism=(
            None if payload.get("missing_mechanism") is None else str(payload["missing_mechanism"])
        ),
        missing_mar_observed_fraction=(
            None
            if payload.get("missing_mar_observed_fraction") is None
            else float(payload["missing_mar_observed_fraction"])
        ),
        missing_mar_logit_scale=(
            None
            if payload.get("missing_mar_logit_scale") is None
            else float(payload["missing_mar_logit_scale"])
        ),
        missing_mnar_logit_scale=(
            None
            if payload.get("missing_mnar_logit_scale") is None
            else float(payload["missing_mnar_logit_scale"])
        ),
    )


def _generator_fingerprint(
    *,
    module_name: str,
    callable_name: str,
    inputs: Mapping[str, Any],
) -> tuple[str, Path]:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(f"failed to import corpus recipe generator module {module_name!r}") from exc
    module_file = getattr(module, "__file__", None)
    if not isinstance(module_file, str) or not module_file.strip():
        raise RuntimeError(f"corpus recipe generator module {module_name!r} does not define __file__")
    resolved_module_path = Path(module_file).expanduser().resolve()
    fingerprint = sha256_text(
        json.dumps(
            {
                "module": module_name,
                "callable": callable_name,
                "inputs": _copy_jsonable(inputs),
                "module_sha256": sha256_path(resolved_module_path),
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )[:12]
    return fingerprint, resolved_module_path


def _generator_callable(
    *,
    module_name: str,
    callable_name: str,
) -> Callable[..., Mapping[str, Any]]:
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(f"failed to import corpus recipe generator module {module_name!r}") from exc
    try:
        generator = getattr(module, callable_name)
    except AttributeError as exc:
        raise RuntimeError(
            f"corpus recipe generator module {module_name!r} does not define {callable_name!r}"
        ) from exc
    if not callable(generator):
        raise RuntimeError(
            f"corpus recipe generator target {module_name!r}.{callable_name} must be callable"
        )
    return cast(Callable[..., Mapping[str, Any]], generator)


def _generator_invocation_payloads(
    payload: Mapping[str, Any],
    *,
    recipe_id: str,
) -> tuple[tuple[DagzooInvocationRecipe, ...], dict[str, Any], dict[str, Any]]:
    generator_payload = _ensure_mapping(
        payload.get("generator"),
        context=f"recipe {recipe_id!r}.generator",
    )
    module_name = _ensure_non_empty_string(
        generator_payload.get("module"),
        context=f"recipe {recipe_id!r}.generator.module",
    )
    callable_name = _ensure_non_empty_string(
        generator_payload.get("callable"),
        context=f"recipe {recipe_id!r}.generator.callable",
    )
    inputs = _ensure_mapping(
        generator_payload.get("inputs"),
        context=f"recipe {recipe_id!r}.generator.inputs",
    )
    declared_fingerprint = _ensure_non_empty_string(
        generator_payload.get("fingerprint"),
        context=f"recipe {recipe_id!r}.generator.fingerprint",
    )
    computed_fingerprint, resolved_module_path = _generator_fingerprint(
        module_name=module_name,
        callable_name=callable_name,
        inputs=inputs,
    )
    if declared_fingerprint != computed_fingerprint:
        raise RuntimeError(
            "generator fingerprint mismatch for "
            f"recipe {recipe_id!r}: declared {declared_fingerprint!r}, computed {computed_fingerprint!r}"
        )
    declared_review_summary = _ensure_mapping(
        payload.get("review_summary"),
        context=f"recipe {recipe_id!r}.review_summary",
    )
    if "dagzoo" in payload or "invocations" in payload:
        raise RuntimeError(
            f"generator-backed recipe {recipe_id!r} must not define inline dagzoo/invocations payloads"
        )
    generated = _generator_callable(
        module_name=module_name,
        callable_name=callable_name,
    )(
        recipe_id=recipe_id,
        description=_ensure_non_empty_string(
            payload.get("description"),
            context=f"recipe {recipe_id!r}.description",
        ),
        surface_label=_ensure_non_empty_string(
            payload.get("surface_label"),
            context=f"recipe {recipe_id!r}.surface_label",
        ),
        manifest=_copy_jsonable(_ensure_mapping(payload.get("manifest"), context=f"recipe {recipe_id!r}.manifest")),
        provenance_labels=_copy_jsonable(
            _ensure_mapping(payload.get("provenance_labels"), context=f"recipe {recipe_id!r}.provenance_labels")
        ),
        inputs=_copy_jsonable(inputs),
        recipe_path=payload.get("recipe_path"),
    )
    if not isinstance(generated, Mapping):
        raise RuntimeError(f"corpus recipe generator for {recipe_id!r} must return a mapping payload")
    generated_review_summary = _ensure_mapping(
        generated.get("review_summary"),
        context=f"generated review summary for recipe {recipe_id!r}",
    )
    if generated_review_summary != declared_review_summary:
        raise RuntimeError(
            f"generated review_summary does not match checked-in summary for recipe {recipe_id!r}"
        )
    if "dagzoo" in generated:
        dagzoo_payload = _ensure_mapping(
            generated.get("dagzoo"),
            context=f"generated dagzoo payload for recipe {recipe_id!r}",
        )
        invocations: tuple[DagzooInvocationRecipe, ...] = (
            _invocation_from_payload(dagzoo_payload, default_invocation_id="default"),
        )
    else:
        raw_invocations = generated.get("invocations")
        if not isinstance(raw_invocations, list) or not raw_invocations:
            raise RuntimeError(
                f"corpus recipe generator for {recipe_id!r} must return a non-empty invocations list"
            )
        invocations = tuple(
            _invocation_from_payload(
                _ensure_mapping(item, context=f"generated invocations[{index}] for recipe {recipe_id!r}"),
                default_invocation_id=f"invocation_{index + 1}",
            )
            for index, item in enumerate(raw_invocations)
        )
    return (
        invocations,
        {
            "module": module_name,
            "callable": callable_name,
            "inputs": _copy_jsonable(inputs),
            "fingerprint": declared_fingerprint,
            "module_sha256": sha256_path(resolved_module_path),
        },
        declared_review_summary,
    )


def _recipe_from_payload(payload: Mapping[str, Any], *, recipe_path: Path) -> CorpusRecipe:
    schema = payload.get("schema")
    if schema != CORPUS_RECIPE_SCHEMA:
        raise RuntimeError(
            f"corpus recipe schema must be {CORPUS_RECIPE_SCHEMA!r}, got {schema!r}: {recipe_path}"
        )
    recipe_id = _ensure_non_empty_string(payload.get("recipe_id"), context="recipe.recipe_id")
    kind = _ensure_non_empty_string(payload.get("kind"), context=f"recipe {recipe_id!r}.kind")
    if kind not in _VALID_RECIPE_KINDS:
        raise RuntimeError(f"unsupported corpus recipe kind {kind!r}: {recipe_path}")
    description = _ensure_non_empty_string(
        payload.get("description"),
        context=f"recipe {recipe_id!r}.description",
    )
    surface_label = _ensure_non_empty_string(
        payload.get("surface_label"),
        context=f"recipe {recipe_id!r}.surface_label",
    )
    manifest_policy = _manifest_policy_from_payload(payload)
    provenance_labels = _ensure_mapping(
        payload.get("provenance_labels"),
        context=f"recipe {recipe_id!r}.provenance_labels",
    )
    generator_summary: dict[str, Any] | None = None
    review_summary: dict[str, Any] | None = None
    if kind == RECIPE_KIND_DAGZOO_SINGLE:
        dagzoo_payload = _ensure_mapping(
            payload.get("dagzoo"),
            context=f"recipe {recipe_id!r}.dagzoo",
        )
        invocations: tuple[DagzooInvocationRecipe, ...] = (
            _invocation_from_payload(dagzoo_payload, default_invocation_id="default"),
        )
    elif kind == RECIPE_KIND_DAGZOO_MULTI:
        raw_invocations = payload.get("invocations")
        if not isinstance(raw_invocations, list) or not raw_invocations:
            raise RuntimeError(f"recipe {recipe_id!r}.invocations must be a non-empty list")
        invocations = tuple(
            _invocation_from_payload(
                _ensure_mapping(item, context=f"recipe {recipe_id!r}.invocations[{index}]"),
                default_invocation_id=f"invocation_{index + 1}",
            )
            for index, item in enumerate(raw_invocations)
        )
    else:
        generated_payload = dict(payload)
        generated_payload["recipe_path"] = str(recipe_path.expanduser().resolve())
        invocations, generator_summary, review_summary = _generator_invocation_payloads(
            generated_payload,
            recipe_id=recipe_id,
        )
    return CorpusRecipe(
        recipe_id=recipe_id,
        kind=kind,
        description=description,
        surface_label=surface_label,
        manifest_policy=manifest_policy,
        invocations=invocations,
        provenance_labels=provenance_labels,
        generator=generator_summary,
        review_summary=review_summary,
        recipe_path=recipe_path.expanduser().resolve(),
    )


def _recipe_paths_from_index(
    *,
    index_path: Path,
    root: Path,
    context: str,
    allow_missing: bool,
) -> dict[str, Path]:
    resolved_index_path = index_path.expanduser().resolve()
    if allow_missing and not resolved_index_path.exists():
        return {}
    index = _load_yaml_mapping(resolved_index_path, context=context)
    if index.get("schema") != CORPUS_RECIPE_INDEX_SCHEMA:
        raise RuntimeError(
            f"corpus recipe index schema must be {CORPUS_RECIPE_INDEX_SCHEMA!r}, "
            f"got {index.get('schema')!r}: {resolved_index_path}"
        )
    recipes = _ensure_mapping(index.get("recipes"), context=f"{context} recipes")
    recipe_paths: dict[str, Path] = {}
    for recipe_id, raw_entry in recipes.items():
        if not isinstance(raw_entry, Mapping):
            raise RuntimeError(f"{context} entry {recipe_id!r} must be a mapping")
        recipe_paths[str(recipe_id)] = _recipe_path_from_index_entry(recipe_id, raw_entry, root=root)
    return recipe_paths


def _resolved_recipe_paths(
    *,
    repo_root: Path | None = None,
    sweep_id: str | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Path]:
    resolved_repo_root = (repo_root or _repo_root()).expanduser().resolve()
    recipe_paths = _recipe_paths_from_index(
        index_path=corpus_recipe_index_path(repo_root=resolved_repo_root),
        root=corpus_recipes_root(repo_root=resolved_repo_root),
        context="corpus recipe index",
        allow_missing=False,
    )
    if sweep_id is None:
        return recipe_paths
    recipe_paths.update(
        _recipe_paths_from_index(
            index_path=sweep_corpus_recipe_index_path(
                sweep_id,
                repo_root=resolved_repo_root,
                sweeps_root=sweeps_root,
            ),
            root=sweep_corpus_recipes_root(
                sweep_id,
                repo_root=resolved_repo_root,
                sweeps_root=sweeps_root,
            ),
            context=f"sweep-local corpus recipe index for {sweep_id!r}",
            allow_missing=True,
        )
    )
    return recipe_paths


def _load_recipe_from_path(recipe_id: str, recipe_path: Path) -> CorpusRecipe:
    try:
        payload = _load_yaml_mapping(recipe_path, context=f"corpus recipe {recipe_id!r}")
    except FileNotFoundError as exc:
        raise RuntimeError(f"corpus recipe {recipe_id!r} does not exist: {recipe_path}") from exc
    return _recipe_from_payload(payload, recipe_path=recipe_path)


def load_corpus_recipe(
    recipe_id: str,
    *,
    repo_root: Path | None = None,
    sweep_id: str | None = None,
    sweeps_root: Path | None = None,
) -> CorpusRecipe:
    recipe_paths = _resolved_recipe_paths(
        repo_root=repo_root,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )
    recipe_path = recipe_paths.get(recipe_id)
    if recipe_path is None:
        raise RuntimeError(f"unknown corpus recipe: {recipe_id!r}")
    return _load_recipe_from_path(recipe_id, recipe_path)


def list_corpus_recipes(
    *,
    repo_root: Path | None = None,
    sweep_id: str | None = None,
    sweeps_root: Path | None = None,
) -> list[CorpusRecipe]:
    recipe_paths = _resolved_recipe_paths(
        repo_root=repo_root,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )
    return [
        _load_recipe_from_path(recipe_id, recipe_paths[recipe_id])
        for recipe_id in sorted(recipe_paths)
    ]


def _global_recipe_paths(*, repo_root: Path | None = None) -> dict[str, Path]:
    resolved_repo_root = (repo_root or _repo_root()).expanduser().resolve()
    return _recipe_paths_from_index(
        index_path=corpus_recipe_index_path(repo_root=resolved_repo_root),
        root=corpus_recipes_root(repo_root=resolved_repo_root),
        context="corpus recipe index",
        allow_missing=True,
    )


def _sweep_recipe_paths(
    sweep_id: str,
    *,
    repo_root: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Path] | None:
    resolved_repo_root = (repo_root or _repo_root()).expanduser().resolve()
    index_path = sweep_corpus_recipe_index_path(
        sweep_id,
        repo_root=resolved_repo_root,
        sweeps_root=sweeps_root,
    )
    if not index_path.expanduser().resolve().exists():
        return None
    return _recipe_paths_from_index(
        index_path=index_path,
        root=sweep_corpus_recipes_root(
            sweep_id,
            repo_root=resolved_repo_root,
            sweeps_root=sweeps_root,
        ),
        context=f"sweep-local corpus recipe index for {sweep_id!r}",
        allow_missing=False,
    )


def _stable_recipe_locator(
    recipe_path: Path,
    *,
    repo_root: Path | None = None,
) -> tuple[str | None, str]:
    resolved_recipe_path = recipe_path.expanduser().resolve()
    resolved_repo_root = (repo_root or _repo_root()).expanduser().resolve()
    try:
        relative_path = resolved_recipe_path.relative_to(resolved_repo_root).as_posix()
    except ValueError:
        return None, str(resolved_recipe_path)
    return relative_path, relative_path


def _recipe_identity_payload(recipe: CorpusRecipe) -> dict[str, Any]:
    payload = recipe.to_dict()
    payload.pop("recipe_path", None)
    return payload


def _recipe_storage_context(
    recipe: CorpusRecipe,
    *,
    repo_root: Path | None = None,
) -> CorpusRecipeStorageContext:
    resolved_recipe_path = recipe.recipe_path.expanduser().resolve()
    global_recipe_path = _global_recipe_paths(repo_root=repo_root).get(recipe.recipe_id)
    recipe_relative_path, identity_source = _stable_recipe_locator(
        resolved_recipe_path,
        repo_root=repo_root,
    )
    uses_scoped_identity = (
        global_recipe_path is None
        or global_recipe_path.expanduser().resolve() != resolved_recipe_path
    )
    recipe_identity_payload = {
        "locator": identity_source,
        "recipe": _recipe_identity_payload(recipe),
    }
    return CorpusRecipeStorageContext(
        recipe_identity=sha256_text(
            json.dumps(
                recipe_identity_payload,
                sort_keys=True,
                separators=(",", ":"),
            )
        )[:12],
        recipe_relative_path=recipe_relative_path,
        uses_scoped_identity=uses_scoped_identity,
    )


def corpus_id_for_manifest(
    *,
    recipe_id: str,
    manifest_sha256: str,
    recipe_identity: str | None = None,
) -> str:
    corpus_id = f"{recipe_id}__{manifest_sha256[:12]}"
    if recipe_identity is None:
        return corpus_id
    return f"{corpus_id}__{recipe_identity[:12]}"


def corpus_record_path(
    *,
    recipe_id: str,
    corpus_id: str,
    repo_root: Path | None = None,
) -> Path:
    return corpus_outputs_root(repo_root=repo_root) / recipe_id / corpus_id / "corpus_record.json"


def _latest_pointer_path(
    *,
    recipe_id: str,
    repo_root: Path | None = None,
    recipe_identity: str | None = None,
) -> Path:
    latest_name = "latest.json" if recipe_identity is None else f"latest__{recipe_identity}.json"
    return corpus_outputs_root(repo_root=repo_root) / recipe_id / latest_name


def _write_latest_pointer(
    *,
    recipe_id: str,
    corpus_id: str,
    corpus_ref: str,
    record_path: Path,
    recipe_path: Path,
    recipe_identity: str,
    repo_root: Path | None = None,
    scoped_recipe_identity: str | None = None,
) -> Path:
    payload = {
        "schema": CORPUS_LATEST_SCHEMA,
        "generated_at_utc": utc_now(),
        "recipe_id": str(recipe_id),
        "corpus_id": str(corpus_id),
        "corpus_ref": str(corpus_ref),
        "corpus_record_path": str(record_path.expanduser().resolve()),
        "recipe_path": str(recipe_path.expanduser().resolve()),
        "recipe_identity": str(recipe_identity),
    }
    recipe_relative_path, _identity_source = _stable_recipe_locator(recipe_path, repo_root=repo_root)
    if recipe_relative_path is not None:
        payload["recipe_relative_path"] = recipe_relative_path
    latest_path = _latest_pointer_path(
        recipe_id=recipe_id,
        repo_root=repo_root,
        recipe_identity=scoped_recipe_identity,
    )
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return latest_path


def _load_latest_pointer(
    recipe_id: str,
    *,
    repo_root: Path | None = None,
    recipe_identity: str | None = None,
) -> dict[str, Any] | None:
    latest_path = _latest_pointer_path(
        recipe_id=recipe_id,
        repo_root=repo_root,
        recipe_identity=recipe_identity,
    )
    if not latest_path.exists():
        return None
    payload = _read_json_mapping(latest_path, context=f"corpus latest pointer for {recipe_id!r}")
    if payload.get("schema") != CORPUS_LATEST_SCHEMA:
        raise RuntimeError(
            f"corpus latest pointer schema must be {CORPUS_LATEST_SCHEMA!r}, got {payload.get('schema')!r}: {latest_path}"
        )
    return payload


def _load_corpus_record_payload(record_path: Path, *, context: str) -> dict[str, Any]:
    if not record_path.exists():
        raise RuntimeError(f"corpus record does not exist: {record_path}")
    payload = _read_json_mapping(record_path, context=context)
    if payload.get("schema") != CORPUS_RECORD_SCHEMA:
        raise RuntimeError(
            f"corpus record schema must be {CORPUS_RECORD_SCHEMA!r}, got {payload.get('schema')!r}: {record_path}"
        )
    if "dagzoo_provenance_summary" not in payload and isinstance(payload.get("dagzoo_provenance"), Mapping):
        raw_provenance = cast(Mapping[str, Any], payload["dagzoo_provenance"])
        raw_corpus_ref = payload.get("corpus_ref", raw_provenance.get("corpus_ref"))
        raw_corpus_id = payload.get("corpus_id", raw_provenance.get("corpus_id"))
        raw_surface_label = payload.get("surface_label")
        payload["dagzoo_provenance_summary"] = build_dagzoo_provenance_summary(
            recipe=payload.get("recipe"),
            corpus_ref=_ensure_non_empty_string(raw_corpus_ref, context="corpus record corpus_ref"),
            corpus_id=_ensure_non_empty_string(raw_corpus_id, context="corpus record corpus_id"),
            provenance=raw_provenance,
            surface_label=_optional_string(raw_surface_label),
        )
    return payload
