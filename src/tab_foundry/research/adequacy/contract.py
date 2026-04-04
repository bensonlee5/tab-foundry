"""Latent-target contract inspection for adequacy pilot corpora."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from tab_foundry.data.corpus_lookup import hydrate_corpus_record_manifest_characteristics
from tab_foundry.data.dataset import load_manifest_record_catalog
from tab_foundry.research.synthetic_adequacy import SyntheticAdequacyBlock

from .shared import (
    _CANARY_BLOCK_ID,
    _LATENT_TARGET_DERIVATION,
    _ensure_mapping,
    _finite_float_or_none,
    _int_or_none,
    _normalize_contract_check_mode,
    _optional_mapping,
)


def _manifest_path_from_corpus_record(corpus_record: Mapping[str, Any]) -> Path:
    manifest = _ensure_mapping(corpus_record.get("manifest"), context="corpus_record.manifest")
    raw_manifest_path = manifest.get("manifest_path")
    if not isinstance(raw_manifest_path, str) or not raw_manifest_path.strip():
        raise RuntimeError("corpus_record.manifest.manifest_path must be a non-empty string")
    return Path(raw_manifest_path).expanduser().resolve()


def _row_total_from_record(record: Mapping[str, Any]) -> int:
    return int(record.get("n_train", 0)) + int(record.get("n_test", 0))


def _classification_manifest_records(manifest_path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(manifest_path)
    records = []
    for raw_record in table.to_pylist():
        record = {str(key): value for key, value in cast(Mapping[str, Any], raw_record).items()}
        if str(record.get("task", "")).strip().lower() != "classification":
            continue
        records.append(record)
    return records


def _normalized_value_count_payload(values: pa.Array | pa.ChunkedArray) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for entry in pc.value_counts(values).to_pylist():
        normalized = "missing"
        value = entry.get("values")
        if value is not None:
            normalized_value = str(value).strip()
            normalized = normalized_value or "missing"
        counts[normalized] += int(entry.get("counts", 0))
    return {key: int(counts[key]) for key in sorted(counts)}


def _positive_int_value_count_payload(values: pa.Array | pa.ChunkedArray) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in pc.value_counts(values).to_pylist():
        raw_value = entry.get("values")
        raw_count = entry.get("counts", 0)
        if raw_value is None or raw_count is None:
            continue
        try:
            normalized_value = int(raw_value)
            normalized_count = int(raw_count)
        except (TypeError, ValueError):
            continue
        if normalized_value <= 0 or normalized_count <= 0:
            continue
        counts[str(normalized_value)] = normalized_count
    return {key: counts[key] for key in sorted(counts, key=int)}


def _manifest_contract_stats_from_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    row_total_counts = Counter(_row_total_from_record(record) for record in records)
    split_counts = Counter(str(record.get("split", "")).strip() or "missing" for record in records)
    return {
        "source": "full_manifest_records",
        "classification_task_count": len(records),
        "row_total_counts": {
            str(row_total): int(count)
            for row_total, count in sorted(row_total_counts.items())
            if int(row_total) > 0 and int(count) > 0
        },
        "split_counts": {
            split: int(count)
            for split, count in sorted(split_counts.items())
            if int(count) > 0
        },
    }


def _manifest_contract_stats_from_characteristics(
    corpus_record: Mapping[str, Any],
) -> dict[str, Any] | None:
    manifest = _optional_mapping(corpus_record.get("manifest"))
    if manifest is None:
        return None
    characteristics = _optional_mapping(manifest.get("characteristics"))
    if characteristics is None:
        return None
    classification_task_count = _int_or_none(characteristics.get("classification_task_count"))
    raw_row_total_counts = _optional_mapping(characteristics.get("row_total_counts"))
    raw_split_counts = _optional_mapping(characteristics.get("classification_split_counts"))
    if (
        classification_task_count is None
        or raw_row_total_counts is None
        or raw_split_counts is None
    ):
        return None
    row_total_counts: dict[str, int] = {}
    for row_total, count in raw_row_total_counts.items():
        parsed_count = _int_or_none(count)
        if parsed_count is None or parsed_count <= 0:
            continue
        row_total_counts[str(row_total)] = parsed_count
    split_counts: dict[str, int] = {}
    for split, count in raw_split_counts.items():
        parsed_count = _int_or_none(count)
        if parsed_count is None or parsed_count <= 0:
            continue
        normalized_split = str(split).strip() or "missing"
        split_counts[normalized_split] = split_counts.get(normalized_split, 0) + parsed_count
    return {
        "source": "manifest_characteristics",
        "classification_task_count": int(classification_task_count),
        "row_total_counts": {
            key: row_total_counts[key]
            for key in sorted(row_total_counts, key=int)
        },
        "split_counts": {
            key: split_counts[key]
            for key in sorted(split_counts)
        },
    }


def _scan_manifest_contract_stats_fast(manifest_path: Path) -> dict[str, Any]:
    projected = pq.read_table(
        manifest_path,
        columns=["task", "split", "n_train", "n_test"],
    )
    if "task" not in projected.column_names:
        return {
            "source": "arrow_projected_scan",
            "classification_task_count": 0,
            "row_total_counts": {},
            "split_counts": {},
        }
    classification_mask = pc.fill_null(
        pc.equal(projected["task"], pa.scalar("classification")),
        False,
    )
    filtered = projected.filter(classification_mask)
    classification_task_count = int(filtered.num_rows)
    split_counts = (
        {}
        if "split" not in filtered.column_names
        else _normalized_value_count_payload(filtered["split"])
    )
    row_total_counts: dict[str, int] = {}
    if {"n_train", "n_test"}.issubset(set(filtered.column_names)):
        n_train = pc.cast(filtered["n_train"], pa.int64(), safe=False)
        n_test = pc.cast(filtered["n_test"], pa.int64(), safe=False)
        valid_mask = pc.fill_null(pc.and_(pc.is_valid(n_train), pc.is_valid(n_test)), False)
        total_rows = pc.add(pc.filter(n_train, valid_mask), pc.filter(n_test, valid_mask))
        positive_rows = pc.filter(
            total_rows,
            pc.fill_null(pc.greater(total_rows, pa.scalar(0, type=pa.int64())), False),
        )
        row_total_counts = _positive_int_value_count_payload(positive_rows)
    return {
        "source": "arrow_projected_scan",
        "classification_task_count": classification_task_count,
        "row_total_counts": row_total_counts,
        "split_counts": split_counts,
    }


def _load_manifest_contract_stats(
    *,
    manifest_path: Path,
    corpus_record: Mapping[str, Any],
) -> dict[str, Any]:
    characteristics_stats = _manifest_contract_stats_from_characteristics(corpus_record)
    if characteristics_stats is not None:
        return characteristics_stats
    return _scan_manifest_contract_stats_fast(manifest_path)


def validate_latent_target_metadata(
    catalog_record: Mapping[str, Any],
    *,
    n_features: int | None = None,
) -> dict[str, Any]:
    missing_reasons: list[str] = []
    target_derivation = catalog_record.get("target_derivation")
    if target_derivation != _LATENT_TARGET_DERIVATION:
        missing_reasons.append(
            f"catalog.target_derivation must equal {_LATENT_TARGET_DERIVATION!r}"
        )

    feature_count = int(n_features) if n_features is not None else None
    target_relevant_feature_count: int | None = None
    target_relevant_feature_fraction: float | None = None
    target_relevance = _optional_mapping(catalog_record.get("target_relevance"))
    if target_relevance is None:
        missing_reasons.append("catalog.target_relevance is missing")
    else:
        target_relevant_feature_count = _int_or_none(target_relevance.get("feature_count"))
        if target_relevant_feature_count is None:
            missing_reasons.append("catalog.target_relevance.feature_count must be an integer")
        elif target_relevant_feature_count < 0:
            missing_reasons.append(
                "catalog.target_relevance.feature_count must be non-negative"
            )

        target_relevant_feature_fraction = _finite_float_or_none(
            target_relevance.get("feature_fraction")
        )
        if target_relevant_feature_fraction is None or not (
            0.0 <= target_relevant_feature_fraction <= 1.0
        ):
            missing_reasons.append(
                "catalog.target_relevance.feature_fraction must be finite in [0, 1]"
            )

        if (
            feature_count is not None
            and target_relevant_feature_count is not None
            and target_relevant_feature_count > feature_count
        ):
            missing_reasons.append(
                "catalog.target_relevance.feature_count must be within [0, n_features]"
            )
        if (
            feature_count is not None
            and target_relevant_feature_count is not None
            and target_relevant_feature_fraction is not None
            and feature_count > 0
        ):
            expected_fraction = float(target_relevant_feature_count) / float(feature_count)
            if not math.isclose(
                target_relevant_feature_fraction,
                expected_fraction,
                rel_tol=1.0e-9,
                abs_tol=1.0e-9,
            ):
                missing_reasons.append(
                    "catalog.target_relevance.feature_fraction does not match feature_count / n_features"
                )

    return {
        "present": not missing_reasons,
        "target_derivation": (
            None if target_derivation is None else str(target_derivation)
        ),
        "feature_count": feature_count,
        "target_relevant_feature_count": target_relevant_feature_count,
        "target_relevant_feature_fraction": target_relevant_feature_fraction,
        "missing_reasons": missing_reasons,
    }


def inspect_corpus_latent_target_contract(
    *,
    block: SyntheticAdequacyBlock,
    corpus_record: Mapping[str, Any],
    mode: str = "full",
) -> dict[str, Any]:
    resolved_mode = _normalize_contract_check_mode(mode)
    hydrated_corpus_record = _ensure_mapping(corpus_record, context="corpus_record")
    if resolved_mode == "fast" and block.block_id != _CANARY_BLOCK_ID:
        try:
            hydrated_corpus_record = hydrate_corpus_record_manifest_characteristics(
                hydrated_corpus_record
            )
        except Exception:
            hydrated_corpus_record = _ensure_mapping(corpus_record, context="corpus_record")
    manifest_path = _manifest_path_from_corpus_record(hydrated_corpus_record)
    records: list[dict[str, Any]] | None = None
    if resolved_mode == "full" or block.block_id == _CANARY_BLOCK_ID:
        records = _classification_manifest_records(manifest_path)
        contract_stats = _manifest_contract_stats_from_records(records)
    else:
        contract_stats = _load_manifest_contract_stats(
            manifest_path=manifest_path,
            corpus_record=hydrated_corpus_record,
        )
    row_total_counts = {
        str(key): int(value)
        for key, value in cast(Mapping[str, Any], contract_stats["row_total_counts"]).items()
    }
    split_counts = {
        str(key): int(value)
        for key, value in cast(Mapping[str, Any], contract_stats["split_counts"]).items()
    }
    catalog_validation_mode = (
        "sampled_records"
        if resolved_mode == "full" or block.block_id == _CANARY_BLOCK_ID
        else "skipped"
    )

    sample_records: list[dict[str, Any]] = []
    missing_reasons: list[str] = []
    if catalog_validation_mode == "sampled_records":
        if records is None:
            records = _classification_manifest_records(manifest_path)
        for row_total in block.n_ladder:
            sample_record = next(
                (record for record in records if _row_total_from_record(record) == int(row_total)),
                None,
            )
            if sample_record is None:
                missing_reasons.append(
                    f"manifest is missing a classification record for row_total={int(row_total)}"
                )
                continue
            catalog_record = load_manifest_record_catalog(
                manifest_path,
                record=sample_record,
            )
            validation = validate_latent_target_metadata(
                catalog_record,
                n_features=int(sample_record["n_features"]),
            )
            sample_entry = {
                "row_total": int(row_total),
                "dataset_id": str(sample_record.get("dataset_id", "unknown")),
                "split": str(sample_record.get("split", "unknown")),
                "dataset_index": int(sample_record["dataset_index"]),
                "n_train": int(sample_record["n_train"]),
                "n_test": int(sample_record["n_test"]),
                "n_features": int(sample_record["n_features"]),
                "n_classes": int(sample_record["n_classes"]),
                **validation,
            }
            sample_records.append(sample_entry)
            if not bool(validation["present"]):
                for reason in cast(list[str], validation["missing_reasons"]):
                    missing_reasons.append(f"row_total={int(row_total)}: {reason}")
    else:
        for row_total in block.n_ladder:
            if int(row_total_counts.get(str(int(row_total)), 0)) > 0:
                continue
            missing_reasons.append(
                f"manifest is missing a classification record for row_total={int(row_total)}"
            )

    provenance_summary = (
        _optional_mapping(hydrated_corpus_record.get("dagzoo_provenance_summary")) or {}
    )
    dagzoo_provenance = _optional_mapping(hydrated_corpus_record.get("dagzoo_provenance")) or {}
    if provenance_summary.get("target_derivation") != _LATENT_TARGET_DERIVATION:
        missing_reasons.append(
            f"corpus_record.dagzoo_provenance_summary.target_derivation must equal {_LATENT_TARGET_DERIVATION!r}"
        )

    invocation_payloads = cast(list[Any], dagzoo_provenance.get("invocations", []))
    target_accepted_datasets = 0
    if not invocation_payloads:
        missing_reasons.append("corpus_record.dagzoo_provenance.invocations is missing")
    for invocation in invocation_payloads:
        if not isinstance(invocation, Mapping):
            missing_reasons.append("corpus_record.dagzoo_provenance.invocations must contain mappings")
            continue
        normalized_invocation = {
            str(key): value for key, value in cast(Mapping[str, Any], invocation).items()
        }
        invocation_id = str(normalized_invocation.get("invocation_id", "unknown"))
        requested_count = _int_or_none(normalized_invocation.get("num_datasets"))
        if requested_count is None or requested_count <= 0:
            missing_reasons.append(
                f"invocation {invocation_id!r} is missing a positive num_datasets target"
            )
            continue
        target_accepted_datasets += requested_count
        filter_payload = _optional_mapping(normalized_invocation.get("filter"))
        if filter_payload is None:
            missing_reasons.append(
                f"invocation {invocation_id!r} is missing accepted_only filter provenance"
            )
            continue
        if str(filter_payload.get("filter_policy", "")).strip() != "accepted_only":
            missing_reasons.append(
                f"invocation {invocation_id!r} filter_policy must equal 'accepted_only'"
            )
        curated_accepted = _int_or_none(filter_payload.get("curated_accepted_datasets"))
        if curated_accepted != requested_count:
            missing_reasons.append(
                f"invocation {invocation_id!r} curated_accepted_datasets "
                f"must equal authored target {requested_count}, got {curated_accepted!r}"
            )
        accepted_count = _int_or_none(filter_payload.get("accepted_datasets"))
        if accepted_count is None or accepted_count < requested_count:
            missing_reasons.append(
                f"invocation {invocation_id!r} accepted_datasets must be at least {requested_count}"
            )
        for required_path_key in ("filter_manifest_path", "filter_summary_path", "curated_dir"):
            raw_path = filter_payload.get(required_path_key)
            if not isinstance(raw_path, str) or not raw_path.strip():
                missing_reasons.append(
                    f"invocation {invocation_id!r} filter provenance is missing {required_path_key}"
                )
                continue
            if not Path(raw_path).expanduser().resolve().exists():
                missing_reasons.append(
                    f"invocation {invocation_id!r} {required_path_key} does not exist"
                )

    accepted_datasets = _int_or_none(provenance_summary.get("accepted_datasets"))
    curated_accepted_datasets = _int_or_none(provenance_summary.get("curated_accepted_datasets"))
    if str(provenance_summary.get("filter_policy", "")).strip() != "accepted_only":
        missing_reasons.append(
            "corpus_record.dagzoo_provenance_summary.filter_policy must equal 'accepted_only'"
        )
    if target_accepted_datasets > 0:
        if accepted_datasets is None or accepted_datasets < target_accepted_datasets:
            missing_reasons.append(
                "corpus_record.dagzoo_provenance_summary.accepted_datasets must meet the authored target"
            )
        if curated_accepted_datasets != target_accepted_datasets:
            missing_reasons.append(
                "corpus_record.dagzoo_provenance_summary.curated_accepted_datasets must equal the authored target"
            )

    return {
        "required": True,
        "present": not missing_reasons,
        "contract_check_mode": resolved_mode,
        "stats_source": str(contract_stats["source"]),
        "catalog_validation_mode": catalog_validation_mode,
        "provenance": {
            "target_derivation": provenance_summary.get("target_derivation"),
            "target_relevant_feature_count_range": provenance_summary.get(
                "target_relevant_feature_count_range"
            ),
            "target_relevant_feature_fraction_range": provenance_summary.get(
                "target_relevant_feature_fraction_range"
            ),
        },
        "filter_provenance": {
            "filter_policy": provenance_summary.get("filter_policy"),
            "target_accepted_datasets": target_accepted_datasets,
            "accepted_datasets": accepted_datasets,
            "rejected_datasets": _int_or_none(provenance_summary.get("rejected_datasets")),
            "curated_accepted_datasets": curated_accepted_datasets,
            "acceptance_rate": _finite_float_or_none(provenance_summary.get("acceptance_rate")),
        },
        "manifest_path": str(manifest_path),
        "classification_task_count": int(contract_stats["classification_task_count"]),
        "row_total_counts": row_total_counts,
        "split_counts": split_counts,
        "sample_records": sample_records,
        "missing_reasons": missing_reasons,
    }


__all__ = [
    "inspect_corpus_latent_target_contract",
    "validate_latent_target_metadata",
]
