"""Invocation materialization helpers for corpus recipes."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import json
import math
from pathlib import Path
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence, cast
import yaml

from .dagzoo_handoff_support import (
    DagzooGeneratedIdentityAccumulator,
    DagzooHandoffInfo,
    load_dagzoo_handoff_info,
    verify_dagzoo_handoff_matches_generated_corpus,
)
from tab_foundry.hashing import sha256_path

from .corpus_loading import (
    DagzooInvocationRecipe,
    _copy_jsonable,
    _deep_merge_payload,
    _load_yaml_mapping,
    _repo_root,
    _resolve_from_root,
    load_corpus_recipe,
)
from .corpus_materialization_shared import (
    _ACCEPTED_ONLY_MAX_GENERATED_MULTIPLIER,
    _INITIAL_ACCEPTED_ONLY_EXPECTED_ACCEPTANCE_RATE,
    _SUBPROCESS_POLL_INTERVAL_SECONDS,
    _clamp_expected_acceptance_rate,
    _drop_none_values,
    _float_or_none,
    _int_or_none,
    _read_json_mapping,
    _resolve_materialize_processes,
    _resolve_materialize_worker_threads,
    _snapshot_tree,
)
from .dagzoo_workflow import (
    DagzooFilterConfig,
    DagzooGenerateConfig,
    build_dagzoo_filter_argv,
    build_dagzoo_generate_argv,
    run_dagzoo_filter,
    run_dagzoo_generate,
)


@dataclass(slots=True)
class _ActiveInvocationProcess:
    process: subprocess.Popen[str]
    spec: DagzooInvocationRecipe


def _invocation_filter_payloads(
    invocation_payloads: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    return [
        {str(key): value for key, value in cast(Mapping[str, Any], payload["filter"]).items()}
        for payload in invocation_payloads
        if isinstance(payload.get("filter"), Mapping)
    ]


def _dagzoo_public_catalog_paths(generated_dir: Path) -> list[Path]:
    resolved_generated_dir = generated_dir.expanduser().resolve()
    catalog_paths = sorted(resolved_generated_dir.rglob("dataset_catalog.ndjson"))
    if catalog_paths:
        return catalog_paths
    metadata_paths = sorted(resolved_generated_dir.rglob("metadata.ndjson"))
    if metadata_paths:
        return metadata_paths
    raise RuntimeError(
        "dagzoo generated directory does not contain dataset_catalog.ndjson or metadata.ndjson: "
        f"{resolved_generated_dir}"
    )


def _scan_dagzoo_generated_identity(generated_dir: Path) -> DagzooGeneratedIdentityAccumulator:
    scanned_identity = DagzooGeneratedIdentityAccumulator()
    for catalog_path in _dagzoo_public_catalog_paths(generated_dir):
        for line_number, raw_line in enumerate(
            catalog_path.read_text(encoding="utf-8").splitlines(),
            start=1,
        ):
            if not raw_line.strip():
                continue
            try:
                payload = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    "failed to parse dagzoo catalog record while verifying handoff: "
                    f"path={catalog_path}, line={line_number}"
                ) from exc
            if not isinstance(payload, Mapping):
                raise RuntimeError(
                    "dagzoo catalog NDJSON record must decode to an object: "
                    f"path={catalog_path}, line={line_number}"
                )
            dataset_index = payload.get("dataset_index")
            if dataset_index is None:
                raise RuntimeError(
                    "dagzoo catalog record missing dataset_index while verifying handoff: "
                    f"path={catalog_path}, line={line_number}"
                )
            scanned_identity.add_record(
                payload,
                record_path=catalog_path,
                dataset_index=int(dataset_index),
            )
    return scanned_identity


def _invocation_paths(*, corpus_root: Path, invocation_id: str) -> tuple[Path, Path]:
    invocation_root = corpus_root / "invocations" / invocation_id
    return invocation_root, invocation_root / "handoff_manifest.json"


def _invocation_rounds_root(*, corpus_root: Path, invocation_id: str) -> Path:
    invocation_root, _handoff_manifest_path = _invocation_paths(
        corpus_root=corpus_root,
        invocation_id=invocation_id,
    )
    return invocation_root / ".rounds"


def _invocation_round_root(*, corpus_root: Path, invocation_id: str, round_index: int) -> Path:
    return _invocation_rounds_root(
        corpus_root=corpus_root,
        invocation_id=invocation_id,
    ) / f"round_{int(round_index):02d}"


def _invocation_filter_root(*, corpus_root: Path, invocation_id: str) -> Path:
    invocation_root, _handoff_manifest_path = _invocation_paths(
        corpus_root=corpus_root,
        invocation_id=invocation_id,
    )
    return invocation_root / "filter"


def _invocation_curated_root(*, corpus_root: Path, invocation_id: str) -> Path:
    invocation_root, _handoff_manifest_path = _invocation_paths(
        corpus_root=corpus_root,
        invocation_id=invocation_id,
    )
    return invocation_root / "curated"


def _invocation_materialization_summary_path(*, corpus_root: Path, invocation_id: str) -> Path:
    invocation_root, _handoff_manifest_path = _invocation_paths(
        corpus_root=corpus_root,
        invocation_id=invocation_id,
    )
    return invocation_root / "materialization_summary.json"


def _invocation_rendered_config_path(*, corpus_root: Path, invocation_id: str) -> Path:
    invocation_root, _handoff_manifest_path = _invocation_paths(
        corpus_root=corpus_root,
        invocation_id=invocation_id,
    )
    return invocation_root / "dagzoo_config.yaml"


def _verified_invocation_handoff(
    *,
    corpus_root: Path,
    spec: DagzooInvocationRecipe,
) -> DagzooHandoffInfo:
    _invocation_root, handoff_manifest_path = _invocation_paths(
        corpus_root=corpus_root,
        invocation_id=spec.invocation_id,
    )
    handoff = load_dagzoo_handoff_info(handoff_manifest_path)
    verify_dagzoo_handoff_matches_generated_corpus(
        handoff,
        scanned_identity=_scan_dagzoo_generated_identity(handoff.generated_dir),
    )
    return handoff


def _manifest_source_root(*, handoff: DagzooHandoffInfo, filter_policy: str) -> Path:
    normalized_filter_policy = str(filter_policy).strip()
    if normalized_filter_policy == "accepted_only":
        if handoff.curated_dir is None:
            raise RuntimeError(
                "filter_policy='accepted_only' requires a curated dagzoo corpus. "
                "Run `dagzoo filter --in "
                f"{handoff.generated_dir} --out <filter_dir> --curated-out <curated_dir>` first."
            )
        return handoff.curated_dir
    return handoff.generated_dir


def _invocation_requested_config_ref(spec: DagzooInvocationRecipe) -> str:
    config_ref = spec.config_ref if spec.config_ref is not None else spec.base_config_ref
    if config_ref is None:
        raise RuntimeError(f"invocation {spec.invocation_id!r} does not define a dagzoo config")
    return str(config_ref)


def _invocation_dagzoo_config_path(
    *,
    dagzoo_root: Path,
    corpus_root: Path,
    spec: DagzooInvocationRecipe,
    write_rendered_config: bool,
) -> Path:
    if spec.config_ref is not None:
        return Path(str(spec.config_ref))
    if spec.base_config_ref is None:
        raise RuntimeError(f"invocation {spec.invocation_id!r} does not define a dagzoo config")
    rendered_config_path = _invocation_rendered_config_path(
        corpus_root=corpus_root,
        invocation_id=spec.invocation_id,
    )
    if write_rendered_config:
        base_config_path = _resolve_from_root(dagzoo_root, Path(spec.base_config_ref))
        merged_payload = _deep_merge_payload(
            _load_yaml_mapping(
                base_config_path,
                context=f"dagzoo base config for invocation {spec.invocation_id!r}",
            ),
            spec.config_overrides,
        )
        rendered_config_path.parent.mkdir(parents=True, exist_ok=True)
        rendered_config_path.write_text(
            yaml.safe_dump(merged_payload, sort_keys=False, allow_unicode=False),
            encoding="utf-8",
        )
    return rendered_config_path.resolve()


def _invocation_fixed_dimension(
    spec: DagzooInvocationRecipe,
    *,
    base_key: str,
) -> int | None:
    dataset_overrides = spec.config_overrides.get("dataset")
    if not isinstance(dataset_overrides, Mapping):
        return None
    direct_value = _int_or_none(dataset_overrides.get(base_key))
    if direct_value is not None:
        return direct_value
    minimum = _int_or_none(dataset_overrides.get(f"{base_key}_min"))
    maximum = _int_or_none(dataset_overrides.get(f"{base_key}_max"))
    if minimum is not None and maximum is not None and minimum == maximum:
        return minimum
    return None


def _invocation_shape_key(
    spec: DagzooInvocationRecipe,
) -> tuple[int | None, int | None, int | None]:
    row_total = _int_or_none(spec.rows)
    n_features = _invocation_fixed_dimension(spec, base_key="n_features")
    n_classes = _invocation_fixed_dimension(spec, base_key="n_classes")
    return (row_total, n_features, n_classes)


def _dagzoo_generate_config(
    *,
    dagzoo_root: Path,
    corpus_root: Path,
    spec: DagzooInvocationRecipe,
    write_rendered_config: bool,
    handoff_root: Path | None = None,
    num_datasets: int | None = None,
    worker_threads: int | None = None,
) -> DagzooGenerateConfig:
    resolved_handoff_root = (
        handoff_root.expanduser().resolve()
        if handoff_root is not None
        else _invocation_paths(
            corpus_root=corpus_root,
            invocation_id=spec.invocation_id,
        )[0]
    )
    return DagzooGenerateConfig(
        dagzoo_root=dagzoo_root,
        dagzoo_config=_invocation_dagzoo_config_path(
            dagzoo_root=dagzoo_root,
            corpus_root=corpus_root,
            spec=spec,
            write_rendered_config=write_rendered_config,
        ),
        handoff_root=resolved_handoff_root,
        num_datasets=int(spec.num_datasets if num_datasets is None else num_datasets),
        seed=spec.seed,
        rows=spec.rows,
        device=spec.device,
        hardware_policy=str(spec.hardware_policy),
        diagnostics=bool(spec.diagnostics),
        diagnostics_out_dir=(
            None if spec.diagnostics_out_dir is None else Path(str(spec.diagnostics_out_dir))
        ),
        missing_rate=spec.missing_rate,
        missing_mechanism=spec.missing_mechanism,
        missing_mar_observed_fraction=spec.missing_mar_observed_fraction,
        missing_mar_logit_scale=spec.missing_mar_logit_scale,
        missing_mnar_logit_scale=spec.missing_mnar_logit_scale,
        worker_threads=worker_threads,
        set_overrides=(),
    )


def _stringify_command(argv: list[str]) -> str:
    return " ".join(str(part) for part in argv)


def _aggregate_handoff_provenance(
    handoff_provenances: list[Mapping[str, Any]],
) -> dict[str, Any] | None:
    target_derivations = sorted(
        {
            str(value).strip()
            for provenance in handoff_provenances
            for value in (provenance.get("target_derivation"),)
            if isinstance(value, str) and str(value).strip()
        }
    )
    count_bounds = [
        bound
        for provenance in handoff_provenances
        for bound in (
            provenance.get("target_relevant_feature_count_range"),
            provenance.get("target_parent_count_range"),
        )
        if isinstance(bound, Mapping)
    ]
    fraction_bounds = [
        bound
        for provenance in handoff_provenances
        for bound in (
            provenance.get("target_relevant_feature_fraction_range"),
            provenance.get("target_parent_fraction_range"),
        )
        if isinstance(bound, Mapping)
    ]
    minimum_counts = [
        int(bound["min"])
        for bound in count_bounds
        if bound.get("min") is not None
    ]
    maximum_counts = [
        int(bound["max"])
        for bound in count_bounds
        if bound.get("max") is not None
    ]
    minimum_fractions = [
        float(bound["min"])
        for bound in fraction_bounds
        if bound.get("min") is not None
    ]
    maximum_fractions = [
        float(bound["max"])
        for bound in fraction_bounds
        if bound.get("max") is not None
    ]
    payload = _drop_none_values(
        {
            "target_derivation": (
                target_derivations[0] if len(target_derivations) == 1 else None
            ),
            "target_relevant_feature_count_range": (
                {
                    "min": min(minimum_counts),
                    "max": max(maximum_counts),
                }
                if minimum_counts and maximum_counts
                else None
            ),
            "target_relevant_feature_fraction_range": (
                {
                    "min": min(minimum_fractions),
                    "max": max(maximum_fractions),
                }
                if minimum_fractions and maximum_fractions
                else None
            ),
        }
    )
    return payload or None


def _elapsed_seconds_since(start_time: float) -> float:
    return max(0.0, float(time.perf_counter() - start_time))


def _sum_elapsed_seconds(*values: float | None) -> float | None:
    resolved = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not resolved:
        return None
    return float(sum(resolved))


def _datasets_per_minute(
    *,
    dataset_count: int | None,
    elapsed_seconds: float | None,
) -> float | None:
    if dataset_count is None or elapsed_seconds is None or float(elapsed_seconds) <= 0.0:
        return None
    return float(dataset_count) / float(elapsed_seconds) * 60.0


def _generated_datasets_from_handoff(
    handoff: DagzooHandoffInfo,
    *,
    fallback: int,
) -> int:
    summary = handoff.to_summary_dict()
    if isinstance(summary, Mapping):
        resolved = _int_or_none(summary.get("generated_datasets"))
        if resolved is not None and resolved >= 0:
            return int(resolved)
    return int(fallback)


def _copy_curated_round_shards(
    *,
    round_curated_dir: Path,
    final_curated_dir: Path,
    next_shard_index: int,
    max_datasets: int | None = None,
) -> tuple[int, int]:
    if not round_curated_dir.exists():
        return next_shard_index, 0
    final_curated_dir.mkdir(parents=True, exist_ok=True)
    remaining_datasets = None if max_datasets is None else max(0, int(max_datasets))
    copied_datasets = 0
    for shard_dir in sorted(path for path in round_curated_dir.glob("shard_*") if path.is_dir()):
        if remaining_datasets == 0:
            break
        destination = final_curated_dir / f"shard_{next_shard_index:05d}"
        resolved_shard_dir = shard_dir
        if not any(
            candidate.exists()
            for candidate in (
                resolved_shard_dir / "dataset_catalog.ndjson",
                resolved_shard_dir / "metadata.ndjson",
            )
        ):
            nested_shards = sorted(
                path for path in resolved_shard_dir.glob("shard_*") if path.is_dir()
            )
            if len(nested_shards) == 1:
                resolved_shard_dir = nested_shards[0]
        shard_catalog_path = next(
            (
                candidate
                for candidate in (
                    resolved_shard_dir / "dataset_catalog.ndjson",
                    resolved_shard_dir / "metadata.ndjson",
                )
                if candidate.exists()
            ),
            None,
        )
        if shard_catalog_path is None:
            raise RuntimeError(
                f"curated shard is missing dataset catalog metadata: {resolved_shard_dir}"
            )
        catalog_lines = [
            line
            for line in shard_catalog_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        if not catalog_lines:
            continue
        if remaining_datasets is None or len(catalog_lines) <= remaining_datasets:
            _snapshot_tree(resolved_shard_dir, destination)
            copied_in_shard = len(catalog_lines)
        else:
            selected_lines = catalog_lines[:remaining_datasets]
            selected_dataset_indices = [
                int(
                    cast(Mapping[str, Any], json.loads(raw_line)).get(
                        "dataset_index",
                        -1,
                    )
                )
                for raw_line in selected_lines
            ]
            if any(dataset_index < 0 for dataset_index in selected_dataset_indices):
                raise RuntimeError(
                    "curated shard catalog record is missing dataset_index while trimming "
                    f"accepted_only output: {shard_catalog_path}"
                )
            destination.mkdir(parents=True, exist_ok=True)
            (destination / shard_catalog_path.name).write_text(
                "\n".join(selected_lines) + "\n",
                encoding="utf-8",
            )
            for split_filename in ("train.parquet", "test.parquet"):
                split_path = resolved_shard_dir / split_filename
                if not split_path.exists():
                    raise RuntimeError(
                        f"curated shard is missing split parquet while trimming output: {split_path}"
                    )
                table = pq.read_table(split_path)
                dataset_index_values = pa.array(
                    selected_dataset_indices,
                    type=table["dataset_index"].type,
                )
                mask = pc.is_in(table["dataset_index"], value_set=dataset_index_values)
                pq.write_table(
                    table.filter(mask),
                    destination / split_filename,
                    compression="zstd",
                )
            copied_in_shard = len(selected_lines)
        next_shard_index += 1
        copied_datasets += copied_in_shard
        if remaining_datasets is not None:
            remaining_datasets -= copied_in_shard
    return next_shard_index, copied_datasets


def _write_accepted_only_filter_artifacts(
    *,
    filter_root: Path,
    rounds: Sequence[Mapping[str, Any]],
    target_accepted_datasets: int,
    total_generated_datasets: int,
    accepted_datasets: int,
    rejected_datasets: int,
    curated_accepted_datasets: int,
    materialize_worker_threads: int | None = None,
) -> None:
    filter_root.mkdir(parents=True, exist_ok=True)
    manifest_path = filter_root / "filter_manifest.ndjson"
    with manifest_path.open("w", encoding="utf-8") as handle:
        for round_payload in rounds:
            round_manifest_path = Path(str(round_payload["filter_manifest_path"]))
            if not round_manifest_path.exists():
                raise RuntimeError(f"round filter manifest is missing: {round_manifest_path}")
            text = round_manifest_path.read_text(encoding="utf-8")
            if text:
                handle.write(text)
                if not text.endswith("\n"):
                    handle.write("\n")
    summary_path = filter_root / "filter_summary.json"
    acceptance_rate = (
        None
        if total_generated_datasets <= 0
        else float(accepted_datasets) / float(total_generated_datasets)
    )
    summary_path.write_text(
        json.dumps(
            {
                "filter_policy": "accepted_only",
                "round_count": len(rounds),
                "target_accepted_datasets": int(target_accepted_datasets),
                "total_datasets": int(total_generated_datasets),
                "accepted_datasets": int(accepted_datasets),
                "rejected_datasets": int(rejected_datasets),
                "curated_accepted_datasets": int(curated_accepted_datasets),
                "acceptance_rate": acceptance_rate,
                "generated_budget_cap": (
                    int(target_accepted_datasets) * _ACCEPTED_ONLY_MAX_GENERATED_MULTIPLIER
                ),
                "materialize_worker_threads": (
                    None
                    if materialize_worker_threads is None
                    else int(materialize_worker_threads)
                ),
                "rounds": [
                    {
                        "round_index": int(round_payload["round_index"]),
                        "requested_generated_datasets": int(
                            round_payload["requested_generated_datasets"]
                        ),
                        "generated_datasets": int(round_payload["generated_datasets"]),
                        "accepted_datasets": int(round_payload["accepted_datasets"]),
                        "rejected_datasets": int(round_payload["rejected_datasets"]),
                        "curated_accepted_datasets": int(
                            round_payload["curated_accepted_datasets"]
                        ),
                    }
                    for round_payload in rounds
                ],
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _materialize_accepted_only_invocation(
    *,
    dagzoo_root: Path,
    corpus_root: Path,
    spec: DagzooInvocationRecipe,
    materialize_worker_threads: int | None = None,
    initial_expected_acceptance_rate: float | None = None,
) -> None:
    invocation_start_time = time.perf_counter()
    invocation_root, _handoff_manifest_path = _invocation_paths(
        corpus_root=corpus_root,
        invocation_id=spec.invocation_id,
    )
    invocation_root.mkdir(parents=True, exist_ok=True)
    accepted_target = int(spec.num_datasets)
    generated_budget_cap = accepted_target * _ACCEPTED_ONLY_MAX_GENERATED_MULTIPLIER
    final_filter_root = _invocation_filter_root(
        corpus_root=corpus_root,
        invocation_id=spec.invocation_id,
    )
    final_curated_root = _invocation_curated_root(
        corpus_root=corpus_root,
        invocation_id=spec.invocation_id,
    )

    total_generated_datasets = 0
    accepted_datasets = 0
    rejected_datasets = 0
    curated_accepted_datasets = 0
    next_shard_index = 0
    round_payloads: list[dict[str, Any]] = []
    handoff_provenances: list[Mapping[str, Any]] = []

    round_index = 1
    while (
        curated_accepted_datasets < accepted_target
        and total_generated_datasets < generated_budget_cap
    ):
        round_start_time = time.perf_counter()
        remaining_generated_budget = generated_budget_cap - total_generated_datasets
        remaining_accepted_datasets = accepted_target - curated_accepted_datasets
        requested_generated_datasets = remaining_accepted_datasets
        if total_generated_datasets <= 0:
            expected_acceptance_rate = _clamp_expected_acceptance_rate(
                initial_expected_acceptance_rate
            )
            if expected_acceptance_rate is None:
                expected_acceptance_rate = _INITIAL_ACCEPTED_ONLY_EXPECTED_ACCEPTANCE_RATE
            requested_generated_datasets = int(
                math.ceil(
                    float(remaining_accepted_datasets) / float(expected_acceptance_rate)
                )
            )
        elif curated_accepted_datasets > 0:
            empirical_acceptance_rate = (
                float(curated_accepted_datasets) / float(total_generated_datasets)
            )
            if 0.0 < empirical_acceptance_rate < 1.0:
                requested_generated_datasets = max(
                    requested_generated_datasets,
                    int(
                        math.ceil(
                            float(remaining_accepted_datasets) / empirical_acceptance_rate
                        )
                    ),
                )
        requested_generated_datasets = min(
            requested_generated_datasets,
            remaining_generated_budget,
        )
        if requested_generated_datasets <= 0:
            break
        round_root = _invocation_round_root(
            corpus_root=corpus_root,
            invocation_id=spec.invocation_id,
            round_index=round_index,
        )
        round_root.mkdir(parents=True, exist_ok=True)
        generate_config = _dagzoo_generate_config(
            dagzoo_root=dagzoo_root,
            corpus_root=corpus_root,
            spec=spec,
            write_rendered_config=(round_index == 1),
            handoff_root=round_root,
            num_datasets=requested_generated_datasets,
            worker_threads=materialize_worker_threads,
        )
        generate_start_time = time.perf_counter()
        handoff = run_dagzoo_generate(generate_config)
        generate_elapsed_seconds = _elapsed_seconds_since(generate_start_time)
        filter_config = DagzooFilterConfig(
            dagzoo_root=dagzoo_root,
            input_dir=handoff.generated_dir,
            filter_out_dir=round_root / "filter",
            curated_out_dir=round_root / "curated",
            worker_threads=materialize_worker_threads,
        )
        filter_start_time = time.perf_counter()
        filter_result = run_dagzoo_filter(filter_config)
        measured_filter_elapsed_seconds = _elapsed_seconds_since(filter_start_time)
        filter_elapsed_seconds = (
            float(filter_result.elapsed_seconds)
            if filter_result.elapsed_seconds is not None
            else float(measured_filter_elapsed_seconds)
        )
        if filter_result.curated_out_dir is None:
            raise RuntimeError(
                "dagzoo filter did not produce a curated output directory for accepted_only"
            )

        total_generated_datasets += int(filter_result.total_datasets)
        accepted_datasets += int(filter_result.accepted_datasets)
        rejected_datasets += int(filter_result.rejected_datasets)
        copy_start_time = time.perf_counter()
        next_shard_index, committed_curated_datasets = _copy_curated_round_shards(
            round_curated_dir=filter_result.curated_out_dir,
            final_curated_dir=final_curated_root,
            next_shard_index=next_shard_index,
            max_datasets=accepted_target - curated_accepted_datasets,
        )
        copy_elapsed_seconds = _elapsed_seconds_since(copy_start_time)
        curated_accepted_datasets += int(committed_curated_datasets)
        handoff_provenance = getattr(handoff, "provenance", None)
        if isinstance(handoff_provenance, Mapping):
            handoff_provenances.append(
                {str(key): value for key, value in handoff_provenance.items()}
            )
        round_elapsed_seconds = _elapsed_seconds_since(round_start_time)
        upstream_elapsed_seconds = _sum_elapsed_seconds(
            generate_elapsed_seconds,
            filter_elapsed_seconds,
        )
        local_overhead_elapsed_seconds = max(
            0.0,
            float(round_elapsed_seconds)
            - float(upstream_elapsed_seconds or 0.0)
            - float(copy_elapsed_seconds),
        )
        round_payloads.append(
            {
                "round_index": round_index,
                "requested_generated_datasets": requested_generated_datasets,
                "generated_datasets": int(filter_result.total_datasets),
                "accepted_datasets": int(filter_result.accepted_datasets),
                "rejected_datasets": int(filter_result.rejected_datasets),
                "filter_curated_accepted_datasets": int(filter_result.curated_accepted_datasets),
                "curated_accepted_datasets": int(committed_curated_datasets),
                "filter_manifest_path": str(filter_result.manifest_path),
                "filter_summary_path": str(filter_result.summary_path),
                "generate_elapsed_seconds": float(generate_elapsed_seconds),
                "filter_elapsed_seconds": float(filter_elapsed_seconds),
                "filter_datasets_per_minute": filter_result.datasets_per_minute,
                "copy_elapsed_seconds": float(copy_elapsed_seconds),
                "upstream_elapsed_seconds": upstream_elapsed_seconds,
                "local_overhead_elapsed_seconds": float(local_overhead_elapsed_seconds),
                "round_elapsed_seconds": float(round_elapsed_seconds),
            }
        )
        if curated_accepted_datasets >= accepted_target:
            break
        round_index += 1

    if curated_accepted_datasets != accepted_target:
        raise RuntimeError(
            "accepted_only materialization exhausted the generated dataset budget before "
            "reaching the requested accepted dataset target "
            f"for invocation {spec.invocation_id!r}: "
            f"accepted={curated_accepted_datasets} target={accepted_target} "
            f"rounds={len(round_payloads)} generated={total_generated_datasets} "
            f"generated_budget_cap={generated_budget_cap}"
        )
    if total_generated_datasets > generated_budget_cap:
        raise RuntimeError(
            "accepted_only materialization exceeded the generated dataset budget "
            f"for invocation {spec.invocation_id!r}: "
            f"generated={total_generated_datasets} budget_cap={generated_budget_cap}"
        )

    _write_accepted_only_filter_artifacts(
        filter_root=final_filter_root,
        rounds=round_payloads,
        target_accepted_datasets=accepted_target,
        total_generated_datasets=total_generated_datasets,
        accepted_datasets=accepted_datasets,
        rejected_datasets=rejected_datasets,
        curated_accepted_datasets=curated_accepted_datasets,
        materialize_worker_threads=materialize_worker_threads,
    )

    total_generate_elapsed_seconds = _sum_elapsed_seconds(
        *[
            _float_or_none(round_payload.get("generate_elapsed_seconds"))
            for round_payload in round_payloads
        ]
    )
    total_filter_elapsed_seconds = _sum_elapsed_seconds(
        *[
            _float_or_none(round_payload.get("filter_elapsed_seconds"))
            for round_payload in round_payloads
        ]
    )
    total_copy_elapsed_seconds = _sum_elapsed_seconds(
        *[
            _float_or_none(round_payload.get("copy_elapsed_seconds"))
            for round_payload in round_payloads
        ]
    )
    invocation_elapsed_seconds = _elapsed_seconds_since(invocation_start_time)
    upstream_elapsed_seconds = _sum_elapsed_seconds(
        total_generate_elapsed_seconds,
        total_filter_elapsed_seconds,
    )
    local_overhead_elapsed_seconds = max(
        0.0,
        float(invocation_elapsed_seconds)
        - float(upstream_elapsed_seconds or 0.0)
        - float(total_copy_elapsed_seconds or 0.0),
    )
    materialization_summary_path = _invocation_materialization_summary_path(
        corpus_root=corpus_root,
        invocation_id=spec.invocation_id,
    )
    materialization_summary_path.write_text(
        json.dumps(
            {
                "filter_policy": "accepted_only",
                "target_accepted_datasets": accepted_target,
                "generated_datasets": total_generated_datasets,
                "accepted_datasets": accepted_datasets,
                "rejected_datasets": rejected_datasets,
                "curated_accepted_datasets": curated_accepted_datasets,
                "acceptance_rate": (
                    None
                    if total_generated_datasets <= 0
                    else float(accepted_datasets) / float(total_generated_datasets)
                ),
                "round_count": len(round_payloads),
                "materialize_worker_threads": (
                    None
                    if materialize_worker_threads is None
                    else int(materialize_worker_threads)
                ),
                "generate_elapsed_seconds": total_generate_elapsed_seconds,
                "filter_elapsed_seconds": total_filter_elapsed_seconds,
                "copy_elapsed_seconds": total_copy_elapsed_seconds,
                "upstream_elapsed_seconds": upstream_elapsed_seconds,
                "local_overhead_elapsed_seconds": float(local_overhead_elapsed_seconds),
                "invocation_elapsed_seconds": float(invocation_elapsed_seconds),
                "initial_expected_acceptance_rate": (
                    None
                    if initial_expected_acceptance_rate is None
                    else float(initial_expected_acceptance_rate)
                ),
                "rounds": [
                    {
                        "round_index": int(round_payload["round_index"]),
                        "requested_generated_datasets": int(
                            round_payload["requested_generated_datasets"]
                        ),
                        "generated_datasets": int(round_payload["generated_datasets"]),
                        "accepted_datasets": int(round_payload["accepted_datasets"]),
                        "rejected_datasets": int(round_payload["rejected_datasets"]),
                        "filter_curated_accepted_datasets": int(
                            round_payload["filter_curated_accepted_datasets"]
                        ),
                        "curated_accepted_datasets": int(
                            round_payload["curated_accepted_datasets"]
                        ),
                        "generate_elapsed_seconds": round_payload.get(
                            "generate_elapsed_seconds"
                        ),
                        "filter_elapsed_seconds": round_payload.get(
                            "filter_elapsed_seconds"
                        ),
                        "filter_datasets_per_minute": round_payload.get(
                            "filter_datasets_per_minute"
                        ),
                        "copy_elapsed_seconds": round_payload.get("copy_elapsed_seconds"),
                        "upstream_elapsed_seconds": round_payload.get(
                            "upstream_elapsed_seconds"
                        ),
                        "local_overhead_elapsed_seconds": round_payload.get(
                            "local_overhead_elapsed_seconds"
                        ),
                        "round_elapsed_seconds": round_payload.get(
                            "round_elapsed_seconds"
                        ),
                    }
                    for round_payload in round_payloads
                ],
                "handoff_provenance": _aggregate_handoff_provenance(handoff_provenances),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def _materialize_invocation(
    *,
    dagzoo_root: Path,
    corpus_root: Path,
    spec: DagzooInvocationRecipe,
    filter_policy: str,
    materialize_worker_threads: int | None = None,
    initial_expected_acceptance_rate: float | None = None,
) -> None:
    if str(filter_policy).strip() == "accepted_only":
        _materialize_accepted_only_invocation(
            dagzoo_root=dagzoo_root,
            corpus_root=corpus_root,
            spec=spec,
            materialize_worker_threads=materialize_worker_threads,
            initial_expected_acceptance_rate=initial_expected_acceptance_rate,
        )
        return
    invocation_root, _handoff_manifest_path = _invocation_paths(
        corpus_root=corpus_root,
        invocation_id=spec.invocation_id,
    )
    invocation_root.mkdir(parents=True, exist_ok=True)
    invocation_start_time = time.perf_counter()
    generate_start_time = time.perf_counter()
    handoff = run_dagzoo_generate(
        _dagzoo_generate_config(
            dagzoo_root=dagzoo_root,
            corpus_root=corpus_root,
            spec=spec,
            write_rendered_config=True,
            worker_threads=materialize_worker_threads,
        )
    )
    generate_elapsed_seconds = _elapsed_seconds_since(generate_start_time)
    invocation_elapsed_seconds = _elapsed_seconds_since(invocation_start_time)
    handoff_provenance = getattr(handoff, "provenance", None)
    materialization_summary_path = _invocation_materialization_summary_path(
        corpus_root=corpus_root,
        invocation_id=spec.invocation_id,
    )
    materialization_summary_path.write_text(
        json.dumps(
            {
                "filter_policy": str(filter_policy).strip(),
                "generated_datasets": _generated_datasets_from_handoff(
                    handoff,
                    fallback=int(spec.num_datasets),
                ),
                "materialize_worker_threads": (
                    None
                    if materialize_worker_threads is None
                    else int(materialize_worker_threads)
                ),
                "generate_elapsed_seconds": float(generate_elapsed_seconds),
                "upstream_elapsed_seconds": float(generate_elapsed_seconds),
                "local_overhead_elapsed_seconds": max(
                    0.0,
                    float(invocation_elapsed_seconds) - float(generate_elapsed_seconds),
                ),
                "invocation_elapsed_seconds": float(invocation_elapsed_seconds),
                "handoff_provenance": (
                    None
                    if not isinstance(handoff_provenance, Mapping)
                    else _drop_none_values(
                        {str(key): value for key, value in handoff_provenance.items()}
                    )
                ),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )


def materialize_recipe_invocation(
    *,
    recipe_id: str,
    invocation_id: str,
    dagzoo_root: Path,
    corpus_root: Path,
    materialize_worker_threads: int | None = None,
    initial_expected_acceptance_rate: float | None = None,
    repo_root: Path | None = None,
    sweep_id: str | None = None,
    sweeps_root: Path | None = None,
) -> None:
    resolved_repo_root = (repo_root or _repo_root()).expanduser().resolve()
    recipe = load_corpus_recipe(
        recipe_id,
        repo_root=resolved_repo_root,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )
    spec = next(
        (
            invocation
            for invocation in recipe.invocations
            if str(invocation.invocation_id) == str(invocation_id)
        ),
        None,
    )
    if spec is None:
        raise RuntimeError(
            f"recipe {recipe_id!r} does not define invocation {invocation_id!r}"
        )
    _materialize_invocation(
        dagzoo_root=dagzoo_root.expanduser().resolve(),
        corpus_root=corpus_root.expanduser().resolve(),
        spec=spec,
        filter_policy=str(recipe.manifest_policy.filter_policy),
        materialize_worker_threads=materialize_worker_threads,
        initial_expected_acceptance_rate=initial_expected_acceptance_rate,
    )


def _invocation_worker_argv(
    *,
    recipe_id: str,
    invocation_id: str,
    dagzoo_root: Path,
    corpus_root: Path,
    repo_root: Path,
    materialize_worker_threads: int | None,
    initial_expected_acceptance_rate: float | None,
    sweep_id: str | None,
    sweeps_root: Path | None,
) -> list[str]:
    argv = [
        sys.executable,
        "-m",
        "tab_foundry.data.corpus_materialization_worker",
        "--recipe-id",
        str(recipe_id),
        "--invocation-id",
        str(invocation_id),
        "--dagzoo-root",
        str(dagzoo_root.expanduser().resolve()),
        "--corpus-root",
        str(corpus_root.expanduser().resolve()),
        "--repo-root",
        str(repo_root.expanduser().resolve()),
    ]
    if materialize_worker_threads is not None:
        argv.extend(
            ["--materialize-worker-threads", str(int(materialize_worker_threads))]
        )
    if initial_expected_acceptance_rate is not None:
        argv.extend(
            [
                "--initial-expected-acceptance-rate",
                str(float(initial_expected_acceptance_rate)),
            ]
        )
    if sweep_id is not None:
        argv.extend(["--sweep-id", str(sweep_id)])
    if sweeps_root is not None:
        argv.extend(["--sweeps-root", str(sweeps_root.expanduser().resolve())])
    return argv


def _terminate_active_invocation_subprocesses(
    active_processes: Mapping[int, _ActiveInvocationProcess],
) -> None:
    for active_process in active_processes.values():
        process = active_process.process
        if process.poll() is None:
            process.terminate()
    deadline = time.monotonic() + 5.0
    for active_process in active_processes.values():
        process = active_process.process
        if process.poll() is not None:
            continue
        remaining = deadline - time.monotonic()
        try:
            process.wait(timeout=max(0.0, remaining))
        except subprocess.TimeoutExpired:
            process.kill()
    for active_process in active_processes.values():
        process = active_process.process
        if process.poll() is None:
            process.wait()


def _materialize_invocations_with_subprocess_fanout(
    *,
    recipe_id: str,
    invocations: Sequence[DagzooInvocationRecipe],
    dagzoo_root: Path,
    corpus_root: Path,
    repo_root: Path,
    sweep_id: str | None,
    sweeps_root: Path | None,
    materialize_processes: int | None,
    materialize_worker_threads: int | None,
) -> None:
    max_processes = min(_resolve_materialize_processes(materialize_processes), len(invocations))
    resolved_worker_threads = _resolve_materialize_worker_threads(
        materialize_worker_threads,
        materialize_processes=max_processes,
    )
    shape_acceptance_rates: dict[tuple[int | None, int | None, int | None], list[float]] = {}
    row_acceptance_rates: dict[int, list[float]] = {}
    if max_processes <= 1:
        for spec in invocations:
            shape_key = _invocation_shape_key(spec)
            shape_rates = shape_acceptance_rates.get(shape_key)
            initial_expected_acceptance_rate = None
            if shape_rates:
                initial_expected_acceptance_rate = _clamp_expected_acceptance_rate(
                    sum(shape_rates) / float(len(shape_rates))
                )
            elif shape_key[0] is not None:
                row_rates = row_acceptance_rates.get(int(shape_key[0]))
                if row_rates:
                    initial_expected_acceptance_rate = _clamp_expected_acceptance_rate(
                        sum(row_rates) / float(len(row_rates))
                    )
            materialize_recipe_invocation(
                recipe_id=recipe_id,
                invocation_id=str(spec.invocation_id),
                dagzoo_root=dagzoo_root,
                corpus_root=corpus_root,
                materialize_worker_threads=resolved_worker_threads,
                initial_expected_acceptance_rate=initial_expected_acceptance_rate,
                repo_root=repo_root,
                sweep_id=sweep_id,
                sweeps_root=sweeps_root,
            )
            summary_path = _invocation_materialization_summary_path(
                corpus_root=corpus_root,
                invocation_id=spec.invocation_id,
            )
            if summary_path.exists():
                summary_payload = _read_json_mapping(
                    summary_path,
                    context=(
                        f"accepted_only materialization summary for invocation "
                        f"{spec.invocation_id!r}"
                    ),
                )
                acceptance_rate = _clamp_expected_acceptance_rate(
                    _float_or_none(summary_payload.get("acceptance_rate"))
                )
                if acceptance_rate is not None:
                    shape_acceptance_rates.setdefault(shape_key, []).append(acceptance_rate)
                    row_total = shape_key[0]
                    if row_total is not None:
                        row_acceptance_rates.setdefault(int(row_total), []).append(
                            acceptance_rate
                        )
        return

    pending = deque(invocations)
    active_processes: dict[int, _ActiveInvocationProcess] = {}
    try:
        while pending or active_processes:
            while pending and len(active_processes) < max_processes:
                spec = pending.popleft()
                shape_key = _invocation_shape_key(spec)
                shape_rates = shape_acceptance_rates.get(shape_key)
                initial_expected_acceptance_rate = None
                if shape_rates:
                    initial_expected_acceptance_rate = _clamp_expected_acceptance_rate(
                        sum(shape_rates) / float(len(shape_rates))
                    )
                elif shape_key[0] is not None:
                    row_rates = row_acceptance_rates.get(int(shape_key[0]))
                    if row_rates:
                        initial_expected_acceptance_rate = _clamp_expected_acceptance_rate(
                            sum(row_rates) / float(len(row_rates))
                        )
                process = subprocess.Popen(
                    _invocation_worker_argv(
                        recipe_id=recipe_id,
                        invocation_id=str(spec.invocation_id),
                        dagzoo_root=dagzoo_root,
                        corpus_root=corpus_root,
                        repo_root=repo_root,
                        materialize_worker_threads=resolved_worker_threads,
                        initial_expected_acceptance_rate=initial_expected_acceptance_rate,
                        sweep_id=sweep_id,
                        sweeps_root=sweeps_root,
                    ),
                    cwd=repo_root,
                    text=True,
                )
                active_processes[int(process.pid)] = _ActiveInvocationProcess(
                    process=process,
                    spec=spec,
                )

            completed_pid: int | None = None
            completed_active_process: _ActiveInvocationProcess | None = None
            completed_returncode: int | None = None
            while completed_active_process is None:
                for pid, active_process in list(active_processes.items()):
                    process = active_process.process
                    returncode = process.poll()
                    if returncode is None:
                        continue
                    completed_pid = pid
                    completed_active_process = active_process
                    completed_returncode = int(returncode)
                    break
                if completed_active_process is None:
                    time.sleep(_SUBPROCESS_POLL_INTERVAL_SECONDS)
            assert completed_pid is not None
            del active_processes[completed_pid]
            if completed_returncode != 0:
                assert completed_active_process is not None
                raise RuntimeError(
                    "invocation materialization subprocess failed: "
                    f"invocation_id={completed_active_process.spec.invocation_id} "
                    f"returncode={completed_returncode} "
                    f"argv={completed_active_process.process.args!r}"
                )
            assert completed_active_process is not None
            summary_path = _invocation_materialization_summary_path(
                corpus_root=corpus_root,
                invocation_id=completed_active_process.spec.invocation_id,
            )
            if summary_path.exists():
                summary_payload = _read_json_mapping(
                    summary_path,
                    context=(
                        f"accepted_only materialization summary for invocation "
                        f"{completed_active_process.spec.invocation_id!r}"
                    ),
                )
                acceptance_rate = _clamp_expected_acceptance_rate(
                    _float_or_none(summary_payload.get("acceptance_rate"))
                )
                if acceptance_rate is not None:
                    shape_key = _invocation_shape_key(completed_active_process.spec)
                    shape_acceptance_rates.setdefault(shape_key, []).append(acceptance_rate)
                    row_total = shape_key[0]
                    if row_total is not None:
                        row_acceptance_rates.setdefault(int(row_total), []).append(
                            acceptance_rate
                        )
        return
    finally:
        if active_processes:
            _terminate_active_invocation_subprocesses(active_processes)


def _invocation_record_payload(
    *,
    dagzoo_root: Path,
    corpus_root: Path,
    spec: DagzooInvocationRecipe,
) -> dict[str, Any]:
    invocation_root, handoff_manifest_path = _invocation_paths(
        corpus_root=corpus_root,
        invocation_id=spec.invocation_id,
    )
    materialization_summary_path = _invocation_materialization_summary_path(
        corpus_root=corpus_root,
        invocation_id=spec.invocation_id,
    )
    generate_config = _dagzoo_generate_config(
        dagzoo_root=dagzoo_root,
        corpus_root=corpus_root,
        spec=spec,
        write_rendered_config=False,
    )
    resolved_config_path = _resolve_from_root(dagzoo_root, generate_config.dagzoo_config)
    payload = {
        "invocation_id": str(spec.invocation_id),
        "requested_config_ref": _invocation_requested_config_ref(spec),
        "num_datasets": int(spec.num_datasets),
        "seed": None if spec.seed is None else int(spec.seed),
        "rows": spec.rows,
        "device": spec.device,
        "hardware_policy": str(spec.hardware_policy),
        "resolved_config_path": str(resolved_config_path),
        "invocation_root": str(invocation_root.resolve()),
    }
    summary_payload: dict[str, Any] | None = None
    if materialization_summary_path.exists():
        summary_payload = _read_json_mapping(
            materialization_summary_path,
            context=f"materialization summary for invocation {spec.invocation_id!r}",
        )
        timing_payload = _drop_none_values(
            {
                "generated_datasets": _int_or_none(summary_payload.get("generated_datasets")),
                "generate_elapsed_seconds": _float_or_none(
                    summary_payload.get("generate_elapsed_seconds")
                ),
                "filter_elapsed_seconds": _float_or_none(
                    summary_payload.get("filter_elapsed_seconds")
                ),
                "copy_elapsed_seconds": _float_or_none(
                    summary_payload.get("copy_elapsed_seconds")
                ),
                "upstream_elapsed_seconds": _float_or_none(
                    summary_payload.get("upstream_elapsed_seconds")
                ),
                "local_overhead_elapsed_seconds": _float_or_none(
                    summary_payload.get("local_overhead_elapsed_seconds")
                ),
                "invocation_elapsed_seconds": _float_or_none(
                    summary_payload.get("invocation_elapsed_seconds")
                ),
                "materialize_worker_threads": _int_or_none(
                    summary_payload.get("materialize_worker_threads")
                ),
                "round_count": _int_or_none(summary_payload.get("round_count")),
            }
        )
        if timing_payload:
            payload["materialization_timing"] = timing_payload
    if (
        summary_payload is not None
        and str(summary_payload.get("filter_policy")).strip() == "accepted_only"
    ):
        materialize_worker_threads = _int_or_none(
            summary_payload.get("materialize_worker_threads")
        )
        rounds_payload = []
        command_list: list[str] = []
        for round_payload in cast(list[Any], summary_payload.get("rounds", [])):
            if not isinstance(round_payload, Mapping):
                raise RuntimeError(
                    "accepted_only materialization summary rounds must contain mappings"
                )
            normalized_round_payload = {
                str(key): value for key, value in round_payload.items()
            }
            round_index = int(normalized_round_payload["round_index"])
            requested_generated_datasets = int(
                normalized_round_payload["requested_generated_datasets"]
            )
            round_root = _invocation_round_root(
                corpus_root=corpus_root,
                invocation_id=spec.invocation_id,
                round_index=round_index,
            )
            round_handoff = load_dagzoo_handoff_info(round_root / "handoff_manifest.json")
            round_generate_config = _dagzoo_generate_config(
                dagzoo_root=dagzoo_root,
                corpus_root=corpus_root,
                spec=spec,
                write_rendered_config=False,
                handoff_root=round_root,
                num_datasets=requested_generated_datasets,
            )
            round_filter_config = DagzooFilterConfig(
                dagzoo_root=dagzoo_root,
                input_dir=round_handoff.generated_dir,
                filter_out_dir=round_root / "filter",
                curated_out_dir=round_root / "curated",
                worker_threads=materialize_worker_threads,
            )
            generate_command = _stringify_command(
                build_dagzoo_generate_argv(round_generate_config)
            )
            filter_command = _stringify_command(
                build_dagzoo_filter_argv(round_filter_config)
            )
            command_list.extend([generate_command, filter_command])
            round_entry: dict[str, Any] = {
                "round_index": round_index,
                "requested_generated_datasets": requested_generated_datasets,
                "generated_datasets": int(normalized_round_payload["generated_datasets"]),
                "accepted_datasets": int(normalized_round_payload["accepted_datasets"]),
                "rejected_datasets": int(normalized_round_payload["rejected_datasets"]),
                "filter_curated_accepted_datasets": int(
                    normalized_round_payload.get(
                        "filter_curated_accepted_datasets",
                        normalized_round_payload["curated_accepted_datasets"],
                    )
                ),
                "curated_accepted_datasets": int(
                    normalized_round_payload["curated_accepted_datasets"]
                ),
                "generate_command": generate_command,
                "filter_command": filter_command,
                "handoff": round_handoff.to_summary_dict(),
            }
            round_handoff_provenance = getattr(round_handoff, "provenance", None)
            if isinstance(round_handoff_provenance, Mapping):
                round_entry["handoff_provenance"] = _drop_none_values(
                    {str(key): value for key, value in round_handoff_provenance.items()}
                )
            round_timing_payload = _drop_none_values(
                {
                    "generate_elapsed_seconds": _float_or_none(
                        normalized_round_payload.get("generate_elapsed_seconds")
                    ),
                    "filter_elapsed_seconds": _float_or_none(
                        normalized_round_payload.get("filter_elapsed_seconds")
                    ),
                    "filter_datasets_per_minute": _float_or_none(
                        normalized_round_payload.get("filter_datasets_per_minute")
                    ),
                    "copy_elapsed_seconds": _float_or_none(
                        normalized_round_payload.get("copy_elapsed_seconds")
                    ),
                    "upstream_elapsed_seconds": _float_or_none(
                        normalized_round_payload.get("upstream_elapsed_seconds")
                    ),
                    "local_overhead_elapsed_seconds": _float_or_none(
                        normalized_round_payload.get("local_overhead_elapsed_seconds")
                    ),
                    "round_elapsed_seconds": _float_or_none(
                        normalized_round_payload.get("round_elapsed_seconds")
                    ),
                }
            )
            if round_timing_payload:
                round_entry["materialization_timing"] = round_timing_payload
            rounds_payload.append(round_entry)
        payload.update(
            {
                "filter_policy": "accepted_only",
                "command": command_list[0] if command_list else None,
                "commands": command_list,
                "rounds": rounds_payload,
                "filter": _drop_none_values(
                    {
                        "filter_policy": "accepted_only",
                        "target_accepted_datasets": int(
                            summary_payload.get("target_accepted_datasets", spec.num_datasets)
                        ),
                        "generated_datasets": int(summary_payload.get("generated_datasets", 0)),
                        "accepted_datasets": int(summary_payload.get("accepted_datasets", 0)),
                        "rejected_datasets": int(summary_payload.get("rejected_datasets", 0)),
                        "curated_accepted_datasets": int(
                            summary_payload.get("curated_accepted_datasets", 0)
                        ),
                        "acceptance_rate": summary_payload.get("acceptance_rate"),
                        "round_count": int(summary_payload.get("round_count", len(rounds_payload))),
                        "filter_manifest_path": str(
                            (
                                _invocation_filter_root(
                                    corpus_root=corpus_root,
                                    invocation_id=spec.invocation_id,
                                )
                                / "filter_manifest.ndjson"
                            ).resolve()
                        ),
                        "filter_summary_path": str(
                            (
                                _invocation_filter_root(
                                    corpus_root=corpus_root,
                                    invocation_id=spec.invocation_id,
                                )
                                / "filter_summary.json"
                            ).resolve()
                        ),
                        "curated_dir": str(
                            _invocation_curated_root(
                                corpus_root=corpus_root,
                                invocation_id=spec.invocation_id,
                            ).resolve()
                        ),
                    }
                ),
            }
        )
        handoff_provenance = summary_payload.get("handoff_provenance")
        if isinstance(handoff_provenance, Mapping):
            payload["handoff_provenance"] = _drop_none_values(
                {str(key): value for key, value in handoff_provenance.items()}
            )
    else:
        handoff = load_dagzoo_handoff_info(handoff_manifest_path)
        payload.update(
            {
                "command": _stringify_command(build_dagzoo_generate_argv(generate_config)),
                "handoff": handoff.to_summary_dict(),
            }
        )
        handoff_provenance = (
            summary_payload.get("handoff_provenance")
            if isinstance(summary_payload, Mapping)
            else getattr(handoff, "provenance", None)
        )
        if isinstance(handoff_provenance, Mapping):
            payload["handoff_provenance"] = _drop_none_values(
                {str(key): value for key, value in handoff_provenance.items()}
            )
    if spec.config_ref is not None:
        payload["config_ref"] = str(spec.config_ref)
    if spec.base_config_ref is not None:
        rendered_config_path = _invocation_rendered_config_path(
            corpus_root=corpus_root,
            invocation_id=spec.invocation_id,
        )
        payload["base_config_ref"] = str(spec.base_config_ref)
        payload["config_overrides"] = _copy_jsonable(spec.config_overrides)
        payload["rendered_config_path"] = str(rendered_config_path.resolve())
        payload["rendered_config_sha256"] = sha256_path(rendered_config_path)
    return payload
