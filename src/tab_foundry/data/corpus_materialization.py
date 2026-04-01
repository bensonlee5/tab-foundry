"""Corpus materialization helpers."""

from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, Sequence, cast

import yaml

from tab_realdata_hub.manifest import build_manifest

from tab_foundry.hashing import sha256_path
from tab_foundry.timestamps import utc_now

from .corpus_loading import (
    CORPUS_RECORD_SCHEMA,
    DagzooInvocationRecipe,
    _copy_jsonable,
    _deep_merge_payload,
    _ensure_non_empty_string,
    _load_yaml_mapping,
    _latest_pointer_path,
    _recipe_storage_context,
    _repo_root,
    _resolve_from_root,
    _write_latest_pointer,
    build_dagzoo_provenance_summary,
    corpus_id_for_manifest,
    corpus_outputs_root,
    load_corpus_recipe,
)
from .corpus_lookup import _load_reusable_corpus_record, _record_matches_recipe
from .manifest_characteristics import inspect_manifest_summary
from tab_realdata_hub.dagzoo_handoff import (
    DagzooGeneratedIdentityAccumulator,
    DagzooHandoffInfo,
    load_dagzoo_handoff_info,
    verify_dagzoo_handoff_matches_generated_corpus,
)
from .dagzoo_workflow import (
    DagzooFilterConfig,
    build_dagzoo_filter_argv,
    DagzooGenerateConfig,
    build_dagzoo_generate_argv,
    run_dagzoo_filter,
    run_dagzoo_generate,
)


_ACCEPTED_ONLY_MAX_ROUNDS = 8
_ACCEPTED_ONLY_MAX_GENERATED_MULTIPLIER = 4


def _git_info(root: Path) -> dict[str, Any] | None:
    if not root.exists():
        return None

    def _capture(*argv: str) -> str | None:
        completed = subprocess.run(
            list(argv),
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return None
        value = completed.stdout.strip()
        return value or None

    head = _capture("git", "rev-parse", "HEAD")
    if head is None:
        return None
    describe = _capture("git", "describe", "--always", "--dirty", "--tags")
    status = _capture("git", "status", "--short")
    return {
        "head": head,
        "describe": describe,
        "dirty": bool(status),
    }


def _drop_none_values(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in payload.items()
        if value is not None
    }


def _read_json_mapping(path: Path, *, context: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{context} must decode to a JSON object: {path}")
    return {str(key): value for key, value in cast(Mapping[str, Any], payload).items()}


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


def _dagzoo_generate_config(
    *,
    dagzoo_root: Path,
    corpus_root: Path,
    spec: DagzooInvocationRecipe,
    write_rendered_config: bool,
    handoff_root: Path | None = None,
    num_datasets: int | None = None,
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


def _copy_curated_round_shards(
    *,
    round_curated_dir: Path,
    final_curated_dir: Path,
    next_shard_index: int,
) -> int:
    if not round_curated_dir.exists():
        return next_shard_index
    final_curated_dir.mkdir(parents=True, exist_ok=True)
    for shard_dir in sorted(path for path in round_curated_dir.glob("shard_*") if path.is_dir()):
        destination = final_curated_dir / f"shard_{next_shard_index:05d}"
        shutil.copytree(shard_dir, destination)
        next_shard_index += 1
    return next_shard_index


def _write_accepted_only_filter_artifacts(
    *,
    filter_root: Path,
    rounds: Sequence[Mapping[str, Any]],
    target_accepted_datasets: int,
    total_generated_datasets: int,
    accepted_datasets: int,
    rejected_datasets: int,
    curated_accepted_datasets: int,
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
                "round_budget_cap": _ACCEPTED_ONLY_MAX_ROUNDS,
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
) -> None:
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

    for round_index in range(1, _ACCEPTED_ONLY_MAX_ROUNDS + 1):
        if curated_accepted_datasets >= accepted_target:
            break
        if total_generated_datasets >= generated_budget_cap:
            break

        remaining_generated_budget = generated_budget_cap - total_generated_datasets
        requested_generated_datasets = min(
            accepted_target - curated_accepted_datasets,
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
        )
        handoff = run_dagzoo_generate(generate_config)
        filter_config = DagzooFilterConfig(
            dagzoo_root=dagzoo_root,
            input_dir=handoff.generated_dir,
            filter_out_dir=round_root / "filter",
            curated_out_dir=round_root / "curated",
        )
        filter_result = run_dagzoo_filter(filter_config)
        if filter_result.curated_out_dir is None:
            raise RuntimeError(
                "dagzoo filter did not produce a curated output directory for accepted_only"
            )

        total_generated_datasets += int(filter_result.total_datasets)
        accepted_datasets += int(filter_result.accepted_datasets)
        rejected_datasets += int(filter_result.rejected_datasets)
        curated_accepted_datasets += int(filter_result.curated_accepted_datasets)
        next_shard_index = _copy_curated_round_shards(
            round_curated_dir=filter_result.curated_out_dir,
            final_curated_dir=final_curated_root,
            next_shard_index=next_shard_index,
        )
        handoff_provenance = getattr(handoff, "provenance", None)
        if isinstance(handoff_provenance, Mapping):
            handoff_provenances.append(
                {str(key): value for key, value in handoff_provenance.items()}
            )
        round_payloads.append(
            {
                "round_index": round_index,
                "requested_generated_datasets": requested_generated_datasets,
                "generated_datasets": int(filter_result.total_datasets),
                "accepted_datasets": int(filter_result.accepted_datasets),
                "rejected_datasets": int(filter_result.rejected_datasets),
                "curated_accepted_datasets": int(filter_result.curated_accepted_datasets),
                "filter_manifest_path": str(filter_result.manifest_path),
                "filter_summary_path": str(filter_result.summary_path),
            }
        )
        if curated_accepted_datasets >= accepted_target:
            break

    if curated_accepted_datasets != accepted_target:
        raise RuntimeError(
            "accepted_only materialization did not reach the requested accepted dataset target "
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


def materialize_corpus_ref(
    *,
    corpus_ref: str,
    dagzoo_root: Path,
    force: bool = False,
    repo_root: Path | None = None,
    sweep_id: str | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    normalized_corpus_ref = _ensure_non_empty_string(corpus_ref, context="corpus_ref")
    recipe_id, separator, corpus_id = normalized_corpus_ref.partition("/")
    if not separator:
        return materialize_corpus_recipe(
            recipe_id=recipe_id,
            dagzoo_root=dagzoo_root,
            force=force,
            repo_root=repo_root,
            sweep_id=sweep_id,
            sweeps_root=sweeps_root,
        )

    if not force:
        existing_record = _load_reusable_corpus_record(
            normalized_corpus_ref,
            repo_root=repo_root,
            sweep_id=sweep_id,
            sweeps_root=sweeps_root,
        )
        if existing_record is not None:
            return existing_record

    record = materialize_corpus_recipe(
        recipe_id=recipe_id,
        dagzoo_root=dagzoo_root,
        force=force,
        repo_root=repo_root,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )
    materialized_corpus_ref = _ensure_non_empty_string(
        record.get("corpus_ref"),
        context="materialized corpus record corpus_ref",
    )
    if materialized_corpus_ref != normalized_corpus_ref:
        raise RuntimeError(
            f"requested corpus_ref {normalized_corpus_ref!r} is pinned to an exact corpus id, "
            f"but materializing recipe {recipe_id!r} produced {materialized_corpus_ref!r}"
        )
    return record


def _materialize_invocation(
    *,
    dagzoo_root: Path,
    corpus_root: Path,
    spec: DagzooInvocationRecipe,
    filter_policy: str,
) -> None:
    if str(filter_policy).strip() == "accepted_only":
        _materialize_accepted_only_invocation(
            dagzoo_root=dagzoo_root,
            corpus_root=corpus_root,
            spec=spec,
        )
        return
    invocation_root, _handoff_manifest_path = _invocation_paths(
        corpus_root=corpus_root,
        invocation_id=spec.invocation_id,
    )
    invocation_root.mkdir(parents=True, exist_ok=True)
    run_dagzoo_generate(
        _dagzoo_generate_config(
            dagzoo_root=dagzoo_root,
            corpus_root=corpus_root,
            spec=spec,
            write_rendered_config=True,
        )
    )


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
    if materialization_summary_path.exists():
        summary_payload = _read_json_mapping(
            materialization_summary_path,
            context=f"materialization summary for invocation {spec.invocation_id!r}",
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
        handoff_provenance = getattr(handoff, "provenance", None)
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


def _manifest_characteristics_sidecar_path(*, corpus_root: Path) -> Path:
    return corpus_root / "manifest_characteristics.json"


def materialize_corpus_recipe(
    *,
    recipe_id: str,
    dagzoo_root: Path,
    force: bool = False,
    repo_root: Path | None = None,
    sweep_id: str | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    resolved_repo_root = (repo_root or _repo_root()).expanduser().resolve()
    resolved_dagzoo_root = dagzoo_root.expanduser().resolve()
    recipe = load_corpus_recipe(
        recipe_id,
        repo_root=resolved_repo_root,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )
    storage = _recipe_storage_context(recipe, repo_root=resolved_repo_root)
    if not force:
        existing_record = _load_reusable_corpus_record(
            recipe.recipe_id,
            repo_root=resolved_repo_root,
            sweep_id=sweep_id,
            sweeps_root=sweeps_root,
        )
        if existing_record is not None and _record_matches_recipe(
            existing_record,
            recipe,
            storage=storage,
        ):
            return existing_record

    recipe_root = corpus_outputs_root(repo_root=resolved_repo_root) / recipe.recipe_id
    stage_root = recipe_root / ".staging"
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)

    try:
        filter_policy = str(recipe.manifest_policy.filter_policy)
        for spec in recipe.invocations:
            _materialize_invocation(
                dagzoo_root=resolved_dagzoo_root,
                corpus_root=stage_root,
                spec=spec,
                filter_policy=filter_policy,
            )
        if filter_policy == "accepted_only":
            generated_roots = [
                _invocation_curated_root(
                    corpus_root=stage_root,
                    invocation_id=spec.invocation_id,
                )
                for spec in recipe.invocations
            ]
            dagzoo_handoff_manifest_path = None
        elif len(recipe.invocations) == 1:
            single_handoff = load_dagzoo_handoff_info(
                _invocation_paths(
                    corpus_root=stage_root,
                    invocation_id=recipe.invocations[0].invocation_id,
                )[1]
            )
            generated_roots = [
                _manifest_source_root(
                    handoff=single_handoff,
                    filter_policy=str(recipe.manifest_policy.filter_policy),
                )
            ]
            dagzoo_handoff_manifest_path = single_handoff.handoff_manifest_path
        else:
            verified_handoffs = [
                _verified_invocation_handoff(
                    corpus_root=stage_root,
                    spec=spec,
                )
                for spec in recipe.invocations
            ]
            generated_roots = [
                _manifest_source_root(
                    handoff=handoff,
                    filter_policy=str(recipe.manifest_policy.filter_policy),
                )
                for handoff in verified_handoffs
            ]
            dagzoo_handoff_manifest_path = None
        manifest_path = stage_root / "manifest.parquet"
        _ = build_manifest(
            data_roots=generated_roots,
            out_path=manifest_path,
            train_ratio=float(recipe.manifest_policy.train_ratio),
            val_ratio=float(recipe.manifest_policy.val_ratio),
            filter_policy=str(recipe.manifest_policy.filter_policy),
            missing_value_policy=str(recipe.manifest_policy.missing_value_policy),
            dagzoo_handoff_manifest_path=dagzoo_handoff_manifest_path,
        )
        manifest_sha256 = sha256_path(manifest_path)
        corpus_id = corpus_id_for_manifest(
            recipe_id=recipe.recipe_id,
            manifest_sha256=manifest_sha256,
            recipe_identity=(
                storage.recipe_identity if storage.uses_scoped_identity else None
            ),
        )
        corpus_ref = f"{recipe.recipe_id}/{corpus_id}"
        final_root = recipe_root / corpus_id
        if final_root.exists():
            if force:
                shutil.rmtree(final_root)
            else:
                existing_record = _load_reusable_corpus_record(
                    corpus_ref,
                    repo_root=resolved_repo_root,
                    sweep_id=sweep_id,
                    sweeps_root=sweeps_root,
                )
                if existing_record is not None and _record_matches_recipe(
                    existing_record,
                    recipe,
                    storage=storage,
                ):
                    shutil.rmtree(stage_root)
                    return existing_record
                shutil.rmtree(final_root)
        shutil.move(str(stage_root), str(final_root))
        resolved_manifest_path = final_root / "manifest.parquet"
        invocation_payloads = [
            _invocation_record_payload(
                dagzoo_root=resolved_dagzoo_root,
                corpus_root=final_root,
                spec=spec,
            )
            for spec in recipe.invocations
        ]
        dagzoo_provenance_summary = build_dagzoo_provenance_summary(
            recipe=recipe,
            corpus_ref=corpus_ref,
            corpus_id=corpus_id,
            provenance={
                "invocations": invocation_payloads,
            },
        )
        invocation_filter_payloads = _invocation_filter_payloads(invocation_payloads)
        dagzoo_provenance = _drop_none_values(
            {
                "corpus_ref": corpus_ref,
                "recipe_id": recipe.recipe_id,
                "corpus_id": corpus_id,
                "recipe_kind": recipe.kind,
                "corpus_variant": recipe.provenance_labels.get("corpus_variant", recipe.surface_label),
                "comparator_role": recipe.provenance_labels.get("comparator_role"),
                "config_refs": sorted(
                    {_invocation_requested_config_ref(invocation) for invocation in recipe.invocations}
                ),
                "commands": [
                    command
                    for payload in invocation_payloads
                    for command in (
                        cast(list[Any], payload.get("commands"))
                        if isinstance(payload.get("commands"), list)
                        else (
                            [payload.get("command")]
                            if payload.get("command") is not None
                            else []
                        )
                    )
                    if isinstance(command, str) and command.strip()
                ],
                "filter_policy": dagzoo_provenance_summary.get("filter_policy"),
                "accepted_datasets": dagzoo_provenance_summary.get("accepted_datasets"),
                "rejected_datasets": dagzoo_provenance_summary.get("rejected_datasets"),
                "curated_accepted_datasets": dagzoo_provenance_summary.get(
                    "curated_accepted_datasets"
                ),
                "acceptance_rate": dagzoo_provenance_summary.get("acceptance_rate"),
                "filter_manifest_paths": [
                    filter_payload.get("filter_manifest_path")
                    for filter_payload in invocation_filter_payloads
                    if filter_payload.get("filter_manifest_path") is not None
                ],
                "filter_summary_paths": [
                    filter_payload.get("filter_summary_path")
                    for filter_payload in invocation_filter_payloads
                    if filter_payload.get("filter_summary_path") is not None
                ],
                "curated_root_lineage": [
                    filter_payload.get("curated_dir")
                    for filter_payload in invocation_filter_payloads
                    if filter_payload.get("curated_dir") is not None
                ],
                "invocations": invocation_payloads,
                "dagzoo_git": _git_info(resolved_dagzoo_root),
                "target_derivation": dagzoo_provenance_summary.get("target_derivation"),
                "target_relevant_feature_count_range": dagzoo_provenance_summary.get(
                    "target_relevant_feature_count_range"
                ),
                "target_relevant_feature_fraction_range": dagzoo_provenance_summary.get(
                    "target_relevant_feature_fraction_range"
                ),
            }
        )
        manifest_inspection = inspect_manifest_summary(resolved_manifest_path)
        manifest_persisted_summary = manifest_inspection.get("persisted_summary")
        characteristics_sidecar_path = _manifest_characteristics_sidecar_path(corpus_root=final_root)
        record: dict[str, Any] = {
            "schema": CORPUS_RECORD_SCHEMA,
            "generated_at_utc": utc_now(),
            "recipe_id": recipe.recipe_id,
            "corpus_id": corpus_id,
            "corpus_ref": corpus_ref,
            "recipe_path": str(recipe.recipe_path),
            "recipe_identity": storage.recipe_identity,
            "recipe_relative_path": storage.recipe_relative_path,
            "surface_label": recipe.surface_label,
            "surface_label_recommendation": recipe.surface_label,
            "recipe": recipe.to_dict(),
            "artifacts": {
                "corpus_root": str(final_root.resolve()),
                "manifest_path": str(resolved_manifest_path.resolve()),
                "latest_pointer_path": str(
                    _latest_pointer_path(
                        recipe_id=recipe.recipe_id,
                        repo_root=resolved_repo_root,
                        recipe_identity=(
                            storage.recipe_identity if storage.uses_scoped_identity else None
                        ),
                    )
                ),
            },
            "manifest": {
                "manifest_path": str(resolved_manifest_path.resolve()),
                "manifest_sha256": manifest_sha256,
                "inspection": manifest_inspection,
                "characteristics": {
                    "persisted_summary": manifest_persisted_summary,
                    "sidecar_path": str(characteristics_sidecar_path.resolve()),
                    "cache_status": "deferred",
                },
            },
            "dagzoo_provenance": dagzoo_provenance,
            "dagzoo_provenance_summary": dagzoo_provenance_summary,
        }
        record_path = final_root / "corpus_record.json"
        record["corpus_record_path"] = str(record_path.resolve())
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        latest_path = _write_latest_pointer(
            recipe_id=recipe.recipe_id,
            corpus_id=corpus_id,
            corpus_ref=corpus_ref,
            record_path=record_path,
            recipe_path=recipe.recipe_path,
            recipe_identity=storage.recipe_identity,
            repo_root=resolved_repo_root,
            scoped_recipe_identity=(
                storage.recipe_identity if storage.uses_scoped_identity else None
            ),
        )
        record["artifacts"]["latest_pointer_path"] = str(latest_path.resolve())
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return record
    finally:
        if stage_root.exists():
            shutil.rmtree(stage_root)
