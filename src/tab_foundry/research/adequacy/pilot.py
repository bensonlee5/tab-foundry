"""Lean synthetic adequacy pilot for TF-RD-010."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

from tab_foundry.data.corpus_materialization import materialize_corpus_refs_batch
from tab_foundry.research.synthetic_adequacy import (
    SyntheticAdequacyBlock,
    load_synthetic_adequacy_spec,
)

from . import canary as canary_module
from . import contract as contract_module
from . import production_control as production_control_module
from . import reporting as reporting_module
from . import shared as shared_module

default_pilot_output_root = shared_module.default_pilot_output_root
validate_latent_target_metadata = contract_module.validate_latent_target_metadata
inspect_corpus_latent_target_contract = contract_module.inspect_corpus_latent_target_contract
score_task_local_predictors = canary_module.score_task_local_predictors
score_canary_block = canary_module.score_canary_block
build_production_control_config = production_control_module.build_production_control_config
run_production_control_pilot = production_control_module.run_production_control_pilot
render_adequacy_pilot_markdown = reporting_module.render_adequacy_pilot_markdown
select_provisional_interpretation = reporting_module.select_provisional_interpretation


def run_adequacy_pilot(
    *,
    adequacy_id: str,
    dagzoo_root: Path,
    device: str = shared_module._SUPPORTED_DEVICE,
    force: bool = False,
    materialize_processes: int | None = None,
    materialize_worker_threads: int | None = None,
    contract_check: str = "fast",
    out_root: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    shared_module._ensure_supported_configuration(adequacy_id=adequacy_id, device=device)
    resolved_contract_check = shared_module._normalize_contract_check_mode(contract_check)
    spec = load_synthetic_adequacy_spec(adequacy_id, repo_root=repo_root)
    pilot_root = (
        out_root.expanduser().resolve()
        if out_root is not None
        else default_pilot_output_root(adequacy_id, repo_root=repo_root)
    )
    pilot_root.mkdir(parents=True, exist_ok=True)

    materialized_corpora: dict[str, dict[str, Any]] = {}
    latent_target_contract: dict[str, dict[str, Any]] = {}
    canary_summary: dict[str, Any] | None = None
    blocking_summary_written = False

    production_block = next(
        (
            block
            for block in spec.blocks
            if block.block_id == shared_module._PRODUCTION_BLOCK_ID
        ),
        None,
    )
    batched_blocks = [
        block
        for block in spec.blocks
        if production_block is None or block.block_id != shared_module._PRODUCTION_BLOCK_ID
    ]
    blocks_by_recipe_id: dict[str, list[SyntheticAdequacyBlock]] = {}
    for block in batched_blocks:
        blocks_by_recipe_id.setdefault(
            shared_module._recipe_id_from_corpus_ref(str(block.corpus_ref)),
            [],
        ).append(block)

    summary_md_path = pilot_root / shared_module._SUMMARY_MARKDOWN_NAME
    canary_recipe_id = next(
        (
            shared_module._recipe_id_from_corpus_ref(str(block.corpus_ref))
            for block in batched_blocks
            if block.block_id == shared_module._CANARY_BLOCK_ID
        ),
        None,
    )

    def _on_corpus_materialized(corpus_record: dict[str, Any]) -> None:
        nonlocal blocking_summary_written, canary_summary
        recipe_id = str(corpus_record["recipe_id"])
        for block in blocks_by_recipe_id.get(recipe_id, []):
            materialized_corpora[block.block_id] = reporting_module._materialized_corpus_payload(
                block=block,
                corpus_record=corpus_record,
                materialization_state="finalized",
            )
            latent_target_contract[block.block_id] = (
                contract_module.inspect_corpus_latent_target_contract(
                    block=block,
                    corpus_record=corpus_record,
                    mode=resolved_contract_check,
                )
            )
            filter_provenance = shared_module._optional_mapping(
                latent_target_contract[block.block_id].get("filter_provenance")
            )
            if filter_provenance is not None:
                materialized_corpora[block.block_id]["filter_provenance"] = filter_provenance
            if block.block_id != shared_module._CANARY_BLOCK_ID:
                continue
            canary_summary = canary_module.score_canary_block(
                block,
                corpus_record=corpus_record,
            )
            contract_payload = latent_target_contract[block.block_id]
            if bool(contract_payload.get("required")) and not bool(contract_payload.get("present")):
                reporting_module._write_blocking_summary(
                    adequacy_id=adequacy_id,
                    contract_check_mode=resolved_contract_check,
                    blocked_sweeps=spec.blocked_sweeps,
                    pilot_root=pilot_root,
                    materialized_corpora=materialized_corpora,
                    latent_target_contract=latent_target_contract,
                    canary_summary=canary_summary,
                    definition=spec.decision_buckets.get("generator_problem"),
                    reasoning=[
                        "latent-target contract validation failed for "
                        + str(block.block_id)
                    ],
                )
                blocking_summary_written = True
                raise RuntimeError(
                    "latent-target contract validation failed for the canary corpus; "
                    f"wrote blocking summary to {summary_md_path.resolve()}"
                )
            canary_failure_reasons = reporting_module._canary_failure_reasons(canary_summary)
            if canary_failure_reasons:
                reporting_module._write_blocking_summary(
                    adequacy_id=adequacy_id,
                    contract_check_mode=resolved_contract_check,
                    blocked_sweeps=spec.blocked_sweeps,
                    pilot_root=pilot_root,
                    materialized_corpora=materialized_corpora,
                    latent_target_contract=latent_target_contract,
                    canary_summary=canary_summary,
                    definition=spec.decision_buckets.get("generator_problem"),
                    reasoning=canary_failure_reasons,
                )
                blocking_summary_written = True
                raise RuntimeError(
                    "canary baseline validation failed for the adequacy pilot; "
                    f"wrote blocking summary to {summary_md_path.resolve()}"
                )

    if batched_blocks:
        try:
            _ = materialize_corpus_refs_batch(
                corpus_refs=[block.corpus_ref for block in batched_blocks],
                dagzoo_root=dagzoo_root,
                force=force,
                materialize_processes=materialize_processes,
                materialize_worker_threads=materialize_worker_threads,
                prioritized_recipe_ids=(
                    [] if canary_recipe_id is None else [canary_recipe_id]
                ),
                on_corpus_materialized=_on_corpus_materialized,
                repo_root=repo_root,
            )
        except Exception as exc:
            if blocking_summary_written:
                raise
            reporting_module._write_blocking_summary(
                adequacy_id=adequacy_id,
                contract_check_mode=resolved_contract_check,
                blocked_sweeps=spec.blocked_sweeps,
                pilot_root=pilot_root,
                materialized_corpora=materialized_corpora,
                latent_target_contract=latent_target_contract,
                canary_summary=canary_summary,
                definition=spec.decision_buckets.get("generator_problem"),
                reasoning=[
                    "corpus materialization or validation failed: "
                    f"{type(exc).__name__}: {exc}"
                ],
            )
            blocking_summary_written = True
            raise RuntimeError(
                "adequacy pilot blocked during corpus materialization or validation; "
                f"wrote blocking summary to {summary_md_path.resolve()}"
            ) from exc

    missing_block_ids = [
        block.block_id for block in batched_blocks if block.block_id not in materialized_corpora
    ]
    if missing_block_ids:
        reporting_module._write_blocking_summary(
            adequacy_id=adequacy_id,
            contract_check_mode=resolved_contract_check,
            blocked_sweeps=spec.blocked_sweeps,
            pilot_root=pilot_root,
            materialized_corpora=materialized_corpora,
            latent_target_contract=latent_target_contract,
            canary_summary=canary_summary,
            definition=spec.decision_buckets.get("generator_problem"),
            reasoning=[
                "missing materialized adequacy blocks: "
                + ", ".join(sorted(missing_block_ids))
            ],
        )
        raise RuntimeError(
            "adequacy pilot did not materialize every required corpus; "
            f"wrote blocking summary to {summary_md_path.resolve()}"
        )

    missing_contract_blocks = [
        block_id
        for block_id, payload in latent_target_contract.items()
        if bool(payload.get("required")) and not bool(payload.get("present"))
    ]
    if missing_contract_blocks:
        reporting_module._write_blocking_summary(
            adequacy_id=adequacy_id,
            contract_check_mode=resolved_contract_check,
            blocked_sweeps=spec.blocked_sweeps,
            pilot_root=pilot_root,
            materialized_corpora=materialized_corpora,
            latent_target_contract=latent_target_contract,
            canary_summary=canary_summary,
            definition=spec.decision_buckets.get("generator_problem"),
            reasoning=[
                "latent-target contract validation failed for "
                + ", ".join(sorted(missing_contract_blocks))
            ],
        )
        raise RuntimeError(
            "latent-target contract validation failed for one or more adequacy blocks; "
            f"wrote blocking summary to {(pilot_root / shared_module._SUMMARY_MARKDOWN_NAME).resolve()}"
        )

    canary_failure_reasons = reporting_module._canary_failure_reasons(canary_summary)
    if canary_failure_reasons:
        reporting_module._write_blocking_summary(
            adequacy_id=adequacy_id,
            contract_check_mode=resolved_contract_check,
            blocked_sweeps=spec.blocked_sweeps,
            pilot_root=pilot_root,
            materialized_corpora=materialized_corpora,
            latent_target_contract=latent_target_contract,
            canary_summary=canary_summary,
            definition=spec.decision_buckets.get("generator_problem"),
            reasoning=canary_failure_reasons,
        )
        raise RuntimeError(
            "canary baseline validation failed for the adequacy pilot; "
            f"wrote blocking summary to {(pilot_root / shared_module._SUMMARY_MARKDOWN_NAME).resolve()}"
        )

    if production_block is None:
        raise RuntimeError(
            f"adequacy spec {adequacy_id!r} is missing the {shared_module._PRODUCTION_BLOCK_ID!r} block"
        )
    production_resolution = production_control_module._resolve_production_control_corpus(
        requested_corpus_ref=str(production_block.corpus_ref),
        pilot_root=pilot_root,
        dagzoo_root=dagzoo_root,
        force=force,
        repo_root=repo_root,
    )
    production_corpus_record = cast(
        Mapping[str, Any],
        production_resolution["corpus_record"],
    )
    materialized_corpora[production_block.block_id] = reporting_module._materialized_corpus_payload(
        block=production_block,
        corpus_record=production_corpus_record,
        materialization_state=str(production_resolution["materialization_state"]),
    )
    latent_target_contract[production_block.block_id] = (
        contract_module.inspect_corpus_latent_target_contract(
            block=production_block,
            corpus_record=production_corpus_record,
            mode=resolved_contract_check,
        )
    )
    production_filter_provenance = shared_module._optional_mapping(
        latent_target_contract[production_block.block_id].get("filter_provenance")
    )
    if production_filter_provenance is not None:
        materialized_corpora[production_block.block_id]["filter_provenance"] = production_filter_provenance

    missing_contract_blocks = [
        block_id
        for block_id, payload in latent_target_contract.items()
        if bool(payload.get("required")) and not bool(payload.get("present"))
    ]
    if missing_contract_blocks:
        reporting_module._write_blocking_summary(
            adequacy_id=adequacy_id,
            contract_check_mode=resolved_contract_check,
            blocked_sweeps=spec.blocked_sweeps,
            pilot_root=pilot_root,
            materialized_corpora=materialized_corpora,
            latent_target_contract=latent_target_contract,
            canary_summary=canary_summary,
            definition=spec.decision_buckets.get("generator_problem"),
            reasoning=[
                "latent-target contract validation failed for "
                + ", ".join(sorted(missing_contract_blocks))
            ],
        )
        raise RuntimeError(
            "latent-target contract validation failed for one or more adequacy blocks; "
            f"wrote blocking summary to {(pilot_root / shared_module._SUMMARY_MARKDOWN_NAME).resolve()}"
        )

    production_control_corpus_payload = materialized_corpora[shared_module._PRODUCTION_BLOCK_ID]
    production_control_manifest_path = production_control_corpus_payload.get("manifest_path")
    production_control_summary = production_control_module.run_production_control_pilot(
        requested_corpus_ref=cast(
            str,
            production_control_corpus_payload["requested_corpus_ref"],
        ),
        corpus_ref=cast(
            str | None,
            production_control_corpus_payload.get("materialized_corpus_ref"),
        ),
        manifest_path=(
            None
            if not isinstance(production_control_manifest_path, str)
            or not production_control_manifest_path.strip()
            else Path(production_control_manifest_path).expanduser().resolve()
        ),
        materialization_state=cast(
            str,
            production_control_corpus_payload["materialization_state"],
        ),
        pilot_root=pilot_root,
        device=device,
        force=force,
    )

    interpretation = reporting_module.select_provisional_interpretation(
        decision_buckets=spec.decision_buckets,
        latent_target_contract=latent_target_contract,
        canary_summary=canary_summary,
        production_control_summary=production_control_summary,
    )
    return reporting_module.write_completed_summary(
        adequacy_id=adequacy_id,
        contract_check_mode=resolved_contract_check,
        blocked_sweeps=spec.blocked_sweeps,
        pilot_root=pilot_root,
        materialized_corpora=materialized_corpora,
        latent_target_contract=latent_target_contract,
        canary_summary=canary_summary,
        production_control_summary=production_control_summary,
        interpretation=interpretation,
    )


def finalize_adequacy_pilot(
    *,
    adequacy_id: str,
    dagzoo_root: Path,
    contract_check: str = "fast",
    out_root: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    shared_module._ensure_supported_configuration(
        adequacy_id=adequacy_id,
        device=shared_module._SUPPORTED_DEVICE,
    )
    resolved_contract_check = shared_module._normalize_contract_check_mode(contract_check)
    resolved_repo_root = (repo_root or shared_module._repo_root()).expanduser().resolve()
    spec = load_synthetic_adequacy_spec(adequacy_id, repo_root=resolved_repo_root)
    pilot_root = (
        out_root.expanduser().resolve()
        if out_root is not None
        else default_pilot_output_root(adequacy_id, repo_root=resolved_repo_root)
    )
    pilot_root.mkdir(parents=True, exist_ok=True)

    materialized_corpora: dict[str, dict[str, Any]] = {}
    latent_target_contract: dict[str, dict[str, Any]] = {}
    canary_summary: dict[str, Any] | None = None

    production_block = next(
        (
            block
            for block in spec.blocks
            if block.block_id == shared_module._PRODUCTION_BLOCK_ID
        ),
        None,
    )
    if production_block is None:
        raise RuntimeError(
            f"adequacy spec {adequacy_id!r} is missing the {shared_module._PRODUCTION_BLOCK_ID!r} block"
        )

    for block in spec.blocks:
        if block.block_id == shared_module._PRODUCTION_BLOCK_ID:
            continue
        corpus_record = production_control_module.load_corpus_record(
            str(block.corpus_ref),
            repo_root=resolved_repo_root,
        )
        materialized_corpora[block.block_id] = reporting_module._materialized_corpus_payload(
            block=block,
            corpus_record=corpus_record,
            materialization_state="finalized",
        )
        latent_target_contract[block.block_id] = (
            contract_module.inspect_corpus_latent_target_contract(
                block=block,
                corpus_record=corpus_record,
                mode=resolved_contract_check,
            )
        )
        filter_provenance = shared_module._optional_mapping(
            latent_target_contract[block.block_id].get("filter_provenance")
        )
        if filter_provenance is not None:
            materialized_corpora[block.block_id]["filter_provenance"] = filter_provenance
        if block.block_id == shared_module._CANARY_BLOCK_ID:
            canary_summary = canary_module.score_canary_block(
                block,
                corpus_record=corpus_record,
            )

    production_resolution = production_control_module._resolve_production_control_corpus(
        requested_corpus_ref=str(production_block.corpus_ref),
        pilot_root=pilot_root,
        dagzoo_root=dagzoo_root,
        force=False,
        repo_root=resolved_repo_root,
    )
    production_corpus_record = cast(
        Mapping[str, Any],
        production_resolution["corpus_record"],
    )
    materialized_corpora[production_block.block_id] = reporting_module._materialized_corpus_payload(
        block=production_block,
        corpus_record=production_corpus_record,
        materialization_state=str(production_resolution["materialization_state"]),
    )
    latent_target_contract[production_block.block_id] = (
        contract_module.inspect_corpus_latent_target_contract(
            block=production_block,
            corpus_record=production_corpus_record,
            mode=resolved_contract_check,
        )
    )
    production_filter_provenance = shared_module._optional_mapping(
        latent_target_contract[production_block.block_id].get("filter_provenance")
    )
    if production_filter_provenance is not None:
        materialized_corpora[production_block.block_id]["filter_provenance"] = (
            production_filter_provenance
        )

    production_control_summary = production_control_module._summarize_existing_production_control_pilot(
        requested_corpus_ref=str(production_block.corpus_ref),
        pilot_root=pilot_root,
    )
    interpretation = reporting_module.select_provisional_interpretation(
        decision_buckets=spec.decision_buckets,
        latent_target_contract=latent_target_contract,
        canary_summary=canary_summary,
        production_control_summary=production_control_summary,
    )
    return reporting_module.write_completed_summary(
        adequacy_id=adequacy_id,
        contract_check_mode=resolved_contract_check,
        blocked_sweeps=spec.blocked_sweeps,
        pilot_root=pilot_root,
        materialized_corpora=materialized_corpora,
        latent_target_contract=latent_target_contract,
        canary_summary=canary_summary,
        production_control_summary=production_control_summary,
        interpretation=interpretation,
    )


__all__ = [
    "build_production_control_config",
    "default_pilot_output_root",
    "finalize_adequacy_pilot",
    "inspect_corpus_latent_target_contract",
    "render_adequacy_pilot_markdown",
    "run_adequacy_pilot",
    "run_production_control_pilot",
    "score_canary_block",
    "score_task_local_predictors",
    "select_provisional_interpretation",
    "validate_latent_target_metadata",
]
