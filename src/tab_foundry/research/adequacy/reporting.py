"""Summary and interpretation helpers for adequacy pilot."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from .contract import _manifest_path_from_corpus_record
from .shared import (
    _ABSOLUTE_CANARY_IMPROVEMENT_THRESHOLD,
    _SUMMARY_JSON_NAME,
    _SUMMARY_MARKDOWN_NAME,
    _ensure_mapping,
    _finite_float_or_none,
    _int_or_none,
    _json_safe,
    _optional_mapping,
    _write_json,
)


def _canary_failure_reasons(canary_summary: Mapping[str, Any] | None) -> list[str]:
    if canary_summary is None:
        return ["canary baseline summary is missing"]
    if int(canary_summary.get("predictor_error_count", 0)) > 0:
        return ["one or more canary baseline tasks failed to score cleanly"]
    comparisons = _optional_mapping(canary_summary.get("comparisons")) or {}
    if not comparisons:
        return ["canary baseline comparisons are missing"]
    healthy_buckets = 0
    checked_buckets = 0
    for comparison in comparisons.values():
        comparison_payload = _optional_mapping(comparison)
        if comparison_payload is None:
            continue
        improvement = _finite_float_or_none(
            comparison_payload.get("chance_minus_logistic_log_loss")
        )
        if improvement is None:
            continue
        checked_buckets += 1
        if improvement >= _ABSOLUTE_CANARY_IMPROVEMENT_THRESHOLD:
            healthy_buckets += 1
    if checked_buckets == 0:
        return ["canary baseline comparisons did not produce any scored row-total buckets"]
    if healthy_buckets * 2 < checked_buckets:
        return [
            "logistic regression does not beat chance by a convincing margin on most canary row totals"
        ]
    return []


def _production_training_problem_reasons(production_summary: Mapping[str, Any] | None) -> list[str]:
    if production_summary is None:
        return []
    if production_summary.get("status") == "error":
        error_payload = _optional_mapping(production_summary.get("error")) or {}
        return [
            "production-control sandwich pilot errored: "
            f"{error_payload.get('type', 'RuntimeError')}: {error_payload.get('message', 'unknown error')}"
        ]
    run_inspect_payload = _optional_mapping(production_summary.get("run_inspect")) or {}
    health = _optional_mapping(run_inspect_payload.get("health"))
    if health is None:
        return []
    verdict = str(health.get("verdict", "")).strip().lower()
    metrics = _optional_mapping(health.get("metrics")) or {}
    initial_train_loss = _finite_float_or_none(metrics.get("initial_train_loss"))
    final_train_loss = _finite_float_or_none(metrics.get("final_train_loss"))
    if verdict == "fail":
        return ["production-control sandwich pilot tripped the run-health fail thresholds"]
    if (
        initial_train_loss is not None
        and final_train_loss is not None
        and final_train_loss >= initial_train_loss * 0.98
    ):
        return ["production-control sandwich pilot does not show a meaningful train-loss reduction"]
    return []


def select_provisional_interpretation(
    *,
    decision_buckets: Mapping[str, str],
    latent_target_contract: Mapping[str, Mapping[str, Any]],
    canary_summary: Mapping[str, Any] | None,
    production_control_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    missing_contract_blocks = [
        block_id
        for block_id, payload in latent_target_contract.items()
        if bool(payload.get("required")) and not bool(payload.get("present"))
    ]
    if missing_contract_blocks:
        reasoning = [
            "latent-target contract validation failed for "
            + ", ".join(sorted(missing_contract_blocks))
        ]
        bucket = "generator_problem"
    else:
        canary_reasons = _canary_failure_reasons(canary_summary)
        if canary_reasons:
            reasoning = canary_reasons
            bucket = "generator_problem"
        else:
            production_reasons = _production_training_problem_reasons(production_control_summary)
            if production_reasons:
                reasoning = production_reasons
                bucket = "training_regime_problem"
            else:
                reasoning = [
                    "latent-target lineage metadata validates, the canary baselines beat chance, and the "
                    "single production-control CPU pilot is not decisively broken"
                ]
                bucket = "inconclusive"
    return {
        "bucket": bucket,
        "definition": decision_buckets.get(bucket),
        "reasoning": reasoning,
    }


def _markdown_float(value: Any) -> str:
    numeric = _finite_float_or_none(value)
    return "n/a" if numeric is None else f"{numeric:.4f}"


def render_adequacy_pilot_markdown(summary: Mapping[str, Any]) -> str:
    materialized_corpora = _optional_mapping(summary.get("materialized_corpora")) or {}
    latent_target_contract = _optional_mapping(summary.get("latent_target_contract")) or {}
    canary_summary = _optional_mapping(summary.get("canary_baselines"))
    production_control_summary = _optional_mapping(summary.get("production_control_pilot"))
    interpretation = _ensure_mapping(
        summary.get("provisional_interpretation"),
        context="summary.provisional_interpretation",
    )
    contract_check = _optional_mapping(summary.get("contract_check")) or {}

    lines = [
        f"# {summary['adequacy_id']} adequacy pilot",
        "",
        f"- Status: `{summary['status']}`",
        f"- Contract check: `{contract_check.get('mode', 'fast')}`",
        f"- Provisional interpretation: `{interpretation['bucket']}`",
        f"- Blocked sweeps remain: {', '.join(cast(list[str], summary['blocked_sweeps']))}",
        "",
        "## Corpora",
        "",
        "| Block | Requested | Resolved | State | Latent target contract | Curated accepted | Acceptance rate |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for block_id, payload in materialized_corpora.items():
        corpus_payload = _ensure_mapping(payload, context=f"materialized_corpora.{block_id}")
        contract_payload = _optional_mapping(latent_target_contract.get(block_id)) or {}
        filter_payload = _optional_mapping(contract_payload.get("filter_provenance")) or {}
        target_accepted = _int_or_none(filter_payload.get("target_accepted_datasets"))
        curated_accepted = _int_or_none(filter_payload.get("curated_accepted_datasets"))
        curated_display = "n/a"
        if target_accepted is not None:
            curated_display = (
                f"`{curated_accepted}`/`{target_accepted}`"
                if curated_accepted is not None
                else f"`?`/`{target_accepted}`"
            )
        resolved_corpus = (
            corpus_payload.get("materialized_corpus_ref")
            or corpus_payload.get("manifest_path")
            or "n/a"
        )
        lines.append(
            "| "
            + " | ".join(
                [
                    block_id,
                    f"`{corpus_payload['requested_corpus_ref']}`",
                    f"`{resolved_corpus}`",
                    f"`{corpus_payload.get('materialization_state', 'unknown')}`",
                    "`present`" if contract_payload.get("present") else "`missing`",
                    curated_display,
                    _markdown_float(filter_payload.get("acceptance_rate")),
                ]
            )
            + " |"
        )
    if canary_summary is not None:
        lines.extend(
            [
                "",
                "## Canary Baselines",
                "",
                "| n | chance log loss | logistic log loss | chance - logistic |",
                "| --- | --- | --- | --- |",
            ]
        )
        scores_by_predictor = _optional_mapping(canary_summary.get("scores_by_predictor")) or {}
        chance_scores = _optional_mapping(scores_by_predictor.get("chance")) or {}
        logistic_scores = _optional_mapping(scores_by_predictor.get("logistic_regression")) or {}
        comparisons = _optional_mapping(canary_summary.get("comparisons")) or {}
        for row_total in ("128", "256", "512", "1024"):
            chance_payload = _optional_mapping(chance_scores.get(row_total)) or {}
            logistic_payload = _optional_mapping(logistic_scores.get(row_total)) or {}
            comparison_payload = _optional_mapping(comparisons.get(row_total)) or {}
            lines.append(
                "| "
                + " | ".join(
                    [
                        row_total,
                        _markdown_float(chance_payload.get("label_target_log_loss_per_test_cell")),
                        _markdown_float(logistic_payload.get("label_target_log_loss_per_test_cell")),
                        _markdown_float(comparison_payload.get("chance_minus_logistic_log_loss")),
                    ]
                )
                + " |"
            )
    if production_control_summary is not None:
        lines.extend(
            [
                "",
                "## Production Control Pilot",
                "",
                f"- Status: `{production_control_summary.get('status', 'unknown')}`",
                f"- Run dir: `{production_control_summary.get('run_dir', 'unknown')}`",
            ]
        )
        metrics = _optional_mapping(production_control_summary.get("metrics")) or {}
        if metrics:
            lines.append(
                "- Validation losses: "
                f"best={_markdown_float(metrics.get('best_val_loss'))}, "
                f"final={_markdown_float(metrics.get('final_val_loss'))}"
            )
        run_inspect_payload = _optional_mapping(production_control_summary.get("run_inspect")) or {}
        health = _optional_mapping(run_inspect_payload.get("health")) or {}
        if health:
            lines.append(f"- Health verdict: `{health.get('verdict', 'unknown')}`")
            if health.get("summary") is not None:
                lines.append(f"- Health summary: {health['summary']}")

    reasoning = cast(list[str], interpretation.get("reasoning", []))
    if reasoning:
        lines.extend(
            [
                "",
                "## Interpretation Notes",
                "",
            ]
        )
        for reason in reasoning:
            lines.append(f"- {reason}")
    return "\n".join(lines) + "\n"


def _summary_paths(pilot_root: Path) -> dict[str, str]:
    return {
        "summary_json": str((pilot_root / _SUMMARY_JSON_NAME).resolve()),
        "summary_md": str((pilot_root / _SUMMARY_MARKDOWN_NAME).resolve()),
    }


def _write_blocking_summary(
    *,
    adequacy_id: str,
    contract_check_mode: str,
    blocked_sweeps: Sequence[str],
    pilot_root: Path,
    materialized_corpora: Mapping[str, Any],
    latent_target_contract: Mapping[str, Any],
    canary_summary: Mapping[str, Any] | None,
    definition: str | None,
    reasoning: list[str],
) -> None:
    summary = {
        "adequacy_id": adequacy_id,
        "contract_check": {"mode": contract_check_mode},
        "status": "blocked",
        "blocked_sweeps": list(blocked_sweeps),
        "materialized_corpora": _json_safe(materialized_corpora),
        "latent_target_contract": _json_safe(latent_target_contract),
        "canary_baselines": _json_safe(canary_summary),
        "production_control_pilot": None,
        "provisional_interpretation": {
            "bucket": "generator_problem",
            "definition": definition,
            "reasoning": list(reasoning),
        },
        "summary_paths": _summary_paths(pilot_root),
    }
    _write_json(pilot_root / _SUMMARY_JSON_NAME, summary)
    (pilot_root / _SUMMARY_MARKDOWN_NAME).write_text(
        render_adequacy_pilot_markdown(summary),
        encoding="utf-8",
    )


def _materialized_corpus_payload(
    *,
    block: Any,
    corpus_record: Mapping[str, Any],
    materialization_state: str,
) -> dict[str, Any]:
    raw_materialized_corpus_ref = corpus_record.get("corpus_ref")
    raw_corpus_record_path = corpus_record.get("corpus_record_path")
    raw_corpus_id = corpus_record.get("corpus_id")
    return {
        "requested_corpus_ref": block.corpus_ref,
        "materialized_corpus_ref": (
            None
            if not isinstance(raw_materialized_corpus_ref, str) or not raw_materialized_corpus_ref.strip()
            else str(raw_materialized_corpus_ref)
        ),
        "materialization_state": str(materialization_state),
        "recipe_id": str(corpus_record["recipe_id"]),
        "corpus_id": None if raw_corpus_id is None else str(raw_corpus_id),
        "surface_label": str(corpus_record["surface_label"]),
        "manifest_path": str(_manifest_path_from_corpus_record(corpus_record)),
        "corpus_record_path": (
            None
            if not isinstance(raw_corpus_record_path, str) or not raw_corpus_record_path.strip()
            else str(raw_corpus_record_path)
        ),
    }


def write_completed_summary(
    *,
    adequacy_id: str,
    contract_check_mode: str,
    blocked_sweeps: Sequence[str],
    pilot_root: Path,
    materialized_corpora: Mapping[str, Any],
    latent_target_contract: Mapping[str, Any],
    canary_summary: Mapping[str, Any] | None,
    production_control_summary: Mapping[str, Any] | None,
    interpretation: Mapping[str, Any],
) -> dict[str, Any]:
    summary = {
        "adequacy_id": adequacy_id,
        "contract_check": {"mode": contract_check_mode},
        "status": "completed",
        "blocked_sweeps": list(blocked_sweeps),
        "materialized_corpora": materialized_corpora,
        "latent_target_contract": latent_target_contract,
        "canary_baselines": canary_summary,
        "production_control_pilot": production_control_summary,
        "provisional_interpretation": dict(interpretation),
        "summary_paths": _summary_paths(pilot_root),
    }
    _write_json(pilot_root / _SUMMARY_JSON_NAME, summary)
    (pilot_root / _SUMMARY_MARKDOWN_NAME).write_text(
        render_adequacy_pilot_markdown(summary),
        encoding="utf-8",
    )
    return summary


__all__ = [
    "render_adequacy_pilot_markdown",
    "select_provisional_interpretation",
]
