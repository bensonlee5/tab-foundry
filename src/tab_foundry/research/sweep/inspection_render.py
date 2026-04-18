"""Text rendering for sweep inspection payloads."""

from __future__ import annotations

import json
from typing import Any, Mapping, cast


def render_sweep_row_text(payload: Mapping[str, Any]) -> str:
    queue = cast(Mapping[str, Any], payload["queue"])
    row = cast(Mapping[str, Any], payload["row"])
    target = cast(Mapping[str, Any], payload["target"])
    navigation = cast(Mapping[str, Any] | None, payload.get("navigation"))
    row_summary = cast(Mapping[str, Any] | None, payload.get("row_summary"))
    selector_summary = cast(Mapping[str, Any] | None, payload.get("selector_summary"))
    resolved = cast(Mapping[str, Any], target["resolved"])
    model = cast(Mapping[str, Any], resolved["model"])
    data = cast(Mapping[str, Any] | None, resolved.get("data"))
    preprocessing = cast(Mapping[str, Any] | None, resolved.get("preprocessing"))
    training = cast(Mapping[str, Any] | None, resolved.get("training"))
    parameter_counts = cast(Mapping[str, Any], model["parameter_counts"])
    lines = [
        "Sweep row inspection.",
        f"sweep_id={queue['sweep_id']}",
        f"order={int(row['order']):02d}",
        f"delta_id={row['delta_id']}",
        f"status={row['status']}",
        f"decision={row.get('decision') or 'n/a'}",
        f"run_id={row.get('run_id') or 'n/a'}",
        f"parent_delta_ref={row.get('parent_delta_ref') or 'n/a'}",
        f"benchmark_checkpoint_selection={row.get('benchmark_checkpoint_selection') or 'all'}",
        f"training_experiment={queue['training_experiment']}",
        f"training_config_profile={queue['training_config_profile']}",
        f"surface_role={queue['surface_role']}",
        f"model.stage_label={model.get('stage_label')}",
        f"model.arch={model.get('arch')}",
        f"model.parameters.total={parameter_counts['total_params']}",
        f"model.parameters.trainable={parameter_counts['trainable_params']}",
    ]
    if navigation is not None:
        lineage = navigation.get("lineage")
        if isinstance(lineage, list) and lineage:
            lines.append(
                "lineage=" + " -> ".join(str(cast(Mapping[str, Any], entry)["sweep_id"]) for entry in lineage if isinstance(entry, Mapping))
            )
        contract = navigation.get("contract")
        if isinstance(contract, Mapping):
            lines.append(f"benchmark_manifest_path={contract.get('benchmark_manifest_path')}")
            if contract.get("uses_default_anchor_benchmark") is not None:
                lines.append(
                    "uses_default_anchor_benchmark="
                    + ("yes" if contract.get("uses_default_anchor_benchmark") else "no")
                )
            default_anchor_benchmark = contract.get("default_anchor_benchmark")
            if isinstance(default_anchor_benchmark, Mapping):
                lines.append(
                    "default_anchor_benchmark_manifest_path="
                    f"{default_anchor_benchmark.get('benchmark_manifest_path')}"
                )
            lines.append(f"control_baseline_id={contract.get('control_baseline_id')}")
            lines.append(f"carried_in_family_baseline_run_id={contract.get('carried_in_family_baseline_run_id')}")
            if contract.get("corpus_ref") is not None:
                lines.append(f"data.corpus_ref.locked={contract.get('corpus_ref')}")
            formal_external_reference = contract.get("formal_external_reference")
            if isinstance(formal_external_reference, Mapping):
                lines.append(
                    "formal_external_reference="
                    + json.dumps(dict(formal_external_reference), sort_keys=True)
                )
        winner = navigation.get("winner")
        if isinstance(winner, Mapping):
            winner_parts = [f"order {int(winner['order']):02d}"]
            if winner.get("geometry_label") is not None:
                winner_parts.append(str(winner["geometry_label"]))
            winner_parts.append(f"log_loss={float(winner['benchmark_log_loss']):.6f}")
            if winner.get("throughput_tokens_per_second") is not None:
                winner_parts.append(
                    f"throughput={float(winner['throughput_tokens_per_second']):.1f} tok/s"
                )
            if winner.get("end_to_end_wall_seconds") is not None:
                winner_parts.append(
                    f"wall={float(winner['end_to_end_wall_seconds']):.1f}s"
                )
            lines.append("sweep_winner=" + ", ".join(winner_parts))
        linked_scaling_studies = navigation.get("linked_scaling_study_ids")
        if isinstance(linked_scaling_studies, list) and linked_scaling_studies:
            lines.append(
                "linked_scaling_studies="
                + ", ".join(str(value) for value in linked_scaling_studies)
            )
    if row_summary is not None:
        lines.append(
            "pareto_admissible="
            + (
                "yes"
                if row_summary.get("pareto_admissible") is True
                else ("no" if row_summary.get("pareto_admissible") is False else "n/a")
            )
        )
        lines.append(
            "geometry_pareto_admissible="
            + (
                "yes"
                if row_summary.get("geometry_pareto_admissible") is True
                else (
                    "no"
                    if row_summary.get("geometry_pareto_admissible") is False
                    else "n/a"
                )
            )
        )
        lines.append(
            f"selector_geometry_label={row_summary.get('selector_geometry_label') or 'n/a'}"
        )
        lines.append(
            f"selector_prescription_label={row_summary.get('selector_prescription_label') or 'n/a'}"
        )
        if row_summary.get("end_to_end_wall_seconds") is not None:
            lines.append(
                "end_to_end_wall_seconds="
                f"{float(row_summary['end_to_end_wall_seconds']):.1f}"
            )
        if row_summary.get("throughput_tokens_per_second") is not None:
            lines.append(
                "throughput_tokens_per_second="
                f"{float(row_summary['throughput_tokens_per_second']):.1f}"
            )
    if selector_summary is not None:
        best_row = selector_summary.get("best_row")
        if isinstance(best_row, Mapping):
            lines.append(
                "selector_best_row="
                f"order {int(best_row['order']):02d}, "
                f"geometry={best_row.get('geometry_label') or 'n/a'}, "
                f"prescription={best_row.get('prescription_label') or 'n/a'}, "
                f"log_loss={float(best_row['final_log_loss']):.6f}, "
                f"wall={float(best_row['end_to_end_wall_seconds']):.1f}s"
            )
        kept_contract = selector_summary.get("kept_contract")
        if isinstance(kept_contract, Mapping):
            lines.append(
                "selector_kept_contract="
                f"{kept_contract['prescription_label']} "
                f"(frontier_geometries={int(kept_contract['geometry_count'])}, "
                f"mean_wall={float(kept_contract['mean_end_to_end_wall_seconds']):.1f}s, "
                f"mean_log_loss={float(kept_contract['mean_benchmark_log_loss']):.6f})"
            )
        elif selector_summary.get("no_universal_kept_contract") is True:
            lines.append("selector_kept_contract=none")
    if data is not None:
        lines.append(f"data.surface_label={data.get('surface_label')}")
        if data.get("corpus_ref") is not None:
            lines.append(f"data.corpus_ref={data.get('corpus_ref')}")
        if data.get("recipe_id") is not None:
            lines.append(f"data.recipe_id={data.get('recipe_id')}")
        if data.get("corpus_id") is not None:
            lines.append(f"data.corpus_id={data.get('corpus_id')}")
    if preprocessing is not None:
        lines.append(f"preprocessing.surface_label={preprocessing.get('surface_label')}")
    if training is not None:
        lines.append(f"training.surface_label={training.get('surface_label')}")
    reuse_train_artifact = cast(Mapping[str, Any] | None, row.get("reuse_train_artifact"))
    if reuse_train_artifact is not None:
        lines.append(f"reuse_train_artifact.run_dir={reuse_train_artifact.get('run_dir')}")
        lines.append(
            "reuse_train_artifact.training_surface_fingerprint="
            f"{reuse_train_artifact.get('training_surface_fingerprint')}"
        )
    module_selection = model.get("module_selection")
    if isinstance(module_selection, Mapping):
        lines.append(f"model.module_selection={json.dumps(module_selection, sort_keys=True)}")
    module_hyperparameters = model.get("module_hyperparameters")
    if isinstance(module_hyperparameters, Mapping):
        lines.append(
            f"model.module_hyperparameters={json.dumps(module_hyperparameters, sort_keys=True)}"
        )
    metrics = target.get("metrics")
    if isinstance(metrics, Mapping):
        lines.append(f"metrics={json.dumps(dict(metrics), sort_keys=True)}")
    artifacts = cast(Mapping[str, Any], target["artifacts"])
    for key in (
        "run_dir",
        "benchmark_dir",
        "training_surface_record_json",
        "comparison_summary_json",
    ):
        entry = artifacts.get(key)
        if isinstance(entry, Mapping):
            lines.append(f"{key}={entry['path']}")
    return "\n".join(lines)
