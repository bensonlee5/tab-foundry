"""Text rendering for sweep inspection payloads."""

from __future__ import annotations

import json
from typing import Any, Mapping, cast


def render_sweep_row_text(payload: Mapping[str, Any]) -> str:
    queue = cast(Mapping[str, Any], payload["queue"])
    row = cast(Mapping[str, Any], payload["row"])
    target = cast(Mapping[str, Any], payload["target"])
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
