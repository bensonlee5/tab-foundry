"""Pilot orchestration for the adversarial Dagzoo prior optimizer."""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
from omegaconf import OmegaConf

from tab_foundry.bench.artifacts import write_json
from tab_foundry.bench.openml_benchmark import (
    evaluate_tab_foundry_run,
    load_benchmark_manifest_datasets,
)
from tab_foundry.config import compose_config
from tab_foundry.data.corpus_loading import (
    CORPUS_RECIPE_SCHEMA,
    RECIPE_KIND_DAGZOO_MULTI,
    CorpusManifestPolicy,
    CorpusRecipe,
    DagzooInvocationRecipe,
)
from tab_foundry.data.corpus_materialization import materialize_corpus_recipe_object
from tab_foundry.training.trainer import train
from tab_foundry.types import TrainResult

from .config import (
    RobustPriorStudyConfig,
    default_robust_prior_study_path,
    load_robust_prior_study_config,
)
from .proposer import fit_proposer_distribution, sample_proposal
from .scoring import score_probe_manifest
from .search_space import RobustPriorProposal, robust_prior_search_space_v1


@dataclass(frozen=True, slots=True)
class RobustPriorPaths:
    """Filesystem layout for one robust-prior pilot."""

    study_root: Path

    @property
    def summary_path(self) -> Path:
        return self.study_root / "summary.json"

    def round_root(self, round_index: int) -> Path:
        return self.study_root / f"round_{round_index:02d}"


def _paths_for_config(config: RobustPriorStudyConfig) -> RobustPriorPaths:
    return RobustPriorPaths(study_root=config.output_root_path())


def _read_json_mapping(path: Path, *, context: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{context} must decode to a JSON object: {path}")
    return {str(key): value for key, value in cast(Mapping[str, Any], payload).items()}


def _anchor_training_surface_record_path(checkpoint_path: Path) -> Path:
    return checkpoint_path.expanduser().resolve().parent.parent / "training_surface_record.json"


def _control_corpus_ref(
    config: RobustPriorStudyConfig,
    *,
    anchor_checkpoint_path: Path,
) -> str:
    if config.control_corpus_ref is not None:
        return str(config.control_corpus_ref)
    surface_path = _anchor_training_surface_record_path(anchor_checkpoint_path)
    payload = _read_json_mapping(surface_path, context="anchor training surface record")
    data_payload = payload.get("data")
    if not isinstance(data_payload, Mapping):
        raise RuntimeError("anchor training surface record is missing the data payload")
    corpus_ref = data_payload.get("corpus_ref")
    if not isinstance(corpus_ref, str) or not corpus_ref.strip():
        raise RuntimeError(
            "robust-prior pilot requires control_corpus_ref in config or corpus_ref in the "
            f"anchor training surface record: {surface_path}"
        )
    return str(corpus_ref)


def _first_stage_template(cfg: Any) -> dict[str, Any]:
    stages = getattr(cfg.schedule, "stages", None)
    if not isinstance(stages, list) or not stages:
        return {"name": "round", "steps": 1, "lr_max": 8.0e-4}
    first_stage = stages[0]
    if isinstance(first_stage, Mapping):
        return {str(key): value for key, value in first_stage.items()}
    return {"name": "round", "steps": 1, "lr_max": 8.0e-4}


def _configure_round_training_cfg(
    *,
    config: RobustPriorStudyConfig,
    output_dir: Path,
    corpus_ref: str,
    initial_checkpoint_path: Path,
) -> Any:
    cfg = compose_config([f"experiment={config.base_experiment}"])
    cfg.runtime.output_dir = str(output_dir)
    cfg.runtime.max_steps = int(config.train_steps_per_round)
    cfg.runtime.eval_every = max(1, min(int(config.train_steps_per_round), 25))
    cfg.runtime.checkpoint_every = max(1, min(int(config.train_steps_per_round), 25))
    cfg.logging.use_wandb = bool(config.logging_use_wandb)
    OmegaConf.update(cfg, "data.source", "manifest", merge=False, force_add=True)
    OmegaConf.update(cfg, "data.requested_corpus_ref", str(corpus_ref), merge=False, force_add=True)
    OmegaConf.update(cfg, "data.corpus_ref", str(corpus_ref), merge=False, force_add=True)
    OmegaConf.update(cfg, "data.manifest_path", None, merge=False, force_add=True)
    cfg.training.initial_checkpoint_path = str(initial_checkpoint_path.expanduser().resolve())
    stage = _first_stage_template(cfg)
    stage["name"] = "round"
    stage["steps"] = int(config.train_steps_per_round)
    cfg.schedule.stages = [stage]
    for section_name, overrides in config.training_overrides.items():
        section = cfg.get(section_name)
        if section is None or not isinstance(overrides, Mapping):
            continue
        for key, value in overrides.items():
            setattr(section, str(key), value)
    return cfg


def _selected_checkpoint(result: TrainResult) -> Path:
    checkpoint = result.best_checkpoint or result.latest_checkpoint
    if checkpoint is None:
        raise RuntimeError(f"training did not produce a checkpoint under {result.output_dir}")
    return checkpoint.expanduser().resolve()


def _benchmark_summary_for_run(
    *,
    run_dir: Path,
    benchmark_manifest_path: Path,
    benchmark_device: str,
    checkpoint_selection: str,
) -> dict[str, Any]:
    datasets, task_records, benchmark_surface = load_benchmark_manifest_datasets(
        benchmark_manifest_path=benchmark_manifest_path,
    )
    records = evaluate_tab_foundry_run(
        run_dir,
        datasets=datasets,
        task_type=str(benchmark_surface["task_type"]),
        device=benchmark_device,
        allow_missing_values=bool(benchmark_surface["allow_missing_values"]),
        checkpoint_selection=checkpoint_selection,
    )
    successful = [
        dict(record)
        for record in records
        if record.get("log_loss") is not None and math.isfinite(float(record["log_loss"]))
    ]
    successful.sort(key=lambda record: int(record["step"]))
    if not successful:
        raise RuntimeError(f"benchmark evaluation produced no successful checkpoints: {run_dir}")
    best = min(successful, key=lambda record: float(record["log_loss"]))
    final = successful[-1]
    return {
        "objective_metric": "final_log_loss_at_matched_regime_budget",
        "benchmark_manifest_path": str(benchmark_manifest_path.expanduser().resolve()),
        "task_count": int(len(task_records)),
        "best_log_loss": float(best["log_loss"]),
        "best_step": int(best["step"]),
        "final_log_loss": float(final["log_loss"]),
        "final_step": int(final["step"]),
        "best_to_final_log_loss_delta": float(final["log_loss"]) - float(best["log_loss"]),
        "curve_records": successful,
    }


def _recipe_manifest_policy() -> CorpusManifestPolicy:
    return CorpusManifestPolicy(
        train_ratio=0.90,
        val_ratio=0.05,
        filter_policy="accepted_only",
        missing_value_policy="allow_any",
    )


def _trial_probe_recipe(
    *,
    config: RobustPriorStudyConfig,
    round_index: int,
    trial_index: int,
    proposal: RobustPriorProposal,
    num_datasets: int,
) -> CorpusRecipe:
    search_space = robust_prior_search_space_v1()
    recipe_id = f"{config.study_id}__probe_r{round_index:02d}_t{trial_index:02d}"
    invocation = DagzooInvocationRecipe(
        invocation_id="probe",
        config_ref=None,
        base_config_ref=str(config.dagzoo_base_config_ref),
        config_overrides=search_space.proposal_to_overrides(proposal),
        num_datasets=int(num_datasets),
        seed=int(round_index * 1000 + trial_index),
        rows=None,
        device="cpu",
        hardware_policy="none",
        diagnostics=False,
        diagnostics_out_dir=None,
        missing_rate=None,
        missing_mechanism=None,
        missing_mar_observed_fraction=None,
        missing_mar_logit_scale=None,
        missing_mnar_logit_scale=None,
    )
    return CorpusRecipe(
        recipe_id=recipe_id,
        kind=RECIPE_KIND_DAGZOO_MULTI,
        description=f"Robust prior probe round {round_index} trial {trial_index}",
        surface_label=f"{config.study_id}_probe",
        manifest_policy=_recipe_manifest_policy(),
        invocations=(invocation,),
        provenance_labels={
            "schema": CORPUS_RECIPE_SCHEMA,
            "study_id": config.study_id,
            "round_index": int(round_index),
            "trial_index": int(trial_index),
            "proposal": proposal.to_dict(),
        },
        generator=None,
        review_summary={
            "study_id": config.study_id,
            "round_index": int(round_index),
            "trial_index": int(trial_index),
        },
        recipe_path=config.output_root_path() / f"probe_recipe_r{round_index:02d}_t{trial_index:02d}.yaml",
    )


def _round_training_recipe(
    *,
    config: RobustPriorStudyConfig,
    round_index: int,
    selected_trials: Sequence[Mapping[str, Any]],
) -> CorpusRecipe:
    search_space = robust_prior_search_space_v1()
    candidate_count = max(1, len(selected_trials))
    base_count = int(config.round_train_datasets) // candidate_count
    remainder = int(config.round_train_datasets) % candidate_count
    invocations: list[DagzooInvocationRecipe] = []
    for index, trial in enumerate(selected_trials, start=1):
        proposal = RobustPriorProposal(**dict(cast(Mapping[str, Any], trial["proposal"])))
        num_datasets = int(base_count + (1 if index <= remainder else 0))
        invocations.append(
            DagzooInvocationRecipe(
                invocation_id=f"candidate_{index:02d}",
                config_ref=None,
                base_config_ref=str(config.dagzoo_base_config_ref),
                config_overrides=search_space.proposal_to_overrides(proposal),
                num_datasets=max(1, int(num_datasets)),
                seed=int(round_index * 1000 + index),
                rows=None,
                device="cpu",
                hardware_policy="none",
                diagnostics=False,
                diagnostics_out_dir=None,
                missing_rate=None,
                missing_mechanism=None,
                missing_mar_observed_fraction=None,
                missing_mar_logit_scale=None,
                missing_mnar_logit_scale=None,
            )
        )
    return CorpusRecipe(
        recipe_id=f"{config.study_id}__round_{round_index:02d}_adversarial",
        kind=RECIPE_KIND_DAGZOO_MULTI,
        description=f"Robust prior round {round_index} adversarial training corpus",
        surface_label=f"{config.study_id}_round_{round_index:02d}_adversarial",
        manifest_policy=_recipe_manifest_policy(),
        invocations=tuple(invocations),
        provenance_labels={
            "study_id": config.study_id,
            "round_index": int(round_index),
            "curriculum_id": "robust_prior_adversarial",
            "curriculum_mix": [
                {
                    "proposal": dict(cast(Mapping[str, Any], trial["proposal"])),
                    "normalized_gap": cast(Mapping[str, Any], trial["aggregate"]).get(
                        "normalized_gap"
                    ),
                }
                for trial in selected_trials
            ],
        },
        generator=None,
        review_summary={"selected_candidate_count": int(len(selected_trials))},
        recipe_path=config.output_root_path() / f"round_{round_index:02d}_adversarial_recipe.yaml",
    )


def _trial_distance_vector(trial: Mapping[str, Any]) -> np.ndarray:
    aggregate = cast(Mapping[str, Any], trial.get("aggregate", {}))
    proposal_vector = cast(Mapping[str, Any], trial.get("proposal_vector", {}))
    values = np.asarray(
        [
            float(aggregate.get("normalized_gap", 0.0)),
            float(aggregate.get("depth_ratio", 0.0) or 0.0),
            float(aggregate.get("feature_count_center", 0.0)),
            float(aggregate.get("categorical_ratio_center", 0.0)),
            float(aggregate.get("class_count_center", 0.0)),
            float(proposal_vector.get("mechanism_nonlinearity_mass", 0.0)),
            float(proposal_vector.get("shift_enabled", 0.0)),
            float(proposal_vector.get("shift_graph_scale", 0.0)),
            float(proposal_vector.get("shift_variance_scale", 0.0)),
        ],
        dtype=np.float64,
    )
    return values


def _select_diverse_candidates(
    *,
    trials: Sequence[Mapping[str, Any]],
    topk: int,
    minimum_distance: float,
) -> list[dict[str, Any]]:
    feasible = [dict(trial) for trial in trials if bool(trial.get("feasible", False))]
    feasible.sort(
        key=lambda trial: float(cast(Mapping[str, Any], trial["aggregate"]).get("normalized_gap", 0.0)),
        reverse=True,
    )
    if not feasible:
        return []
    selected = [feasible[0]]
    remaining = feasible[1:]
    while remaining and len(selected) < int(topk):
        best_index = None
        best_score = None
        for index, candidate in enumerate(remaining):
            candidate_vector = _trial_distance_vector(candidate)
            distances = [
                float(np.linalg.norm(candidate_vector - _trial_distance_vector(chosen)))
                for chosen in selected
            ]
            min_distance = min(distances) if distances else 0.0
            score = (
                min_distance,
                float(cast(Mapping[str, Any], candidate["aggregate"]).get("normalized_gap", 0.0)),
            )
            if best_score is None or score > best_score:
                best_index = index
                best_score = score
        if best_index is None:
            break
        candidate = remaining.pop(best_index)
        candidate_distance = (
            float(best_score[0]) if best_score is not None else 0.0
        )
        if candidate_distance < float(minimum_distance) and len(selected) >= 1:
            break
        selected.append(candidate)
    return selected


def _score_trial(
    *,
    config: RobustPriorStudyConfig,
    search_space: Any,
    dagzoo_root: Path,
    round_index: int,
    trial_index: int,
    proposal: RobustPriorProposal,
    checkpoint_path: Path,
) -> dict[str, Any]:
    round_root = _paths_for_config(config).round_root(round_index)
    probe_recipe = _trial_probe_recipe(
        config=config,
        round_index=round_index,
        trial_index=trial_index,
        proposal=proposal,
        num_datasets=int(config.probe_datasets_per_trial),
    )
    corpus_record = materialize_corpus_recipe_object(
        recipe=probe_recipe,
        dagzoo_root=dagzoo_root,
        materialize_worker_threads=config.materialize_worker_threads,
    )
    manifest_path = Path(str(cast(Mapping[str, Any], corpus_record["manifest"])["manifest_path"]))
    probe_score = score_probe_manifest(
        manifest_path=manifest_path,
        checkpoint_path=checkpoint_path,
        device=str(config.benchmark_device),
        seed=int(round_index * 10_000 + trial_index),
        class_entropy_floor=float(config.guardrails.class_entropy_floor),
        min_class_prior_headroom=float(config.guardrails.min_class_prior_headroom),
        authored_depth_ratio_band=(
            max(
                0.0,
                search_space.authored_depth_ratio_band(proposal)[0]
                - float(config.guardrails.depth_ratio_tolerance),
            ),
            min(
                1.0,
                search_space.authored_depth_ratio_band(proposal)[1]
                + float(config.guardrails.depth_ratio_tolerance),
            ),
        ),
    )
    payload = {
        "trial_index": int(trial_index),
        "proposal": proposal.to_dict(),
        "proposal_vector": search_space.proposal_vector(proposal),
        "corpus_ref": str(corpus_record.get("corpus_ref", "")),
        "corpus_record_path": str(corpus_record.get("corpus_record_path", "")),
        **probe_score.as_dict(),
    }
    write_json(round_root / f"trial_{trial_index:02d}.json", payload)
    return payload


def run_robust_prior_pilot(
    *,
    study_id: str | None = None,
    study_path: Path | None = None,
    studies_root: Path | None = None,
    dagzoo_root: Path,
) -> dict[str, Any]:
    """Run one robust-prior pilot end to end."""

    config = load_robust_prior_study_config(
        study_id=study_id,
        study_path=study_path,
        studies_root=studies_root,
    )
    paths = _paths_for_config(config)
    paths.study_root.mkdir(parents=True, exist_ok=True)
    write_json(paths.study_root / "config.json", config.as_dict())
    search_space = robust_prior_search_space_v1()
    rng = np.random.default_rng(0)
    dagzoo_root = dagzoo_root.expanduser().resolve()
    anchor_checkpoint_path = config.anchor_checkpoint()
    benchmark_manifest_path = config.benchmark_manifest()
    control_corpus_ref = _control_corpus_ref(config, anchor_checkpoint_path=anchor_checkpoint_path)
    trial_history: list[dict[str, Any]] = []
    round_summaries: list[dict[str, Any]] = []
    adversarial_checkpoint = anchor_checkpoint_path
    control_checkpoint = anchor_checkpoint_path
    final_decision = "defer"
    defer_reason = None

    for round_index in range(1, int(config.outer_rounds) + 1):
        round_root = paths.round_root(round_index)
        round_root.mkdir(parents=True, exist_ok=True)
        proposer = fit_proposer_distribution(
            search_space=search_space,
            trial_history=trial_history,
            seed=int(round_index),
        )
        round_trials: list[dict[str, Any]] = []
        seen_encodings: set[tuple[int, ...]] = set()
        for trial_index in range(1, int(config.trials_per_round) + 1):
            while True:
                proposal, sampling = sample_proposal(
                    search_space=search_space,
                    rng=rng,
                    probabilities=None if proposer is None else proposer.probabilities,
                    exploration_rate=float(config.exploration_rate),
                    entropy_floor_ratio=float(config.entropy_floor_ratio),
                )
                encoded = tuple(search_space.encode(proposal))
                if encoded not in seen_encodings:
                    seen_encodings.add(encoded)
                    break
            trial_payload = _score_trial(
                config=config,
                search_space=search_space,
                dagzoo_root=dagzoo_root,
                round_index=round_index,
                trial_index=trial_index,
                proposal=proposal,
                checkpoint_path=adversarial_checkpoint,
            )
            trial_payload["sampling"] = sampling
            round_trials.append(trial_payload)
            trial_history.append(trial_payload)
        selected_trials = _select_diverse_candidates(
            trials=round_trials,
            topk=int(config.topk_training_candidates),
            minimum_distance=float(config.guardrails.diversity_min_distance),
        )
        round_summary: dict[str, Any] = {
            "round_index": int(round_index),
            "trial_count": int(len(round_trials)),
            "selected_candidate_count": int(len(selected_trials)),
            "proposer": None if proposer is None else {
                "probabilities": proposer.probabilities,
                "fit_summary": proposer.fit_summary,
            },
            "selected_trials": selected_trials,
        }
        if not selected_trials:
            defer_reason = "no_feasible_diverse_candidates"
            round_summary["status"] = "deferred"
            write_json(round_root / "round_summary.json", round_summary)
            round_summaries.append(round_summary)
            break
        adversarial_recipe = _round_training_recipe(
            config=config,
            round_index=round_index,
            selected_trials=selected_trials,
        )
        adversarial_record = materialize_corpus_recipe_object(
            recipe=adversarial_recipe,
            dagzoo_root=dagzoo_root,
            materialize_worker_threads=config.materialize_worker_threads,
        )
        adversarial_corpus_ref = str(adversarial_record["corpus_ref"])
        adversarial_train_cfg = _configure_round_training_cfg(
            config=config,
            output_dir=round_root / "adversarial_train",
            corpus_ref=adversarial_corpus_ref,
            initial_checkpoint_path=adversarial_checkpoint,
        )
        adversarial_result = train(adversarial_train_cfg)
        adversarial_checkpoint = _selected_checkpoint(adversarial_result)
        adversarial_benchmark = _benchmark_summary_for_run(
            run_dir=adversarial_result.output_dir,
            benchmark_manifest_path=benchmark_manifest_path,
            benchmark_device=str(config.benchmark_device),
            checkpoint_selection=str(config.benchmark_checkpoint_selection),
        )
        control_benchmark = None
        control_result_payload = None
        if config.matched_control:
            control_train_cfg = _configure_round_training_cfg(
                config=config,
                output_dir=round_root / "control_train",
                corpus_ref=control_corpus_ref,
                initial_checkpoint_path=control_checkpoint,
            )
            control_result = train(control_train_cfg)
            control_checkpoint = _selected_checkpoint(control_result)
            control_benchmark = _benchmark_summary_for_run(
                run_dir=control_result.output_dir,
                benchmark_manifest_path=benchmark_manifest_path,
                benchmark_device=str(config.benchmark_device),
                checkpoint_selection=str(config.benchmark_checkpoint_selection),
            )
            control_result_payload = {
                "output_dir": str(control_result.output_dir.resolve()),
                "best_checkpoint": (
                    None
                    if control_result.best_checkpoint is None
                    else str(control_result.best_checkpoint.resolve())
                ),
                "latest_checkpoint": (
                    None
                    if control_result.latest_checkpoint is None
                    else str(control_result.latest_checkpoint.resolve())
                ),
                "global_step": int(control_result.global_step),
            }
        round_summary.update(
            {
                "status": "completed",
                "adversarial_corpus_ref": adversarial_corpus_ref,
                "adversarial_result": {
                    "output_dir": str(adversarial_result.output_dir.resolve()),
                    "best_checkpoint": (
                        None
                        if adversarial_result.best_checkpoint is None
                        else str(adversarial_result.best_checkpoint.resolve())
                    ),
                    "latest_checkpoint": (
                        None
                        if adversarial_result.latest_checkpoint is None
                        else str(adversarial_result.latest_checkpoint.resolve())
                    ),
                    "global_step": int(adversarial_result.global_step),
                },
                "control_result": control_result_payload,
                "benchmark": {
                    "adversarial": adversarial_benchmark,
                    "control": control_benchmark,
                },
            }
        )
        write_json(round_root / "round_summary.json", round_summary)
        round_summaries.append(round_summary)
    if round_summaries:
        final_round = round_summaries[-1]
        benchmark_payload = cast(Mapping[str, Any], final_round.get("benchmark", {}))
        adversarial_metrics = cast(Mapping[str, Any], benchmark_payload.get("adversarial", {}))
        control_metrics = cast(Mapping[str, Any], benchmark_payload.get("control", {}))
        if adversarial_metrics.get("final_log_loss") is not None:
            if (
                control_metrics.get("final_log_loss") is not None
                and float(adversarial_metrics["final_log_loss"])
                < float(control_metrics["final_log_loss"])
            ):
                final_decision = "keep"
            elif defer_reason is None:
                defer_reason = "matched_control_not_beaten"
    summary = {
        "study_id": config.study_id,
        "description": config.description,
        "study_root": str(paths.study_root.resolve()),
        "anchor_checkpoint_path": str(anchor_checkpoint_path),
        "control_corpus_ref": control_corpus_ref,
        "benchmark_manifest_path": str(benchmark_manifest_path.resolve()),
        "search_space_id": search_space.search_space_id,
        "round_summaries": round_summaries,
        "final_decision": final_decision,
        "defer_reason": defer_reason,
    }
    write_json(paths.summary_path, summary)
    return summary


def inspect_robust_prior_pilot(
    *,
    study_id: str | None = None,
    study_path: Path | None = None,
    studies_root: Path | None = None,
) -> dict[str, Any]:
    """Inspect one completed or partial robust-prior pilot from artifacts."""

    config = load_robust_prior_study_config(
        study_id=study_id,
        study_path=study_path,
        studies_root=studies_root,
    )
    paths = _paths_for_config(config)
    summary_path = paths.summary_path
    if not summary_path.exists():
        raise RuntimeError(f"robust-prior summary does not exist: {summary_path}")
    payload = _read_json_mapping(summary_path, context="robust-prior summary")
    resolved_config_path = (
        study_path.expanduser().resolve()
        if study_path is not None
        else default_robust_prior_study_path(str(study_id), studies_root=studies_root)
    )
    payload["config_path"] = str(resolved_config_path)
    return payload


def render_robust_prior_text(payload: Mapping[str, Any]) -> str:
    """Render one compact human-readable robust-prior summary."""

    study_id = str(payload.get("study_id", "unknown"))
    final_decision = str(payload.get("final_decision", "unknown"))
    lines = [
        f"study_id: {study_id}",
        f"final_decision: {final_decision}",
    ]
    defer_reason = payload.get("defer_reason")
    if defer_reason is not None:
        lines.append(f"defer_reason: {defer_reason}")
    round_summaries = payload.get("round_summaries")
    if isinstance(round_summaries, list):
        lines.append(f"rounds: {len(round_summaries)}")
        for round_summary in round_summaries:
            if not isinstance(round_summary, Mapping):
                continue
            round_index = int(round_summary.get("round_index", 0))
            status = str(round_summary.get("status", "unknown"))
            selected = int(round_summary.get("selected_candidate_count", 0))
            lines.append(
                f"round_{round_index:02d}: status={status} selected_candidates={selected}"
            )
            benchmark_payload = round_summary.get("benchmark")
            if isinstance(benchmark_payload, Mapping):
                adversarial = benchmark_payload.get("adversarial")
                control = benchmark_payload.get("control")
                if isinstance(adversarial, Mapping) and adversarial.get("final_log_loss") is not None:
                    lines.append(
                        "  adversarial_final_log_loss="
                        f"{float(adversarial['final_log_loss']):.6f}"
                    )
                if isinstance(control, Mapping) and control.get("final_log_loss") is not None:
                    lines.append(
                        "  control_final_log_loss="
                        f"{float(control['final_log_loss']):.6f}"
                    )
    return "\n".join(lines)
