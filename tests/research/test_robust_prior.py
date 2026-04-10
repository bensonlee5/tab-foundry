from __future__ import annotations

from pathlib import Path
from typing import Any

from click.testing import CliRunner
import pytest
import torch

import tab_foundry.cli as cli_module
import tab_foundry.cli.research_robust_prior as robust_prior_cli_module
import tab_foundry.research.robust_prior.pilot as pilot_module
from tab_foundry.research.robust_prior.proposer import sample_proposal
from tab_foundry.research.robust_prior.scoring import ProbeDatasetScore, ProbeTrialScore, compute_gap_metrics
from tab_foundry.research.robust_prior.search_space import RobustPriorProposal, robust_prior_search_space_v1
from tab_foundry.types import TrainResult


def test_robust_prior_search_space_round_trips_and_maps_to_dagzoo_overrides() -> None:
    search_space = robust_prior_search_space_v1()
    proposal = RobustPriorProposal(
        feature_count_bucket="wide",
        class_count_bucket="medium",
        categorical_ratio_bucket="mixed_high",
        max_categorical_cardinality_bucket="card24",
        graph_node_bucket="graph_large",
        target_depth_bucket="deep",
        mechanism_preset="compositional",
        shift_preset="mixed",
        noise_preset="mixture",
    )

    encoded = search_space.encode(proposal)
    decoded = search_space.decode(encoded)
    overrides = search_space.proposal_to_overrides(proposal)

    assert decoded == proposal
    assert overrides["dataset"]["n_train"] == 768
    assert overrides["dataset"]["n_test"] == 256
    assert overrides["dataset"]["max_categorical_cardinality"] == 24
    assert overrides["graph"]["n_nodes_max"] == 28
    assert overrides["graph"]["target_depth_nodes_min"] >= 1
    assert overrides["noise"]["family"] == "mixture"
    assert overrides["shift"]["mode"] == "mixed"
    assert overrides["mechanism"]["function_family_mix"]["piecewise"] == pytest.approx(2.25)


def test_compute_gap_metrics_matches_hand_worked_example() -> None:
    metrics = compute_gap_metrics(
        tfm_log_loss=0.80,
        baseline_log_losses={
            "catboost": 0.50,
            "random_forest": 0.55,
            "logistic_regression": 0.70,
            "mlp": 0.60,
        },
        class_prior_log_loss=1.10,
    )

    assert metrics["raw_gap"] == pytest.approx(0.30)
    assert metrics["class_prior_headroom"] == pytest.approx(0.60)
    assert metrics["normalized_gap"] == pytest.approx(0.50)


def test_sample_proposal_applies_entropy_floor_and_uniform_exploration() -> None:
    search_space = robust_prior_search_space_v1()
    probabilities = {
        dimension.name: [1.0] + [0.0] * (len(dimension.values) - 1)
        for dimension in search_space.dimensions
    }

    proposal, sampling = sample_proposal(
        search_space=search_space,
        rng=np_random(7),
        probabilities=probabilities,
        exploration_rate=0.20,
        entropy_floor_ratio=0.65,
    )

    assert isinstance(proposal, RobustPriorProposal)
    adjusted = sampling["probabilities"]
    first_dimension = search_space.dimensions[0]
    first_probs = adjusted[first_dimension.name]
    assert pytest.approx(sum(first_probs), rel=1.0e-9, abs=1.0e-9) == 1.0
    assert first_probs[0] < 1.0
    assert all(probability > 0.0 for probability in first_probs)


def test_run_robust_prior_pilot_emits_round_artifacts_and_beats_control(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    study_path = tmp_path / "pilot.yaml"
    anchor_checkpoint = tmp_path / "anchor" / "checkpoints" / "best.pt"
    anchor_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": {}}, anchor_checkpoint)
    benchmark_manifest = tmp_path / "bench" / "manifest.parquet"
    benchmark_manifest.parent.mkdir(parents=True, exist_ok=True)
    benchmark_manifest.write_bytes(b"manifest")
    output_root = tmp_path / "outputs" / "robust_prior"
    study_path.write_text(
        "\n".join(
            [
                "schema: tab-foundry-robust-prior-v1",
                "study_id: smoke",
                "description: smoke",
                f"output_root: {output_root}",
                f"anchor_checkpoint_path: {anchor_checkpoint}",
                "base_experiment: cls_benchmark_staged_corpus",
                "control_corpus_ref: tf_rd_010_dagzoo_medium_control_curated_v5",
                f"benchmark_manifest_path: {benchmark_manifest}",
                "outer_rounds: 1",
                "trials_per_round: 3",
                "probe_datasets_per_trial: 2",
                "topk_training_candidates: 2",
                "round_train_datasets: 8",
                "train_steps_per_round: 2",
                "matched_control: true",
                "logging_use_wandb: false",
                "benchmark_device: cpu",
                "benchmark_checkpoint_selection: all",
                "guardrails:",
                "  class_entropy_floor: 0.1",
                "  min_class_prior_headroom: 0.01",
                "  depth_ratio_tolerance: 0.2",
                "  diversity_min_distance: 0.0",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    captured_recipes: list[str] = []

    def _fake_materialize_corpus_recipe_object(*, recipe, **_kwargs):
        captured_recipes.append(recipe.recipe_id)
        manifest_path = tmp_path / f"{recipe.recipe_id}.parquet"
        manifest_path.write_bytes(b"manifest")
        return {
            "corpus_ref": f"{recipe.recipe_id}/materialized",
            "corpus_record_path": str(tmp_path / f"{recipe.recipe_id}.json"),
            "manifest": {"manifest_path": str(manifest_path)},
        }

    def _fake_score_probe_manifest(**_kwargs):
        dataset_score = ProbeDatasetScore(
            dataset_id="dataset_000001",
            tfm_log_loss=0.80,
            class_prior_log_loss=1.20,
            baseline_log_losses={"catboost": 0.50},
            raw_gap=0.30,
            normalized_gap=0.43,
            class_prior_headroom=0.70,
            class_entropy=0.9,
            graph_target_depth_ratio=0.55,
            feature_count_center=32.0,
            class_count_center=4.0,
            categorical_ratio_center=0.4,
        )
        return ProbeTrialScore(
            dataset_scores=(dataset_score,),
            aggregate={
                "raw_gap": 0.30,
                "normalized_gap": 0.43,
                "class_prior_headroom": 0.70,
                "class_entropy": 0.9,
                "depth_ratio": 0.55,
                "feature_count_center": 32.0,
                "class_count_center": 4.0,
                "categorical_ratio_center": 0.4,
            },
            feasible=True,
        )

    train_calls = 0

    def _fake_train(cfg):
        nonlocal train_calls
        train_calls += 1
        output_dir = Path(str(cfg.runtime.output_dir)).expanduser().resolve()
        checkpoint = output_dir / "checkpoints" / "best.pt"
        checkpoint.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model": {}}, checkpoint)
        (output_dir / "training_surface_record.json").write_text("{}", encoding="utf-8")
        return TrainResult(
            output_dir=output_dir,
            best_checkpoint=checkpoint,
            latest_checkpoint=checkpoint,
            global_step=int(cfg.runtime.max_steps),
            metrics={},
        )

    benchmark_calls: list[str] = []

    def _fake_benchmark_summary_for_run(*, run_dir: Path, **_kwargs):
        benchmark_calls.append(str(run_dir))
        is_control = "control_train" in str(run_dir)
        return {
            "objective_metric": "final_log_loss_at_matched_regime_budget",
            "final_log_loss": 0.48 if is_control else 0.42,
            "best_log_loss": 0.46 if is_control else 0.41,
            "best_to_final_log_loss_delta": 0.02 if is_control else 0.01,
            "curve_records": [{"step": 2, "log_loss": 0.48 if is_control else 0.42}],
        }

    monkeypatch.setattr(pilot_module, "materialize_corpus_recipe_object", _fake_materialize_corpus_recipe_object)
    monkeypatch.setattr(pilot_module, "score_probe_manifest", _fake_score_probe_manifest)
    monkeypatch.setattr(pilot_module, "train", _fake_train)
    monkeypatch.setattr(pilot_module, "_benchmark_summary_for_run", _fake_benchmark_summary_for_run)

    payload = pilot_module.run_robust_prior_pilot(
        study_path=study_path,
        dagzoo_root=tmp_path / "dagzoo",
    )

    assert payload["final_decision"] == "keep"
    assert len(payload["round_summaries"]) == 1
    assert train_calls == 2
    assert any(recipe_id.endswith("_adversarial") for recipe_id in captured_recipes)
    inspect_payload = pilot_module.inspect_robust_prior_pilot(study_path=study_path)
    assert inspect_payload["final_decision"] == "keep"
    assert (output_root / "summary.json").exists()
    assert benchmark_calls


def test_research_robust_prior_cli_dispatches_to_run_and_inspect(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    run_called: dict[str, Any] = {}
    inspect_called: dict[str, Any] = {}
    monkeypatch.setattr(
        robust_prior_cli_module,
        "run_robust_prior_pilot",
        lambda **kwargs: run_called.update(kwargs) or {"study_id": "smoke", "round_summaries": []},
    )
    monkeypatch.setattr(
        robust_prior_cli_module,
        "inspect_robust_prior_pilot",
        lambda **kwargs: inspect_called.update(kwargs) or {"study_id": "smoke", "round_summaries": []},
    )

    run_result = CliRunner().invoke(
        cli_module.cli,
        [
            "research",
            "robust-prior",
            "run",
            "--study",
            "anchor_pilot_v1",
            "--dagzoo-root",
            "/tmp/dagzoo",
            "--json",
        ],
    )
    inspect_result = CliRunner().invoke(
        cli_module.cli,
        [
            "research",
            "robust-prior",
            "inspect",
            "--study",
            "anchor_pilot_v1",
            "--json",
        ],
    )

    assert run_result.exit_code == 0
    assert inspect_result.exit_code == 0
    assert run_called["study_id"] == "anchor_pilot_v1"
    assert Path(str(run_called["dagzoo_root"])).resolve() == Path("/tmp/dagzoo").resolve()
    assert inspect_called["study_id"] == "anchor_pilot_v1"


def np_random(seed: int) -> Any:
    import numpy as np

    return np.random.default_rng(seed)
