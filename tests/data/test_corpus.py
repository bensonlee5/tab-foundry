from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import time
from typing import Any

import pytest
import pyarrow.parquet as pq
import yaml

import tab_foundry.benchmark_registry as registry_module
import tab_foundry.data.corpus_loading as corpus_loading_module
import tab_foundry.data.corpus_lookup as corpus_lookup_module
import tab_foundry.data.corpus_materialization as corpus_materialization_module
import tab_foundry.data.corpus_materialization_batch as corpus_materialization_batch_module
import tab_foundry.data.corpus_materialization_invocation as corpus_materialization_invocation_module
import tab_foundry.data.corpus_materialization_recipe_worker as recipe_worker_module
from tab_foundry.data.corpus_loading import (
    _generator_fingerprint,
    build_dagzoo_provenance_summary,
    corpus_id_for_manifest,
    corpus_outputs_root,
    corpus_recipe_index_path,
    corpus_recipes_root,
    list_corpus_recipes,
    load_corpus_recipe,
)
from tab_foundry.data.dagzoo_workflow import DagzooFilterResult
from tab_foundry.data.corpus_lookup import load_corpus_record
from tab_foundry.data.corpus_materialization import (
    default_materialize_processes,
    default_materialize_worker_threads,
    finalize_staged_corpus_recipe,
    load_staged_corpus_recipe_preview,
    materialize_corpus_ref,
    materialize_corpus_recipe,
)
from tab_foundry.data.corpus_reporting import (
    corpus_compare_payload,
    corpus_results_payload,
)
from tab_realdata_hub.dagzoo_handoff import (
    DAGZOO_HANDOFF_SCHEMA_NAME,
    DAGZOO_HANDOFF_SCHEMA_VERSION,
    load_dagzoo_handoff_info,
    stable_dagzoo_generated_corpus_id,
)
from tab_realdata_hub.manifest import build_manifest
from tab_foundry.data.surface import resolve_data_surface
from tab_foundry.hashing import sha256_path

from tests.support import manifest_and_dataset_cases as cases


REPO_ROOT = Path(__file__).resolve().parents[2]
_TEST_GENERATE_RUN_ID = "1" * 32
_TEST_DATASET_ID = "3" * 32
_TEST_GENERATED_CORPUS_ID = stable_dagzoo_generated_corpus_id(
    generate_run_id=_TEST_GENERATE_RUN_ID,
    dataset_ids=[_TEST_DATASET_ID],
)


def test_corpus_default_paths_follow_shared_repo_root() -> None:
    assert corpus_recipes_root() == REPO_ROOT / "reference" / "corpus_recipes"
    assert corpus_recipe_index_path() == REPO_ROOT / "reference" / "corpus_recipes" / "index.yaml"
    assert corpus_outputs_root() == REPO_ROOT / "outputs" / "corpora"


def test_build_dagzoo_provenance_summary_preserves_latent_target_provenance() -> None:
    recipe = load_corpus_recipe("tf_rd_010_dagzoo_medium_control_v4", repo_root=REPO_ROOT)
    summary = build_dagzoo_provenance_summary(
        recipe=recipe,
        corpus_ref="tf_rd_010_dagzoo_medium_control_v4/abc123",
        corpus_id="abc123",
        provenance={
            "target_derivation": "tabiclv2_latent_node",
            "target_relevant_feature_count_range": {"min": 5, "max": 7},
            "target_relevant_feature_fraction_range": {"min": 0.5, "max": 0.75},
            "invocations": [
                {
                    "handoff_provenance": {
                        "target_derivation": "tabiclv2_latent_node",
                        "target_relevant_feature_count_range": {"min": 6, "max": 8},
                        "target_relevant_feature_fraction_range": {"min": 0.6, "max": 0.8},
                    }
                }
            ],
        },
    )

    assert summary["recipe_id"] == "tf_rd_010_dagzoo_medium_control_v4"
    assert summary["corpus_variant"] == "tf_rd_010_dagzoo_medium_control"
    assert summary["target_derivation"] == "tabiclv2_latent_node"
    assert summary["target_relevant_feature_count_range"] == {"min": 5, "max": 7}
    assert summary["target_relevant_feature_fraction_range"] == {"min": 0.5, "max": 0.75}
    assert summary["review_summary"]["target_derivation"] == "tabiclv2_latent_node"


def test_build_dagzoo_provenance_summary_falls_back_to_recipe_metadata_without_handoff_provenance() -> None:
    recipe = load_corpus_recipe("tf_rd_010_dagzoo_medium_control_v4", repo_root=REPO_ROOT)

    summary = build_dagzoo_provenance_summary(
        recipe=recipe,
        corpus_ref="tf_rd_010_dagzoo_medium_control_v4/abc123",
        corpus_id="abc123",
        provenance={},
    )

    assert summary["target_derivation"] == "tabiclv2_latent_node"
    assert summary.get("target_relevant_feature_count_range") is None
    assert summary.get("target_relevant_feature_fraction_range") is None


def test_build_dagzoo_provenance_summary_aggregates_materialization_timing() -> None:
    recipe = load_corpus_recipe("tf_rd_010_dagzoo_medium_control_v4", repo_root=REPO_ROOT)

    summary = build_dagzoo_provenance_summary(
        recipe=recipe,
        corpus_ref="tf_rd_010_dagzoo_medium_control_v4/abc123",
        corpus_id="abc123",
        provenance={
            "materialization_timing": {
                "recipe_elapsed_seconds": 30.0,
                "invocation_fanout_elapsed_seconds": 20.0,
                "manifest_build_elapsed_seconds": 4.0,
                "promotion_elapsed_seconds": 3.0,
            },
            "invocations": [
                {
                    "materialization_timing": {
                        "generated_datasets": 8,
                        "round_count": 1,
                        "generate_elapsed_seconds": 5.0,
                        "filter_elapsed_seconds": 2.0,
                        "copy_elapsed_seconds": 1.0,
                        "upstream_elapsed_seconds": 7.0,
                        "local_overhead_elapsed_seconds": 0.5,
                        "invocation_elapsed_seconds": 8.5,
                    }
                },
                {
                    "materialization_timing": {
                        "generated_datasets": 6,
                        "round_count": 2,
                        "generate_elapsed_seconds": 4.0,
                        "upstream_elapsed_seconds": 4.0,
                        "local_overhead_elapsed_seconds": 0.25,
                        "invocation_elapsed_seconds": 4.25,
                    }
                },
            ],
        },
    )

    assert summary["materialization_timing"] == {
        "recipe_elapsed_seconds": 30.0,
        "invocation_fanout_elapsed_seconds": 20.0,
        "manifest_build_elapsed_seconds": 4.0,
        "promotion_elapsed_seconds": 3.0,
        "timed_invocation_count": 2,
        "cumulative_round_count": 3,
        "cumulative_generated_datasets": 14,
        "cumulative_generate_elapsed_seconds": 9.0,
        "cumulative_filter_elapsed_seconds": 2.0,
        "cumulative_copy_elapsed_seconds": 1.0,
        "cumulative_upstream_elapsed_seconds": 11.0,
        "cumulative_local_overhead_elapsed_seconds": 0.75,
        "cumulative_invocation_elapsed_seconds": 12.75,
    }


def _patch_corpus_repo_root(monkeypatch: pytest.MonkeyPatch, repo_root: Path) -> None:
    monkeypatch.setattr(corpus_loading_module, "_repo_root", lambda: repo_root)
    monkeypatch.setattr(corpus_lookup_module, "_repo_root", lambda: repo_root)


def _patch_dagzoo_generate(monkeypatch: pytest.MonkeyPatch, replacement: Any) -> None:
    monkeypatch.setattr(corpus_materialization_invocation_module, "run_dagzoo_generate", replacement)


def _patch_dagzoo_filter(monkeypatch: pytest.MonkeyPatch, replacement: Any) -> None:
    monkeypatch.setattr(corpus_materialization_invocation_module, "run_dagzoo_filter", replacement)


def _write_recipe_registry(repo_root: Path) -> None:
    recipe_root = repo_root / "reference" / "corpus_recipes"
    recipe_root.mkdir(parents=True, exist_ok=True)
    (recipe_root / "index.yaml").write_text(
        "\n".join(
            [
                "schema: tab-foundry-corpus-recipe-index-v1",
                "recipes:",
                "  current_recipe:",
                "    path: current_recipe.yaml",
                "  size_recipe:",
                "    path: size_recipe.yaml",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (recipe_root / "current_recipe.yaml").write_text(
        "\n".join(
            [
                "schema: tab-foundry-corpus-recipe-v1",
                "recipe_id: current_recipe",
                "kind: dagzoo_single_invocation",
                "description: Current corpus test recipe.",
                "surface_label: anchor_manifest_default",
                "manifest:",
                "  train_ratio: 0.9",
                "  val_ratio: 0.05",
                "  filter_policy: include_all",
                "  missing_value_policy: allow_any",
                "provenance_labels:",
                "  corpus_variant: current_corpus_default",
                "  comparator_role: control",
                "dagzoo:",
                "  config_ref: configs/default.yaml",
                "  num_datasets: 8",
                "  seed: 1",
                "  device: cpu",
                "  hardware_policy: none",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (recipe_root / "size_recipe.yaml").write_text(
        "\n".join(
            [
                "schema: tab-foundry-corpus-recipe-v1",
                "recipe_id: size_recipe",
                "kind: dagzoo_multi_invocation_manifest",
                "description: Size ladder test recipe.",
                "surface_label: tf_rd_013_dagzoo_shape_aware_size_small",
                "manifest:",
                "  train_ratio: 0.9",
                "  val_ratio: 0.05",
                "  filter_policy: include_all",
                "  missing_value_policy: allow_any",
                "provenance_labels:",
                "  corpus_variant: dagzoo_shape_aware_size_small",
                "  comparator_role: promoted_anchor_candidate",
                "invocations:",
                "  - invocation_id: benchmark_cpu",
                "    config_ref: configs/benchmark_cpu.yaml",
                "    num_datasets: 4",
                "    seed: 1",
                "    device: cpu",
                "    hardware_policy: none",
                "  - invocation_id: large_shape",
                "    config_ref: configs/benchmark_cuda_h100_large_shape.yaml",
                "    num_datasets: 2",
                "    seed: 1",
                "    device: cpu",
                "    hardware_policy: none",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _register_recipe_fixture(
    repo_root: Path,
    *,
    recipe_id: str,
    filename: str,
    contents: str,
) -> None:
    recipe_root = repo_root / "reference" / "corpus_recipes"
    index_path = recipe_root / "index.yaml"
    index_payload = yaml.safe_load(index_path.read_text(encoding="utf-8"))
    assert isinstance(index_payload, dict)
    recipes = index_payload.setdefault("recipes", {})
    assert isinstance(recipes, dict)
    recipes[recipe_id] = {"path": filename}
    index_path.write_text(yaml.safe_dump(index_payload, sort_keys=False), encoding="utf-8")
    (recipe_root / filename).write_text(contents, encoding="utf-8")


def _write_adequacy_recipe_fixture(repo_root: Path) -> None:
    _register_recipe_fixture(
        repo_root,
        recipe_id="adequacy_recipe",
        filename="adequacy_recipe.yaml",
        contents="\n".join(
            [
                "schema: tab-foundry-corpus-recipe-v1",
                "recipe_id: adequacy_recipe",
                "kind: dagzoo_single_invocation",
                "description: Adequacy fallback fixture.",
                "surface_label: adequacy_surface",
                "manifest:",
                "  train_ratio: 0.9",
                "  val_ratio: 0.05",
                "  filter_policy: include_all",
                "  missing_value_policy: allow_any",
                "provenance_labels:",
                "  corpus_variant: adequacy_surface",
                "  comparator_role: control",
                "  target_derivation: tabiclv2_latent_node",
                "review_summary:",
                "  config_refs:",
                "  - configs/default.yaml",
                "  invocation_count: 1",
                "  manifest_record_count: 1",
                "  target_derivation: tabiclv2_latent_node",
                "dagzoo:",
                "  base_config_ref: configs/default.yaml",
                "  config_overrides:",
                "    dataset: {}",
                "  num_datasets: 8",
                "  seed: 1",
                "  device: cpu",
                "  hardware_policy: none",
            ]
        )
        + "\n",
    )


def _write_accepted_only_recipe_fixture(
    repo_root: Path,
    *,
    recipe_id: str = "accepted_only_recipe",
    filename: str = "accepted_only_recipe.yaml",
    num_datasets: int = 2,
) -> None:
    _register_recipe_fixture(
        repo_root,
        recipe_id=recipe_id,
        filename=filename,
        contents="\n".join(
            [
                "schema: tab-foundry-corpus-recipe-v1",
                f"recipe_id: {recipe_id}",
                "kind: dagzoo_single_invocation",
                "description: Accepted-only corpus fixture.",
                "surface_label: accepted_only_surface",
                "manifest:",
                "  train_ratio: 0.9",
                "  val_ratio: 0.05",
                "  filter_policy: accepted_only",
                "  missing_value_policy: allow_any",
                "provenance_labels:",
                "  corpus_variant: accepted_only_surface",
                "  comparator_role: control",
                "  target_derivation: tabiclv2_latent_node",
                "review_summary:",
                "  config_refs:",
                "  - configs/default.yaml",
                "  invocation_count: 1",
                f"  manifest_record_count: {num_datasets}",
                "  target_derivation: tabiclv2_latent_node",
                "dagzoo:",
                "  base_config_ref: configs/default.yaml",
                "  config_overrides:",
                "    dataset: {}",
                f"  num_datasets: {num_datasets}",
                "  seed: 1",
                "  device: cpu",
                "  hardware_policy: none",
            ]
        )
        + "\n",
    )


def _write_multi_invocation_accepted_only_recipe_fixture(
    repo_root: Path,
    *,
    recipe_id: str = "accepted_only_multi_recipe",
    filename: str = "accepted_only_multi_recipe.yaml",
    invocation_ids: tuple[str, str] = ("slow", "fast"),
    num_datasets_per_invocation: int = 2,
) -> None:
    manifest_record_count = int(num_datasets_per_invocation) * len(invocation_ids)
    _register_recipe_fixture(
        repo_root,
        recipe_id=recipe_id,
        filename=filename,
        contents="\n".join(
            [
                "schema: tab-foundry-corpus-recipe-v1",
                f"recipe_id: {recipe_id}",
                "kind: dagzoo_multi_invocation_manifest",
                "description: Accepted-only multi-invocation corpus fixture.",
                "surface_label: accepted_only_multi_surface",
                "manifest:",
                "  train_ratio: 0.9",
                "  val_ratio: 0.05",
                "  filter_policy: accepted_only",
                "  missing_value_policy: allow_any",
                "provenance_labels:",
                "  corpus_variant: accepted_only_multi_surface",
                "  comparator_role: control",
                "  target_derivation: tabiclv2_latent_node",
                "review_summary:",
                "  config_refs:",
                "  - configs/default.yaml",
                f"  invocation_count: {len(invocation_ids)}",
                f"  manifest_record_count: {manifest_record_count}",
                "  target_derivation: tabiclv2_latent_node",
                "invocations:",
                *[
                    line
                    for invocation_id in invocation_ids
                    for line in (
                        f"  - invocation_id: {invocation_id}",
                        "    config_ref: configs/default.yaml",
                        f"    num_datasets: {num_datasets_per_invocation}",
                        "    seed: 1",
                        "    device: cpu",
                        "    hardware_policy: none",
                    )
                ],
            ]
        )
        + "\n",
    )


def _write_generator_recipe_registry(repo_root: Path) -> None:
    recipe_root = repo_root / "reference" / "corpus_recipes"
    recipe_root.mkdir(parents=True, exist_ok=True)
    inputs = {
        "invocation_dataset_counts": {
            "benchmark_cpu": 1,
            "default_medium": 2,
            "large_shape": 1,
        }
    }
    fingerprint, _module_path = _generator_fingerprint(
        module_name="tab_foundry.data.corpus_generators.tf_rd_013",
        callable_name="build_shape_aware_size_recipe",
        inputs=inputs,
    )
    (recipe_root / "index.yaml").write_text(
        "\n".join(
            [
                "schema: tab-foundry-corpus-recipe-index-v1",
                "recipes:",
                "  generated_recipe:",
                "    path: generated_recipe.yaml",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (recipe_root / "generated_recipe.yaml").write_text(
        "\n".join(
            [
                "schema: tab-foundry-corpus-recipe-v1",
                "recipe_id: generated_recipe",
                "kind: dagzoo_python_generated",
                "description: Generated recipe fixture.",
                "surface_label: generated_surface",
                "manifest:",
                "  train_ratio: 0.9",
                "  val_ratio: 0.05",
                "  filter_policy: include_all",
                "  missing_value_policy: allow_any",
                "provenance_labels:",
                "  corpus_variant: generated_surface",
                "  comparator_role: exploratory",
                "generator:",
                "  module: tab_foundry.data.corpus_generators.tf_rd_013",
                "  callable: build_shape_aware_size_recipe",
                "  inputs:",
                "    invocation_dataset_counts:",
                "      benchmark_cpu: 1",
                "      default_medium: 2",
                "      large_shape: 1",
                f"  fingerprint: {fingerprint}",
                "review_summary:",
                "  config_refs:",
                "  - configs/benchmark_cpu.yaml",
                "  - configs/default.yaml",
                "  - configs/benchmark_cuda_h100_large_shape.yaml",
                "  invocation_count: 3",
                "  manifest_record_count: 4",
                "  invocation_dataset_counts:",
                "    benchmark_cpu: 1",
                "    default_medium: 2",
                "    large_shape: 1",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_sweep_recipe_registry(repo_root: Path, *, sweep_id: str) -> None:
    recipe_root = (
        repo_root
        / "reference"
        / "system_delta_sweeps"
        / sweep_id
        / "corpus_recipes"
    )
    recipe_root.mkdir(parents=True, exist_ok=True)
    (recipe_root / "index.yaml").write_text(
        "\n".join(
            [
                "schema: tab-foundry-corpus-recipe-index-v1",
                "recipes:",
                "  current_recipe:",
                "    path: current_recipe.yaml",
                "  sweep_recipe:",
                "    path: sweep_recipe.yaml",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (recipe_root / "current_recipe.yaml").write_text(
        "\n".join(
            [
                "schema: tab-foundry-corpus-recipe-v1",
                "recipe_id: current_recipe",
                "kind: dagzoo_single_invocation",
                "description: Sweep-local override-backed recipe shadowing the global current recipe.",
                "surface_label: sweep_local_current",
                "manifest:",
                "  train_ratio: 0.9",
                "  val_ratio: 0.05",
                "  filter_policy: include_all",
                "  missing_value_policy: allow_any",
                "provenance_labels:",
                "  corpus_variant: sweep_local_current_variant",
                "  comparator_role: exploratory",
                "dagzoo:",
                "  base_config_ref: configs/base_override.yaml",
                "  config_overrides:",
                "    seed: 7",
                "    dataset:",
                "      rows: 256",
                "      shape: sweep_local",
                "    generator:",
                "      mode: harder",
                "  num_datasets: 16",
                "  seed: 3",
                "  device: cpu",
                "  hardware_policy: none",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (recipe_root / "sweep_recipe.yaml").write_text(
        "\n".join(
            [
                "schema: tab-foundry-corpus-recipe-v1",
                "recipe_id: sweep_recipe",
                "kind: dagzoo_single_invocation",
                "description: Sweep-local config_ref recipe.",
                "surface_label: sweep_only_current",
                "manifest:",
                "  train_ratio: 0.9",
                "  val_ratio: 0.05",
                "  filter_policy: include_all",
                "  missing_value_policy: allow_any",
                "provenance_labels:",
                "  corpus_variant: sweep_only_current_variant",
                "  comparator_role: exploratory",
                "dagzoo:",
                "  config_ref: configs/default.yaml",
                "  num_datasets: 6",
                "  seed: 5",
                "  device: cpu",
                "  hardware_policy: none",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_matching_sweep_recipe_registry(repo_root: Path, *, sweep_id: str) -> None:
    recipe_root = (
        repo_root
        / "reference"
        / "system_delta_sweeps"
        / sweep_id
        / "corpus_recipes"
    )
    recipe_root.mkdir(parents=True, exist_ok=True)
    (recipe_root / "index.yaml").write_text(
        "\n".join(
            [
                "schema: tab-foundry-corpus-recipe-index-v1",
                "recipes:",
                "  current_recipe:",
                "    path: current_recipe.yaml",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (recipe_root / "current_recipe.yaml").write_text(
        "\n".join(
            [
                "schema: tab-foundry-corpus-recipe-v1",
                "recipe_id: current_recipe",
                "kind: dagzoo_single_invocation",
                "description: Sweep-local recipe matching the global current recipe output.",
                "surface_label: sweep_local_current",
                "manifest:",
                "  train_ratio: 0.9",
                "  val_ratio: 0.05",
                "  filter_policy: include_all",
                "  missing_value_policy: allow_any",
                "provenance_labels:",
                "  corpus_variant: sweep_local_current_variant",
                "  comparator_role: exploratory",
                "dagzoo:",
                "  config_ref: configs/default.yaml",
                "  num_datasets: 8",
                "  seed: 1",
                "  device: cpu",
                "  hardware_policy: none",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_broken_sweep_recipe_registry(repo_root: Path, *, sweep_id: str) -> None:
    recipe_root = (
        repo_root
        / "reference"
        / "system_delta_sweeps"
        / sweep_id
        / "corpus_recipes"
    )
    recipe_root.mkdir(parents=True, exist_ok=True)
    (recipe_root / "index.yaml").write_text(
        "\n".join(
            [
                "schema: tab-foundry-corpus-recipe-index-v1",
                "recipes:",
                "  current_recipe:",
                "    path: missing_recipe.yaml",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_handoff_manifest(
    handoff_root: Path,
    *,
    generated_dir_rel: str = "generated",
    curated_dir_rel: str | None = None,
    generate_run_id: str = _TEST_GENERATE_RUN_ID,
    generated_corpus_id: str = _TEST_GENERATED_CORPUS_ID,
) -> Path:
    handoff_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_name": DAGZOO_HANDOFF_SCHEMA_NAME,
        "schema_version": DAGZOO_HANDOFF_SCHEMA_VERSION,
        "identity": {
            "source_family": "dagzoo.fixed_layout_scm",
            "generate_run_id": generate_run_id,
            "generated_corpus_id": generated_corpus_id,
        },
        "artifacts_relative": {
            "generated_dir": generated_dir_rel,
        },
    }
    if curated_dir_rel is not None:
        payload["artifacts_relative"]["curated_dir"] = curated_dir_rel
    handoff_manifest_path = handoff_root / "handoff_manifest.json"
    handoff_manifest_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return handoff_manifest_path


def _write_generated_dataset(
    generated_dir: Path,
    *,
    seed: int,
    generate_run_id: str = _TEST_GENERATE_RUN_ID,
    dataset_id: str = _TEST_DATASET_ID,
) -> None:
    x_train, y_train, x_test, y_test = cases._classification_arrays(seed=seed)
    metadata = cases._classification_metadata(
        n_features=x_train.shape[1],
        seed=seed,
        filter_status="accepted",
        filter_accepted=True,
    )
    metadata["dataset_id"] = dataset_id
    metadata["split_groups"] = {"request_run": generate_run_id}
    cases._write_packed_shard(
        generated_dir / "shard_00000",
        datasets=[
            {
                "dataset_index": 0,
                "x_train": x_train,
                "y_train": y_train,
                "x_test": x_test,
                "y_test": y_test,
                "feature_types": ["floating"] * x_train.shape[1],
                "metadata": metadata,
            }
        ],
    )


def _fake_run_dagzoo_generate(config) -> object:
    handoff_root = Path(str(config.handoff_root)).expanduser().resolve()
    generated_dir = handoff_root / "generated"
    _write_generated_dataset(generated_dir, seed=max(int(config.num_datasets), 1))
    handoff_manifest_path = _write_handoff_manifest(handoff_root)
    return load_dagzoo_handoff_info(handoff_manifest_path)


def _fake_run_dagzoo_generate_many(config) -> object:
    handoff_root = Path(str(config.handoff_root)).expanduser().resolve()
    generated_dir = handoff_root / "generated"
    generated_dir.mkdir(parents=True, exist_ok=True)
    _write_curated_datasets(generated_dir, dataset_count=max(int(config.num_datasets), 1), seed_base=10)
    handoff_manifest_path = _write_handoff_manifest(handoff_root)
    return load_dagzoo_handoff_info(handoff_manifest_path)


def _write_fake_dagzoo_python(dagzoo_root: Path) -> Path:
    interpreter = dagzoo_root / ".venv" / "bin" / "python"
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
    interpreter.chmod(0o755)
    return interpreter


def _write_curated_datasets(curated_dir: Path, *, dataset_count: int, seed_base: int = 100) -> None:
    for dataset_offset in range(int(dataset_count)):
        _write_generated_dataset(
            curated_dir / f"shard_{dataset_offset:05d}",
            seed=seed_base + dataset_offset,
            dataset_id=f"{seed_base + dataset_offset:032x}",
        )


def _fake_run_dagzoo_filter(config) -> DagzooFilterResult:
    filter_root = Path(str(config.filter_out_dir)).expanduser().resolve()
    curated_dir = Path(str(config.curated_out_dir)).expanduser().resolve()
    filter_root.mkdir(parents=True, exist_ok=True)
    curated_dir.mkdir(parents=True, exist_ok=True)
    _write_curated_datasets(curated_dir, dataset_count=1)
    manifest_path = filter_root / "filter_manifest.ndjson"
    summary_path = filter_root / "filter_summary.json"
    manifest_path.write_text("{}\n", encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "total_datasets": 1,
                "accepted_datasets": 1,
                "rejected_datasets": 0,
                "curated_out_dir": str(curated_dir.resolve()),
                "curated_accepted_datasets": 1,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return DagzooFilterResult(
        manifest_path=manifest_path.resolve(),
        summary_path=summary_path.resolve(),
        total_datasets=1,
        accepted_datasets=1,
        rejected_datasets=0,
        elapsed_seconds=0.1,
        datasets_per_minute=600.0,
        curated_out_dir=curated_dir.resolve(),
        curated_accepted_datasets=1,
    )


def _fake_run_dagzoo_filter_all(config) -> DagzooFilterResult:
    filter_root = Path(str(config.filter_out_dir)).expanduser().resolve()
    curated_dir = Path(str(config.curated_out_dir)).expanduser().resolve()
    input_dir = Path(str(config.input_dir)).expanduser().resolve()
    filter_root.mkdir(parents=True, exist_ok=True)
    curated_dir.mkdir(parents=True, exist_ok=True)
    dataset_count = len(sorted(input_dir.glob("shard_*")))
    _write_curated_datasets(curated_dir, dataset_count=max(dataset_count, 1), seed_base=200)
    manifest_path = filter_root / "filter_manifest.ndjson"
    summary_path = filter_root / "filter_summary.json"
    manifest_path.write_text("{}\n" * max(1, dataset_count), encoding="utf-8")
    summary_path.write_text(
        json.dumps(
            {
                "total_datasets": dataset_count,
                "accepted_datasets": dataset_count,
                "rejected_datasets": 0,
                "curated_out_dir": str(curated_dir.resolve()),
                "curated_accepted_datasets": dataset_count,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return DagzooFilterResult(
        manifest_path=manifest_path.resolve(),
        summary_path=summary_path.resolve(),
        total_datasets=dataset_count,
        accepted_datasets=dataset_count,
        rejected_datasets=0,
        elapsed_seconds=0.1,
        datasets_per_minute=600.0,
        curated_out_dir=curated_dir.resolve(),
        curated_accepted_datasets=dataset_count,
    )


def _rewrite_curated_root_as_legacy_shards(
    curated_root: Path,
    *,
    dataset_count: int,
    seed_base: int = 300,
) -> None:
    if curated_root.exists():
        shutil.rmtree(curated_root)
    curated_root.mkdir(parents=True, exist_ok=True)
    _write_curated_datasets(curated_root, dataset_count=dataset_count, seed_base=seed_base)
    for catalog_path in curated_root.rglob("dataset_catalog.parquet"):
        catalog_path.unlink()


def _round_sequence_fake_run_dagzoo_filter(
    curated_counts: list[int],
    *,
    total_datasets_per_round: int = 2,
):
    call_counter = {"count": 0}

    def _run(config) -> DagzooFilterResult:
        call_counter["count"] += 1
        call_index = call_counter["count"] - 1
        curated_count = curated_counts[call_index] if call_index < len(curated_counts) else 0
        filter_root = Path(str(config.filter_out_dir)).expanduser().resolve()
        curated_dir = Path(str(config.curated_out_dir)).expanduser().resolve()
        filter_root.mkdir(parents=True, exist_ok=True)
        curated_dir.mkdir(parents=True, exist_ok=True)
        _write_curated_datasets(
            curated_dir,
            dataset_count=curated_count,
            seed_base=200 + call_index * 10,
        )
        manifest_path = filter_root / "filter_manifest.ndjson"
        summary_path = filter_root / "filter_summary.json"
        manifest_path.write_text("{}\n" * max(1, curated_count), encoding="utf-8")
        summary_path.write_text(
            json.dumps(
                {
                    "total_datasets": total_datasets_per_round,
                    "accepted_datasets": curated_count,
                    "rejected_datasets": max(0, total_datasets_per_round - curated_count),
                    "curated_out_dir": str(curated_dir.resolve()),
                    "curated_accepted_datasets": curated_count,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return DagzooFilterResult(
            manifest_path=manifest_path.resolve(),
            summary_path=summary_path.resolve(),
            total_datasets=total_datasets_per_round,
            accepted_datasets=curated_count,
            rejected_datasets=max(0, total_datasets_per_round - curated_count),
            elapsed_seconds=0.1,
            datasets_per_minute=600.0,
            curated_out_dir=curated_dir.resolve(),
            curated_accepted_datasets=curated_count,
        )

    return _run


def _fake_run_dagzoo_generate_mismatched_handoff(config) -> object:
    handoff_root = Path(str(config.handoff_root)).expanduser().resolve()
    generated_dir = handoff_root / "generated"
    _write_generated_dataset(generated_dir, seed=max(int(config.num_datasets), 1))
    handoff_manifest_path = _write_handoff_manifest(
        handoff_root,
        generated_corpus_id="4" * 32,
    )
    return load_dagzoo_handoff_info(handoff_manifest_path)


def _counting_fake_run_dagzoo_generate(call_counter: list[int]):
    def _run(config) -> object:
        call_counter[0] += 1
        return _fake_run_dagzoo_generate(config)

    return _run


def _initialize_repo_workspace(repo_root: Path) -> None:
    _write_recipe_registry(repo_root)
    dagzoo_root = repo_root / ".." / "dagzoo"
    _write_fake_dagzoo_python(dagzoo_root)
    (dagzoo_root / "configs").mkdir(parents=True, exist_ok=True)
    (dagzoo_root / "configs" / "default.yaml").write_text("seed: 1\n", encoding="utf-8")
    (dagzoo_root / "configs" / "benchmark_cpu.yaml").write_text("seed: 1\n", encoding="utf-8")
    (dagzoo_root / "configs" / "benchmark_cuda_h100_large_shape.yaml").write_text(
        "seed: 1\n",
        encoding="utf-8",
    )
    (dagzoo_root / "configs" / "base_override.yaml").write_text(
        "\n".join(
            [
                "seed: 1",
                "dataset:",
                "  rows: 128",
                "  shape: anchor",
                "generator:",
                "  mode: baseline",
            ]
        )
        + "\n",
        encoding="utf-8",
    )


def _write_legacy_unscoped_corpus_record(
    repo_root: Path,
    *,
    sweep_id: str,
    recipe_id: str,
    seed: int,
) -> dict[str, Any]:
    recipe = load_corpus_recipe(recipe_id, repo_root=repo_root, sweep_id=sweep_id)
    corpus_root = repo_root / "outputs" / "corpora" / recipe_id / ".legacy_unscoped"
    invocation_root = corpus_root / "invocations" / "default"
    generated_dir = invocation_root / "generated"
    _write_generated_dataset(generated_dir, seed=seed)
    _write_handoff_manifest(invocation_root)
    manifest_path = corpus_root / "manifest.parquet"
    _ = build_manifest(
        data_roots=[generated_dir],
        out_path=manifest_path,
        train_ratio=float(recipe.manifest_policy.train_ratio),
        val_ratio=float(recipe.manifest_policy.val_ratio),
        filter_policy=str(recipe.manifest_policy.filter_policy),
        missing_value_policy=str(recipe.manifest_policy.missing_value_policy),
    )
    manifest_sha256 = sha256_path(manifest_path)
    legacy_corpus_id = corpus_id_for_manifest(
        recipe_id=recipe.recipe_id,
        manifest_sha256=manifest_sha256,
    )
    final_root = corpus_root.parent / legacy_corpus_id
    if final_root.exists():
        shutil.rmtree(final_root)
    shutil.move(str(corpus_root), str(final_root))
    final_invocation_root = final_root / "invocations" / "default"
    final_generated_dir = final_invocation_root / "generated"
    final_handoff_manifest_path = final_invocation_root / "handoff_manifest.json"
    final_manifest_path = final_root / "manifest.parquet"
    record_path = final_root / "corpus_record.json"
    latest_pointer_path = final_root.parent / "latest.json"
    legacy_record = {
        "schema": "tab-foundry-corpus-record-v1",
        "generated_at_utc": "2026-03-24T00:00:00Z",
        "recipe_id": recipe.recipe_id,
        "corpus_id": legacy_corpus_id,
        "corpus_ref": f"{recipe.recipe_id}/{legacy_corpus_id}",
        "recipe_path": str(recipe.recipe_path),
        "surface_label": recipe.surface_label,
        "surface_label_recommendation": recipe.surface_label,
        "recipe": recipe.to_dict(),
        "artifacts": {
            "corpus_root": str(final_root.resolve()),
            "manifest_path": str(final_manifest_path.resolve()),
            "latest_pointer_path": str(latest_pointer_path.resolve()),
        },
        "manifest": {
            "manifest_path": str(final_manifest_path.resolve()),
            "manifest_sha256": manifest_sha256,
            "inspection": {"total_records": 1},
            "characteristics": {"record_count": 1},
        },
        "dagzoo_provenance": {
            "corpus_ref": f"{recipe.recipe_id}/{legacy_corpus_id}",
            "recipe_id": recipe.recipe_id,
            "corpus_id": legacy_corpus_id,
            "commands": [],
            "config_refs": [],
            "curated_root_lineage": [],
            "invocations": [
                {
                    "invocation_id": "default",
                    "invocation_root": str(final_invocation_root.resolve()),
                    "handoff": {
                        "handoff_manifest_path": str(final_handoff_manifest_path.resolve()),
                        "generated_dir": str(final_generated_dir.resolve()),
                    },
                }
            ],
        },
        "corpus_record_path": str(record_path.resolve()),
    }
    record_path.write_text(json.dumps(legacy_record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    latest_pointer_path.write_text(
        json.dumps(
            {
                "schema": "tab-foundry-corpus-latest-v1",
                "generated_at_utc": "2026-03-24T00:00:00Z",
                "recipe_id": recipe.recipe_id,
                "corpus_id": legacy_corpus_id,
                "corpus_ref": legacy_record["corpus_ref"],
                "corpus_record_path": str(record_path.resolve()),
                "recipe_path": str(recipe.recipe_path),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return legacy_record


def test_load_and_list_corpus_recipes(repo_tmp_path: Path) -> None:
    _write_recipe_registry(repo_tmp_path)

    recipes = list_corpus_recipes(repo_root=repo_tmp_path)

    assert [recipe.recipe_id for recipe in recipes] == ["current_recipe", "size_recipe"]
    current = load_corpus_recipe("current_recipe", repo_root=repo_tmp_path)
    assert current.kind == "dagzoo_single_invocation"
    assert current.surface_label == "anchor_manifest_default"
    assert current.invocations[0].config_ref == "configs/default.yaml"


def test_load_generator_backed_recipe_expands_checked_in_summary(repo_tmp_path: Path) -> None:
    _write_generator_recipe_registry(repo_tmp_path)

    recipe = load_corpus_recipe("generated_recipe", repo_root=repo_tmp_path)

    assert recipe.kind == "dagzoo_python_generated"
    assert recipe.generator is not None
    assert recipe.generator["module"] == "tab_foundry.data.corpus_generators.tf_rd_013"
    assert recipe.review_summary == {
        "config_refs": [
            "configs/benchmark_cpu.yaml",
            "configs/default.yaml",
            "configs/benchmark_cuda_h100_large_shape.yaml",
        ],
        "invocation_count": 3,
        "manifest_record_count": 4,
        "invocation_dataset_counts": {
            "benchmark_cpu": 1,
            "default_medium": 2,
            "large_shape": 1,
        },
    }
    assert [invocation.invocation_id for invocation in recipe.invocations] == [
        "benchmark_cpu",
        "default_medium",
        "large_shape",
    ]
    assert [invocation.num_datasets for invocation in recipe.invocations] == [1, 2, 1]


def test_load_and_list_corpus_recipes_include_sweep_local_shadowing(repo_tmp_path: Path) -> None:
    _write_sweep_recipe_registry(repo_tmp_path, sweep_id="tf_rd_local")

    recipes = list_corpus_recipes(repo_root=repo_tmp_path, sweep_id="tf_rd_local")

    assert [recipe.recipe_id for recipe in recipes] == ["current_recipe", "size_recipe", "sweep_recipe"]
    current = load_corpus_recipe("current_recipe", repo_root=repo_tmp_path, sweep_id="tf_rd_local")
    assert current.surface_label == "sweep_local_current"
    assert current.recipe_path == (
        repo_tmp_path
        / "reference"
        / "system_delta_sweeps"
        / "tf_rd_local"
        / "corpus_recipes"
        / "current_recipe.yaml"
    )
    assert current.invocations[0].config_ref is None
    assert current.invocations[0].base_config_ref == "configs/base_override.yaml"
    size = load_corpus_recipe("size_recipe", repo_root=repo_tmp_path, sweep_id="tf_rd_local")
    assert size.recipe_path == repo_tmp_path / "reference" / "corpus_recipes" / "size_recipe.yaml"


def test_load_corpus_record_raises_for_broken_sweep_local_recipe(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    sweep_id = "tf_rd_broken"
    _write_broken_sweep_recipe_registry(repo_tmp_path, sweep_id=sweep_id)
    _patch_dagzoo_generate(monkeypatch, _fake_run_dagzoo_generate)
    global_record = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=True,
        repo_root=repo_tmp_path,
    )

    with pytest.raises(RuntimeError, match="corpus recipe 'current_recipe' does not exist"):
        load_corpus_record("current_recipe", repo_root=repo_tmp_path, sweep_id=sweep_id)

    loaded_global = load_corpus_record("current_recipe", repo_root=repo_tmp_path)
    assert loaded_global["corpus_ref"] == global_record["corpus_ref"]


@pytest.fixture
def repo_tmp_path(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    _initialize_repo_workspace(repo_root)
    return repo_root


def test_default_materialization_budget_balances_processes_and_worker_threads() -> None:
    assert default_materialize_processes(cpu_count=10) == 8
    assert default_materialize_worker_threads(cpu_count=10, materialize_processes=8) == 1
    assert default_materialize_processes(cpu_count=4) == 3
    assert default_materialize_worker_threads(cpu_count=4, materialize_processes=3) == 1


def test_materialize_corpus_recipe_writes_corpus_record_and_latest_pointer(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _patch_dagzoo_generate(monkeypatch, _fake_run_dagzoo_generate)

    record = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=True,
        repo_root=repo_tmp_path,
    )

    manifest_sha = str(record["manifest"]["manifest_sha256"])
    expected_corpus_id = corpus_id_for_manifest(
        recipe_id="current_recipe",
        manifest_sha256=manifest_sha,
    )
    assert record["corpus_id"] == expected_corpus_id
    assert record["corpus_ref"] == f"current_recipe/{expected_corpus_id}"
    assert Path(str(record["manifest"]["manifest_path"])).exists()
    assert Path(str(record["corpus_record_path"])).exists()
    latest_pointer_path = repo_tmp_path / "outputs" / "corpora" / "current_recipe" / "latest.json"
    latest_payload = json.loads(latest_pointer_path.read_text(encoding="utf-8"))
    assert latest_payload["corpus_ref"] == record["corpus_ref"]
    loaded = load_corpus_record("current_recipe", repo_root=repo_tmp_path)
    assert loaded["corpus_ref"] == record["corpus_ref"]
    assert loaded["dagzoo_provenance"]["config_refs"] == ["configs/default.yaml"]
    invocation = loaded["dagzoo_provenance"]["invocations"][0]
    materialization_summary_path = (
        Path(str(invocation["invocation_root"])) / "materialization_summary.json"
    )
    materialization_summary = json.loads(
        materialization_summary_path.read_text(encoding="utf-8")
    )
    assert materialization_summary["filter_policy"] == "include_all"
    assert materialization_summary["generated_datasets"] == invocation["num_datasets"]
    assert materialization_summary["generate_elapsed_seconds"] >= 0.0
    assert materialization_summary["upstream_elapsed_seconds"] >= 0.0
    assert materialization_summary["local_overhead_elapsed_seconds"] >= 0.0
    assert materialization_summary["invocation_elapsed_seconds"] >= 0.0

    invocation_timing = invocation["materialization_timing"]
    assert invocation_timing["generated_datasets"] == invocation["num_datasets"]
    assert invocation_timing["generate_elapsed_seconds"] == pytest.approx(
        materialization_summary["generate_elapsed_seconds"]
    )
    assert invocation_timing["upstream_elapsed_seconds"] == pytest.approx(
        materialization_summary["upstream_elapsed_seconds"]
    )
    assert invocation_timing["local_overhead_elapsed_seconds"] == pytest.approx(
        materialization_summary["local_overhead_elapsed_seconds"]
    )
    assert invocation_timing["invocation_elapsed_seconds"] == pytest.approx(
        materialization_summary["invocation_elapsed_seconds"]
    )

    timing_summary = loaded["dagzoo_provenance_summary"]["materialization_timing"]
    assert loaded["dagzoo_provenance"]["materialization_timing"] == timing_summary
    assert timing_summary["timed_invocation_count"] == 1
    assert timing_summary["cumulative_generated_datasets"] == invocation["num_datasets"]
    assert timing_summary["cumulative_generate_elapsed_seconds"] == pytest.approx(
        invocation_timing["generate_elapsed_seconds"]
    )
    assert timing_summary["cumulative_upstream_elapsed_seconds"] == pytest.approx(
        invocation_timing["upstream_elapsed_seconds"]
    )
    assert timing_summary["cumulative_local_overhead_elapsed_seconds"] == pytest.approx(
        invocation_timing["local_overhead_elapsed_seconds"]
    )
    assert timing_summary["cumulative_invocation_elapsed_seconds"] == pytest.approx(
        invocation_timing["invocation_elapsed_seconds"]
    )
    assert timing_summary["recipe_elapsed_seconds"] >= 0.0
    assert timing_summary["invocation_fanout_elapsed_seconds"] >= 0.0
    assert timing_summary["manifest_build_elapsed_seconds"] >= 0.0
    assert timing_summary["promotion_elapsed_seconds"] >= 0.0


def test_materialize_corpus_recipe_delegates_multi_invocation_runs_to_subprocess_fanout(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _patch_dagzoo_generate(monkeypatch, _fake_run_dagzoo_generate)
    captured: dict[str, Any] = {}

    def _fake_subprocess_fanout(
        *,
        recipe_id: str,
        invocations,
        dagzoo_root: Path,
        corpus_root: Path,
        repo_root: Path,
        sweep_id: str | None,
        sweeps_root: Path | None,
        materialize_processes: int | None,
        materialize_worker_threads: int | None,
    ) -> None:
        captured.update(
            {
                "recipe_id": recipe_id,
                "invocation_ids": [str(spec.invocation_id) for spec in invocations],
                "materialize_processes": materialize_processes,
                "materialize_worker_threads": materialize_worker_threads,
            }
        )
        for spec in invocations:
            corpus_materialization_module.materialize_recipe_invocation(
                recipe_id=recipe_id,
                invocation_id=str(spec.invocation_id),
                dagzoo_root=dagzoo_root,
                corpus_root=corpus_root,
                repo_root=repo_root,
                sweep_id=sweep_id,
                sweeps_root=sweeps_root,
            )

    monkeypatch.setattr(
        corpus_materialization_invocation_module,
        "_materialize_invocations_with_subprocess_fanout",
        _fake_subprocess_fanout,
    )

    record = materialize_corpus_recipe(
        recipe_id="size_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=True,
        materialize_processes=3,
        materialize_worker_threads=2,
        repo_root=repo_tmp_path,
    )

    assert captured == {
        "recipe_id": "size_recipe",
        "invocation_ids": ["benchmark_cpu", "large_shape"],
        "materialize_processes": 3,
        "materialize_worker_threads": 2,
    }
    assert Path(str(record["manifest"]["manifest_path"])).exists()


def test_materialize_corpus_recipe_aborts_without_manifest_when_subprocess_fanout_fails(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    def _failing_subprocess_fanout(**_kwargs: Any) -> None:
        raise RuntimeError("fanout failed")

    monkeypatch.setattr(
        corpus_materialization_invocation_module,
        "_materialize_invocations_with_subprocess_fanout",
        _failing_subprocess_fanout,
    )

    with pytest.raises(RuntimeError, match="fanout failed"):
        _ = materialize_corpus_recipe(
            recipe_id="size_recipe",
            dagzoo_root=repo_tmp_path.parent / "dagzoo",
            force=True,
            materialize_processes=3,
            repo_root=repo_tmp_path,
        )

    recipe_root = repo_tmp_path / "outputs" / "corpora" / "size_recipe"
    assert not any(recipe_root.glob("*/manifest.parquet"))


def test_materialize_corpus_refs_batch_delegates_pending_recipes_to_recipe_worker_fanout(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _write_recipe_registry(repo_tmp_path)
    captured: dict[str, Any] = {}
    callback_order: list[str] = []

    def _record_for(recipe_id: str) -> dict[str, Any]:
        return {
            "recipe_id": recipe_id,
            "corpus_ref": f"{recipe_id}/{recipe_id}__123456789abc",
            "manifest": {
                "manifest_path": str((repo_tmp_path / f"{recipe_id}.parquet").resolve())
            },
        }

    def _fake_recipe_worker_fanout(
        *,
        pending_requests,
        materialize_processes: int | None,
        materialize_worker_threads: int | None,
        prioritized_recipe_ids,
        on_recipe_materialized=None,
    ) -> list[dict[str, Any]]:
        captured.update(
            {
                "pending_recipe_ids": [pending.recipe_id for pending in pending_requests],
                "requested_exact_refs": [pending.requested_exact_ref for pending in pending_requests],
                "requires_recipe_record": [
                    pending.requires_recipe_record for pending in pending_requests
                ],
                "materialize_processes": materialize_processes,
                "materialize_worker_threads": materialize_worker_threads,
                "prioritized_recipe_ids": list(prioritized_recipe_ids),
            }
        )
        assert on_recipe_materialized is not None
        completed = [_record_for("current_recipe"), _record_for("size_recipe")]
        for record in completed:
            on_recipe_materialized(record)
        return completed

    monkeypatch.setattr(
        corpus_materialization_batch_module,
        "_materialize_pending_recipes_with_subprocess_fanout",
        _fake_recipe_worker_fanout,
    )

    records = corpus_materialization_module.materialize_corpus_refs_batch(
        corpus_refs=["size_recipe", "current_recipe"],
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=True,
        materialize_processes=3,
        materialize_worker_threads=2,
        prioritized_recipe_ids=["current_recipe"],
        on_corpus_materialized=lambda record: callback_order.append(str(record["recipe_id"])),
        repo_root=repo_tmp_path,
    )

    assert captured == {
        "pending_recipe_ids": ["size_recipe", "current_recipe"],
        "requested_exact_refs": [None, None],
        "requires_recipe_record": [True, True],
        "materialize_processes": 3,
        "materialize_worker_threads": 2,
        "prioritized_recipe_ids": ["current_recipe"],
    }
    assert callback_order == ["current_recipe", "size_recipe"]
    assert [str(record["recipe_id"]) for record in records] == ["size_recipe", "current_recipe"]


def test_materialize_corpus_refs_batch_reuses_cached_exact_ref_without_recipe_worker(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _write_recipe_registry(repo_tmp_path)
    _patch_dagzoo_generate(monkeypatch, _fake_run_dagzoo_generate)
    record = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=True,
        repo_root=repo_tmp_path,
    )

    def _unexpected_recipe_worker_fanout(**_kwargs: Any) -> list[dict[str, Any]]:
        raise AssertionError("cached exact refs should not spawn recipe workers")

    monkeypatch.setattr(
        corpus_materialization_batch_module,
        "_materialize_pending_recipes_with_subprocess_fanout",
        _unexpected_recipe_worker_fanout,
    )

    records = corpus_materialization_module.materialize_corpus_refs_batch(
        corpus_refs=[str(record["corpus_ref"])],
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=False,
        repo_root=repo_tmp_path,
    )

    assert [str(item["corpus_ref"]) for item in records] == [str(record["corpus_ref"])]


def test_materialize_corpus_refs_batch_rejects_conflicting_uncached_exact_refs(
    repo_tmp_path: Path,
) -> None:
    _write_recipe_registry(repo_tmp_path)

    with pytest.raises(RuntimeError, match="multiple pinned corpus ids for recipe 'current_recipe'"):
        _ = corpus_materialization_module.materialize_corpus_refs_batch(
            corpus_refs=[
                "current_recipe/current_recipe__deadbeefdead",
                "current_recipe/current_recipe__feedfacefeed",
            ],
            dagzoo_root=repo_tmp_path.parent / "dagzoo",
            repo_root=repo_tmp_path,
        )


def test_materialize_corpus_refs_batch_rejects_mismatched_pinned_exact_ref_from_recipe_worker(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _write_recipe_registry(repo_tmp_path)

    def _fake_recipe_worker_fanout(
        *,
        pending_requests,
        materialize_processes: int | None,
        materialize_worker_threads: int | None,
        prioritized_recipe_ids,
        on_recipe_materialized=None,
    ) -> list[dict[str, Any]]:
        del pending_requests, materialize_processes, materialize_worker_threads, prioritized_recipe_ids
        assert on_recipe_materialized is not None
        on_recipe_materialized(
            {
                "recipe_id": "current_recipe",
                "corpus_ref": "current_recipe/current_recipe__123456789abc",
                "manifest": {
                    "manifest_path": str((repo_tmp_path / "current_recipe.parquet").resolve())
                },
            }
        )
        return []

    monkeypatch.setattr(
        corpus_materialization_batch_module,
        "_materialize_pending_recipes_with_subprocess_fanout",
        _fake_recipe_worker_fanout,
    )

    with pytest.raises(RuntimeError, match="pinned to an exact corpus id"):
        _ = corpus_materialization_module.materialize_corpus_refs_batch(
            corpus_refs=["current_recipe/current_recipe__deadbeefdead"],
            dagzoo_root=repo_tmp_path.parent / "dagzoo",
            force=False,
            repo_root=repo_tmp_path,
        )


def test_materialize_corpus_refs_batch_aborts_without_manifest_when_recipe_worker_fanout_fails(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _write_recipe_registry(repo_tmp_path)

    def _failing_recipe_worker_fanout(**_kwargs: Any) -> list[dict[str, Any]]:
        raise RuntimeError("recipe worker failed")

    monkeypatch.setattr(
        corpus_materialization_batch_module,
        "_materialize_pending_recipes_with_subprocess_fanout",
        _failing_recipe_worker_fanout,
    )

    with pytest.raises(RuntimeError, match="recipe worker failed"):
        _ = corpus_materialization_module.materialize_corpus_refs_batch(
            corpus_refs=["current_recipe", "size_recipe"],
            dagzoo_root=repo_tmp_path.parent / "dagzoo",
            force=True,
            repo_root=repo_tmp_path,
        )

    for recipe_id in ("current_recipe", "size_recipe"):
        recipe_root = repo_tmp_path / "outputs" / "corpora" / recipe_id
        assert not any(recipe_root.glob("*/manifest.parquet"))


def test_materialize_pending_recipes_with_subprocess_fanout_prioritizes_launch_and_splits_process_budget(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    launched_recipe_ids: list[str] = []
    launched_process_allocations: list[int] = []
    launched_worker_threads: list[int | None] = []
    launched_result_paths: list[Path] = []
    callback_order: list[str] = []
    completion_order = ["size_recipe", "current_recipe", "adequacy_recipe"]
    completed_indices = [0]
    active_processes: dict[str, Any] = {}

    class FakePopen:
        next_pid = 1000

        def __init__(
            self,
            argv: list[str],
            *,
            cwd: Path,
            text: bool,
        ) -> None:
            del cwd, text
            self.args = argv
            self.pid = FakePopen.next_pid
            FakePopen.next_pid += 1
            recipe_id = argv[argv.index("--recipe-id") + 1]
            launched_recipe_ids.append(recipe_id)
            launched_process_allocations.append(
                int(argv[argv.index("--materialize-processes") + 1])
            )
            launched_worker_threads.append(
                (
                    None
                    if "--materialize-worker-threads" not in argv
                    else int(argv[argv.index("--materialize-worker-threads") + 1])
                )
            )
            self.result_path = Path(argv[argv.index("--result-path") + 1])
            launched_result_paths.append(self.result_path)
            self.recipe_id = recipe_id
            self.returncode: int | None = None
            self.result_payload = {
                "recipe_id": recipe_id,
                "corpus_ref": f"{recipe_id}/{recipe_id}__123456789abc",
                "manifest": {
                    "manifest_path": str((repo_tmp_path / f"{recipe_id}.parquet").resolve())
                },
            }
            active_processes[recipe_id] = self

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            if self.returncode is None:
                self.returncode = -15

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    def _fake_sleep(_seconds: float) -> None:
        if completed_indices[0] >= len(completion_order):
            raise AssertionError("scheduler slept after all fake completions were consumed")
        recipe_id = completion_order[completed_indices[0]]
        completed_indices[0] += 1
        active_processes[recipe_id].result_path.write_text(
            json.dumps(active_processes[recipe_id].result_payload),
            encoding="utf-8",
        )
        active_processes[recipe_id].returncode = 0

    monkeypatch.setattr(corpus_materialization_batch_module.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(corpus_materialization_batch_module.time, "sleep", _fake_sleep)

    records = corpus_materialization_batch_module._materialize_pending_recipes_with_subprocess_fanout(
        pending_requests=[
            corpus_materialization_batch_module._PendingRecipeWorkerMaterialization(
                recipe_id="adequacy_recipe",
                dagzoo_root=repo_tmp_path.parent / "dagzoo",
                force=True,
                repo_root=repo_tmp_path,
                requested_exact_ref=None,
                requires_recipe_record=True,
                sweep_id=None,
                sweeps_root=None,
            ),
            corpus_materialization_batch_module._PendingRecipeWorkerMaterialization(
                recipe_id="current_recipe",
                dagzoo_root=repo_tmp_path.parent / "dagzoo",
                force=True,
                repo_root=repo_tmp_path,
                requested_exact_ref=None,
                requires_recipe_record=True,
                sweep_id=None,
                sweeps_root=None,
            ),
            corpus_materialization_batch_module._PendingRecipeWorkerMaterialization(
                recipe_id="size_recipe",
                dagzoo_root=repo_tmp_path.parent / "dagzoo",
                force=True,
                repo_root=repo_tmp_path,
                requested_exact_ref=None,
                requires_recipe_record=True,
                sweep_id=None,
                sweeps_root=None,
            ),
        ],
        materialize_processes=4,
        materialize_worker_threads=None,
        prioritized_recipe_ids=["current_recipe"],
        on_recipe_materialized=lambda record: callback_order.append(str(record["recipe_id"])),
    )

    assert launched_recipe_ids == ["current_recipe", "adequacy_recipe", "size_recipe"]
    assert launched_process_allocations == [2, 1, 1]
    assert launched_worker_threads == [None, None, None]
    assert callback_order == completion_order
    assert [str(record["recipe_id"]) for record in records] == completion_order
    assert all(not path.exists() for path in launched_result_paths)


def test_materialize_pending_recipes_with_subprocess_fanout_forwards_explicit_worker_threads(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    launched_worker_threads: list[int | None] = []
    launched_result_paths: list[Path] = []

    class FakePopen:
        next_pid = 1500

        def __init__(
            self,
            argv: list[str],
            *,
            cwd: Path,
            text: bool,
        ) -> None:
            del cwd, text
            self.args = argv
            self.pid = FakePopen.next_pid
            FakePopen.next_pid += 1
            launched_worker_threads.append(
                (
                    None
                    if "--materialize-worker-threads" not in argv
                    else int(argv[argv.index("--materialize-worker-threads") + 1])
                )
            )
            result_path = Path(argv[argv.index("--result-path") + 1])
            launched_result_paths.append(result_path)
            result_path.write_text(
                json.dumps(
                    {
                        "recipe_id": "current_recipe",
                        "corpus_ref": "current_recipe/current_recipe__123456789abc",
                        "manifest": {
                            "manifest_path": str(
                                (repo_tmp_path / "current_recipe.parquet").resolve()
                            )
                        },
                    }
                ),
                encoding="utf-8",
            )
            self.returncode = 0

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.returncode = -15

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    monkeypatch.setattr(corpus_materialization_batch_module.subprocess, "Popen", FakePopen)

    records = corpus_materialization_batch_module._materialize_pending_recipes_with_subprocess_fanout(
        pending_requests=[
            corpus_materialization_batch_module._PendingRecipeWorkerMaterialization(
                recipe_id="current_recipe",
                dagzoo_root=repo_tmp_path.parent / "dagzoo",
                force=True,
                repo_root=repo_tmp_path,
                requested_exact_ref=None,
                requires_recipe_record=True,
                sweep_id=None,
                sweeps_root=None,
            ),
        ],
        materialize_processes=2,
        materialize_worker_threads=3,
        prioritized_recipe_ids=(),
    )

    assert launched_worker_threads == [3]
    assert [str(record["recipe_id"]) for record in records] == ["current_recipe"]
    assert all(not path.exists() for path in launched_result_paths)


def test_materialize_pending_recipes_with_subprocess_fanout_terminates_remaining_workers_on_failure(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    process_by_recipe_id: dict[str, Any] = {}
    launched_result_paths: list[Path] = []
    completion_order = ["current_recipe"]
    completed_indices = [0]

    class FakePopen:
        next_pid = 2000

        def __init__(
            self,
            argv: list[str],
            *,
            cwd: Path,
            text: bool,
        ) -> None:
            del cwd, text
            self.args = argv
            self.pid = FakePopen.next_pid
            FakePopen.next_pid += 1
            self.recipe_id = argv[argv.index("--recipe-id") + 1]
            self.result_path = Path(argv[argv.index("--result-path") + 1])
            launched_result_paths.append(self.result_path)
            self.returncode: int | None = None
            self.terminated = False
            process_by_recipe_id[self.recipe_id] = self

        def poll(self) -> int | None:
            return self.returncode

        def terminate(self) -> None:
            self.terminated = True
            if self.returncode is None:
                self.returncode = -15

        def wait(self, timeout: float | None = None) -> int:
            del timeout
            if self.returncode is None:
                self.returncode = 0
            return self.returncode

        def kill(self) -> None:
            self.returncode = -9

    def _fake_sleep(_seconds: float) -> None:
        if completed_indices[0] >= len(completion_order):
            raise AssertionError("scheduler slept after the failure was already injected")
        recipe_id = completion_order[completed_indices[0]]
        completed_indices[0] += 1
        process_by_recipe_id[recipe_id].returncode = 1

    monkeypatch.setattr(corpus_materialization_batch_module.subprocess, "Popen", FakePopen)
    monkeypatch.setattr(corpus_materialization_batch_module.time, "sleep", _fake_sleep)

    with pytest.raises(RuntimeError, match="recipe materialization subprocess failed"):
        _ = corpus_materialization_batch_module._materialize_pending_recipes_with_subprocess_fanout(
            pending_requests=[
                corpus_materialization_batch_module._PendingRecipeWorkerMaterialization(
                    recipe_id="current_recipe",
                    dagzoo_root=repo_tmp_path.parent / "dagzoo",
                    force=True,
                    repo_root=repo_tmp_path,
                    requested_exact_ref=None,
                    requires_recipe_record=True,
                    sweep_id=None,
                    sweeps_root=None,
                ),
                corpus_materialization_batch_module._PendingRecipeWorkerMaterialization(
                    recipe_id="size_recipe",
                    dagzoo_root=repo_tmp_path.parent / "dagzoo",
                    force=True,
                    repo_root=repo_tmp_path,
                    requested_exact_ref=None,
                    requires_recipe_record=True,
                    sweep_id=None,
                    sweeps_root=None,
                ),
            ],
            materialize_processes=2,
            materialize_worker_threads=3,
            prioritized_recipe_ids=(),
        )

    assert process_by_recipe_id["size_recipe"].terminated is True
    assert all(not path.exists() for path in launched_result_paths)


def test_recipe_worker_run_from_args_writes_record_and_preserves_optional_threads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured_threads: list[int | None] = []

    def _fake_materialize_corpus_recipe(
        *,
        recipe_id: str,
        dagzoo_root: Path,
        force: bool = False,
        materialize_processes: int | None = None,
        materialize_worker_threads: int | None = None,
        repo_root: Path | None = None,
        sweep_id: str | None = None,
        sweeps_root: Path | None = None,
    ) -> dict[str, Any]:
        del dagzoo_root, force, materialize_processes, repo_root, sweep_id, sweeps_root
        captured_threads.append(materialize_worker_threads)
        return {
            "recipe_id": recipe_id,
            "corpus_ref": f"{recipe_id}/{recipe_id}__123456789abc",
            "manifest": {
                "manifest_path": str((tmp_path / f"{recipe_id}.parquet").resolve())
            },
        }

    monkeypatch.setattr(
        recipe_worker_module,
        "materialize_corpus_recipe",
        _fake_materialize_corpus_recipe,
    )

    first_result_path = tmp_path / "worker-result-1.json"
    first_exit_code = recipe_worker_module.run_from_args(
        argparse.Namespace(
            recipe_id="current_recipe",
            dagzoo_root=str(tmp_path / "dagzoo"),
            repo_root=str(tmp_path / "repo"),
            result_path=str(first_result_path),
            force=False,
            materialize_processes=2,
            materialize_worker_threads=None,
            sweep_id=None,
            sweeps_root=None,
        )
    )
    second_result_path = tmp_path / "worker-result-2.json"
    second_exit_code = recipe_worker_module.run_from_args(
        argparse.Namespace(
            recipe_id="size_recipe",
            dagzoo_root=str(tmp_path / "dagzoo"),
            repo_root=str(tmp_path / "repo"),
            result_path=str(second_result_path),
            force=False,
            materialize_processes=2,
            materialize_worker_threads=5,
            sweep_id=None,
            sweeps_root=None,
        )
    )

    assert first_exit_code == 0
    assert second_exit_code == 0
    assert captured_threads == [None, 5]
    assert (
        json.loads(first_result_path.read_text(encoding="utf-8"))["recipe_id"]
        == "current_recipe"
    )
    assert json.loads(second_result_path.read_text(encoding="utf-8"))["recipe_id"] == "size_recipe"


def test_materialize_corpus_recipe_backfills_adequacy_metadata_from_recipe_when_handoff_omits_provenance(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _write_adequacy_recipe_fixture(repo_tmp_path)
    _patch_dagzoo_generate(monkeypatch, _fake_run_dagzoo_generate)

    record = materialize_corpus_recipe(
        recipe_id="adequacy_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=True,
        repo_root=repo_tmp_path,
    )

    summary = record["dagzoo_provenance_summary"]
    assert summary["target_derivation"] == "tabiclv2_latent_node"
    assert summary.get("target_relevant_feature_count_range") is None
    assert summary.get("target_relevant_feature_fraction_range") is None

    dagzoo_provenance = record["dagzoo_provenance"]
    assert dagzoo_provenance["target_derivation"] == "tabiclv2_latent_node"
    assert "target_relevant_feature_count_range" not in dagzoo_provenance
    assert "target_relevant_feature_fraction_range" not in dagzoo_provenance


def test_materialize_corpus_recipe_runs_generate_then_filter_for_accepted_only(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _write_accepted_only_recipe_fixture(repo_tmp_path, num_datasets=1)
    _patch_dagzoo_generate(monkeypatch, _fake_run_dagzoo_generate)
    _patch_dagzoo_filter(monkeypatch, _fake_run_dagzoo_filter)

    record = materialize_corpus_recipe(
        recipe_id="accepted_only_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=True,
        materialize_worker_threads=2,
        repo_root=repo_tmp_path,
    )

    dagzoo_provenance = record["dagzoo_provenance"]
    summary = record["dagzoo_provenance_summary"]
    assert dagzoo_provenance["filter_policy"] == "accepted_only"
    assert dagzoo_provenance["accepted_datasets"] == 1
    assert dagzoo_provenance["curated_accepted_datasets"] == 1
    assert len(dagzoo_provenance["filter_manifest_paths"]) == 1
    assert len(dagzoo_provenance["filter_summary_paths"]) == 1
    assert len(dagzoo_provenance["curated_root_lineage"]) == 1
    assert summary["filter_policy"] == "accepted_only"
    assert summary["accepted_datasets"] == 1
    assert summary["curated_accepted_datasets"] == 1
    invocation = dagzoo_provenance["invocations"][0]
    materialization_summary_path = (
        Path(str(invocation["invocation_root"])) / "materialization_summary.json"
    )
    materialization_summary = json.loads(
        materialization_summary_path.read_text(encoding="utf-8")
    )
    assert materialization_summary["filter_policy"] == "accepted_only"
    assert materialization_summary["generated_datasets"] == 1
    assert materialization_summary["round_count"] == 1
    assert materialization_summary["materialize_worker_threads"] == 2
    assert materialization_summary["generate_elapsed_seconds"] >= 0.0
    assert materialization_summary["filter_elapsed_seconds"] == pytest.approx(0.1)
    assert materialization_summary["copy_elapsed_seconds"] >= 0.0
    assert materialization_summary["upstream_elapsed_seconds"] == pytest.approx(
        materialization_summary["generate_elapsed_seconds"] + 0.1
    )
    assert materialization_summary["local_overhead_elapsed_seconds"] >= 0.0
    assert materialization_summary["invocation_elapsed_seconds"] >= 0.0
    assert materialization_summary["rounds"][0]["filter_elapsed_seconds"] == pytest.approx(0.1)
    assert materialization_summary["rounds"][0]["filter_datasets_per_minute"] == pytest.approx(
        600.0
    )

    invocation_timing = invocation["materialization_timing"]
    assert invocation_timing["generated_datasets"] == 1
    assert invocation_timing["round_count"] == 1
    assert invocation_timing["materialize_worker_threads"] == 2
    assert invocation_timing["generate_elapsed_seconds"] == pytest.approx(
        materialization_summary["generate_elapsed_seconds"]
    )
    assert invocation_timing["filter_elapsed_seconds"] == pytest.approx(0.1)
    assert invocation_timing["copy_elapsed_seconds"] == pytest.approx(
        materialization_summary["copy_elapsed_seconds"]
    )
    assert invocation_timing["upstream_elapsed_seconds"] == pytest.approx(
        materialization_summary["upstream_elapsed_seconds"]
    )
    assert invocation_timing["local_overhead_elapsed_seconds"] == pytest.approx(
        materialization_summary["local_overhead_elapsed_seconds"]
    )
    assert invocation_timing["invocation_elapsed_seconds"] == pytest.approx(
        materialization_summary["invocation_elapsed_seconds"]
    )

    round_payload = invocation["rounds"][0]
    assert round_payload["filter_curated_accepted_datasets"] == 1
    round_timing = round_payload["materialization_timing"]
    assert round_timing["generate_elapsed_seconds"] >= 0.0
    assert round_timing["filter_elapsed_seconds"] == pytest.approx(0.1)
    assert round_timing["filter_datasets_per_minute"] == pytest.approx(600.0)
    assert round_timing["copy_elapsed_seconds"] >= 0.0
    assert round_timing["upstream_elapsed_seconds"] == pytest.approx(
        round_timing["generate_elapsed_seconds"] + 0.1
    )
    assert round_timing["local_overhead_elapsed_seconds"] >= 0.0
    assert round_timing["round_elapsed_seconds"] >= 0.0

    timing_summary = summary["materialization_timing"]
    assert dagzoo_provenance["materialization_timing"] == timing_summary
    assert timing_summary["timed_invocation_count"] == 1
    assert timing_summary["cumulative_round_count"] == 1
    assert timing_summary["cumulative_generated_datasets"] == 1
    assert timing_summary["cumulative_generate_elapsed_seconds"] == pytest.approx(
        invocation_timing["generate_elapsed_seconds"]
    )
    assert timing_summary["cumulative_filter_elapsed_seconds"] == pytest.approx(0.1)
    assert timing_summary["cumulative_copy_elapsed_seconds"] == pytest.approx(
        invocation_timing["copy_elapsed_seconds"]
    )
    assert timing_summary["cumulative_upstream_elapsed_seconds"] == pytest.approx(
        invocation_timing["upstream_elapsed_seconds"]
    )
    assert timing_summary["cumulative_local_overhead_elapsed_seconds"] == pytest.approx(
        invocation_timing["local_overhead_elapsed_seconds"]
    )
    assert timing_summary["cumulative_invocation_elapsed_seconds"] == pytest.approx(
        invocation_timing["invocation_elapsed_seconds"]
    )
    assert "filter.n_jobs" not in invocation["rounds"][0]["filter_command"]
    assert Path(str(record["manifest"]["manifest_path"])).exists()


def test_finalize_staged_corpus_recipe_promotes_existing_stage_with_fast_verification(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _write_accepted_only_recipe_fixture(repo_tmp_path, num_datasets=1)
    _patch_dagzoo_generate(monkeypatch, _fake_run_dagzoo_generate)
    _patch_dagzoo_filter(monkeypatch, _fake_run_dagzoo_filter)

    stage_root = (
        repo_tmp_path / "outputs" / "corpora" / "accepted_only_recipe" / ".staging"
    )
    stage_root.mkdir(parents=True, exist_ok=True)
    corpus_materialization_module.materialize_recipe_invocation(
        recipe_id="accepted_only_recipe",
        invocation_id="default",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        corpus_root=stage_root,
        repo_root=repo_tmp_path,
    )

    result = finalize_staged_corpus_recipe(
        recipe_id="accepted_only_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        verify="fast",
        repo_root=repo_tmp_path,
    )

    record = result["record"]
    verification = result["verification"]
    assert verification["mode"] == "fast"
    assert verification["verified_invocations"] == 1
    assert verification["accepted_only"]["target_accepted_datasets"] == 1
    assert verification["accepted_only"]["curated_accepted_datasets"] == 1
    manifest_path = Path(str(record["manifest"]["manifest_path"]))
    assert manifest_path.exists()
    assert stage_root.exists()
    final_root = Path(str(record["artifacts"]["corpus_root"]))
    latest_pointer_path = Path(str(record["artifacts"]["latest_pointer_path"]))
    assert final_root.exists()
    assert latest_pointer_path.exists()
    assert ".staging" not in str(final_root)

    manifest_row = pq.read_table(manifest_path).to_pylist()[0]
    for path_key in ("metadata_path", "catalog_path", "train_path", "test_path"):
        raw_path = manifest_row.get(path_key)
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        resolved_path = Path(raw_path)
        if not resolved_path.is_absolute():
            resolved_path = (manifest_path.parent / resolved_path).resolve()
        else:
            resolved_path = resolved_path.resolve()
        assert str(resolved_path).startswith(str(final_root.resolve()))
        assert ".staging" not in str(resolved_path)

    loaded = load_corpus_record("accepted_only_recipe", repo_root=repo_tmp_path)
    assert loaded["corpus_ref"] == record["corpus_ref"]


def test_compact_staged_corpus_recipe_rewrites_legacy_curated_root_and_finalize_builds_manifest(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _write_accepted_only_recipe_fixture(repo_tmp_path, num_datasets=3)
    _patch_dagzoo_generate(monkeypatch, _fake_run_dagzoo_generate_many)
    _patch_dagzoo_filter(monkeypatch, _fake_run_dagzoo_filter_all)
    real_compact_curated_root = corpus_materialization_invocation_module.compact_curated_root

    def _compact_to_two(
        *,
        source_curated_dir: Path,
        output_curated_dir: Path,
        start_shard_index: int = 0,
        max_datasets: int | None = None,
        target_datasets_per_shard: int = 512,
    ) -> dict[str, Any]:
        del target_datasets_per_shard
        return real_compact_curated_root(
            source_curated_dir=source_curated_dir,
            output_curated_dir=output_curated_dir,
            start_shard_index=start_shard_index,
            target_datasets_per_shard=2,
            max_datasets=max_datasets,
        )

    monkeypatch.setattr(
        corpus_materialization_invocation_module,
        "compact_curated_root",
        _compact_to_two,
    )

    stage_root = (
        repo_tmp_path / "outputs" / "corpora" / "accepted_only_recipe" / ".staging"
    )
    stage_root.mkdir(parents=True, exist_ok=True)
    corpus_materialization_module.materialize_recipe_invocation(
        recipe_id="accepted_only_recipe",
        invocation_id="default",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        corpus_root=stage_root,
        repo_root=repo_tmp_path,
    )
    curated_root = corpus_materialization_invocation_module._invocation_curated_root(
        corpus_root=stage_root,
        invocation_id="default",
    )
    _rewrite_curated_root_as_legacy_shards(curated_root, dataset_count=3)

    compacted = corpus_materialization_module.compact_staged_corpus_recipe(
        recipe_id="accepted_only_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        repo_root=repo_tmp_path,
    )

    assert compacted["recipe_id"] == "accepted_only_recipe"
    assert compacted["invocations"][0]["curated_compaction"] == {
        "target_datasets_per_shard": 2,
        "source_shard_count": 3,
        "output_shard_count": 2,
        "dataset_count": 3,
    }
    shard_dirs = sorted(curated_root.glob("shard_*"))
    assert [path.name for path in shard_dirs] == ["shard_00000", "shard_00001"]
    assert all((path / "dataset_catalog.parquet").exists() for path in shard_dirs)
    assert not list(curated_root.rglob("metadata.ndjson"))

    summary_path = corpus_materialization_invocation_module._invocation_materialization_summary_path(
        corpus_root=stage_root,
        invocation_id="default",
    )
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["curated_compaction"] == {
        "target_datasets_per_shard": 2,
        "source_shard_count": 3,
        "output_shard_count": 2,
        "dataset_count": 3,
    }

    result = finalize_staged_corpus_recipe(
        recipe_id="accepted_only_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        verify="fast",
        repo_root=repo_tmp_path,
        manifest_workers=4,
    )

    assert result["verification"]["mode"] == "fast"
    assert result["verification"]["accepted_only"]["target_accepted_datasets"] == 3
    assert result["verification"]["accepted_only"]["curated_accepted_datasets"] == 3
    assert result["verification"]["accepted_only"]["accepted_datasets"] >= 3
    assert result["record"]["manifest"]["inspection"]["persisted_summary"]["total_records"] == 3


def test_compact_staged_corpus_recipe_refuses_already_compacted_stage_without_force(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _write_accepted_only_recipe_fixture(repo_tmp_path, num_datasets=1)
    _patch_dagzoo_generate(monkeypatch, _fake_run_dagzoo_generate)
    _patch_dagzoo_filter(monkeypatch, _fake_run_dagzoo_filter)

    stage_root = (
        repo_tmp_path / "outputs" / "corpora" / "accepted_only_recipe" / ".staging"
    )
    stage_root.mkdir(parents=True, exist_ok=True)
    corpus_materialization_module.materialize_recipe_invocation(
        recipe_id="accepted_only_recipe",
        invocation_id="default",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        corpus_root=stage_root,
        repo_root=repo_tmp_path,
    )

    with pytest.raises(RuntimeError, match="already contains parquet catalogs"):
        corpus_materialization_module.compact_staged_corpus_recipe(
            recipe_id="accepted_only_recipe",
            dagzoo_root=repo_tmp_path.parent / "dagzoo",
            repo_root=repo_tmp_path,
        )

    forced = corpus_materialization_module.compact_staged_corpus_recipe(
        recipe_id="accepted_only_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        repo_root=repo_tmp_path,
        force=True,
    )

    assert forced["invocations"][0]["curated_compaction"]["dataset_count"] == 1


def test_compact_staged_corpus_recipe_parallelizes_invocations_but_preserves_result_order(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _write_multi_invocation_accepted_only_recipe_fixture(repo_tmp_path)
    _patch_dagzoo_generate(monkeypatch, _fake_run_dagzoo_generate_many)
    _patch_dagzoo_filter(monkeypatch, _fake_run_dagzoo_filter_all)
    real_compact_curated_root = corpus_materialization_invocation_module.compact_curated_root

    def _sleepy_compact(
        *,
        source_curated_dir: Path,
        output_curated_dir: Path,
        start_shard_index: int = 0,
        target_datasets_per_shard: int = 512,
        max_datasets: int | None = None,
    ) -> dict[str, Any]:
        if "slow" in str(source_curated_dir):
            time.sleep(0.1)
        return real_compact_curated_root(
            source_curated_dir=source_curated_dir,
            output_curated_dir=output_curated_dir,
            start_shard_index=start_shard_index,
            target_datasets_per_shard=target_datasets_per_shard,
            max_datasets=max_datasets,
        )

    monkeypatch.setattr(
        corpus_materialization_invocation_module,
        "compact_curated_root",
        _sleepy_compact,
    )

    stage_root = (
        repo_tmp_path / "outputs" / "corpora" / "accepted_only_multi_recipe" / ".staging"
    )
    stage_root.mkdir(parents=True, exist_ok=True)
    for invocation_id in ("slow", "fast"):
        corpus_materialization_module.materialize_recipe_invocation(
            recipe_id="accepted_only_multi_recipe",
            invocation_id=invocation_id,
            dagzoo_root=repo_tmp_path.parent / "dagzoo",
            corpus_root=stage_root,
            repo_root=repo_tmp_path,
        )
        curated_root = corpus_materialization_invocation_module._invocation_curated_root(
            corpus_root=stage_root,
            invocation_id=invocation_id,
        )
        _rewrite_curated_root_as_legacy_shards(curated_root, dataset_count=2)

    progress_invocation_ids: list[str] = []

    compacted = corpus_materialization_module.compact_staged_corpus_recipe(
        recipe_id="accepted_only_multi_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        repo_root=repo_tmp_path,
        compact_workers=2,
        progress_callback=lambda payload: progress_invocation_ids.append(
            str(payload["invocation_id"])
        ),
    )

    assert progress_invocation_ids == ["fast", "slow"]
    assert [item["invocation_id"] for item in compacted["invocations"]] == ["slow", "fast"]
    assert compacted["invocations"][0]["curated_compaction"]["dataset_count"] == 2
    assert compacted["invocations"][1]["curated_compaction"]["dataset_count"] == 2


def test_load_staged_corpus_recipe_preview_returns_stage_metadata(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _write_accepted_only_recipe_fixture(repo_tmp_path, num_datasets=1)
    _patch_dagzoo_generate(monkeypatch, _fake_run_dagzoo_generate)
    _patch_dagzoo_filter(monkeypatch, _fake_run_dagzoo_filter)

    stage_root = (
        repo_tmp_path / "outputs" / "corpora" / "accepted_only_recipe" / ".staging"
    )
    stage_root.mkdir(parents=True, exist_ok=True)
    corpus_materialization_module.materialize_recipe_invocation(
        recipe_id="accepted_only_recipe",
        invocation_id="default",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        corpus_root=stage_root,
        repo_root=repo_tmp_path,
    )

    preview = load_staged_corpus_recipe_preview(
        recipe_id="accepted_only_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        repo_root=repo_tmp_path,
    )

    assert preview["recipe_id"] == "accepted_only_recipe"
    assert preview["surface_label"] == "accepted_only_surface"
    assert preview["stage_root"] == str(stage_root.resolve())
    invocation = preview["invocations"][0]
    assert invocation["invocation_id"] == "default"
    assert invocation["filter"]["target_accepted_datasets"] == 1
    assert invocation["filter"]["curated_accepted_datasets"] == 1
    assert invocation["rounds"][0]["round_index"] == 1


def test_load_staged_corpus_recipe_preview_errors_when_stage_missing(
    repo_tmp_path: Path,
) -> None:
    _write_accepted_only_recipe_fixture(repo_tmp_path, num_datasets=1)

    with pytest.raises(RuntimeError, match="staged corpus root does not exist"):
        _ = load_staged_corpus_recipe_preview(
            recipe_id="accepted_only_recipe",
            dagzoo_root=repo_tmp_path.parent / "dagzoo",
            repo_root=repo_tmp_path,
        )


def test_materialize_corpus_recipe_tops_up_accepted_only_until_target(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _write_accepted_only_recipe_fixture(
        repo_tmp_path,
        recipe_id="accepted_only_topup_recipe",
        filename="accepted_only_topup_recipe.yaml",
        num_datasets=3,
    )
    generate_calls = [0]
    _patch_dagzoo_generate(monkeypatch, _counting_fake_run_dagzoo_generate(generate_calls))
    _patch_dagzoo_filter(
        monkeypatch,
        _round_sequence_fake_run_dagzoo_filter([1, 2], total_datasets_per_round=2),
    )

    record = materialize_corpus_recipe(
        recipe_id="accepted_only_topup_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=True,
        repo_root=repo_tmp_path,
    )

    assert generate_calls[0] == 2
    dagzoo_provenance = record["dagzoo_provenance"]
    invocation = dagzoo_provenance["invocations"][0]
    assert invocation["filter"]["target_accepted_datasets"] == 3
    assert invocation["filter"]["curated_accepted_datasets"] == 3
    assert invocation["filter"]["round_count"] == 2
    assert len(invocation["rounds"]) == 2
    assert dagzoo_provenance["accepted_datasets"] == 3
    assert dagzoo_provenance["rejected_datasets"] == 1


def test_materialize_corpus_recipe_scales_accepted_only_topup_by_observed_acceptance_rate(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _write_accepted_only_recipe_fixture(
        repo_tmp_path,
        recipe_id="accepted_only_adaptive_topup_recipe",
        filename="accepted_only_adaptive_topup_recipe.yaml",
        num_datasets=20,
    )
    requested_num_datasets: list[int] = []
    curated_counts_by_round = [12, 7, 1]

    def _fake_generate(config) -> object:
        requested_num_datasets.append(int(config.num_datasets))
        return _fake_run_dagzoo_generate(config)

    def _long_topup_filter(config) -> DagzooFilterResult:
        round_index = len(requested_num_datasets) - 1
        curated_count = curated_counts_by_round[round_index]
        total_datasets = requested_num_datasets[round_index]
        filter_root = Path(str(config.filter_out_dir)).expanduser().resolve()
        curated_dir = Path(str(config.curated_out_dir)).expanduser().resolve()
        filter_root.mkdir(parents=True, exist_ok=True)
        curated_dir.mkdir(parents=True, exist_ok=True)
        _write_curated_datasets(
            curated_dir,
            dataset_count=curated_count,
            seed_base=500 + round_index * 10,
        )
        manifest_path = filter_root / "filter_manifest.ndjson"
        summary_path = filter_root / "filter_summary.json"
        manifest_path.write_text("{}\n" * max(1, curated_count), encoding="utf-8")
        summary_path.write_text(
            json.dumps(
                {
                    "total_datasets": total_datasets,
                    "accepted_datasets": curated_count,
                    "rejected_datasets": max(0, total_datasets - curated_count),
                    "curated_out_dir": str(curated_dir.resolve()),
                    "curated_accepted_datasets": curated_count,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return DagzooFilterResult(
            manifest_path=manifest_path.resolve(),
            summary_path=summary_path.resolve(),
            total_datasets=total_datasets,
            accepted_datasets=curated_count,
            rejected_datasets=max(0, total_datasets - curated_count),
            elapsed_seconds=0.1,
            datasets_per_minute=600.0,
            curated_out_dir=curated_dir.resolve(),
            curated_accepted_datasets=curated_count,
        )

    _patch_dagzoo_generate(monkeypatch, _fake_generate)
    _patch_dagzoo_filter(monkeypatch, _long_topup_filter)

    record = materialize_corpus_recipe(
        recipe_id="accepted_only_adaptive_topup_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=True,
        repo_root=repo_tmp_path,
    )

    invocation = record["dagzoo_provenance"]["invocations"][0]
    assert requested_num_datasets == [29, 20, 3]
    assert invocation["filter"]["round_count"] == 3
    assert invocation["filter"]["curated_accepted_datasets"] == 20
    assert record["dagzoo_provenance"]["accepted_datasets"] == 20


def test_materialize_corpus_recipe_trims_overaccepted_curated_output_to_exact_target(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _write_accepted_only_recipe_fixture(
        repo_tmp_path,
        recipe_id="accepted_only_overaccept_recipe",
        filename="accepted_only_overaccept_recipe.yaml",
        num_datasets=32,
    )
    requested_num_datasets: list[int] = []
    curated_counts_by_round = [23, 11]

    def _fake_generate(config) -> object:
        requested_num_datasets.append(int(config.num_datasets))
        return _fake_run_dagzoo_generate(config)

    def _overaccept_filter(config) -> DagzooFilterResult:
        round_index = len(requested_num_datasets) - 1
        curated_count = curated_counts_by_round[round_index]
        total_datasets = requested_num_datasets[round_index]
        filter_root = Path(str(config.filter_out_dir)).expanduser().resolve()
        curated_dir = Path(str(config.curated_out_dir)).expanduser().resolve()
        filter_root.mkdir(parents=True, exist_ok=True)
        curated_dir.mkdir(parents=True, exist_ok=True)
        _write_curated_datasets(
            curated_dir,
            dataset_count=curated_count,
            seed_base=700 + round_index * 100,
        )
        manifest_path = filter_root / "filter_manifest.ndjson"
        summary_path = filter_root / "filter_summary.json"
        manifest_path.write_text("{}\n" * max(1, curated_count), encoding="utf-8")
        summary_path.write_text(
            json.dumps(
                {
                    "total_datasets": total_datasets,
                    "accepted_datasets": curated_count,
                    "rejected_datasets": max(0, total_datasets - curated_count),
                    "curated_out_dir": str(curated_dir.resolve()),
                    "curated_accepted_datasets": curated_count,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return DagzooFilterResult(
            manifest_path=manifest_path.resolve(),
            summary_path=summary_path.resolve(),
            total_datasets=total_datasets,
            accepted_datasets=curated_count,
            rejected_datasets=max(0, total_datasets - curated_count),
            elapsed_seconds=0.1,
            datasets_per_minute=600.0,
            curated_out_dir=curated_dir.resolve(),
            curated_accepted_datasets=curated_count,
        )

    _patch_dagzoo_generate(monkeypatch, _fake_generate)
    _patch_dagzoo_filter(monkeypatch, _overaccept_filter)

    record = materialize_corpus_recipe(
        recipe_id="accepted_only_overaccept_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=True,
        repo_root=repo_tmp_path,
    )

    invocation = record["dagzoo_provenance"]["invocations"][0]
    assert requested_num_datasets == [46, 18]
    assert invocation["filter"]["curated_accepted_datasets"] == 32
    assert invocation["filter"]["accepted_datasets"] == 34

    manifest_table = pq.read_table(Path(str(record["manifest"]["manifest_path"])))
    assert manifest_table.num_rows == 32


def test_copy_curated_round_shards_trims_partial_shard_to_dataset_limit(
    tmp_path: Path,
) -> None:
    round_curated_dir = tmp_path / "round_curated"
    final_curated_dir = tmp_path / "final_curated"
    shard_dir = round_curated_dir / "shard_00000"
    shard_dir.mkdir(parents=True, exist_ok=True)
    datasets: list[dict[str, Any]] = []
    for dataset_index, seed in enumerate((901, 902), start=0):
        x_train, y_train, x_test, y_test = cases._classification_arrays(seed=seed)
        metadata = cases._classification_metadata(
            n_features=x_train.shape[1],
            seed=seed,
            filter_status="accepted",
            filter_accepted=True,
        )
        metadata["dataset_id"] = f"{seed:032x}"
        datasets.append(
            {
                "dataset_index": dataset_index,
                "x_train": x_train,
                "y_train": y_train,
                "x_test": x_test,
                "y_test": y_test,
                "feature_types": ["floating"] * x_train.shape[1],
                "metadata": metadata,
            }
        )
    cases._write_packed_shard(shard_dir, datasets=datasets)

    next_shard_index, copied_datasets, compaction_summary = corpus_materialization_invocation_module._copy_curated_round_shards(
        round_curated_dir=round_curated_dir,
        final_curated_dir=final_curated_dir,
        next_shard_index=0,
        max_datasets=1,
    )

    trimmed_shard = final_curated_dir / "shard_00000"
    catalog_rows = pq.read_table(trimmed_shard / "dataset_catalog.parquet").to_pylist()
    train_dataset_indices = pq.read_table(trimmed_shard / "train.parquet")["dataset_index"].to_pylist()
    test_dataset_indices = pq.read_table(trimmed_shard / "test.parquet")["dataset_index"].to_pylist()

    assert next_shard_index == 1
    assert copied_datasets == 1
    assert len(catalog_rows) == 1
    assert set(train_dataset_indices) == {0}
    assert set(test_dataset_indices) == {0}
    assert compaction_summary["source_shard_count"] == 1
    assert compaction_summary["output_shard_count"] == 1
    assert compaction_summary["dataset_count"] == 1


def test_copy_curated_round_shards_compacts_full_shards_into_parquet_catalogs(
    tmp_path: Path,
) -> None:
    round_curated_dir = tmp_path / "round_curated"
    final_curated_dir = tmp_path / "final_curated"
    shard_dir = round_curated_dir / "shard_00000"
    shard_dir.mkdir(parents=True, exist_ok=True)
    datasets: list[dict[str, Any]] = []
    for dataset_index, seed in enumerate((1101, 1102), start=0):
        x_train, y_train, x_test, y_test = cases._classification_arrays(seed=seed)
        metadata = cases._classification_metadata(
            n_features=x_train.shape[1],
            seed=seed,
            filter_status="accepted",
            filter_accepted=True,
        )
        metadata["dataset_id"] = f"{seed:032x}"
        datasets.append(
            {
                "dataset_index": dataset_index,
                "x_train": x_train,
                "y_train": y_train,
                "x_test": x_test,
                "y_test": y_test,
                "feature_types": ["floating"] * x_train.shape[1],
                "metadata": metadata,
            }
        )
    cases._write_packed_shard(shard_dir, datasets=datasets)

    next_shard_index, copied_datasets, compaction_summary = corpus_materialization_invocation_module._copy_curated_round_shards(
        round_curated_dir=round_curated_dir,
        final_curated_dir=final_curated_dir,
        next_shard_index=0,
        max_datasets=None,
    )

    destination_shard = final_curated_dir / "shard_00000"
    assert next_shard_index == 1
    assert copied_datasets == 2
    assert (destination_shard / "dataset_catalog.parquet").exists()
    catalog_rows = pq.read_table(destination_shard / "dataset_catalog.parquet").to_pylist()
    assert len(catalog_rows) == 2
    assert compaction_summary["source_shard_count"] == 1
    assert compaction_summary["output_shard_count"] == 1
    assert compaction_summary["dataset_count"] == 2


def test_materialize_corpus_recipe_clamps_accepted_only_round_to_remaining_budget(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _write_accepted_only_recipe_fixture(
        repo_tmp_path,
        recipe_id="accepted_only_budget_recipe",
        filename="accepted_only_budget_recipe.yaml",
        num_datasets=3,
    )
    requested_num_datasets: list[int] = []

    def _fake_generate(config) -> object:
        requested_num_datasets.append(int(config.num_datasets))
        return _fake_run_dagzoo_generate(config)

    def _budgeted_filter(config) -> DagzooFilterResult:
        round_root = Path(str(config.filter_out_dir)).expanduser().resolve().parent
        round_name = round_root.name
        round_summaries = {
            "round_01": (3, 0),
            "round_02": (3, 0),
            "round_03": (3, 0),
            "round_04": (1, 0),
            "round_05": (2, 0),
        }
        total_datasets, curated_count = round_summaries.get(round_name, (0, 0))
        filter_root = Path(str(config.filter_out_dir)).expanduser().resolve()
        curated_dir = Path(str(config.curated_out_dir)).expanduser().resolve()
        filter_root.mkdir(parents=True, exist_ok=True)
        curated_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = filter_root / "filter_manifest.ndjson"
        summary_path = filter_root / "filter_summary.json"
        manifest_path.write_text("{}\n" * max(1, curated_count), encoding="utf-8")
        summary_path.write_text(
            json.dumps(
                {
                    "total_datasets": total_datasets,
                    "accepted_datasets": curated_count,
                    "rejected_datasets": max(0, total_datasets - curated_count),
                    "curated_out_dir": str(curated_dir.resolve()),
                    "curated_accepted_datasets": curated_count,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        return DagzooFilterResult(
            manifest_path=manifest_path.resolve(),
            summary_path=summary_path.resolve(),
            total_datasets=total_datasets,
            accepted_datasets=curated_count,
            rejected_datasets=max(0, total_datasets - curated_count),
            elapsed_seconds=0.1,
            datasets_per_minute=600.0,
            curated_out_dir=curated_dir.resolve(),
            curated_accepted_datasets=curated_count,
        )

    _patch_dagzoo_generate(monkeypatch, _fake_generate)
    _patch_dagzoo_filter(monkeypatch, _budgeted_filter)

    with pytest.raises(
        RuntimeError,
        match="exhausted the generated dataset budget before reaching the requested accepted dataset target",
    ):
        _ = materialize_corpus_recipe(
            recipe_id="accepted_only_budget_recipe",
            dagzoo_root=repo_tmp_path.parent / "dagzoo",
            force=True,
            repo_root=repo_tmp_path,
        )

    assert requested_num_datasets == [5, 3, 3, 3, 2]


def test_materialize_corpus_recipe_fails_when_accepted_only_target_cannot_be_met(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _write_accepted_only_recipe_fixture(
        repo_tmp_path,
        recipe_id="accepted_only_failure_recipe",
        filename="accepted_only_failure_recipe.yaml",
        num_datasets=2,
    )
    _patch_dagzoo_generate(monkeypatch, _fake_run_dagzoo_generate)
    _patch_dagzoo_filter(
        monkeypatch,
        _round_sequence_fake_run_dagzoo_filter([0, 0, 0, 0], total_datasets_per_round=2),
    )

    with pytest.raises(
        RuntimeError,
        match="exhausted the generated dataset budget before reaching the requested accepted dataset target",
    ):
        _ = materialize_corpus_recipe(
            recipe_id="accepted_only_failure_recipe",
            dagzoo_root=repo_tmp_path.parent / "dagzoo",
            force=True,
            repo_root=repo_tmp_path,
        )


def test_load_corpus_record_backfills_legacy_dagzoo_provenance_summary(
    repo_tmp_path: Path,
) -> None:
    _write_recipe_registry(repo_tmp_path)
    legacy_record = _write_legacy_unscoped_corpus_record(
        repo_root=repo_tmp_path,
        sweep_id=None,
        recipe_id="current_recipe",
        seed=16,
    )

    loaded = load_corpus_record("current_recipe", repo_root=repo_tmp_path)

    assert loaded["corpus_ref"] == legacy_record["corpus_ref"]
    summary = loaded["dagzoo_provenance_summary"]
    assert summary["corpus_ref"] == legacy_record["corpus_ref"]
    assert summary["recipe_id"] == "current_recipe"
    assert summary["corpus_id"] == legacy_record["corpus_id"]
    assert summary["recipe_kind"] == "dagzoo_single_invocation"
    assert summary["surface_label"] == "anchor_manifest_default"
    assert summary["corpus_variant"] == "current_corpus_default"
    assert summary["comparator_role"] == "control"
    assert summary["config_refs"] == ["configs/default.yaml"]
    assert summary["provenance_labels"] == {
        "corpus_variant": "current_corpus_default",
        "comparator_role": "control",
    }
    assert summary["invocation_count"] == 1
    if "filter_policy" in summary:
        assert summary["filter_policy"] == "include_all"


def test_materialize_corpus_recipe_does_not_force_fixed_layout_batch_size_cap(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    captured: list[object] = []

    def _run(config) -> object:
        captured.append(config)
        return _fake_run_dagzoo_generate(config)

    _patch_dagzoo_generate(monkeypatch, _run)

    _ = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=True,
        repo_root=repo_tmp_path,
    )

    assert len(captured) == 1
    config = captured[0]
    assert getattr(config, "set_overrides") == ()


def test_materialize_corpus_recipe_defers_manifest_characteristics_until_hydration(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _patch_dagzoo_generate(monkeypatch, _fake_run_dagzoo_generate)

    record = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=True,
        repo_root=repo_tmp_path,
    )

    manifest = record["manifest"]
    characteristics = manifest["characteristics"]
    sidecar_path = Path(str(characteristics["sidecar_path"]))
    assert characteristics["cache_status"] == "deferred"
    assert sidecar_path.name == "manifest_characteristics.json"
    assert not sidecar_path.exists()
    assert characteristics["persisted_summary"]["total_records"] == 1
    assert "record_count" not in characteristics

    unloaded = load_corpus_record(record["corpus_ref"], repo_root=repo_tmp_path)
    unloaded_characteristics = unloaded["manifest"]["characteristics"]
    assert unloaded_characteristics["cache_status"] == "deferred"
    assert "record_count" not in unloaded_characteristics

    hydrated = load_corpus_record(
        record["corpus_ref"],
        repo_root=repo_tmp_path,
        hydrate_characteristics=True,
    )
    hydrated_characteristics = hydrated["manifest"]["characteristics"]
    assert sidecar_path.exists()
    assert hydrated_characteristics["record_count"] == 1
    assert hydrated_characteristics["persisted_summary"]["total_records"] == 1


def test_materialize_corpus_recipe_prefers_sweep_local_override_and_persists_rendered_config(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    sweep_id = "tf_rd_local"
    dagzoo_root = repo_tmp_path.parent / "dagzoo"
    _write_sweep_recipe_registry(repo_tmp_path, sweep_id=sweep_id)
    config_files_before = sorted(
        str(path.relative_to(dagzoo_root))
        for path in (dagzoo_root / "configs").rglob("*.yaml")
    )
    captured_runs: list[dict[str, object]] = []

    def _capturing_run(config) -> object:
        resolved_config = Path(str(config.dagzoo_config))
        if not resolved_config.is_absolute():
            resolved_config = (Path(str(config.dagzoo_root)) / resolved_config).resolve()
        captured_runs.append(
            {
                "dagzoo_config": str(config.dagzoo_config),
                "resolved_config_path": str(resolved_config),
                "rendered_payload": yaml.safe_load(resolved_config.read_text(encoding="utf-8")),
            }
        )
        return _fake_run_dagzoo_generate(config)

    _patch_dagzoo_generate(monkeypatch, _capturing_run)

    global_record = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=dagzoo_root,
        force=True,
        repo_root=repo_tmp_path,
    )
    local_record = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=dagzoo_root,
        force=False,
        repo_root=repo_tmp_path,
        sweep_id=sweep_id,
    )
    reused = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=dagzoo_root,
        force=False,
        repo_root=repo_tmp_path,
        sweep_id=sweep_id,
    )

    assert len(captured_runs) == 2
    assert global_record["recipe_path"] != local_record["recipe_path"]
    assert global_record["corpus_ref"] != local_record["corpus_ref"]
    assert local_record["recipe_path"] == str(
        (
            repo_tmp_path
            / "reference"
            / "system_delta_sweeps"
            / sweep_id
            / "corpus_recipes"
            / "current_recipe.yaml"
        ).resolve()
    )
    assert local_record["surface_label"] == "sweep_local_current"
    assert reused["corpus_ref"] == local_record["corpus_ref"]
    local_run = captured_runs[1]
    assert Path(str(local_run["dagzoo_config"])).is_absolute()
    assert ".staging" in str(local_run["dagzoo_config"])
    assert local_run["rendered_payload"] == {
        "seed": 7,
        "dataset": {"rows": 256, "shape": "sweep_local"},
        "generator": {"mode": "harder"},
    }
    invocation = local_record["dagzoo_provenance"]["invocations"][0]
    rendered_config_path = Path(str(invocation["rendered_config_path"]))
    assert rendered_config_path.exists()
    assert rendered_config_path.parent == Path(str(invocation["invocation_root"]))
    assert invocation["base_config_ref"] == "configs/base_override.yaml"
    assert invocation["config_overrides"] == {
        "seed": 7,
        "dataset": {"rows": 256, "shape": "sweep_local"},
        "generator": {"mode": "harder"},
    }
    assert invocation["resolved_config_path"] == str(rendered_config_path.resolve())
    assert invocation["rendered_config_sha256"] == sha256_path(rendered_config_path)
    assert local_record["dagzoo_provenance"]["config_refs"] == ["configs/base_override.yaml"]
    assert local_record["recipe_relative_path"] == (
        f"reference/system_delta_sweeps/{sweep_id}/corpus_recipes/current_recipe.yaml"
    )
    assert str(local_record["corpus_id"]).endswith(f"__{local_record['recipe_identity']}")
    global_latest_pointer_path = repo_tmp_path / "outputs" / "corpora" / "current_recipe" / "latest.json"
    global_latest_payload = json.loads(global_latest_pointer_path.read_text(encoding="utf-8"))
    assert global_latest_payload["corpus_ref"] == global_record["corpus_ref"]
    local_latest_pointer_path = Path(str(local_record["artifacts"]["latest_pointer_path"]))
    assert local_latest_pointer_path.name == f"latest__{local_record['recipe_identity']}.json"
    local_latest_payload = json.loads(local_latest_pointer_path.read_text(encoding="utf-8"))
    assert local_latest_payload["corpus_ref"] == local_record["corpus_ref"]
    loaded_global = load_corpus_record("current_recipe", repo_root=repo_tmp_path)
    loaded_local = load_corpus_record("current_recipe", repo_root=repo_tmp_path, sweep_id=sweep_id)
    assert loaded_global["corpus_ref"] == global_record["corpus_ref"]
    assert loaded_local["corpus_ref"] == local_record["corpus_ref"]
    config_files_after = sorted(
        str(path.relative_to(dagzoo_root))
        for path in (dagzoo_root / "configs").rglob("*.yaml")
    )
    assert config_files_after == config_files_before


def test_shadowed_sweep_local_corpus_refs_stay_distinct_when_manifest_hash_matches(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    sweep_id = "tf_rd_matching"
    _write_matching_sweep_recipe_registry(repo_tmp_path, sweep_id=sweep_id)
    call_counter = [0]
    _patch_dagzoo_generate(
        monkeypatch,
        _counting_fake_run_dagzoo_generate(call_counter),
    )

    global_record = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=True,
        repo_root=repo_tmp_path,
    )
    local_record = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=False,
        repo_root=repo_tmp_path,
        sweep_id=sweep_id,
    )

    assert call_counter == [2]
    assert global_record["manifest"]["manifest_sha256"] == local_record["manifest"]["manifest_sha256"]
    assert global_record["corpus_ref"] != local_record["corpus_ref"]
    assert global_record["corpus_id"] != local_record["corpus_id"]
    loaded_global = load_corpus_record(global_record["corpus_ref"], repo_root=repo_tmp_path)
    loaded_local = load_corpus_record(local_record["corpus_ref"], repo_root=repo_tmp_path)
    assert loaded_global["recipe_path"] == global_record["recipe_path"]
    assert loaded_local["recipe_path"] == local_record["recipe_path"]
    assert loaded_global["surface_label"] == "anchor_manifest_default"
    assert loaded_local["surface_label"] == "sweep_local_current"


def test_materialize_shadowed_recipe_rebuilds_legacy_unscoped_record_and_ignores_stale_latest(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    sweep_id = "tf_rd_local"
    _write_sweep_recipe_registry(repo_tmp_path, sweep_id=sweep_id)
    call_counter = [0]
    _patch_dagzoo_generate(
        monkeypatch,
        _counting_fake_run_dagzoo_generate(call_counter),
    )

    global_record = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=True,
        repo_root=repo_tmp_path,
    )
    legacy_record = _write_legacy_unscoped_corpus_record(
        repo_root=repo_tmp_path,
        sweep_id=sweep_id,
        recipe_id="current_recipe",
        seed=16,
    )

    loaded_global = load_corpus_record("current_recipe", repo_root=repo_tmp_path)
    rebuilt_local = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=False,
        repo_root=repo_tmp_path,
        sweep_id=sweep_id,
    )

    assert call_counter == [2]
    assert loaded_global["corpus_ref"] == global_record["corpus_ref"]
    assert loaded_global["recipe_path"] == global_record["recipe_path"]
    assert rebuilt_local["corpus_ref"] != legacy_record["corpus_ref"]
    assert rebuilt_local["corpus_id"] != legacy_record["corpus_id"]
    assert rebuilt_local["recipe_path"] == str(
        (
            repo_tmp_path
            / "reference"
            / "system_delta_sweeps"
            / sweep_id
            / "corpus_recipes"
            / "current_recipe.yaml"
        ).resolve()
    )


def test_sweep_only_corpus_refs_stay_distinct_when_manifest_hash_matches(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    sweep_a = "tf_rd_local_a"
    sweep_b = "tf_rd_local_b"
    _write_sweep_recipe_registry(repo_tmp_path, sweep_id=sweep_a)
    _write_sweep_recipe_registry(repo_tmp_path, sweep_id=sweep_b)
    call_counter = [0]
    _patch_dagzoo_generate(
        monkeypatch,
        _counting_fake_run_dagzoo_generate(call_counter),
    )

    record_a = materialize_corpus_recipe(
        recipe_id="sweep_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=False,
        repo_root=repo_tmp_path,
        sweep_id=sweep_a,
    )
    record_b = materialize_corpus_recipe(
        recipe_id="sweep_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=False,
        repo_root=repo_tmp_path,
        sweep_id=sweep_b,
    )

    assert call_counter == [2]
    assert record_a["manifest"]["manifest_sha256"] == record_b["manifest"]["manifest_sha256"]
    assert record_a["corpus_ref"] != record_b["corpus_ref"]
    assert record_a["corpus_id"] != record_b["corpus_id"]
    latest_pointer_a = Path(str(record_a["artifacts"]["latest_pointer_path"]))
    latest_pointer_b = Path(str(record_b["artifacts"]["latest_pointer_path"]))
    assert latest_pointer_a != latest_pointer_b
    assert latest_pointer_a.name == f"latest__{record_a['recipe_identity']}.json"
    assert latest_pointer_b.name == f"latest__{record_b['recipe_identity']}.json"
    stored_record_a = json.loads(Path(str(record_a["corpus_record_path"])).read_text(encoding="utf-8"))
    stored_record_b = json.loads(Path(str(record_b["corpus_record_path"])).read_text(encoding="utf-8"))
    assert stored_record_a["recipe_path"] == record_a["recipe_path"]
    assert stored_record_b["recipe_path"] == record_b["recipe_path"]
    loaded_a = load_corpus_record("sweep_recipe", repo_root=repo_tmp_path, sweep_id=sweep_a)
    loaded_b = load_corpus_record("sweep_recipe", repo_root=repo_tmp_path, sweep_id=sweep_b)
    assert loaded_a["corpus_ref"] == record_a["corpus_ref"]
    assert loaded_b["corpus_ref"] == record_b["corpus_ref"]


def test_materialize_sweep_only_recipe_rebuilds_legacy_unscoped_record(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    sweep_id = "tf_rd_local"
    _write_sweep_recipe_registry(repo_tmp_path, sweep_id=sweep_id)
    call_counter = [0]
    _patch_dagzoo_generate(
        monkeypatch,
        _counting_fake_run_dagzoo_generate(call_counter),
    )

    legacy_record = _write_legacy_unscoped_corpus_record(
        repo_root=repo_tmp_path,
        sweep_id=sweep_id,
        recipe_id="sweep_recipe",
        seed=16,
    )
    rebuilt = materialize_corpus_recipe(
        recipe_id="sweep_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=False,
        repo_root=repo_tmp_path,
        sweep_id=sweep_id,
    )

    assert call_counter == [1]
    assert rebuilt["corpus_ref"] != legacy_record["corpus_ref"]
    assert rebuilt["corpus_id"] != legacy_record["corpus_id"]
    assert str(rebuilt["corpus_id"]).endswith(f"__{rebuilt['recipe_identity']}")
    assert Path(str(rebuilt["artifacts"]["latest_pointer_path"])).name == (
        f"latest__{rebuilt['recipe_identity']}.json"
    )
    loaded = load_corpus_record("sweep_recipe", repo_root=repo_tmp_path, sweep_id=sweep_id)
    assert loaded["corpus_ref"] == rebuilt["corpus_ref"]


def test_materialize_corpus_ref_rejects_mismatched_explicit_corpus_id(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _patch_dagzoo_generate(monkeypatch, _fake_run_dagzoo_generate)

    with pytest.raises(RuntimeError, match="pinned to an exact corpus id"):
        materialize_corpus_ref(
            corpus_ref="current_recipe/current_recipe__deadbeefdead",
            dagzoo_root=repo_tmp_path.parent / "dagzoo",
            force=False,
            repo_root=repo_tmp_path,
        )


def test_scoped_recipe_identity_is_path_independent_across_repo_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root_a = tmp_path / "repo_a"
    repo_root_b = tmp_path / "repo_b"
    repo_root_a.mkdir(parents=True, exist_ok=True)
    repo_root_b.mkdir(parents=True, exist_ok=True)
    _initialize_repo_workspace(repo_root_a)
    _initialize_repo_workspace(repo_root_b)
    _write_sweep_recipe_registry(repo_root_a, sweep_id="tf_rd_local")
    _write_sweep_recipe_registry(repo_root_b, sweep_id="tf_rd_local")
    _patch_dagzoo_generate(monkeypatch, _fake_run_dagzoo_generate)

    record_a = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_root_a.parent / "dagzoo",
        force=False,
        repo_root=repo_root_a,
        sweep_id="tf_rd_local",
    )
    record_b = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_root_b.parent / "dagzoo",
        force=False,
        repo_root=repo_root_b,
        sweep_id="tf_rd_local",
    )

    assert record_a["recipe_relative_path"] == record_b["recipe_relative_path"]
    assert record_a["recipe_relative_path"] == (
        "reference/system_delta_sweeps/tf_rd_local/corpus_recipes/current_recipe.yaml"
    )
    assert record_a["recipe_identity"] == record_b["recipe_identity"]
    assert str(record_a["corpus_id"]).endswith(f"__{record_a['recipe_identity']}")
    assert str(record_b["corpus_id"]).endswith(f"__{record_b['recipe_identity']}")
    assert Path(str(record_a["artifacts"]["latest_pointer_path"])).name == (
        Path(str(record_b["artifacts"]["latest_pointer_path"])).name
    )


def test_load_copied_corpus_record_relocates_embedded_paths_across_repo_roots(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    repo_root_a = tmp_path / "repo_a"
    repo_root_b = tmp_path / "repo_b"
    repo_root_a.mkdir(parents=True, exist_ok=True)
    repo_root_b.mkdir(parents=True, exist_ok=True)
    _initialize_repo_workspace(repo_root_a)
    _initialize_repo_workspace(repo_root_b)
    _patch_dagzoo_generate(monkeypatch, _fake_run_dagzoo_generate)

    record_a = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_root_a.parent / "dagzoo",
        force=True,
        repo_root=repo_root_a,
    )
    source_recipe_root = repo_root_a / "outputs" / "corpora" / "current_recipe"
    target_recipe_root = repo_root_b / "outputs" / "corpora" / "current_recipe"
    target_recipe_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source_recipe_root, target_recipe_root)

    loaded = load_corpus_record("current_recipe", repo_root=repo_root_b)

    assert loaded["corpus_ref"] == record_a["corpus_ref"]
    assert Path(str(loaded["corpus_record_path"])).exists()
    assert not str(loaded["corpus_record_path"]).startswith(str(repo_root_a))
    assert not str(loaded["manifest"]["manifest_path"]).startswith(str(repo_root_a))
    assert Path(str(loaded["manifest"]["manifest_path"])).exists()
    assert str(loaded["artifacts"]["corpus_root"]).startswith(str(repo_root_b))
    assert Path(str(loaded["artifacts"]["corpus_root"])).exists()
    assert str(loaded["manifest"]["characteristics"]["sidecar_path"]).startswith(str(repo_root_b))
    invocation = loaded["dagzoo_provenance"]["invocations"][0]
    assert str(invocation["invocation_root"]).startswith(str(repo_root_b))
    assert Path(str(invocation["invocation_root"])).exists()
    assert str(invocation["handoff"]["generated_dir"]).startswith(str(repo_root_b))
    assert Path(str(invocation["handoff"]["generated_dir"])).exists()

    _patch_corpus_repo_root(monkeypatch, repo_root_b)
    resolved = resolve_data_surface(
        {
            "source": "manifest",
            "corpus_ref": "current_recipe",
        }
    )
    assert resolved.manifest_path is not None
    assert resolved.manifest_path.exists()
    assert str(resolved.manifest_path).startswith(str(repo_root_b))


def test_materialize_corpus_recipe_reuses_complete_cached_corpus(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    call_counter = [0]
    _patch_dagzoo_generate(
        monkeypatch,
        _counting_fake_run_dagzoo_generate(call_counter),
    )

    record = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=True,
        repo_root=repo_tmp_path,
    )
    reused = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=False,
        repo_root=repo_tmp_path,
    )

    assert call_counter == [1]
    assert reused["corpus_ref"] == record["corpus_ref"]
    assert Path(str(reused["manifest"]["manifest_path"])).exists()


def test_materialize_corpus_recipe_rebuilds_when_recipe_contents_change(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    call_counter = [0]
    _patch_dagzoo_generate(
        monkeypatch,
        _counting_fake_run_dagzoo_generate(call_counter),
    )

    original = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=True,
        repo_root=repo_tmp_path,
    )
    recipe_path = repo_tmp_path / "reference" / "corpus_recipes" / "current_recipe.yaml"
    payload = yaml.safe_load(recipe_path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    payload["dagzoo"]["num_datasets"] = 9
    recipe_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )

    rebuilt = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=False,
        repo_root=repo_tmp_path,
    )

    assert call_counter == [2]
    assert rebuilt["corpus_ref"] != original["corpus_ref"]
    assert rebuilt["recipe_identity"] != original["recipe_identity"]


def test_materialize_corpus_recipe_rebuilds_when_cached_manifest_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    call_counter = [0]
    _patch_dagzoo_generate(
        monkeypatch,
        _counting_fake_run_dagzoo_generate(call_counter),
    )

    record = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=True,
        repo_root=repo_tmp_path,
    )
    Path(str(record["manifest"]["manifest_path"])).unlink()

    rebuilt = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=False,
        repo_root=repo_tmp_path,
    )

    assert call_counter == [2]
    assert rebuilt["corpus_ref"] == record["corpus_ref"]
    assert Path(str(rebuilt["manifest"]["manifest_path"])).exists()


def test_materialize_corpus_recipe_rebuilds_when_cached_record_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    call_counter = [0]
    _patch_dagzoo_generate(
        monkeypatch,
        _counting_fake_run_dagzoo_generate(call_counter),
    )

    record = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=True,
        repo_root=repo_tmp_path,
    )
    Path(str(record["corpus_record_path"])).unlink()

    rebuilt = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=False,
        repo_root=repo_tmp_path,
    )

    assert call_counter == [2]
    assert rebuilt["corpus_ref"] == record["corpus_ref"]
    assert Path(str(rebuilt["corpus_record_path"])).exists()
    assert Path(str(rebuilt["manifest"]["manifest_path"])).exists()


def test_materialize_corpus_recipe_rebuilds_when_cached_invocation_artifact_is_missing(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    call_counter = [0]
    _patch_dagzoo_generate(
        monkeypatch,
        _counting_fake_run_dagzoo_generate(call_counter),
    )

    record = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=True,
        repo_root=repo_tmp_path,
    )
    invocation = record["dagzoo_provenance"]["invocations"][0]
    shutil.rmtree(Path(str(invocation["handoff"]["generated_dir"])))

    rebuilt = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=False,
        repo_root=repo_tmp_path,
    )

    assert call_counter == [2]
    assert rebuilt["corpus_ref"] == record["corpus_ref"]
    assert Path(str(rebuilt["manifest"]["manifest_path"])).exists()
    assert Path(str(rebuilt["dagzoo_provenance"]["invocations"][0]["handoff"]["generated_dir"])).exists()


def test_materialize_corpus_recipe_rejects_single_invocation_handoff_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _patch_dagzoo_generate(monkeypatch, _fake_run_dagzoo_generate_mismatched_handoff)

    with pytest.raises(RuntimeError, match="generated_corpus_id"):
        materialize_corpus_recipe(
            recipe_id="current_recipe",
            dagzoo_root=repo_tmp_path.parent / "dagzoo",
            force=True,
            repo_root=repo_tmp_path,
        )


def test_materialize_corpus_recipe_rejects_multi_invocation_handoff_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _patch_dagzoo_generate(monkeypatch, _fake_run_dagzoo_generate_mismatched_handoff)

    with pytest.raises(RuntimeError, match="generated_corpus_id"):
        materialize_corpus_recipe(
            recipe_id="size_recipe",
            dagzoo_root=repo_tmp_path.parent / "dagzoo",
            force=True,
            materialize_processes=1,
            repo_root=repo_tmp_path,
        )


def test_corpus_compare_payload_reports_differences(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _patch_dagzoo_generate(monkeypatch, _fake_run_dagzoo_generate)
    _ = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=True,
        repo_root=repo_tmp_path,
    )
    _ = materialize_corpus_recipe(
        recipe_id="size_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=True,
        materialize_processes=1,
        repo_root=repo_tmp_path,
    )

    payload = corpus_compare_payload(
        left="current_recipe",
        right="size_recipe",
        repo_root=repo_tmp_path,
    )

    assert payload["difference_count"] > 0
    assert payload["left"]["recipe_id"] == "current_recipe"
    assert payload["right"]["recipe_id"] == "size_recipe"


def test_corpus_results_payload_groups_runs_by_corpus_ref(repo_tmp_path: Path) -> None:
    corpus_root = repo_tmp_path / "outputs" / "corpora" / "current_recipe" / "current_recipe__123456789abc"
    corpus_root.mkdir(parents=True, exist_ok=True)
    corpus_record_path = corpus_root / "corpus_record.json"
    corpus_record = {
        "schema": "tab-foundry-corpus-record-v1",
        "generated_at_utc": "2026-03-23T00:00:00Z",
        "recipe_id": "current_recipe",
        "corpus_id": "current_recipe__123456789abc",
        "corpus_ref": "current_recipe/current_recipe__123456789abc",
        "corpus_record_path": str(corpus_record_path),
        "recipe_path": str((repo_tmp_path / "reference" / "corpus_recipes" / "current_recipe.yaml").resolve()),
        "surface_label": "anchor_manifest_default",
        "recipe": {"invocations": []},
        "manifest": {
            "manifest_path": str((corpus_root / "manifest.parquet").resolve()),
            "manifest_sha256": "a" * 64,
            "inspection": {"total_records": 1},
            "characteristics": {"record_count": 1},
        },
        "dagzoo_provenance": {},
    }
    corpus_record_path.write_text(json.dumps(corpus_record, indent=2, sort_keys=True), encoding="utf-8")
    latest_pointer = repo_tmp_path / "outputs" / "corpora" / "current_recipe" / "latest.json"
    latest_pointer.write_text(
        json.dumps(
            {
                "schema": "tab-foundry-corpus-latest-v1",
                "generated_at_utc": "2026-03-23T00:00:00Z",
                "recipe_id": "current_recipe",
                "corpus_id": "current_recipe__123456789abc",
                "corpus_ref": "current_recipe/current_recipe__123456789abc",
                "corpus_record_path": str(corpus_record_path),
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    run_root = repo_tmp_path / "outputs" / "run_a" / "train"
    run_root.mkdir(parents=True, exist_ok=True)
    training_surface_record_path = run_root / "training_surface_record.json"
    training_surface_record_path.write_text(
        json.dumps(
            {
                "data": {
                    "corpus_ref": "current_recipe/current_recipe__123456789abc",
                    "recipe_id": "current_recipe",
                    "corpus_id": "current_recipe__123456789abc",
                }
            },
            indent=2,
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(
        registry_module,
        "load_benchmark_run_registry",
        lambda _path: {
            "runs": {
                "run_a": {
                    "run_id": "run_a",
                    "experiment": "cls_benchmark_staged",
                    "config_profile": "cls_benchmark_staged",
                    "decision": "defer",
                    "surface_labels": {"data": "anchor_manifest_default"},
                    "tab_foundry_metrics": {"best_roc_auc": 0.71, "final_roc_auc": 0.70},
                    "artifacts": {
                        "training_surface_record_path": str(training_surface_record_path),
                    },
                    "sweep": {
                        "sweep_id": "tf_rd_013_dagzoo_size_ladder_v1",
                        "delta_id": "delta_training_current_corpus_uncapped",
                        "queue_order": 1,
                    },
                }
            }
        },
    )

    try:
        payload = corpus_results_payload(
            corpus_ref="current_recipe",
            registry_path=repo_tmp_path / "registry.json",
            repo_root=repo_tmp_path,
        )
    finally:
        monkeypatch.undo()

    assert payload["run_count"] == 1
    assert payload["runs"][0]["run_id"] == "run_a"


def test_resolve_data_surface_hydrates_corpus_ref(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _patch_dagzoo_generate(monkeypatch, _fake_run_dagzoo_generate)
    _ = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=True,
        repo_root=repo_tmp_path,
    )
    _patch_corpus_repo_root(monkeypatch, repo_tmp_path)

    resolved = resolve_data_surface(
        {
            "source": "manifest",
            "corpus_ref": "current_recipe",
        }
    )

    assert resolved.surface_label == "anchor_manifest_default"
    assert resolved.corpus_ref is not None
    assert resolved.recipe_id == "current_recipe"
    assert resolved.manifest_path is not None and resolved.manifest_path.exists()
    assert resolved.allow_missing_values is True
    assert {
        key: resolved.dagzoo_provenance[key]
        for key in (
            "corpus_ref",
            "recipe_id",
            "corpus_id",
            "recipe_kind",
            "surface_label",
            "corpus_variant",
            "comparator_role",
            "config_refs",
            "invocation_count",
            "provenance_labels",
        )
    } == {
        "corpus_ref": resolved.corpus_ref,
        "recipe_id": "current_recipe",
        "corpus_id": resolved.corpus_id,
        "recipe_kind": "dagzoo_single_invocation",
        "surface_label": "anchor_manifest_default",
        "corpus_variant": "current_corpus_default",
        "comparator_role": "control",
        "config_refs": ["configs/default.yaml"],
        "invocation_count": 1,
        "provenance_labels": {
            "corpus_variant": "current_corpus_default",
            "comparator_role": "control",
        },
    }
    timing_summary = resolved.dagzoo_provenance["materialization_timing"]
    assert timing_summary["timed_invocation_count"] == 1
    assert timing_summary["cumulative_generated_datasets"] == 8
    assert timing_summary["cumulative_generate_elapsed_seconds"] >= 0.0
    assert timing_summary["cumulative_upstream_elapsed_seconds"] >= 0.0
    assert timing_summary["cumulative_local_overhead_elapsed_seconds"] >= 0.0
    assert timing_summary["cumulative_invocation_elapsed_seconds"] >= 0.0


def test_resolve_data_surface_rejects_removed_row_cap_subsampling(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _patch_dagzoo_generate(monkeypatch, _fake_run_dagzoo_generate)
    _ = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=True,
        repo_root=repo_tmp_path,
    )
    _patch_corpus_repo_root(monkeypatch, repo_tmp_path)

    with pytest.raises(ValueError, match="Row subsampling is no longer supported"):
        _ = resolve_data_surface(
            {
                "source": "manifest",
                "corpus_ref": "current_recipe",
                "train_row_cap": 32,
            }
        )


def test_resolve_data_surface_uses_sweep_lookup_hint_for_shadowed_corpus_ref(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    sweep_id = "tf_rd_local"
    _write_sweep_recipe_registry(repo_tmp_path, sweep_id=sweep_id)
    _patch_dagzoo_generate(monkeypatch, _fake_run_dagzoo_generate)
    global_record = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=True,
        repo_root=repo_tmp_path,
    )
    local_record = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=False,
        repo_root=repo_tmp_path,
        sweep_id=sweep_id,
    )
    _patch_corpus_repo_root(monkeypatch, repo_tmp_path)

    resolved_global = resolve_data_surface(
        {
            "source": "manifest",
            "corpus_ref": "current_recipe",
        }
    )
    resolved_local = resolve_data_surface(
        {
            "source": "manifest",
            "corpus_ref": "current_recipe",
            "surface_overrides": {
                "corpus_lookup_sweep_id": sweep_id,
                "corpus_lookup_sweeps_root": str(
                    (repo_tmp_path / "reference" / "system_delta_sweeps").resolve()
                ),
            },
        }
    )

    assert resolved_global.corpus_ref == global_record["corpus_ref"]
    assert resolved_local.corpus_ref == local_record["corpus_ref"]
    assert "corpus_lookup_sweep_id" not in resolved_local.overrides
    assert "corpus_lookup_sweeps_root" not in resolved_local.overrides


def test_resolve_data_surface_raises_for_broken_sweep_lookup_hint(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    sweep_id = "tf_rd_broken"
    _write_broken_sweep_recipe_registry(repo_tmp_path, sweep_id=sweep_id)
    _patch_dagzoo_generate(monkeypatch, _fake_run_dagzoo_generate)
    _ = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=True,
        repo_root=repo_tmp_path,
    )
    _patch_corpus_repo_root(monkeypatch, repo_tmp_path)

    with pytest.raises(RuntimeError, match="corpus recipe 'current_recipe' does not exist"):
        resolve_data_surface(
            {
                "source": "manifest",
                "corpus_ref": "current_recipe",
                "surface_overrides": {
                    "corpus_lookup_sweep_id": sweep_id,
                },
            }
        )


def test_resolve_data_surface_explicit_allow_missing_values_override_beats_corpus_policy(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    _patch_dagzoo_generate(monkeypatch, _fake_run_dagzoo_generate)
    _ = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=True,
        repo_root=repo_tmp_path,
    )
    _patch_corpus_repo_root(monkeypatch, repo_tmp_path)

    resolved = resolve_data_surface(
        {
            "source": "manifest",
            "corpus_ref": "current_recipe",
            "allow_missing_values": False,
        }
    )

    assert resolved.allow_missing_values is False
