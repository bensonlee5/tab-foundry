from __future__ import annotations

import json
from pathlib import Path
import shutil
from typing import Any

import pytest
import yaml

import tab_foundry.benchmark_registry as registry_module
import tab_foundry.data.corpus_loading as corpus_loading_module
import tab_foundry.data.corpus_lookup as corpus_lookup_module
import tab_foundry.data.corpus_materialization as corpus_materialization_module
from tab_foundry.data.corpus_loading import (
    _generator_fingerprint,
    corpus_id_for_manifest,
    corpus_outputs_root,
    corpus_recipe_index_path,
    corpus_recipes_root,
    list_corpus_recipes,
    load_corpus_recipe,
)
from tab_foundry.data.corpus_lookup import load_corpus_record
from tab_foundry.data.corpus_materialization import (
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


def _patch_corpus_repo_root(monkeypatch: pytest.MonkeyPatch, repo_root: Path) -> None:
    monkeypatch.setattr(corpus_loading_module, "_repo_root", lambda: repo_root)
    monkeypatch.setattr(corpus_lookup_module, "_repo_root", lambda: repo_root)


def _patch_dagzoo_generate(monkeypatch: pytest.MonkeyPatch, replacement: Any) -> None:
    monkeypatch.setattr(corpus_materialization_module, "run_dagzoo_generate", replacement)


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
            "run_root": ".",
            "generated_dir": generated_dir_rel,
        },
        "defaults": {
            "recommended_training_corpus": "generated",
            "recommended_training_artifact_key": "generated_dir",
            "curation_policy": "none",
        },
    }
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
    assert loaded["dagzoo_provenance_summary"] == {
        "corpus_ref": legacy_record["corpus_ref"],
        "recipe_id": "current_recipe",
        "corpus_id": legacy_record["corpus_id"],
        "recipe_kind": "dagzoo_single_invocation",
        "surface_label": "anchor_manifest_default",
        "corpus_variant": "current_corpus_default",
        "comparator_role": "control",
        "config_refs": ["configs/default.yaml"],
        "provenance_labels": {
            "corpus_variant": "current_corpus_default",
            "comparator_role": "control",
        },
        "invocation_count": 1,
    }


def test_materialize_corpus_recipe_caps_cpu_fixed_layout_batch_size(
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
    assert getattr(config, "set_overrides") == ("runtime.fixed_layout_batch_size_cap=128",)


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
    assert resolved.dagzoo_provenance == {
        "corpus_ref": resolved.corpus_ref,
        "recipe_id": "current_recipe",
        "corpus_id": resolved.corpus_id,
        "recipe_kind": "dagzoo_single_invocation",
        "surface_label": "anchor_manifest_default",
        "corpus_variant": "current_corpus_default",
        "comparator_role": "control",
        "config_refs": ["configs/default.yaml"],
        "provenance_labels": {
            "corpus_variant": "current_corpus_default",
            "comparator_role": "control",
        },
    }


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
