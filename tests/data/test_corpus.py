from __future__ import annotations

import json
from pathlib import Path

import pytest

import tab_foundry.bench.benchmark_run_registry as registry_module
from tab_foundry.data import corpus as corpus_module
from tab_foundry.data.corpus import (
    corpus_compare_payload,
    corpus_id_for_manifest,
    corpus_results_payload,
    list_corpus_recipes,
    load_corpus_record,
    load_corpus_recipe,
    materialize_corpus_recipe,
)
from tab_foundry.data.dagzoo_handoff import (
    DAGZOO_HANDOFF_SCHEMA_NAME,
    DAGZOO_HANDOFF_SCHEMA_VERSION,
    load_dagzoo_handoff_info,
)
from tab_foundry.data.surface import resolve_data_surface

from . import manifest_and_dataset_cases as cases


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


def _write_handoff_manifest(handoff_root: Path, *, generated_dir_rel: str = "generated") -> Path:
    handoff_root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_name": DAGZOO_HANDOFF_SCHEMA_NAME,
        "schema_version": DAGZOO_HANDOFF_SCHEMA_VERSION,
        "identity": {
            "source_family": "dagzoo.fixed_layout_scm",
            "generate_run_id": "1" * 32,
            "generated_corpus_id": "2" * 32,
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


def _write_generated_dataset(generated_dir: Path, *, seed: int) -> None:
    x_train, y_train, x_test, y_test = cases._classification_arrays(seed=seed)
    metadata = cases._classification_metadata(
        n_features=x_train.shape[1],
        seed=seed,
        filter_status="accepted",
        filter_accepted=True,
    )
    metadata["dataset_id"] = "3" * 32
    metadata["split_groups"] = {"request_run": "1" * 32}
    cases._write_packed_shard(
        generated_dir / "shard_00000",
        datasets=[
            {
                "dataset_index": 0,
                "x_train": x_train,
                "y_train": y_train,
                "x_test": x_test,
                "y_test": y_test,
                "feature_types": ["num"] * x_train.shape[1],
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


def test_load_and_list_corpus_recipes(repo_tmp_path: Path) -> None:
    _write_recipe_registry(repo_tmp_path)

    recipes = list_corpus_recipes(repo_root=repo_tmp_path)

    assert [recipe.recipe_id for recipe in recipes] == ["current_recipe", "size_recipe"]
    current = load_corpus_recipe("current_recipe", repo_root=repo_tmp_path)
    assert current.kind == "dagzoo_single_invocation"
    assert current.surface_label == "anchor_manifest_default"
    assert current.invocations[0].config_ref == "configs/default.yaml"


@pytest.fixture
def repo_tmp_path(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    repo_root.mkdir(parents=True, exist_ok=True)
    _write_recipe_registry(repo_root)
    dagzoo_root = repo_root / ".." / "dagzoo"
    (dagzoo_root / "configs").mkdir(parents=True, exist_ok=True)
    for config_name in (
        "default.yaml",
        "benchmark_cpu.yaml",
        "benchmark_cuda_h100_large_shape.yaml",
    ):
        (dagzoo_root / "configs" / config_name).write_text("seed: 1\n", encoding="utf-8")
    return repo_root


def test_materialize_corpus_recipe_writes_corpus_record_and_latest_pointer(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    monkeypatch.setattr(corpus_module, "run_dagzoo_generate", _fake_run_dagzoo_generate)

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


def test_corpus_compare_payload_reports_differences(
    monkeypatch: pytest.MonkeyPatch,
    repo_tmp_path: Path,
) -> None:
    monkeypatch.setattr(corpus_module, "run_dagzoo_generate", _fake_run_dagzoo_generate)
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
    monkeypatch.setattr(corpus_module, "run_dagzoo_generate", _fake_run_dagzoo_generate)
    _ = materialize_corpus_recipe(
        recipe_id="current_recipe",
        dagzoo_root=repo_tmp_path.parent / "dagzoo",
        force=True,
        repo_root=repo_tmp_path,
    )
    monkeypatch.setattr(corpus_module, "_repo_root", lambda: repo_tmp_path)

    resolved = resolve_data_surface(
        {
            "source": "manifest",
            "corpus_ref": "current_recipe",
            "train_row_cap": 32,
        }
    )

    assert resolved.surface_label == "anchor_manifest_default"
    assert resolved.corpus_ref is not None
    assert resolved.recipe_id == "current_recipe"
    assert resolved.manifest_path is not None and resolved.manifest_path.exists()
    assert resolved.train_row_cap == 32
