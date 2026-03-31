from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.data.corpus_loading import load_corpus_recipe


REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE_ROOT = REPO_ROOT / "reference" / "corpus_recipes"
FEATURE_GRAPH_BANDS = {
    6: (2, 12),
    10: (2, 20),
    14: (4, 28),
    20: (6, 40),
}
CLASS_COUNTS = {2, 3, 5, 7, 10}
SEEDS = {1, 2, 3}
EXPECTED_GRID = {
    (feature_count, class_count, seed)
    for feature_count, class_count, seed in product(
        FEATURE_GRAPH_BANDS,
        CLASS_COUNTS,
        SEEDS,
    )
}


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _load_recipe(recipe_id: str) -> dict[str, Any]:
    return load_corpus_recipe(recipe_id, repo_root=REPO_ROOT).to_dict()


def _grid_signature(recipe: dict[str, Any], *, expected_missing_rate: float) -> set[tuple[int, int, int]]:
    invocations = recipe["invocations"]
    assert isinstance(invocations, list)
    assert len(invocations) == 60
    assert sum(invocation["num_datasets"] for invocation in invocations) == 480

    grid: set[tuple[int, int, int]] = set()
    for invocation in invocations:
        assert invocation["base_config_ref"] == "configs/default.yaml"
        assert invocation["num_datasets"] == 8
        assert invocation["device"] == "cpu"
        assert invocation["hardware_policy"] == "none"
        assert invocation["missing_rate"] == expected_missing_rate
        assert invocation["missing_mechanism"] == "mcar"

        dataset = invocation["config_overrides"]["dataset"]
        graph = invocation["config_overrides"]["graph"]
        retry_policy = invocation["config_overrides"]["filter"]

        feature_count = dataset["n_features_min"]
        assert feature_count in FEATURE_GRAPH_BANDS
        assert dataset["n_features_max"] == feature_count
        assert dataset["n_train"] == 64
        assert dataset["n_test"] == 32
        assert dataset["task"] == "classification"
        assert dataset["categorical_ratio_min"] == 0.0
        assert dataset["categorical_ratio_max"] == 1.0
        assert dataset["max_categorical_cardinality"] == 12

        class_count = dataset["n_classes_min"]
        assert class_count in CLASS_COUNTS
        assert dataset["n_classes_max"] == class_count

        seed = invocation["seed"]
        assert seed in SEEDS

        expected_nodes_min, expected_nodes_max = FEATURE_GRAPH_BANDS[feature_count]
        assert graph == {
            "n_nodes_min": expected_nodes_min,
            "n_nodes_max": expected_nodes_max,
        }
        assert retry_policy == {"max_attempts": 32}
        assert invocation["invocation_id"] == f"f{feature_count:02d}_c{class_count:02d}_s{seed:02d}"
        grid.add((feature_count, class_count, seed))

    assert grid == EXPECTED_GRID
    return grid


def test_tf_rd_010_dagzoo_recipe_v2_is_registered() -> None:
    index = _load_yaml(RECIPE_ROOT / "index.yaml")
    recipes = index["recipes"]
    assert recipes["tf_rd_010_dagzoo_aligned_control_v2"] == {
        "path": "tf_rd_010_dagzoo_aligned_control_v2.yaml"
    }
    assert recipes["tf_rd_010_missingness_mcar_strong_v2"] == {
        "path": "tf_rd_010_missingness_mcar_strong_v2.yaml"
    }


def test_tf_rd_010_dagzoo_recipe_v2_uses_balanced_feature_class_grid() -> None:
    aligned_summary = _load_yaml(RECIPE_ROOT / "tf_rd_010_dagzoo_aligned_control_v2.yaml")
    strong_summary = _load_yaml(RECIPE_ROOT / "tf_rd_010_missingness_mcar_strong_v2.yaml")
    aligned = _load_recipe("tf_rd_010_dagzoo_aligned_control_v2")
    strong = _load_recipe("tf_rd_010_missingness_mcar_strong_v2")

    assert aligned["surface_label"] == "tf_rd_010_dagzoo_aligned_control"
    assert strong["surface_label"] == "tf_rd_010_missingness_mcar_strong"
    assert aligned["provenance_labels"]["comparator_role"] == "control"
    assert strong["provenance_labels"]["comparator_role"] == "tf_rd_010_candidate"
    assert strong["provenance_labels"]["perturbation_strength"] == "strong"
    assert aligned_summary["kind"] == "dagzoo_python_generated"
    assert strong_summary["kind"] == "dagzoo_python_generated"
    assert aligned_summary["review_summary"]["invocation_count"] == 60
    assert strong_summary["review_summary"]["manifest_record_count"] == 480

    aligned_grid = _grid_signature(aligned, expected_missing_rate=0.2)
    strong_grid = _grid_signature(strong, expected_missing_rate=0.4)
    assert aligned_grid == strong_grid == EXPECTED_GRID
