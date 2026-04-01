from __future__ import annotations

from itertools import product
from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.data.corpus_loading import load_corpus_recipe


REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE_ROOT = REPO_ROOT / "reference" / "corpus_recipes"
ROW_SPECS = {
    128: (96, 32),
    256: (192, 64),
    512: (384, 128),
    1024: (768, 256),
}
FEATURE_GRAPH_BANDS = {
    6: (2, 12),
    10: (2, 20),
    14: (4, 28),
    20: (2, 20),
}
CLASS_COUNTS = set(range(2, 11))
EXPECTED_GRID = {
    (row_total, feature_count, class_count)
    for row_total, feature_count, class_count in product(
        ROW_SPECS,
        FEATURE_GRAPH_BANDS,
        CLASS_COUNTS,
    )
}


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def test_tf_rd_010_latent_target_recipe_ids_are_registered() -> None:
    index = _load_yaml(RECIPE_ROOT / "index.yaml")
    recipes = index["recipes"]
    assert recipes["tf_rd_010_dagzoo_medium_control_curated_v5"] == {
        "path": "tf_rd_010_dagzoo_medium_control_curated_v5.yaml"
    }
    assert recipes["tf_rd_010_latent_target_canary_curated_v3"] == {
        "path": "tf_rd_010_latent_target_canary_curated_v3.yaml"
    }


def test_tf_rd_010_dagzoo_medium_control_curated_v5_preserves_balanced_shape_without_teacher_controls() -> None:
    recipe = load_corpus_recipe("tf_rd_010_dagzoo_medium_control_curated_v5", repo_root=REPO_ROOT).to_dict()

    assert recipe["surface_label"] == "tf_rd_010_dagzoo_medium_control"
    assert recipe["manifest"]["filter_policy"] == "accepted_only"
    assert recipe["provenance_labels"]["corpus_recipe_version"] == "v5"
    assert recipe["provenance_labels"]["target_derivation"] == "tabiclv2_latent_node"
    assert recipe["review_summary"]["target_derivation"] == "tabiclv2_latent_node"

    invocations = recipe["invocations"]
    assert len(invocations) == 144
    assert sum(invocation["num_datasets"] for invocation in invocations) == 159984

    grid: set[tuple[int, int, int]] = set()
    for invocation in invocations:
        dataset = invocation["config_overrides"]["dataset"]
        graph = invocation["config_overrides"]["graph"]
        retry_policy = invocation["config_overrides"]["filter"]

        feature_count = dataset["n_features_min"]
        class_count = dataset["n_classes_min"]
        row_total = int(dataset["n_train"]) + int(dataset["n_test"])

        assert dataset["task"] == "classification"
        assert dataset["n_features_max"] == feature_count
        assert dataset["n_classes_max"] == class_count
        assert dataset["categorical_ratio_min"] == 0.0
        assert dataset["categorical_ratio_max"] == 1.0
        assert dataset["max_categorical_cardinality"] == 12
        assert "target_parent_prior" not in dataset
        assert "target_parent_count_min" not in dataset
        assert "target_parent_count_max" not in dataset

        assert graph == {
            "n_nodes_min": FEATURE_GRAPH_BANDS[feature_count][0],
            "n_nodes_max": FEATURE_GRAPH_BANDS[feature_count][1],
        }
        assert retry_policy == {"max_attempts": 256}
        assert row_total in ROW_SPECS
        assert (dataset["n_train"], dataset["n_test"]) == ROW_SPECS[row_total]
        grid.add((row_total, feature_count, class_count))

    assert grid == EXPECTED_GRID


def test_tf_rd_010_latent_target_canary_curated_v3_tracks_row_ladder_without_teacher_controls() -> None:
    recipe = load_corpus_recipe("tf_rd_010_latent_target_canary_curated_v3", repo_root=REPO_ROOT).to_dict()

    assert recipe["surface_label"] == "tf_rd_010_latent_target_canary"
    assert recipe["manifest"]["filter_policy"] == "accepted_only"
    assert recipe["provenance_labels"]["target_derivation"] == "tabiclv2_latent_node"
    assert recipe["review_summary"]["grid_family"] == "latent_target_canary_rows_only_v2"

    invocations = recipe["invocations"]
    assert len(invocations) == 4
    assert sum(invocation["num_datasets"] for invocation in invocations) == 128

    seen_rows: set[int] = set()
    for invocation in invocations:
        dataset = invocation["config_overrides"]["dataset"]
        assert dataset["n_features_min"] == dataset["n_features_max"] == 6
        assert dataset["n_classes_min"] == dataset["n_classes_max"] == 2
        assert dataset["categorical_ratio_min"] == dataset["categorical_ratio_max"] == 0.0
        assert "target_parent_prior" not in dataset
        row_total = int(dataset["n_train"]) + int(dataset["n_test"])
        seen_rows.add(row_total)
    assert seen_rows == set(ROW_SPECS)
