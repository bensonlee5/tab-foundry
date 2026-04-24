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
FEATURE_SHAPE_SPECS = {
    (128, 128): {
        "cell_budget_label": "16k_cells",
        "n_train": 96,
        "n_test": 32,
        "cell_count": 16384,
        "graph": (8, 64),
    },
    (256, 64): {
        "cell_budget_label": "16k_cells",
        "n_train": 192,
        "n_test": 64,
        "cell_count": 16384,
        "graph": (4, 64),
    },
    (512, 32): {
        "cell_budget_label": "16k_cells",
        "n_train": 384,
        "n_test": 128,
        "cell_count": 16384,
        "graph": (4, 32),
    },
    (1024, 16): {
        "cell_budget_label": "16k_cells",
        "n_train": 768,
        "n_test": 256,
        "cell_count": 16384,
        "graph": (4, 16),
    },
    (128, 256): {
        "cell_budget_label": "32k_cells",
        "n_train": 96,
        "n_test": 32,
        "cell_count": 32768,
        "graph": (8, 96),
    },
    (256, 128): {
        "cell_budget_label": "32k_cells",
        "n_train": 192,
        "n_test": 64,
        "cell_count": 32768,
        "graph": (8, 64),
    },
    (512, 64): {
        "cell_budget_label": "32k_cells",
        "n_train": 384,
        "n_test": 128,
        "cell_count": 32768,
        "graph": (4, 64),
    },
    (1024, 32): {
        "cell_budget_label": "32k_cells",
        "n_train": 768,
        "n_test": 256,
        "cell_count": 32768,
        "graph": (4, 32),
    },
    (256, 256): {
        "cell_budget_label": "65k_cells",
        "n_train": 192,
        "n_test": 64,
        "cell_count": 65536,
        "graph": (8, 96),
    },
    (512, 128): {
        "cell_budget_label": "65k_cells",
        "n_train": 384,
        "n_test": 128,
        "cell_count": 65536,
        "graph": (8, 64),
    },
    (1024, 64): {
        "cell_budget_label": "65k_cells",
        "n_train": 768,
        "n_test": 256,
        "cell_count": 65536,
        "graph": (4, 64),
    },
    (2048, 32): {
        "cell_budget_label": "65k_cells",
        "n_train": 1536,
        "n_test": 512,
        "cell_count": 65536,
        "graph": (4, 32),
    },
    (512, 256): {
        "cell_budget_label": "131k_cells",
        "n_train": 384,
        "n_test": 128,
        "cell_count": 131072,
        "graph": (8, 96),
    },
    (1024, 128): {
        "cell_budget_label": "131k_cells",
        "n_train": 768,
        "n_test": 256,
        "cell_count": 131072,
        "graph": (8, 64),
    },
    (2048, 64): {
        "cell_budget_label": "131k_cells",
        "n_train": 1536,
        "n_test": 512,
        "cell_count": 131072,
        "graph": (4, 64),
    },
    (4096, 32): {
        "cell_budget_label": "131k_cells",
        "n_train": 3072,
        "n_test": 1024,
        "cell_count": 131072,
        "graph": (4, 32),
    },
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
EXPECTED_FEATURE_SHAPE_GRID = {
    (row_total, feature_count, class_count)
    for (row_total, feature_count), class_count in product(FEATURE_SHAPE_SPECS, CLASS_COUNTS)
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
    assert recipes["tf_rd_010_dagzoo_feature_shape_control_curated_v1"] == {
        "path": "tf_rd_010_dagzoo_feature_shape_control_curated_v1.yaml"
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


def test_tf_rd_010_dagzoo_feature_shape_control_preserves_balanced_latent_target_grid() -> None:
    recipe = load_corpus_recipe("tf_rd_010_dagzoo_feature_shape_control_curated_v1", repo_root=REPO_ROOT).to_dict()

    assert recipe["surface_label"] == "tf_rd_010_dagzoo_feature_shape_control"
    assert recipe["manifest"]["filter_policy"] == "accepted_only"
    assert recipe["provenance_labels"]["corpus_recipe_version"] == "feature_shape_v1"
    assert recipe["provenance_labels"]["target_derivation"] == "tabiclv2_latent_node"
    assert recipe["review_summary"]["grid_family"] == "paired_feature_shape_rows_features_classes_v1"
    assert recipe["review_summary"]["target_derivation"] == "tabiclv2_latent_node"
    assert recipe["review_summary"]["config_refs"] == ["configs/benchmark_cuda_h100_large_shape.yaml"]
    assert recipe["review_summary"]["shape_pair_count"] == 16
    assert {
        (pair["row_total"], pair["feature_count"], pair["cell_budget_label"], pair["cell_count"])
        for pair in recipe["review_summary"]["shape_pairs"]
    } == {
        (row_total, feature_count, spec["cell_budget_label"], spec["cell_count"])
        for (row_total, feature_count), spec in FEATURE_SHAPE_SPECS.items()
    }

    invocations = recipe["invocations"]
    assert len(invocations) == 144
    assert sum(invocation["num_datasets"] for invocation in invocations) == 73728

    grid: set[tuple[int, int, int]] = set()
    classes_by_shape: dict[tuple[int, int], set[int]] = {}
    for invocation in invocations:
        assert invocation["base_config_ref"] == "configs/benchmark_cuda_h100_large_shape.yaml"
        assert invocation["num_datasets"] == 512
        dataset = invocation["config_overrides"]["dataset"]
        graph = invocation["config_overrides"]["graph"]
        retry_policy = invocation["config_overrides"]["filter"]

        feature_count = dataset["n_features_min"]
        class_count = dataset["n_classes_min"]
        row_total = int(dataset["n_train"]) + int(dataset["n_test"])
        shape = (row_total, feature_count)
        shape_spec = FEATURE_SHAPE_SPECS[shape]

        assert dataset["task"] == "classification"
        assert dataset["n_features_max"] == feature_count
        assert dataset["n_classes_max"] == class_count
        assert dataset["categorical_ratio_min"] == 0.0
        assert dataset["categorical_ratio_max"] == 1.0
        assert dataset["max_categorical_cardinality"] == 32
        assert "target_parent_prior" not in dataset
        assert "target_parent_count_min" not in dataset
        assert "target_parent_count_max" not in dataset

        assert graph == {
            "n_nodes_min": shape_spec["graph"][0],
            "n_nodes_max": shape_spec["graph"][1],
        }
        assert retry_policy == {"max_attempts": 256}
        assert (dataset["n_train"], dataset["n_test"]) == (shape_spec["n_train"], shape_spec["n_test"])
        grid.add((row_total, feature_count, class_count))
        classes_by_shape.setdefault(shape, set()).add(class_count)

    assert grid == EXPECTED_FEATURE_SHAPE_GRID
    assert set(classes_by_shape) == set(FEATURE_SHAPE_SPECS)
    assert all(classes == CLASS_COUNTS for classes in classes_by_shape.values())


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
