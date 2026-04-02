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
POSTERIOR_PREDICTIVE_FACTORIZATION = "independent_p_x_complete_and_p_y_given_x_complete"
TARGET_PARENT_PRIOR = "near_max_mixture"
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


def _load_recipe(recipe_id: str) -> dict[str, Any]:
    return load_corpus_recipe(recipe_id, repo_root=REPO_ROOT).to_dict()


def _grid_signature(recipe: dict[str, Any], *, missingness: str | None) -> set[tuple[int, int, int]]:
    invocations = recipe["invocations"]
    assert isinstance(invocations, list)
    assert len(invocations) == 144
    assert sum(invocation["num_datasets"] for invocation in invocations) == 159984

    grid: set[tuple[int, int, int]] = set()
    for invocation in invocations:
        assert invocation["base_config_ref"] == "configs/default.yaml"
        assert invocation["num_datasets"] == 1111
        assert invocation["device"] == "cpu"
        assert invocation["hardware_policy"] == "none"
        dataset = invocation["config_overrides"]["dataset"]
        graph = invocation["config_overrides"]["graph"]
        retry_policy = invocation["config_overrides"]["filter"]

        feature_count = dataset["n_features_min"]
        assert feature_count in FEATURE_GRAPH_BANDS
        assert dataset["n_features_max"] == feature_count
        assert dataset["task"] == "classification"
        assert dataset["categorical_ratio_min"] == 0.0
        assert dataset["categorical_ratio_max"] == 1.0
        assert dataset["max_categorical_cardinality"] == 12
        assert "target_parent_prior" not in dataset
        assert "target_parent_count_min" not in dataset
        assert "target_parent_count_max" not in dataset
        assert "target_parent_near_max_band_min_fraction" not in dataset
        assert "target_parent_below_sqrt_prob" not in dataset
        assert "target_parent_midrange_prob" not in dataset
        assert "diagnostics" not in invocation["config_overrides"]

        class_count = dataset["n_classes_min"]
        assert class_count in CLASS_COUNTS
        assert dataset["n_classes_max"] == class_count

        row_total = int(dataset["n_train"]) + int(dataset["n_test"])
        assert row_total in ROW_SPECS
        assert (dataset["n_train"], dataset["n_test"]) == ROW_SPECS[row_total]
        assert row_total <= 1024

        expected_nodes_min, expected_nodes_max = FEATURE_GRAPH_BANDS[feature_count]
        assert graph == {
            "n_nodes_min": expected_nodes_min,
            "n_nodes_max": expected_nodes_max,
        }
        assert retry_policy == {"max_attempts": 256}
        assert invocation["invocation_id"] == f"r{row_total:04d}_f{feature_count:02d}_c{class_count:02d}"

        if missingness is None:
            assert invocation.get("missing_rate") is None
            assert invocation.get("missing_mechanism") is None
        elif missingness == "mcar":
            assert invocation["missing_rate"] == 0.25
            assert invocation["missing_mechanism"] == "mcar"
        elif missingness == "mar":
            assert invocation["missing_rate"] == 0.25
            assert invocation["missing_mechanism"] == "mar"
            assert invocation["missing_mar_observed_fraction"] == 0.6
            assert invocation["missing_mar_logit_scale"] == 1.4
        elif missingness == "mnar":
            assert invocation["missing_rate"] == 0.25
            assert invocation["missing_mechanism"] == "mnar"
            assert invocation["missing_mnar_logit_scale"] == 1.4
        else:
            raise AssertionError(f"unexpected missingness mode {missingness!r}")

        grid.add((row_total, feature_count, class_count))

    assert grid == EXPECTED_GRID
    return grid


def test_tf_rd_010_dagzoo_recipe_2500_v3_is_registered() -> None:
    index = _load_yaml(RECIPE_ROOT / "index.yaml")
    recipes = index["recipes"]
    assert recipes["tf_rd_010_dagzoo_medium_control_v3"] == {
        "path": "tf_rd_010_dagzoo_medium_control_v3.yaml"
    }
    assert recipes["tf_rd_010_missingness_mcar_v3"] == {
        "path": "tf_rd_010_missingness_mcar_v3.yaml"
    }
    assert recipes["tf_rd_010_missingness_mar_v3"] == {
        "path": "tf_rd_010_missingness_mar_v3.yaml"
    }
    assert recipes["tf_rd_010_missingness_mnar_v3"] == {
        "path": "tf_rd_010_missingness_mnar_v3.yaml"
    }
    assert recipes["tf_rd_010_factorized_canary_v1"] == {
        "path": "tf_rd_010_factorized_canary_v1.yaml"
    }


def test_tf_rd_010_dagzoo_recipe_2500_v3_preserves_the_balanced_front_shape_with_factorization_metadata() -> None:
    control_summary = _load_yaml(RECIPE_ROOT / "tf_rd_010_dagzoo_medium_control_v3.yaml")
    mcar_summary = _load_yaml(RECIPE_ROOT / "tf_rd_010_missingness_mcar_v3.yaml")
    mar_summary = _load_yaml(RECIPE_ROOT / "tf_rd_010_missingness_mar_v3.yaml")
    mnar_summary = _load_yaml(RECIPE_ROOT / "tf_rd_010_missingness_mnar_v3.yaml")
    control = _load_recipe("tf_rd_010_dagzoo_medium_control_v3")
    mcar = _load_recipe("tf_rd_010_missingness_mcar_v3")
    mar = _load_recipe("tf_rd_010_missingness_mar_v3")
    mnar = _load_recipe("tf_rd_010_missingness_mnar_v3")

    for recipe, summary in (
        (control, control_summary),
        (mcar, mcar_summary),
        (mar, mar_summary),
        (mnar, mnar_summary),
    ):
        labels = recipe["provenance_labels"]
        assert labels["corpus_recipe_version"] == "v3"
        assert labels["synthetic_epoch_regime"] == "one_epoch_159984_records_2500_steps"
        assert labels["balanced_front_shape"] == "inherited_from_v1"
        assert labels["manifest_record_count"] == 159984
        assert labels["per_invocation_num_datasets"] == 1111
        assert labels["posterior_predictive_factorization"] == POSTERIOR_PREDICTIVE_FACTORIZATION
        assert labels["teacher_conditional_export"] is True
        assert labels["metric_definition"] == "label-target log loss per test cell"
        assert labels["target_parent_prior"] == TARGET_PARENT_PRIOR
        assert labels["target_parent_mode"] == "max"
        assert labels["target_parent_near_max_band_min_fraction"] == 0.75
        assert labels["target_parent_below_sqrt_prob"] == 0.05
        assert labels["target_parent_midrange_prob"] == 0.20
        assert "equation-(1)" in recipe["description"]
        assert "teacher-conditional export" in recipe["description"]
        assert summary["kind"] == "dagzoo_python_generated"
        assert summary["review_summary"]["invocation_count"] == 144
        assert summary["review_summary"]["manifest_record_count"] == 159984
        assert summary["review_summary"]["posterior_predictive_factorization"] == POSTERIOR_PREDICTIVE_FACTORIZATION
        assert summary["review_summary"]["target_parent_prior"] == TARGET_PARENT_PRIOR
        assert summary["review_summary"]["target_parent_mode"] == "max"
        assert summary["review_summary"]["target_parent_near_max_band_min_fraction"] == 0.75
        assert summary["review_summary"]["target_parent_below_sqrt_prob"] == 0.05
        assert summary["review_summary"]["target_parent_midrange_prob"] == 0.20

    assert control["surface_label"] == "tf_rd_010_dagzoo_medium_control"
    assert mcar["surface_label"] == "tf_rd_010_missingness_mcar"
    assert mar["surface_label"] == "tf_rd_010_missingness_mar"
    assert mnar["surface_label"] == "tf_rd_010_missingness_mnar"

    control_grid = _grid_signature(control, missingness=None)
    mcar_grid = _grid_signature(mcar, missingness="mcar")
    mar_grid = _grid_signature(mar, missingness="mar")
    mnar_grid = _grid_signature(mnar, missingness="mnar")

    assert control_grid == mcar_grid == mar_grid == mnar_grid == EXPECTED_GRID


def test_tf_rd_010_factorized_canary_recipe_tracks_the_row_ladder_only() -> None:
    canary_summary = _load_yaml(RECIPE_ROOT / "tf_rd_010_factorized_canary_v1.yaml")
    canary = _load_recipe("tf_rd_010_factorized_canary_v1")

    assert canary["surface_label"] == "tf_rd_010_factorized_canary"
    assert canary["provenance_labels"]["corpus_recipe_version"] == "v1"
    assert canary["provenance_labels"]["posterior_predictive_factorization"] == POSTERIOR_PREDICTIVE_FACTORIZATION
    assert canary["provenance_labels"]["teacher_conditional_export"] is True
    assert canary["provenance_labels"]["target_parent_prior"] == TARGET_PARENT_PRIOR
    assert canary["provenance_labels"]["target_parent_mode"] == "max"
    assert canary["review_summary"]["grid_family"] == "factorized_canary_rows_only_v1"
    assert canary["review_summary"]["target_parent_prior"] == TARGET_PARENT_PRIOR
    assert canary["review_summary"]["target_parent_mode"] == "max"
    assert canary_summary["review_summary"]["manifest_record_count"] == 128

    invocations = canary["invocations"]
    assert len(invocations) == 4
    assert sum(invocation["num_datasets"] for invocation in invocations) == 128
    seen_rows: set[int] = set()
    for invocation in invocations:
        dataset = invocation["config_overrides"]["dataset"]
        row_total = int(dataset["n_train"]) + int(dataset["n_test"])
        assert row_total in ROW_SPECS
        assert dataset["n_features_min"] == dataset["n_features_max"] == 6
        assert dataset["n_classes_min"] == dataset["n_classes_max"] == 2
        assert dataset["categorical_ratio_min"] == dataset["categorical_ratio_max"] == 0.0
        assert "target_parent_prior" not in dataset
        assert "target_parent_count_min" not in dataset
        assert "target_parent_count_max" not in dataset
        assert "target_parent_near_max_band_min_fraction" not in dataset
        assert "target_parent_below_sqrt_prob" not in dataset
        assert "target_parent_midrange_prob" not in dataset
        assert "diagnostics" not in invocation["config_overrides"]
        seen_rows.add(row_total)
    assert seen_rows == set(ROW_SPECS)
