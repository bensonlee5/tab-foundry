from __future__ import annotations

from pathlib import Path
from typing import Any

from omegaconf import OmegaConf

from tab_foundry.research.system_delta import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]
SWEEP_ID = "tf_rd_013_dagzoo_size_ladder_v1"


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    assert isinstance(payload, dict)
    return payload


def _row_by_ref(queue: dict[str, Any], delta_ref: str) -> dict[str, Any]:
    rows = queue["rows"]
    assert isinstance(rows, list)
    return next(row for row in rows if row["delta_ref"] == delta_ref)


def test_tf_rd_013_size_ladder_queue_uses_corpus_refs() -> None:
    sweep_root = REPO_ROOT / "reference" / "system_delta_sweeps" / SWEEP_ID
    sweep = _load_yaml(sweep_root / "sweep.yaml")
    queue = _load_yaml(sweep_root / "queue.yaml")
    catalog = _load_yaml(REPO_ROOT / "reference" / "system_delta_catalog.yaml")

    notes = sweep["anchor_surface"]["notes"]
    assert isinstance(notes, list)
    assert any("corpus recipes" in note for note in notes)

    row1 = _row_by_ref(queue, "delta_training_current_corpus_uncapped")
    assert row1["data"] == {
        "surface_label": "anchor_manifest_default",
        "corpus_ref": "tf_rd_013_current_corpus_default_v1",
    }

    for delta_ref, expected_ref in (
        ("delta_data_manifest_root_dagzoo_shape_aware_size_small", "tf_rd_013_dagzoo_shape_aware_size_small_v1"),
        ("delta_data_manifest_root_dagzoo_shape_aware_size_medium", "tf_rd_013_dagzoo_shape_aware_size_medium_v1"),
        ("delta_data_manifest_root_dagzoo_shape_aware_size_large", "tf_rd_013_dagzoo_shape_aware_size_large_v1"),
    ):
        row = _row_by_ref(queue, delta_ref)
        assert row["data"]["corpus_ref"] == expected_ref
        assert "manifest_path" not in row["data"]
        assert "dagzoo_provenance" not in row["data"]

    current_default = catalog["deltas"]["delta_training_current_corpus_uncapped"]["default_effective_surface"]["data"]
    assert current_default == {
        "surface_label": "anchor_manifest_default",
        "surface_overrides": {
            "source": "manifest",
            "corpus_ref": "tf_rd_013_current_corpus_default_v1",
        },
    }


def test_tf_rd_013_size_ladder_materialized_queue_preserves_corpus_refs() -> None:
    queue = load_system_delta_queue(
        sweep_id=SWEEP_ID,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )

    row1 = next(row for row in queue["rows"] if row["delta_id"] == "delta_training_current_corpus_uncapped")
    assert row1["data"]["surface_label"] == "anchor_manifest_default"
    assert row1["data"]["corpus_ref"] == "tf_rd_013_current_corpus_default_v1"
