from __future__ import annotations

import pytest

from tab_foundry.research.tf_rd_009_width_depth_derivation import (
    derive_tf_rd_009_width_depth_family,
)


def test_tf_rd_009_width_depth_derivation_matches_canonical_family() -> None:
    derivation = derive_tf_rd_009_width_depth_family()

    assert derivation.in_family_row_labels == (
        "72x1",
        "96x2",
        "112x3",
        "128x4",
        "152x5",
        "176x6",
    )
    assert [row.delta_id for row in derivation.queue_rows] == [
        "delta_tf_rd_009_cls_sandwich_dicl72_layers1_v1",
        "delta_tf_rd_009_cls_sandwich_dicl112_layers3_v1",
        "delta_tf_rd_009_cls_sandwich_dicl128_layers4_v1",
        "delta_tf_rd_009_cls_sandwich_dicl152_layers5_v1",
        "delta_tf_rd_009_cls_sandwich_dicl176_layers6_v1",
    ]


def test_tf_rd_009_width_depth_derivation_matches_note_values() -> None:
    derivation = derive_tf_rd_009_width_depth_family()

    assert derivation.lower_seed.raw_d_icl == pytest.approx(70.59, abs=0.05)
    assert derivation.upper_seed.raw_d_icl == pytest.approx(113.03, abs=0.05)
    assert derivation.interpolated_rows[0].raw_d_icl == pytest.approx(128.37, abs=0.1)
    assert derivation.interpolated_rows[1].raw_d_icl == pytest.approx(149.46, abs=0.1)
    assert derivation.ceiling_probe.raw_d_icl == pytest.approx(173.52, abs=0.1)

    assert derivation.lower_seed.predicted_total_params == pytest.approx(671809.30, abs=1.0)
    assert derivation.upper_seed.predicted_total_params == pytest.approx(2798089.49, abs=1.0)
    assert derivation.interpolated_rows[0].predicted_total_params == pytest.approx(4438957.75, abs=1.0)
    assert derivation.interpolated_rows[1].predicted_total_params == pytest.approx(7366269.02, abs=1.0)
    assert derivation.ceiling_probe.predicted_total_params == pytest.approx(11366075.68, abs=1.0)

    assert derivation.lower_seed.predicted_reserved_vram_gb == pytest.approx(8.05, abs=0.01)
    assert derivation.upper_seed.predicted_reserved_vram_gb == pytest.approx(13.06, abs=0.01)
    assert derivation.interpolated_rows[0].predicted_reserved_vram_gb == pytest.approx(16.93, abs=0.01)
    assert derivation.interpolated_rows[1].predicted_reserved_vram_gb == pytest.approx(23.82, abs=0.01)
    assert derivation.ceiling_probe.predicted_reserved_vram_gb == pytest.approx(33.25, abs=0.01)
