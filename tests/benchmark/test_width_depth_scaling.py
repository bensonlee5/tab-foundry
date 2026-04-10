from __future__ import annotations

import pytest

from tab_foundry.bench.width_depth_scaling import (
    WidthDepthParameterPoint,
    fit_affine_width_depth_parameter_bridge,
    fit_linear,
    log_space_parameter_targets,
    round_to_width_rung,
)


def _tf_rd_009_parameter_points() -> tuple[WidthDepthParameterPoint, ...]:
    return (
        WidthDepthParameterPoint(d_icl=60, layers=2, total_params=646970, row_label="60x2"),
        WidthDepthParameterPoint(d_icl=96, layers=2, total_params=1618286, row_label="96x2"),
        WidthDepthParameterPoint(d_icl=128, layers=2, total_params=2849422, row_label="128x2"),
        WidthDepthParameterPoint(d_icl=88, layers=1, total_params=986886, row_label="88x1"),
        WidthDepthParameterPoint(d_icl=104, layers=3, total_params=2419862, row_label="104x3"),
        WidthDepthParameterPoint(d_icl=112, layers=4, total_params=3410046, row_label="112x4"),
        WidthDepthParameterPoint(d_icl=128, layers=5, total_params=5234830, row_label="128x5"),
        WidthDepthParameterPoint(d_icl=144, layers=6, total_params=7615262, row_label="144x6"),
    )


def test_affine_width_depth_parameter_bridge_matches_tf_rd_009_canonical_fit() -> None:
    fit = fit_affine_width_depth_parameter_bridge(_tf_rd_009_parameter_points())

    assert fit.intercept == pytest.approx(29966.47, abs=0.1)
    assert fit.d_squared_coefficient == pytest.approx(75.38, abs=0.01)
    assert fit.layered_d_squared_coefficient == pytest.approx(48.43, abs=0.01)
    assert fit.expression() == (
        "P_local(d, L) ≈ 29966.47 + 75.38 * d^2 + 48.43 * L * d^2"
    )


def test_tf_rd_009_width_depth_targets_solve_to_expected_raw_and_rounded_widths() -> None:
    fit = fit_affine_width_depth_parameter_bridge(_tf_rd_009_parameter_points())
    vram_fit = fit_linear(
        (
            (646970.0, 8.0546875),
            (1618286.0, 10.1796875),
            (2849422.0, 13.23046875),
        )
    )

    lower_raw = fit.solve_width_for_target_params(layers=1, target_params=646970.0)
    upper_raw = fit.solve_width_for_target_params(layers=3, target_params=2849422.0)
    ceiling_target_params = vram_fit.solve_x(32.5)
    ceiling_raw = fit.solve_width_for_target_params(layers=6, target_params=ceiling_target_params)

    assert lower_raw == pytest.approx(70.59, abs=0.05)
    assert upper_raw == pytest.approx(113.03, abs=0.05)
    assert ceiling_raw == pytest.approx(173.52, abs=0.1)

    assert round_to_width_rung(lower_raw) == 72
    assert round_to_width_rung(upper_raw) == 112
    assert round_to_width_rung(ceiling_raw) == 176


def test_tf_rd_009_upper_interpolation_targets_remain_log_spaced() -> None:
    fit = fit_affine_width_depth_parameter_bridge(_tf_rd_009_parameter_points())

    upper_seed_params = fit.predict_total_params(d_icl=112, layers=3)
    ceiling_params = fit.predict_total_params(d_icl=176, layers=6)
    interpolation_targets = log_space_parameter_targets(
        start_value=upper_seed_params,
        end_value=ceiling_params,
        count=2,
    )

    assert interpolation_targets[0] == pytest.approx(4464552.30, abs=1.0)
    assert interpolation_targets[1] == pytest.approx(7123513.13, abs=1.0)
    assert round_to_width_rung(
        fit.solve_width_for_target_params(layers=4, target_params=interpolation_targets[0])
    ) == 128
    assert round_to_width_rung(
        fit.solve_width_for_target_params(layers=5, target_params=interpolation_targets[1])
    ) == 152
