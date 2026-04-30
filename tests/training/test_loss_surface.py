from __future__ import annotations

import pytest

from tab_foundry.model.spec import model_build_spec_from_mappings
from tab_foundry.training.loss_surface import (
    resolve_classification_z_loss_coeff,
    resolve_moe_load_balance_loss_coeff,
    resolve_moe_load_balance_loss_final_coeff,
    resolve_moe_load_balance_loss_schedule,
    resolve_moe_router_z_loss_coeff,
    resolve_training_loss_surface,
)
from tab_foundry.training.prior.loop import _scheduled_moe_load_balance_loss_coeff


def test_classification_z_loss_coeff_defaults_to_zero() -> None:
    assert resolve_classification_z_loss_coeff({}) == pytest.approx(0.0)
    assert resolve_classification_z_loss_coeff(None) == pytest.approx(0.0)
    assert resolve_moe_load_balance_loss_coeff({}) == pytest.approx(0.0)
    assert resolve_moe_load_balance_loss_schedule({}) == "constant"
    assert resolve_moe_load_balance_loss_final_coeff({}) is None
    assert resolve_moe_router_z_loss_coeff(None) == pytest.approx(0.0)


@pytest.mark.parametrize("value", (-1.0e-4, float("inf"), float("nan"), "not-a-float"))
def test_classification_z_loss_coeff_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="classification_z_loss_coeff"):
        _ = resolve_classification_z_loss_coeff({"classification_z_loss_coeff": value})


@pytest.mark.parametrize("value", (-1.0e-4, float("inf"), float("nan"), "not-a-float"))
def test_moe_aux_loss_coeffs_reject_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="moe_load_balance_loss_coeff"):
        _ = resolve_moe_load_balance_loss_coeff({"moe_load_balance_loss_coeff": value})
    with pytest.raises(ValueError, match="moe_load_balance_loss_final_coeff"):
        _ = resolve_moe_load_balance_loss_final_coeff(
            {"moe_load_balance_loss_final_coeff": value}
        )
    with pytest.raises(ValueError, match="moe_router_z_loss_coeff"):
        _ = resolve_moe_router_z_loss_coeff({"moe_router_z_loss_coeff": value})


def test_moe_load_balance_loss_schedule_resolves_supported_values() -> None:
    assert (
        resolve_moe_load_balance_loss_schedule(
            {"moe_load_balance_loss_schedule": "warmup_decay"}
        )
        == "warmup_decay"
    )
    assert resolve_moe_load_balance_loss_final_coeff(
        {"moe_load_balance_loss_final_coeff": "0.001"}
    ) == pytest.approx(0.001)

    with pytest.raises(ValueError, match="moe_load_balance_loss_schedule"):
        _ = resolve_moe_load_balance_loss_schedule(
            {"moe_load_balance_loss_schedule": "cosine"}
        )


def test_scheduled_moe_load_balance_loss_coeff_decays_after_warmup() -> None:
    assert _scheduled_moe_load_balance_loss_coeff(
        base_coeff=0.01,
        final_coeff=0.001,
        schedule="constant",
        step=5000,
        max_steps=5000,
    ) == pytest.approx(0.01)
    assert _scheduled_moe_load_balance_loss_coeff(
        base_coeff=0.01,
        final_coeff=0.001,
        schedule="linear_decay",
        step=5000,
        max_steps=5000,
    ) == pytest.approx(0.001)
    assert _scheduled_moe_load_balance_loss_coeff(
        base_coeff=0.01,
        final_coeff=0.001,
        schedule="warmup_decay",
        step=1,
        max_steps=5000,
    ) == pytest.approx(0.01)
    assert _scheduled_moe_load_balance_loss_coeff(
        base_coeff=0.01,
        final_coeff=0.001,
        schedule="warmup_decay",
        step=5000,
        max_steps=5000,
    ) == pytest.approx(0.001)


def test_explicit_cell_bpc_resolution_warns_but_still_resolves() -> None:
    model_spec = model_build_spec_from_mappings(
        task="classification",
        primary={"arch": "tabfoundry_sandwich"},
    )

    with pytest.warns(FutureWarning, match="cell_bpc"):
        resolved = resolve_training_loss_surface(
            {"loss_surface": "cell_bpc"},
            model_spec=model_spec,
            backend="manifest",
        )

    assert resolved == "cell_bpc"
