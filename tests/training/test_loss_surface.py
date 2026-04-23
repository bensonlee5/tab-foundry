from __future__ import annotations

import pytest

from tab_foundry.model.spec import model_build_spec_from_mappings
from tab_foundry.training.loss_surface import (
    resolve_classification_z_loss_coeff,
    resolve_training_loss_surface,
)


def test_classification_z_loss_coeff_defaults_to_zero() -> None:
    assert resolve_classification_z_loss_coeff({}) == pytest.approx(0.0)
    assert resolve_classification_z_loss_coeff(None) == pytest.approx(0.0)


@pytest.mark.parametrize("value", (-1.0e-4, float("inf"), float("nan"), "not-a-float"))
def test_classification_z_loss_coeff_rejects_invalid_values(value: object) -> None:
    with pytest.raises(ValueError, match="classification_z_loss_coeff"):
        _ = resolve_classification_z_loss_coeff({"classification_z_loss_coeff": value})


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
