from __future__ import annotations

import pytest

from tab_foundry.model.spec import model_build_spec_from_mappings
from tab_foundry.training.loss_surface import resolve_training_loss_surface


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
