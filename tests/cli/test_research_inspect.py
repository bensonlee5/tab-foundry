from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

import tab_foundry.cli.research_inspect as research_inspect_cli_module


@pytest.mark.parametrize("json_mode", [False, True])
def test_research_inspect_run_from_args_emits_expected_output(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    json_mode: bool,
) -> None:
    payload: dict[str, Any] = {
        "queue": {"sweep_id": "mini_sweep"},
        "row": {"order": 1, "delta_id": "delta_row_cls_pool"},
        "target": {
            "resolved": {
                "model": {"parameter_counts": {"total_params": 1, "trainable_params": 1}},
            }
        },
    }
    captured: dict[str, Any] = {}

    def _fake_inspect_sweep_row(**kwargs: Any) -> dict[str, Any]:
        captured["kwargs"] = kwargs
        return payload

    def _fake_render_sweep_row_text(render_payload: dict[str, Any]) -> str:
        captured["render_payload"] = render_payload
        return "Sweep row inspection."

    monkeypatch.setattr(research_inspect_cli_module, "inspect_sweep_row", _fake_inspect_sweep_row)
    monkeypatch.setattr(research_inspect_cli_module, "render_sweep_row_text", _fake_render_sweep_row_text)

    exit_code = research_inspect_cli_module.run_from_args(
        argparse.Namespace(
            order=1,
            sweep_id="mini_sweep",
            json=json_mode,
            index_path="index.yaml",
            catalog_path="catalog.yaml",
            sweeps_root="reference/system_delta_sweeps",
            registry_path="registry.json",
        )
    )

    assert exit_code == 0
    inspect_kwargs = captured["kwargs"]
    assert inspect_kwargs["order"] == 1
    assert inspect_kwargs["sweep_id"] == "mini_sweep"
    assert inspect_kwargs["index_path"] == Path("index.yaml").expanduser().resolve()
    assert inspect_kwargs["catalog_path"] == Path("catalog.yaml").expanduser().resolve()
    assert inspect_kwargs["sweeps_root"] == Path("reference/system_delta_sweeps").expanduser().resolve()
    assert inspect_kwargs["registry_path"] == Path("registry.json").expanduser().resolve()

    output = capsys.readouterr().out.strip()
    if json_mode:
        assert json.loads(output) == payload
        assert "render_payload" not in captured
    else:
        assert output == "Sweep row inspection."
        assert captured["render_payload"] == payload
