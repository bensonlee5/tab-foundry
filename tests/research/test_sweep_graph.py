from __future__ import annotations

import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import pytest
import torch

import tab_foundry.research.sweep.graph as graph_module
from tab_foundry.model.spec import model_build_spec_from_mappings
from tab_foundry.research.sweep.graph import GraphPaths
from tab_foundry.research.system_delta import load_system_delta_queue


REPO_ROOT = Path(__file__).resolve().parents[2]


def _copy_reference_workspace(tmp_path: Path) -> tuple[Path, Path]:
    reference_root = tmp_path / "reference"
    sweeps_root = reference_root / "system_delta_sweeps"
    source_sweeps_root = REPO_ROOT / "reference" / "system_delta_sweeps"
    sweeps_root.mkdir(parents=True, exist_ok=True)
    (reference_root / "system_delta_catalog.yaml").write_text(
        (REPO_ROOT / "reference" / "system_delta_catalog.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (sweeps_root / "index.yaml").write_text(
        (source_sweeps_root / "index.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    for source_dir in sorted(source_sweeps_root.iterdir()):
        if source_dir.name == "index.yaml" or not source_dir.is_dir():
            continue
        shutil.copytree(source_dir, sweeps_root / source_dir.name)
    return reference_root, sweeps_root


def test_resolve_queue_row_model_spec_matches_selected_sweep_surface() -> None:
    queue = load_system_delta_queue(
        sweep_id="input_norm_followup",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    row = next(row for row in queue["rows"] if int(row["order"]) == 7)

    spec = graph_module.resolve_queue_row_model_spec(row)

    assert spec.arch == "tabfoundry_staged"
    assert spec.stage == "nano_exact"
    assert spec.stage_label == "dpnb_input_norm_anchor_replay_batch64_sqrt"
    assert spec.input_normalization == "train_zscore_clip"
    assert spec.module_overrides == {
        "table_block_style": "prenorm",
        "allow_test_self_attention": False,
        "row_pool": "row_cls",
    }
    assert spec.tfrow_cls_tokens == 2


def test_resolve_anchor_model_spec_prefers_matching_completed_queue_row() -> None:
    queue = load_system_delta_queue(
        sweep_id="input_norm_followup",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )

    spec, metadata = graph_module.resolve_anchor_model_spec(
        queue=queue,
        registry_path=REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json",
    )

    assert metadata["source"] == "queue_row"
    assert metadata["run_id"] == "sd_input_norm_followup_07_dpnb_input_norm_anchor_replay_batch64_sqrt_v2"
    assert metadata["order"] == 7
    assert spec.stage_label == "dpnb_input_norm_anchor_replay_batch64_sqrt"


def test_resolve_anchor_model_spec_uses_training_surface_record_when_queue_row_missing(tmp_path: Path) -> None:
    spec = model_build_spec_from_mappings(
        task="classification",
        primary={
            "arch": "tabfoundry_staged",
            "stage": "nano_exact",
            "stage_label": "registry_record_anchor",
            "input_normalization": "train_zscore_clip",
            "many_class_base": 2,
            "tficl_n_heads": 4,
            "tficl_n_layers": 2,
            "head_hidden_dim": 128,
        },
    )
    record_path = tmp_path / "training_surface_record.json"
    record_path.write_text(
        json.dumps({"model": {"build_spec": spec.to_dict()}}, indent=2),
        encoding="utf-8",
    )
    registry_payload = {
        "runs": {
            "anchor_from_record": {
                "artifacts": {
                    "training_surface_record_path": str(record_path),
                    "best_checkpoint_path": None,
                }
            }
        }
    }
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(graph_module, "load_benchmark_run_registry", lambda _path: registry_payload)
    monkeypatch.setattr(graph_module, "resolve_registry_path_value", lambda value: Path(value))
    try:
        resolved_spec, metadata = graph_module.resolve_anchor_model_spec(
            queue={"anchor_run_id": "anchor_from_record", "rows": []},
            registry_path=tmp_path / "registry.json",
        )
    finally:
        monkeypatch.undo()

    assert metadata["source"] == "training_surface_record"
    assert resolved_spec.stage_label == "registry_record_anchor"


def test_resolve_anchor_model_spec_falls_back_to_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "config": {
                "task": "classification",
                "model": {
                    "arch": "tabfoundry_staged",
                    "stage": "nano_exact",
                    "stage_label": "checkpoint_anchor",
                    "input_normalization": "train_zscore_clip",
                    "many_class_base": 2,
                    "tficl_n_heads": 4,
                    "tficl_n_layers": 2,
                    "head_hidden_dim": 128,
                },
            },
            "model": {},
        },
        checkpoint_path,
    )
    registry_payload = {
        "runs": {
            "anchor_from_checkpoint": {
                "artifacts": {
                    "training_surface_record_path": str(tmp_path / "missing_training_surface_record.json"),
                    "best_checkpoint_path": str(checkpoint_path),
                }
            }
        }
    }
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(graph_module, "load_benchmark_run_registry", lambda _path: registry_payload)
    monkeypatch.setattr(graph_module, "resolve_registry_path_value", lambda value: Path(value))
    try:
        resolved_spec, metadata = graph_module.resolve_anchor_model_spec(
            queue={"anchor_run_id": "anchor_from_checkpoint", "rows": []},
            registry_path=tmp_path / "registry.json",
        )
    finally:
        monkeypatch.undo()

    assert metadata["source"] == "checkpoint"
    assert resolved_spec.stage_label == "checkpoint_anchor"


def test_resolve_anchor_model_spec_errors_when_all_sources_are_missing(tmp_path: Path) -> None:
    registry_payload = {
        "runs": {
            "missing_anchor": {
                "artifacts": {
                    "training_surface_record_path": str(tmp_path / "missing.json"),
                    "best_checkpoint_path": str(tmp_path / "missing.pt"),
                }
            }
        }
    }
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(graph_module, "load_benchmark_run_registry", lambda _path: registry_payload)
    monkeypatch.setattr(graph_module, "resolve_registry_path_value", lambda value: Path(value))
    try:
        with pytest.raises(RuntimeError, match="unable to resolve anchor model spec"):
            _ = graph_module.resolve_anchor_model_spec(
                queue={"anchor_run_id": "missing_anchor", "rows": []},
                registry_path=tmp_path / "registry.json",
            )
    finally:
        monkeypatch.undo()


@pytest.mark.parametrize(
    ("argv", "expected"),
    [
        (["--anchor"], {"anchor": True, "all_rows": False, "orders": [], "delta_refs": []}),
        (["--all-rows"], {"anchor": False, "all_rows": True, "orders": [], "delta_refs": []}),
        (["--order", "7"], {"anchor": False, "all_rows": False, "orders": [7], "delta_refs": []}),
        (
            ["--delta-ref", "dpnb_input_norm_anchor_replay_batch64_sqrt"],
            {
                "anchor": False,
                "all_rows": False,
                "orders": [],
                "delta_refs": ["dpnb_input_norm_anchor_replay_batch64_sqrt"],
            },
        ),
    ],
)
def test_graph_main_parses_cli_selectors(
    argv: list[str],
    expected: dict[str, object],
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_render_sweep_graphs(**kwargs):
        captured.update(kwargs)
        return {"sweep_id": "cuda_stability_followup", "graphs": [], "index_path": str(tmp_path / "index.md")}

    monkeypatch.setattr(graph_module, "render_sweep_graphs", _fake_render_sweep_graphs)

    exit_code = graph_module.main([*argv, "--out-dir", str(tmp_path)])

    assert exit_code == 0
    assert captured["anchor"] == expected["anchor"]
    assert captured["all_rows"] == expected["all_rows"]
    assert captured["orders"] == expected["orders"]
    assert captured["delta_refs"] == expected["delta_refs"]


def test_render_sweep_graphs_rejects_invalid_selector_combinations(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="cannot combine --all-rows"):
        _ = graph_module.render_sweep_graphs(
            sweep_id="input_norm_followup",
            all_rows=True,
            orders=[7],
            out_dir=tmp_path,
        )


def test_render_sweep_graphs_rejects_unknown_delta_ref(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="unknown sweep delta_ref"):
        _ = graph_module.render_sweep_graphs(
            sweep_id="input_norm_followup",
            delta_refs=["missing_delta"],
            out_dir=tmp_path,
        )


def test_render_sweep_graphs_writes_svg_and_index_without_mutating_sweep_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    reference_root, sweeps_root = _copy_reference_workspace(tmp_path)
    sweep_yaml_path = sweeps_root / "input_norm_followup" / "sweep.yaml"
    queue_yaml_path = sweeps_root / "input_norm_followup" / "queue.yaml"
    sweep_before = sweep_yaml_path.read_text(encoding="utf-8")
    queue_before = queue_yaml_path.read_text(encoding="utf-8")

    class _FakeVisualGraph:
        def pipe(self, *, format: str) -> bytes:
            assert format == "svg"
            return b"<svg><!-- fake graph --></svg>"

    class _FakeGraph:
        def __init__(self) -> None:
            self.visual_graph = _FakeVisualGraph()

    def _fake_draw_graph(*args, **kwargs):
        assert args
        return _FakeGraph()

    monkeypatch.setattr(graph_module.shutil, "which", lambda name: "/usr/bin/dot" if name == "dot" else None)
    monkeypatch.setitem(sys.modules, "torchview", SimpleNamespace(draw_graph=_fake_draw_graph))

    out_dir = tmp_path / "graphs"
    result = graph_module.render_sweep_graphs(
        sweep_id="input_norm_followup",
        anchor=True,
        orders=[7],
        out_dir=out_dir,
        paths=GraphPaths(
            index_path=sweeps_root / "index.yaml",
            catalog_path=reference_root / "system_delta_catalog.yaml",
            sweeps_root=sweeps_root,
            registry_path=REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json",
        ),
    )

    graph_paths = [Path(item["path"]) for item in result["graphs"]]
    assert graph_paths
    assert all(path.exists() for path in graph_paths)
    assert all("<svg" in path.read_text(encoding="utf-8") for path in graph_paths)
    index_path = Path(str(result["index_path"]))
    assert index_path.exists()
    index_text = index_path.read_text(encoding="utf-8")
    assert "Sweep Architecture Graphs" in index_text
    assert "Anchor" in index_text
    assert "row:07:dpnb_input_norm_anchor_replay_batch64_sqrt" in index_text
    assert sweep_yaml_path.read_text(encoding="utf-8") == sweep_before
    assert queue_yaml_path.read_text(encoding="utf-8") == queue_before
