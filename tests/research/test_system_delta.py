from __future__ import annotations

from pathlib import Path

from tab_foundry.research.system_delta import (
    load_system_delta_queue,
    next_ready_row,
    render_system_delta_matrix,
    validate_system_delta_queue,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_system_delta_queue_seeds_anchor_only_rows() -> None:
    queue = load_system_delta_queue(REPO_ROOT / "reference" / "system_delta_queue.yaml")

    assert queue["anchor_run_id"] == "01_nano_exact_md_prior_parity_fix_binary_medium_v1"
    rows = queue["rows"]
    assert len(rows) >= 14
    assert next_ready_row(queue)["delta_id"] == "delta_label_token"


def test_system_delta_matrix_render_includes_anchor_and_row_cls_details() -> None:
    queue = load_system_delta_queue(REPO_ROOT / "reference" / "system_delta_queue.yaml")
    matrix = render_system_delta_matrix(
        queue,
        registry_path=REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json",
    )

    assert "01_nano_exact_md_prior_parity_fix_binary_medium_v1" in matrix
    assert "delta_row_cls_pool" in matrix
    assert "tfrow_n_heads" in matrix
    assert "anchor_only" in matrix


def test_system_delta_queue_validation_passes_when_no_rows_are_completed() -> None:
    queue = load_system_delta_queue(REPO_ROOT / "reference" / "system_delta_queue.yaml")

    assert validate_system_delta_queue(
        queue,
        registry_path=REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json",
    ) == []


def test_each_row_declares_exactly_one_override_family() -> None:
    queue = load_system_delta_queue(REPO_ROOT / "reference" / "system_delta_queue.yaml")

    for row in queue["rows"]:
        declared = []
        if row["model"].get("module_overrides"):
            declared.append("model")
        if row["data"].get("surface_overrides"):
            declared.append("data")
        if row["preprocessing"].get("overrides"):
            declared.append("preprocessing")
        assert len(declared) <= 1
        if row["status"] != "blocked_on_policy_impl":
            assert row["dimension_family"] in declared or not declared
