from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import shutil

from omegaconf import OmegaConf

from tab_foundry.research.system_delta import (
    create_sweep,
    load_system_delta_catalog,
    load_system_delta_index,
    load_system_delta_queue,
    load_system_delta_queue_instance,
    load_system_delta_sweep,
    next_ready_row,
    render_system_delta_matrix,
    validate_system_delta_queue,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_active_sweep_materializes_binary_md_v1() -> None:
    index = load_system_delta_index(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")
    sweep = load_system_delta_sweep(
        "binary_md_v1",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
    )
    queue = load_system_delta_queue(
        sweep_id="binary_md_v1",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )

    assert index["active_sweep_id"] == "binary_md_v1"
    assert sweep["sweep_id"] == "binary_md_v1"
    assert queue["sweep_id"] == "binary_md_v1"
    assert queue["generated_from_sweep_id"] == "binary_md_v1"
    expected_next_ready = next(row for row in queue["rows"] if row["status"] == "ready")
    assert next_ready_row(queue) == expected_next_ready


def test_catalog_and_canonical_queue_are_split() -> None:
    catalog = load_system_delta_catalog(REPO_ROOT / "reference" / "system_delta_catalog.yaml")
    queue_instance = load_system_delta_queue_instance(
        "binary_md_v1",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
    )

    assert "delta_label_token" in catalog["deltas"]
    assert queue_instance["sweep_id"] == "binary_md_v1"
    assert queue_instance["rows"][0]["delta_ref"] == "delta_label_token"
    assert "delta_id" not in queue_instance["rows"][0]


def test_system_delta_matrix_render_includes_sweep_and_namespaced_result_card() -> None:
    queue = load_system_delta_queue(
        sweep_id="binary_md_v1",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    matrix = render_system_delta_matrix(
        queue,
        registry_path=REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json",
    )

    assert "reference/system_delta_sweeps/binary_md_v1/queue.yaml" in matrix
    assert "reference/system_delta_catalog.yaml" in matrix
    assert "Sweep id: `binary_md_v1`" in matrix
    assert "delta_row_cls_pool" in matrix
    assert "tfrow_n_heads" in matrix
    assert "outputs/staged_ladder/research/binary_md_v1/delta_row_cls_pool/result_card.md" in matrix
    assert "Legacy stage alias" in matrix
    row_cls_pool = next(row for row in queue["rows"] if row["delta_id"] == "delta_row_cls_pool")
    assert (
        f"| 6 | `delta_row_cls_pool` | row_pool | yes | {row_cls_pool['status']} | row_cls_pool |"
        in matrix
    )


def test_grouped_tokenizer_guard_is_captured_in_catalog_and_materialized_queue() -> None:
    catalog = load_system_delta_catalog(REPO_ROOT / "reference" / "system_delta_catalog.yaml")
    queue = load_system_delta_queue(
        sweep_id="binary_md_v1",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )

    grouped_catalog = catalog["deltas"]["delta_shifted_grouped_tokenizer"]
    grouped_row = next(row for row in queue["rows"] if row["delta_id"] == "delta_shifted_grouped_tokenizer")

    assert grouped_catalog["applicability_guards"][0]["kind"] == "requires_anchor_model_selection"
    assert grouped_catalog["applicability_guards"][0]["key"] == "feature_encoder"
    assert grouped_row["status"] == "blocked_on_surface_semantics"
    assert grouped_row["interpretation_status"] == "blocked"
    assert "genuinely isolatable tokenization experiment" in grouped_row["next_action"]


def test_active_alias_queue_matches_materialized_active_sweep() -> None:
    materialized = load_system_delta_queue(
        sweep_id="binary_md_v1",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    alias_payload = OmegaConf.to_container(
        OmegaConf.load(REPO_ROOT / "reference" / "system_delta_queue.yaml"),
        resolve=True,
    )

    assert alias_payload == materialized


def test_create_sweep_bootstraps_from_catalog_and_applies_guards(tmp_path: Path) -> None:
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
    shutil.copytree(source_sweeps_root / "binary_md_v1", sweeps_root / "binary_md_v1")

    result = create_sweep(
        sweep_id="binary_sm_v2",
        anchor_run_id="01_nano_exact_md_prior_parity_fix_binary_medium_v1",
        parent_sweep_id="binary_md_v1",
        complexity_level="binary_sm",
        benchmark_bundle_path="src/tab_foundry/bench/nanotabpfn_openml_classification_small_v1.json",
        control_baseline_id="cls_benchmark_linear_v2",
        index_path=sweeps_root / "index.yaml",
        catalog_path=reference_root / "system_delta_catalog.yaml",
        registry_path=REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json",
        sweeps_root=sweeps_root,
    )

    created_sweep = load_system_delta_sweep(
        "binary_sm_v2",
        index_path=sweeps_root / "index.yaml",
        sweeps_root=sweeps_root,
    )
    created_queue = load_system_delta_queue_instance(
        "binary_sm_v2",
        index_path=sweeps_root / "index.yaml",
        sweeps_root=sweeps_root,
    )
    materialized = load_system_delta_queue(
        sweep_id="binary_sm_v2",
        index_path=sweeps_root / "index.yaml",
        catalog_path=reference_root / "system_delta_catalog.yaml",
        sweeps_root=sweeps_root,
    )

    assert Path(result["queue_path"]).exists()
    assert (
        created_sweep["anchor_surface"]["notes"][0]
        == "The locked anchor is benchmark registry run "
        "`01_nano_exact_md_prior_parity_fix_binary_medium_v1` on bundle "
        "`nanotabpfn_openml_classification_small` (5 tasks)."
    )
    assert "10-task medium binary bundle" not in created_sweep["anchor_surface"]["notes"][0]
    assert created_queue["rows"][0]["delta_ref"] == "delta_label_token"
    assert materialized["sweep_id"] == "binary_sm_v2"
    assert materialized["rows"][0]["entangled_legacy_stage"] == "label_token"
    grouped_row = next(row for row in created_queue["rows"] if row["delta_ref"] == "delta_shifted_grouped_tokenizer")
    assert grouped_row["status"] == "blocked_on_surface_semantics"
    assert grouped_row["interpretation_status"] == "blocked"
    assert next(row for row in created_queue["rows"] if row["delta_ref"] == "delta_label_token")["status"] == "ready"


def test_system_delta_queue_validation_passes_when_no_rows_are_completed() -> None:
    queue = load_system_delta_queue(
        sweep_id="binary_md_v1",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    queue_without_completed_rows = deepcopy(queue)
    for row in queue_without_completed_rows["rows"]:
        if row["status"] == "completed":
            row["status"] = "ready"

    assert validate_system_delta_queue(
        queue_without_completed_rows,
        registry_path=REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json",
    ) == []


def test_checked_in_system_delta_matrix_matches_rendered_active_sweep() -> None:
    queue = load_system_delta_queue(
        sweep_id="binary_md_v1",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    rendered = render_system_delta_matrix(
        queue,
        registry_path=REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json",
    )
    checked_in = (REPO_ROOT / "reference" / "system_delta_matrix.md").read_text(encoding="utf-8")

    assert checked_in == rendered
