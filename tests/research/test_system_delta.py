from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
import shutil

from omegaconf import OmegaConf
import pytest

import tab_foundry.research.sweep.matrix as matrix_module
from tab_foundry.research.lane_contract import (
    ARCHITECTURE_SCREEN_LANE,
    HYBRID_DIAGNOSTIC_LANE,
    PFN_CONTROL_LANE,
    resolve_training_config_profile,
    resolve_training_experiment,
    resolve_surface_role,
)
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


def _stub_validation_artifacts(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    result_card = tmp_path / "result_card.md"
    result_card.write_text("# Result Card\n", encoding="utf-8")
    training_surface_record = tmp_path / "training_surface_record.json"
    training_surface_record.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(matrix_module, "result_card_path", lambda **_: result_card)
    monkeypatch.setattr(
        matrix_module,
        "resolve_registry_path_value",
        lambda _path: training_surface_record,
    )


def _anchor_dimension_anchor_text(sweep: dict[str, object], *, dimension: str) -> str:
    dimension_table = sweep["anchor_surface"]
    assert isinstance(dimension_table, dict)
    rows = dimension_table["dimension_table"]
    assert isinstance(rows, list)
    match = next(row for row in rows if row["dimension"] == dimension)
    assert isinstance(match, dict)
    value = match["anchor"]
    assert isinstance(value, str)
    return value


def test_active_sweep_materializes_current_active_sweep() -> None:
    index = load_system_delta_index(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")
    active_sweep_id = str(index["active_sweep_id"])
    sweep = load_system_delta_sweep(
        active_sweep_id,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
    )
    queue = load_system_delta_queue(
        sweep_id=active_sweep_id,
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )

    assert sweep["sweep_id"] == active_sweep_id
    assert queue["sweep_id"] == active_sweep_id
    assert queue["generated_from_sweep_id"] == active_sweep_id
    anchor_surface = sweep["anchor_surface"]
    assert isinstance(anchor_surface, dict)
    dimension_table = anchor_surface["dimension_table"]
    assert isinstance(dimension_table, list) and dimension_table
    assert all(
        isinstance(row.get("dimension"), str) and row["dimension"].strip()
        for row in dimension_table
    )
    assert sweep["anchor_run_id"] == queue["anchor_run_id"]
    expected_next_ready = next((row for row in queue["rows"] if row["status"] == "ready"), None)
    assert next_ready_row(queue) == expected_next_ready


def test_resolve_surface_role_classifies_named_sweep_surfaces() -> None:
    assert resolve_surface_role({"training_experiment": "cls_benchmark_linear_simple"}) == PFN_CONTROL_LANE
    assert resolve_surface_role({"training_experiment": "cls_benchmark_linear_simple_prior"}) == PFN_CONTROL_LANE
    assert resolve_surface_role({"training_experiment": "cls_benchmark_staged_prior"}) == HYBRID_DIAGNOSTIC_LANE
    assert resolve_surface_role({"training_experiment": "cls_benchmark_staged"}) == ARCHITECTURE_SCREEN_LANE


def test_legacy_sweep_without_lane_contract_fields_uses_hybrid_defaults() -> None:
    legacy_sweep = {
        "anchor_context": {
            "experiment": "stability_followup",
            "config_profile": "stability_followup",
        }
    }

    assert resolve_training_experiment(legacy_sweep) == "cls_benchmark_staged_prior"
    assert resolve_training_config_profile(legacy_sweep) == "cls_benchmark_staged_prior"
    assert resolve_surface_role(legacy_sweep) == HYBRID_DIAGNOSTIC_LANE


def test_all_checked_in_sweep_manifests_declare_lane_contract_fields() -> None:
    sweeps_root = REPO_ROOT / "reference" / "system_delta_sweeps"
    required_fields = ("training_experiment", "training_config_profile", "surface_role")

    for sweep_path in sorted(sweeps_root.rglob("sweep.yaml")):
        payload = OmegaConf.to_container(OmegaConf.load(sweep_path), resolve=True)
        assert isinstance(payload, dict)
        for field in required_fields:
            value = payload.get(field)
            assert isinstance(value, str) and value.strip(), f"missing {field} in {sweep_path}"


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
    row_cls_pool = next(row for row in queue["rows"] if row["delta_id"] == "delta_row_cls_pool")
    row_cls_pool["benchmark_metrics"] = {
        "column_encoder_final_window_mean_grad_norm": 0.42,
        "row_pool_final_window_mean_grad_norm": 0.55,
        "column_activation_early_to_final_mean_delta": 0.12,
        "row_activation_early_to_final_mean_delta": 0.18,
    }
    matrix = render_system_delta_matrix(
        queue,
        registry_path=REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json",
    )

    assert "reference/system_delta_sweeps/binary_md_v1/queue.yaml" in matrix
    assert "reference/system_delta_catalog.yaml" in matrix
    assert "Sweep id: `binary_md_v1`" in matrix
    assert "Training experiment: `cls_benchmark_staged_prior`" in matrix
    assert "Training config profile: `cls_benchmark_staged_prior`" in matrix
    assert "Surface role: `hybrid_diagnostic`" in matrix
    assert "delta_row_cls_pool" in matrix
    assert "tfrow_n_heads" in matrix
    assert "delta_row_cls_pool_rmsnorm" in matrix
    assert "tfrow_norm" in matrix
    assert "delta_global_rmsnorm" in matrix
    assert "norm_type" in matrix
    assert "delta_training_linear_decay" in matrix
    assert "Training overrides" in matrix
    assert "training=`prior_linear_decay`" in matrix
    assert "- Stage-local stability: column (grad `0.4200`, act delta `+0.1200`); row (grad `0.5500`, act delta `+0.1800`)" in matrix
    assert "- {'" not in matrix
    assert "outputs/staged_ladder/research/binary_md_v1/delta_row_cls_pool/result_card.md" in matrix
    assert "Recipe alias" in matrix
    assert (
        f"| 6 | `delta_row_cls_pool` | row_pool | yes | {row_cls_pool['status']} | row_cls_pool |"
        in matrix
    )
    training_row = next(row for row in queue["rows"] if row["delta_id"] == "delta_training_linear_decay")
    assert (
        f"| {training_row['order']} | `delta_training_linear_decay` | schedule | "
        f"{'yes' if training_row.get('binary_applicable', False) else 'no'} | "
        f"{training_row['status']} | {training_row.get('entangled_legacy_stage', 'none')} |"
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


def test_missingness_rows_are_deferred_from_the_main_campaign() -> None:
    queue = load_system_delta_queue(
        sweep_id="binary_md_v1",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )

    impute_row = next(row for row in queue["rows"] if row["delta_id"] == "delta_preproc_impute_missing_off")
    fill_row = next(row for row in queue["rows"] if row["delta_id"] == "delta_preproc_all_nan_fill_nonzero")

    assert impute_row["status"] == "deferred_separate_workstream"
    assert impute_row["interpretation_status"] == "blocked"
    assert "main no-missing campaign" in impute_row["next_action"]
    assert fill_row["status"] == "deferred_separate_workstream"
    assert fill_row["interpretation_status"] == "blocked"
    assert "missingness workstream" in fill_row["next_action"]


def test_active_alias_queue_matches_materialized_active_sweep() -> None:
    index = load_system_delta_index(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")
    materialized = load_system_delta_queue(
        sweep_id=str(index["active_sweep_id"]),
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    alias_payload = OmegaConf.to_container(
        OmegaConf.load(REPO_ROOT / "reference" / "system_delta_queue.yaml"),
        resolve=True,
    )

    assert alias_payload == materialized


def test_create_sweep_supports_explicit_delta_ref_order(tmp_path: Path) -> None:
    reference_root, sweeps_root = _copy_reference_workspace(tmp_path)

    _ = create_sweep(
        sweep_id="binary_md_followup",
        anchor_run_id="01_nano_exact_md_prior_parity_fix_binary_medium_v1",
        parent_sweep_id="binary_md_v1",
        complexity_level="binary_md",
        benchmark_bundle_path="src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
        control_baseline_id="cls_benchmark_linear_v2",
        delta_refs=[
            "delta_anchor_activation_trace_baseline",
            "delta_shared_feature_norm",
            "delta_shared_feature_norm_with_post_layernorm",
            "delta_shared_feature_norm_with_post_rmsnorm",
        ],
        index_path=sweeps_root / "index.yaml",
        catalog_path=reference_root / "system_delta_catalog.yaml",
        registry_path=REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json",
        sweeps_root=sweeps_root,
    )

    created_queue = load_system_delta_queue_instance(
        "binary_md_followup",
        index_path=sweeps_root / "index.yaml",
        sweeps_root=sweeps_root,
    )

    assert [row["delta_ref"] for row in created_queue["rows"]] == [
        "delta_anchor_activation_trace_baseline",
        "delta_shared_feature_norm",
        "delta_shared_feature_norm_with_post_layernorm",
        "delta_shared_feature_norm_with_post_rmsnorm",
    ]
    shared_row = next(row for row in created_queue["rows"] if row["delta_ref"] == "delta_shared_feature_norm")
    assert shared_row["training"] == {
        "surface_label": "prior_constant_lr_trace_activations",
        "overrides": {"runtime": {"trace_activations": True}},
    }


def test_create_sweep_rejects_unknown_explicit_delta_ref(tmp_path: Path) -> None:
    reference_root, sweeps_root = _copy_reference_workspace(tmp_path)

    with pytest.raises(RuntimeError, match="unknown delta_refs"):
        _ = create_sweep(
            sweep_id="binary_md_bad_subset",
            anchor_run_id="01_nano_exact_md_prior_parity_fix_binary_medium_v1",
            parent_sweep_id="binary_md_v1",
            complexity_level="binary_md",
            benchmark_bundle_path="src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
            control_baseline_id="cls_benchmark_linear_v2",
            delta_refs=["delta_anchor_activation_trace_baseline", "delta_missing_unknown"],
            index_path=sweeps_root / "index.yaml",
            catalog_path=reference_root / "system_delta_catalog.yaml",
            registry_path=REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json",
            sweeps_root=sweeps_root,
        )


def test_create_sweep_bootstraps_from_catalog_and_applies_guards(tmp_path: Path) -> None:
    reference_root, sweeps_root = _copy_reference_workspace(tmp_path)

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
    assert created_sweep["training_experiment"] == "cls_benchmark_staged_prior"
    assert created_sweep["training_config_profile"] == "cls_benchmark_staged_prior"
    assert created_sweep["surface_role"] == "hybrid_diagnostic"
    assert materialized["sweep_id"] == "binary_sm_v2"
    assert materialized["training_experiment"] == "cls_benchmark_staged_prior"
    assert materialized["training_config_profile"] == "cls_benchmark_staged_prior"
    assert materialized["surface_role"] == "hybrid_diagnostic"
    assert materialized["rows"][0]["entangled_legacy_stage"] == "label_token"
    grouped_row = next(row for row in created_queue["rows"] if row["delta_ref"] == "delta_shifted_grouped_tokenizer")
    assert grouped_row["status"] == "blocked_on_surface_semantics"
    assert grouped_row["interpretation_status"] == "blocked"
    assert next(row for row in created_queue["rows"] if row["delta_ref"] == "delta_label_token")["status"] == "ready"


def test_create_sweep_derives_lane_contract_from_overridden_training_experiment(
    tmp_path: Path,
) -> None:
    reference_root, sweeps_root = _copy_reference_workspace(tmp_path)

    _ = create_sweep(
        sweep_id="input_norm_screen_surface",
        anchor_run_id="01_nano_exact_md_prior_parity_fix_binary_medium_v1",
        parent_sweep_id="input_norm_followup",
        complexity_level="binary_md",
        benchmark_bundle_path="src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
        control_baseline_id="cls_benchmark_linear_v2",
        training_experiment="cls_benchmark_staged",
        delta_refs=["delta_anchor_activation_trace_baseline"],
        index_path=sweeps_root / "index.yaml",
        catalog_path=reference_root / "system_delta_catalog.yaml",
        registry_path=REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json",
        sweeps_root=sweeps_root,
    )

    created_sweep = load_system_delta_sweep(
        "input_norm_screen_surface",
        index_path=sweeps_root / "index.yaml",
        sweeps_root=sweeps_root,
    )
    materialized = load_system_delta_queue(
        sweep_id="input_norm_screen_surface",
        index_path=sweeps_root / "index.yaml",
        catalog_path=reference_root / "system_delta_catalog.yaml",
        sweeps_root=sweeps_root,
    )

    assert created_sweep["training_experiment"] == "cls_benchmark_staged"
    assert created_sweep["training_config_profile"] == "cls_benchmark_staged"
    assert created_sweep["surface_role"] == "architecture_screen"
    assert materialized["training_experiment"] == "cls_benchmark_staged"
    assert materialized["training_config_profile"] == "cls_benchmark_staged"
    assert materialized["surface_role"] == "architecture_screen"


def test_create_sweep_preserves_explicit_lane_contract_overrides(tmp_path: Path) -> None:
    reference_root, sweeps_root = _copy_reference_workspace(tmp_path)

    _ = create_sweep(
        sweep_id="input_norm_custom_surface",
        anchor_run_id="01_nano_exact_md_prior_parity_fix_binary_medium_v1",
        parent_sweep_id="input_norm_followup",
        complexity_level="binary_md",
        benchmark_bundle_path="src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
        control_baseline_id="cls_benchmark_linear_v2",
        training_experiment="cls_benchmark_staged",
        training_config_profile="custom_profile",
        surface_role="custom",
        delta_refs=["delta_anchor_activation_trace_baseline"],
        index_path=sweeps_root / "index.yaml",
        catalog_path=reference_root / "system_delta_catalog.yaml",
        registry_path=REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json",
        sweeps_root=sweeps_root,
    )

    created_sweep = load_system_delta_sweep(
        "input_norm_custom_surface",
        index_path=sweeps_root / "index.yaml",
        sweeps_root=sweeps_root,
    )
    materialized = load_system_delta_queue(
        sweep_id="input_norm_custom_surface",
        index_path=sweeps_root / "index.yaml",
        catalog_path=reference_root / "system_delta_catalog.yaml",
        sweeps_root=sweeps_root,
    )

    assert created_sweep["training_experiment"] == "cls_benchmark_staged"
    assert created_sweep["training_config_profile"] == "custom_profile"
    assert created_sweep["surface_role"] == "custom"
    assert materialized["training_experiment"] == "cls_benchmark_staged"
    assert materialized["training_config_profile"] == "custom_profile"
    assert materialized["surface_role"] == "custom"


def test_create_sweep_labels_simple_surface_as_pfn_control(tmp_path: Path) -> None:
    reference_root, sweeps_root = _copy_reference_workspace(tmp_path)

    result = create_sweep(
        sweep_id="input_norm_simple_surface",
        anchor_run_id="01_nano_exact_md_prior_parity_fix_binary_medium_v1",
        parent_sweep_id="input_norm_followup",
        complexity_level="binary_md",
        benchmark_bundle_path="src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
        control_baseline_id="cls_benchmark_linear_v2",
        training_experiment="cls_benchmark_linear_simple",
        delta_refs=["delta_anchor_activation_trace_baseline"],
        index_path=sweeps_root / "index.yaml",
        catalog_path=reference_root / "system_delta_catalog.yaml",
        registry_path=REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json",
        sweeps_root=sweeps_root,
    )

    created_sweep = load_system_delta_sweep(
        "input_norm_simple_surface",
        index_path=sweeps_root / "index.yaml",
        sweeps_root=sweeps_root,
    )
    created_queue = load_system_delta_queue_instance(
        "input_norm_simple_surface",
        index_path=sweeps_root / "index.yaml",
        sweeps_root=sweeps_root,
    )
    materialized = load_system_delta_queue(
        sweep_id="input_norm_simple_surface",
        index_path=sweeps_root / "index.yaml",
        catalog_path=reference_root / "system_delta_catalog.yaml",
        sweeps_root=sweeps_root,
    )

    assert created_sweep["training_experiment"] == "cls_benchmark_linear_simple"
    assert created_sweep["training_config_profile"] == "cls_benchmark_linear_simple"
    assert created_sweep["surface_role"] == "pfn_control"
    assert created_queue["rows"][0]["model"] == {}
    assert created_queue["rows"][0]["training"]["surface_label"] == "prior_constant_lr_trace_activations"
    assert created_queue["rows"][0]["training"]["overrides"]["runtime"]["trace_activations"] is True
    assert materialized["training_experiment"] == "cls_benchmark_linear_simple"
    assert materialized["training_config_profile"] == "cls_benchmark_linear_simple"
    assert materialized["surface_role"] == "pfn_control"
    assert materialized["rows"][0]["model"] == {}
    matrix_text = Path(result["matrix_path"]).read_text(encoding="utf-8")
    assert "Effective labels: model=`cls_benchmark_linear_simple`" in matrix_text


def test_create_sweep_marks_unknown_training_label_for_unlabeled_nonprior_anchor(
    tmp_path: Path,
) -> None:
    reference_root, sweeps_root = _copy_reference_workspace(tmp_path)

    _ = create_sweep(
        sweep_id="binary_sm_simple_anchor",
        anchor_run_id="00_simple_anchor_md",
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
        "binary_sm_simple_anchor",
        index_path=sweeps_root / "index.yaml",
        sweeps_root=sweeps_root,
    )
    materialized = load_system_delta_queue(
        sweep_id="binary_sm_simple_anchor",
        index_path=sweeps_root / "index.yaml",
        catalog_path=reference_root / "system_delta_catalog.yaml",
        sweeps_root=sweeps_root,
    )

    assert _anchor_dimension_anchor_text(created_sweep, dimension="training recipe") == (
        "Training surface label `training surface label unavailable`."
    )
    label_token_row = next(row for row in materialized["rows"] if row["delta_id"] == "delta_label_token")
    assert label_token_row["training"]["surface_label"] == "training surface label unavailable"


def test_load_system_delta_queue_instance_rejects_non_string_notes(tmp_path: Path) -> None:
    reference_root, sweeps_root = _copy_reference_workspace(tmp_path)
    queue_path = sweeps_root / "binary_md_v1" / "queue.yaml"
    queue_payload = OmegaConf.to_container(OmegaConf.load(queue_path), resolve=True)
    assert isinstance(queue_payload, dict)
    rows = queue_payload["rows"]
    assert isinstance(rows, list)
    rows[0]["notes"] = [{"bad": "note"}]
    queue_path.write_text(OmegaConf.to_yaml(OmegaConf.create(queue_payload), resolve=True), encoding="utf-8")

    with pytest.raises(RuntimeError, match="notes\\[0\\] must be a non-empty string"):
        _ = load_system_delta_queue_instance(
            "binary_md_v1",
            index_path=sweeps_root / "index.yaml",
            sweeps_root=sweeps_root,
        )


def test_load_materialized_system_delta_queue_rejects_non_string_notes(tmp_path: Path) -> None:
    queue = load_system_delta_queue(
        sweep_id="binary_md_v1",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    broken_queue = deepcopy(queue)
    broken_queue["rows"][0]["notes"] = [{"bad": "note"}]
    queue_path = tmp_path / "materialized_queue.yaml"
    queue_path.write_text(OmegaConf.to_yaml(OmegaConf.create(broken_queue), resolve=True), encoding="utf-8")

    with pytest.raises(RuntimeError, match="notes\\[0\\] must be a non-empty string"):
        _ = load_system_delta_queue(path=queue_path)


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


def test_system_delta_queue_validation_detects_completed_metric_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_validation_artifacts(monkeypatch, tmp_path)
    queue = load_system_delta_queue(
        sweep_id="input_norm_none_followup",
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    row = next(row for row in queue["rows"] if row["delta_id"] == "dpnb_input_norm_none_batch64_sqrt")
    row["benchmark_metrics"]["final_brier_score"] = 0.0

    issues = validate_system_delta_queue(
        queue,
        registry_path=REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json",
    )

    assert any(
        "dpnb_input_norm_none_batch64_sqrt: benchmark_metrics.final_brier_score mismatch" in issue
        for issue in issues
    )


def test_system_delta_queue_validation_detects_stage_local_metric_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    result_card = tmp_path / "result_card.md"
    result_card.write_text("# Result Card\n", encoding="utf-8")
    training_surface_record = tmp_path / "training_surface_record.json"
    training_surface_record.write_text("{}", encoding="utf-8")
    run_dir = tmp_path / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "telemetry.json").write_text(
        json.dumps(
            {
                "diagnostics": {
                    "stage_local_gradients": {
                        "modules": {
                            "column_encoder": {
                                "windows": {"final_10pct": {"mean_grad_norm": 0.42}}
                            }
                        }
                    },
                    "activation_windows": {
                        "tracked_activations": {
                            "post_column_encoder": {
                                "early_to_final_mean_delta": 0.12,
                                "windows": {"final_10pct": {"mean": 1.3}},
                            }
                        }
                    },
                }
            }
        ),
        encoding="utf-8",
    )
    registry = {
        "runs": {
            "run_1": {
                "artifacts": {
                    "run_dir": str(run_dir),
                    "training_surface_record_path": "training_surface_record.json",
                },
                "tab_foundry_metrics": {},
                "training_diagnostics": {},
                "comparisons": {},
            }
        }
    }

    monkeypatch.setattr(matrix_module, "load_benchmark_run_registry", lambda _path: registry)
    monkeypatch.setattr(matrix_module, "result_card_path", lambda **_: result_card)

    def _resolve_registry_path_value(path: str) -> Path:
        if path == "training_surface_record.json":
            return training_surface_record
        return Path(path)

    monkeypatch.setattr(
        matrix_module,
        "resolve_registry_path_value",
        _resolve_registry_path_value,
    )

    issues = validate_system_delta_queue(
        {
            "sweep_id": "validation_test",
            "rows": [
                {
                    "order": 1,
                    "delta_id": "delta_stage_local",
                    "status": "completed",
                    "run_id": "run_1",
                    "benchmark_metrics": {
                        "column_encoder_final_window_mean_grad_norm": 0.0,
                        "column_activation_early_to_final_mean_delta": 0.12,
                    },
                }
            ],
        }
    )

    assert any(
        "delta_stage_local: benchmark_metrics.column_encoder_final_window_mean_grad_norm mismatch"
        in issue
        for issue in issues
    )


def test_checked_in_completed_system_delta_queues_validate_against_registry(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _stub_validation_artifacts(monkeypatch, tmp_path)
    for sweep_id in (
        "cuda_stability_followup",
        "input_norm_followup",
        "input_norm_none_followup",
        "shared_surface_bridge_v1",
    ):
        queue = load_system_delta_queue(
            sweep_id=sweep_id,
            index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
            catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
        )
        assert validate_system_delta_queue(
            queue,
            registry_path=REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json",
        ) == []


def test_checked_in_system_delta_matrix_matches_rendered_active_sweep() -> None:
    index = load_system_delta_index(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml")
    queue = load_system_delta_queue(
        sweep_id=str(index["active_sweep_id"]),
        index_path=REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml",
        catalog_path=REPO_ROOT / "reference" / "system_delta_catalog.yaml",
    )
    rendered = render_system_delta_matrix(
        queue,
        registry_path=REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json",
    )
    checked_in = (REPO_ROOT / "reference" / "system_delta_matrix.md").read_text(encoding="utf-8")

    assert checked_in == rendered
