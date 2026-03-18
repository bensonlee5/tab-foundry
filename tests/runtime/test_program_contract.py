from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]


def _active_sweep_payload() -> tuple[str, dict[str, object]]:
    index_payload = OmegaConf.to_container(
        OmegaConf.load(REPO_ROOT / "reference" / "system_delta_sweeps" / "index.yaml"),
        resolve=True,
    )
    assert isinstance(index_payload, dict)
    active_sweep_id = index_payload.get("active_sweep_id")
    assert isinstance(active_sweep_id, str) and active_sweep_id.strip()
    sweep_payload = OmegaConf.to_container(
        OmegaConf.load(REPO_ROOT / "reference" / "system_delta_sweeps" / active_sweep_id / "sweep.yaml"),
        resolve=True,
    )
    assert isinstance(sweep_payload, dict)
    return active_sweep_id, sweep_payload


def test_program_contract_has_required_policy_sections() -> None:
    contents = (REPO_ROOT / "program.md").read_text(encoding="utf-8")

    required_headers = [
        "## Objective",
        "## Locked Anchor Surface",
        "## Dimension Families",
        "## Queue And Matrix",
        "## Required Research Package",
        "## Execution Loop",
        "## Decisions",
    ]
    for header in required_headers:
        assert header in contents

    required_statements = [
        "`final_log_loss`",
        "`final_roc_auc`",
        "The primary score remains `final_log_loss`",
        "The benchmark registry is the historical system of record.",
        "Underperformance alone is not enough for `reject`.",
        "This pass is attribution-first. No row becomes the new base during the sweep.",
        "`best_roc_auc` is a tie-breaker and diagnostic, not the main score.",
        "`training_surface_record.json`",
        "Agents should use optional sibling-workspace sources when available, but must",
        "generated compatibility aliases for the active sweep only",
        "Every benchmark-facing run belongs to exactly one `sweep_id`.",
    ]
    for statement in required_statements:
        assert statement in contents


def test_program_contract_required_repo_paths_exist() -> None:
    contents = (REPO_ROOT / "program.md").read_text(encoding="utf-8")
    active_sweep_id, _ = _active_sweep_payload()
    required_repo_paths = [
        "reference/system_delta_catalog.yaml",
        "reference/system_delta_sweeps/index.yaml",
        f"reference/system_delta_sweeps/{active_sweep_id}/queue.yaml",
        f"reference/system_delta_sweeps/{active_sweep_id}/matrix.md",
        "reference/system_delta_queue.yaml",
        "reference/system_delta_matrix.md",
        "reference/system_delta_campaign_template.md",
        "reference/stage_research_sources.yaml",
        "src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json",
        "src/tab_foundry/bench/benchmark_run_registry_v1.json",
    ]
    for relative_path in required_repo_paths:
        assert f"`{relative_path}`" in contents
        assert (REPO_ROOT / relative_path).exists()


def test_program_contract_anchor_is_resolved_via_registry() -> None:
    contents = (REPO_ROOT / "program.md").read_text(encoding="utf-8")
    registry_path = REPO_ROOT / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    _, sweep_payload = _active_sweep_payload()

    anchor_run_id = sweep_payload.get("anchor_run_id")
    assert isinstance(anchor_run_id, str) and anchor_run_id.strip()

    assert f"`{anchor_run_id}`" in contents
    assert "Resolve canonical identity through" in contents
    assert "They may be absent in a fresh clone or CI" in contents

    runs = registry["runs"]
    assert anchor_run_id in runs
    artifacts = runs[anchor_run_id]["artifacts"]
    expected_run_dir = artifacts["run_dir"]
    expected_benchmark_dir = artifacts["benchmark_dir"]
    assert f"`{expected_run_dir}`" in contents
    assert f"`{expected_benchmark_dir}`" in contents
    assert artifacts["run_dir"] == expected_run_dir
    assert artifacts["benchmark_dir"] == expected_benchmark_dir


def test_system_delta_campaign_template_has_required_fields() -> None:
    contents = (REPO_ROOT / "reference" / "system_delta_campaign_template.md").read_text(
        encoding="utf-8"
    )
    required_fields = [
        "`delta_id`",
        "`sweep_id`",
        "`dimension_family`",
        "`comparison_policy: anchor_only`",
        "`training_surface_record.json`",
        "`result_card.md`",
        "`accept_signal`",
        "`needs_followup`",
        "`unambiguously_worse`",
        "adequacy_knobs",
        "`reference/system_delta_sweeps/<sweep_id>/queue.yaml`",
    ]
    for field in required_fields:
        assert field in contents

    assert "outputs/staged_ladder/research/<sweep_id>/<delta_id>/research_card.md" in contents


def test_workflows_runbook_reflects_system_delta_surface() -> None:
    contents = (REPO_ROOT / "docs" / "workflows.md").read_text(encoding="utf-8")

    required_statements = [
        "### System-Delta Sweep Runbook",
        "`reference/system_delta_sweeps/index.yaml`",
        "`reference/system_delta_queue.yaml` and `reference/system_delta_matrix.md`",
        "`cls_benchmark_linear_v2`",
        "`src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`",
        "`training_surface_record.json`",
    ]
    for statement in required_statements:
        assert statement in contents

    forbidden_statements = [
        "### Staged Ladder Runbook",
        "canonical promotion gate",
        "promotes forward by overriding",
    ]
    for statement in forbidden_statements:
        assert statement not in contents


def test_model_config_documents_staged_override_surface() -> None:
    contents = (REPO_ROOT / "docs" / "development" / "model-config.md").read_text(
        encoding="utf-8"
    )

    required_statements = [
        "`stage_label`",
        "`module_overrides`",
        "`feature_encoder`",
        "`post_encoder_norm`",
        "`target_conditioner`",
        "`tokenizer`",
        "`column_encoder`",
        "`row_pool`",
        "`context_encoder`",
        "`head`",
        "`table_block_style`",
        "`allow_test_self_attention`",
        "queue-managed",
        "reference/system_delta_campaign_template.md",
    ]
    for statement in required_statements:
        assert statement in contents

    assert "The current staged ladder is:" not in contents


def test_reference_index_covers_system_delta_surfaces_and_legacy_stage_template_is_removed() -> None:
    reference_index = (REPO_ROOT / "reference" / "README.md").read_text(encoding="utf-8")
    required_entries = [
        "`system_delta_catalog.yaml`",
        "`system_delta_campaign_template.md`",
        "`stage_research_sources.yaml`",
        "`system_delta_sweeps/`",
        "`system_delta_queue.yaml`",
        "`system_delta_matrix.md`",
    ]
    for entry in required_entries:
        assert entry in reference_index

    legacy_template = REPO_ROOT / "reference" / "stage_campaign_template.md"
    assert not legacy_template.exists()


def test_stage_research_source_manifest_schema_is_portable() -> None:
    manifest_path = REPO_ROOT / "reference" / "stage_research_sources.yaml"
    payload = OmegaConf.to_container(OmegaConf.load(manifest_path), resolve=True)

    assert isinstance(payload, dict)
    required_repo_local_sources = payload.get("required_repo_local_sources")
    optional_sibling_workspace_sources = payload.get("optional_sibling_workspace_sources")
    curated_external_sources = payload.get("curated_external_sources")

    assert isinstance(required_repo_local_sources, list) and required_repo_local_sources
    assert isinstance(optional_sibling_workspace_sources, list) and optional_sibling_workspace_sources
    assert isinstance(curated_external_sources, list) and curated_external_sources

    for source in required_repo_local_sources:
        assert isinstance(source, dict)
        path_value = source.get("path")
        assert isinstance(path_value, str) and path_value.strip()
        assert (REPO_ROOT / path_value).resolve().exists()

    for source in optional_sibling_workspace_sources:
        assert isinstance(source, dict)
        path_value = source.get("path")
        optional = source.get("optional")
        role = source.get("role")
        title = source.get("title")
        assert isinstance(title, str) and title.strip()
        assert isinstance(path_value, str) and path_value.strip()
        assert optional is True
        assert isinstance(role, str) and role.strip()

    for source in curated_external_sources:
        assert isinstance(source, dict)
        title = source.get("title")
        url = source.get("url")
        role = source.get("role")
        assert isinstance(title, str) and title.strip()
        assert isinstance(url, str) and url.startswith("https://")
        assert isinstance(role, str) and role.strip()
