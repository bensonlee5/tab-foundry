from __future__ import annotations

import json
from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]


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
        "`final_roc_auc`",
        "The benchmark registry is the historical system of record.",
        "Underperformance alone is not enough for `reject`.",
        "This pass is attribution-first. No row becomes the new base during the sweep.",
        "`best_roc_auc` is a tie-breaker and diagnostic, not the main score.",
        "`training_surface_record.json`",
        "Agents should use optional sibling-workspace sources when available, but must",
    ]
    for statement in required_statements:
        assert statement in contents


def test_program_contract_required_repo_paths_exist() -> None:
    contents = (REPO_ROOT / "program.md").read_text(encoding="utf-8")
    required_repo_paths = [
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

    anchor_run_id = "01_nano_exact_md_prior_parity_fix_binary_medium_v1"
    expected_run_dir = "outputs/staged_ladder/01_nano_exact_md/prior_parity_fix"
    expected_benchmark_dir = "outputs/staged_ladder/01_nano_exact_md/prior_benchmark_binary_medium_v1"

    assert f"`{anchor_run_id}`" in contents
    assert f"`{expected_run_dir}`" in contents
    assert f"`{expected_benchmark_dir}`" in contents
    assert "Resolve canonical identity through" in contents
    assert "They may be absent in a fresh clone or CI" in contents

    runs = registry["runs"]
    assert anchor_run_id in runs
    artifacts = runs[anchor_run_id]["artifacts"]
    assert artifacts["run_dir"] == expected_run_dir
    assert artifacts["benchmark_dir"] == expected_benchmark_dir


def test_system_delta_campaign_template_has_required_fields() -> None:
    contents = (REPO_ROOT / "reference" / "system_delta_campaign_template.md").read_text(
        encoding="utf-8"
    )
    required_fields = [
        "`delta_id`",
        "`dimension_family`",
        "`comparison_policy: anchor_only`",
        "`training_surface_record.json`",
        "`result_card.md`",
        "`accept_signal`",
        "`needs_followup`",
        "`unambiguously_worse`",
        "adequacy_knobs",
    ]
    for field in required_fields:
        assert field in contents


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
