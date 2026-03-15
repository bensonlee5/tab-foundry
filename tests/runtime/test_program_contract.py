from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_program_contract_has_required_policy_sections() -> None:
    program_path = REPO_ROOT / "program.md"
    contents = program_path.read_text(encoding="utf-8")

    required_headers = [
        "## Objective",
        "## Locked Baseline And Comparison Surface",
        "## Search Scope",
        "## Required Research Package",
        "## Execution Loop",
        "## Decisions",
        "## Scaling Confirmation",
    ]
    for header in required_headers:
        assert header in contents

    required_statements = [
        "`final_roc_auc`",
        "The benchmark registry is the historical system of record.",
        "Any mechanism or preprocessing candidate is allowed",
        "Never reject a candidate from one run alone.",
        "Only `keep` candidates enter scaling confirmation.",
        "`best_roc_auc` is a tie-breaker and diagnostic, not the main score.",
        "Agents should use optional sibling-workspace sources when available, but must",
    ]
    for statement in required_statements:
        assert statement in contents


def test_program_contract_required_repo_paths_exist() -> None:
    contents = (REPO_ROOT / "program.md").read_text(encoding="utf-8")
    required_paths = [
        "outputs/staged_ladder/01_nano_exact_md/prior_parity_fix",
        "outputs/staged_ladder/01_nano_exact_md/prior_benchmark_parity_fix",
        "reference/stage_campaign_template.md",
        "reference/stage_research_sources.yaml",
        "src/tab_foundry/bench/nanotabpfn_openml_benchmark_v1.json",
        "src/tab_foundry/bench/benchmark_run_registry_v1.json",
    ]
    for relative_path in required_paths:
        assert f"`{relative_path}`" in contents
        assert (REPO_ROOT / relative_path).exists()


def test_stage_campaign_template_has_required_fields() -> None:
    contents = (REPO_ROOT / "reference" / "stage_campaign_template.md").read_text(encoding="utf-8")
    required_fields = [
        "`candidate_id`",
        "`mechanism_family`",
        "`touched_subsystems`",
        "`comparison_surface`",
        "primary_metric: final_roc_auc",
        "`queue_rationale`",
        "recommended_recipe",
        "preserved_settings",
        "shifted_settings",
        "tunable_params",
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
