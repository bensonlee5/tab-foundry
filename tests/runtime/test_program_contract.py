from __future__ import annotations

from pathlib import Path
import tomllib

from omegaconf import OmegaConf
from tab_foundry.benchmark_registry import default_benchmark_run_registry_path


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
        "`final_log_loss`",
        "`final_brier_score`",
        "`final_roc_auc`",
        "The primary score remains `final_log_loss` on the canonical benchmark bundle",
        "- multiclass classification: `final_log_loss`",
        "The benchmark registry is the historical system of record.",
        "Underperformance alone is not enough for `reject`.",
        "This pass is attribution-first. No row becomes the new base during the sweep.",
        "`best_roc_auc` remains a tie-breaker and diagnostic for classification sweeps,",
        "`training_surface_record.json`",
        "Agents should use optional sibling-workspace sources when available, but must",
        "There is no repo-global active sweep",
        "Every benchmark-facing run belongs to exactly one `sweep_id`.",
        "PFN control lane",
        "hybrid diagnostic lane",
        "canonical architecture-screen surface",
        "Evidence collected only on the hybrid",
        "`screen_only` rows are not benchmark-facing replacements for the anchor.",
    ]
    for statement in required_statements:
        assert statement in contents


def test_program_contract_required_repo_paths_exist() -> None:
    contents = (REPO_ROOT / "program.md").read_text(encoding="utf-8")
    benchmark_registry_relative_path = default_benchmark_run_registry_path().relative_to(REPO_ROOT).as_posix()
    required_path_references = [
        "reference/system_delta_catalog.yaml",
        "reference/system_delta_sweeps/index.yaml",
        "reference/system_delta_sweeps/<sweep_id>/sweep.yaml",
        "reference/system_delta_sweeps/<sweep_id>/queue.yaml",
        "reference/system_delta_sweeps/<sweep_id>/matrix.md",
        "reference/system_delta_campaign_template.md",
        "reference/stage_research_sources.yaml",
        benchmark_registry_relative_path,
    ]
    for relative_path in required_path_references:
        assert f"`{relative_path}`" in contents

    existing_repo_paths = [
        "reference/system_delta_catalog.yaml",
        "reference/system_delta_sweeps/index.yaml",
        "reference/system_delta_campaign_template.md",
        "reference/stage_research_sources.yaml",
        benchmark_registry_relative_path,
    ]
    for relative_path in existing_repo_paths:
        assert (REPO_ROOT / relative_path).exists()


def test_program_contract_describes_registry_resolved_anchor_identity() -> None:
    contents = (REPO_ROOT / "program.md").read_text(encoding="utf-8")
    assert "selected anchor run id: `anchor_run_id` from the chosen `sweep.yaml`" in contents
    assert "Resolve canonical identity through" in contents
    assert "They may be absent in a fresh clone or" in contents
    assert "CI checkout." in contents
    assert "`src/tab_foundry/bench/benchmark_run_registry_v1.json`" in contents


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
        "`training_experiment`",
        "`training_config_profile`",
        "`surface_role`",
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
        "`cls_benchmark_linear_v2`",
        "`src/tab_foundry/bench/nanotabpfn_openml_binary_medium_v1.json`",
        "`training_surface_record.json`",
        "`tab-foundry train legacy-prior staged`",
        "`outputs/staged_ladder/01_nano_exact_md/prior_parity_fix`",
        "`outputs/staged_ladder/01_nano_exact_md/prior_benchmark_binary_medium_v1/comparison_summary.json`",
        "tab-foundry research sweep next --sweep-id <sweep_id>",
        "tab-foundry research sweep execute --sweep-id <sweep_id>",
        "tab-foundry research sweep graph --sweep-id <sweep_id> --anchor",
        "Graphviz `dot`",
        "PFN control lane",
        "Hybrid diagnostic lane",
        "Canonical architecture-screen surface",
        "`screen_only` rows are diagnostic only.",
    ]
    for statement in required_statements:
        assert statement in contents

    forbidden_statements = [
        "### Staged Ladder Runbook",
        "canonical promotion gate",
        "promotes forward by overriding",
        "show-active",
        "set-active",
    ]
    for statement in forbidden_statements:
        assert statement not in contents


def test_python_runtime_and_tooling_contracts_are_aligned() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")
    workflows = (REPO_ROOT / "docs" / "workflows.md").read_text(encoding="utf-8")
    ci_workflow = (REPO_ROOT / ".github" / "workflows" / "test.yml").read_text(encoding="utf-8")
    python_version = (REPO_ROOT / ".python-version").read_text(encoding="utf-8").strip()

    assert pyproject["project"]["requires-python"] == ">=3.14,<3.15"
    assert pyproject["tool"]["ruff"]["target-version"] == "py314"
    assert pyproject["tool"]["mypy"]["python_version"] == "3.14"
    assert python_version == "3.14"
    assert "Python `3.14`" in readme
    assert "Python `3.14`" in workflows
    assert 'python-version: "3.14"' in ci_workflow
    assert "Python `3.13`" not in readme
    assert "Python `3.13`" not in workflows
    assert 'python-version: "3.13"' not in ci_workflow


def test_readme_front_door_contract_matches_current_repo_shape() -> None:
    readme = (REPO_ROOT / "README.md").read_text(encoding="utf-8")

    required_statements = [
        "bensonlee5.github.io/tab-foundry",
        "| If you want to... | Start here | Then go deeper |",
        "`tab-foundry` is the canonical packaged CLI",
        "`./scripts/dev`",
        "`scripts/bench/`",
        "`scripts/materialize_tf_rd_013_support.py`",
        "| Surface | Use it for |",
        "Use `--help` in this order:",
        "tab-foundry --help",
        "tab-foundry <group> --help",
        "tab-foundry <group> <command> --help",
        "| Namespace | Purpose | Read next |",
        "For the canonical leaf-command inventory, use",
        "docs/workflows.md",
        "docs/research-contributors.md",
        "docs/ml-engineering.md",
    ]
    for statement in required_statements:
        assert statement in readme

    forbidden_statements = [
        "## Quickstart",
        "Build a manifest:",
        "Train a smoke profile:",
        "Evaluate a checkpoint:",
        "Export and validate an inference bundle:",
        "## Docs",
        "<summary>Full CLI tree</summary>",
    ]
    for statement in forbidden_statements:
        assert statement not in readme


def test_editable_lockfile_version_matches_pyproject() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    uv_lock = tomllib.loads((REPO_ROOT / "uv.lock").read_text(encoding="utf-8"))

    packages = uv_lock.get("package")
    assert isinstance(packages, list)

    editable_package = next(
        (
            package
            for package in packages
            if isinstance(package, dict)
            and package.get("name") == "tab-foundry"
            and package.get("source") == {"editable": "."}
        ),
        None,
    )

    assert editable_package is not None
    assert editable_package["version"] == pyproject["project"]["version"]


def test_model_config_documents_staged_override_surface() -> None:
    contents = (REPO_ROOT / "docs" / "development" / "model-config.md").read_text(
        encoding="utf-8"
    )

    required_statements = [
        "`stage_label`",
        "`module_overrides`",
        "`feature_encoder`",
        "`post_encoder_norm`",
        "`post_stack_norm`",
        "`target_conditioner`",
        "`tokenizer`",
        "`column_encoder`",
        "`row_pool`",
        "`context_encoder`",
        "`head`",
        "`table_block_style`",
        "`table_block_residual_scale`",
        "`allow_test_self_attention`",
        "queue-managed",
        "reference/system_delta_campaign_template.md",
    ]
    for statement in required_statements:
        assert statement in contents


def test_tabfoundry_sandwich_doc_covers_current_surface() -> None:
    contents = (REPO_ROOT / "docs" / "development" / "tabfoundry-sandwich.md").read_text(
        encoding="utf-8"
    )

    required_statements = [
        "`tabfoundry_sandwich` is a fixed-latent hybrid full-cell / summary-stream",
        "`R * C`",
        "`K * (R + C)`",
        "stage `0`",
        "latent-then-full-cell",
        "per-row self-attention",
        "ISAB",
        "`model.arch`",
        "`d_icl`",
        "`input_normalization`",
        "`many_class_base`",
        "`head_hidden_dim`",
        "`pre_encoder_clip`",
        "`norm_type`",
        "`sandwich_latents`",
        "`sandwich_layers`",
        "`sandwich_heads`",
        "`sandwich_ff_expansion`",
        "`sandwich_summary_tokens_per_axis`",
        "`sandwich_self_attention_per_cross`",
        "`sandwich_pre_row_attention_layers`",
        "`sandwich_pre_column_attention_layers`",
        "`sandwich_pre_column_inducing_tokens`",
        "`TaskBatch.metadata[\"feature_types\"]`",
        "`run_reference_consumer(..., feature_types=[...])`",
        "`forward_batched(..., feature_types=[...])`",
        "`latent_seed`",
        "truncated normal",
        "repeated Perceiver stages",
        "`2 <= num_classes <= many_class_base`",
    ]
    for statement in required_statements:
        assert statement in contents

    forbidden_statements = [
        "earlier sandwich",
        "prior sandwich",
        "intermediate shared-latent",
        "previous sandwich",
        "explicit list is embedded",
        "falls back to all `floating`",
        "optional per-request list",
    ]
    lowered = contents.lower()
    for statement in forbidden_statements:
        assert statement not in lowered

    assert "The current staged ladder is:" not in contents


def test_reference_index_covers_system_delta_surfaces_and_legacy_stage_template_is_removed() -> None:
    reference_index = (REPO_ROOT / "reference" / "README.md").read_text(encoding="utf-8")
    required_entries = [
        "`system_delta_catalog.yaml`",
        "`system_delta_campaign_template.md`",
        "`stage_research_sources.yaml`",
        "`system_delta_sweeps/`",
    ]
    for entry in required_entries:
        assert entry in reference_index

    assert "`system_delta_queue.yaml`" not in reference_index
    assert "`system_delta_matrix.md`" not in reference_index
    assert not (REPO_ROOT / "reference" / "system_delta_queue.yaml").exists()
    assert not (REPO_ROOT / "reference" / "system_delta_matrix.md").exists()

    legacy_template = REPO_ROOT / "reference" / "stage_campaign_template.md"
    assert not legacy_template.exists()


def test_reference_sweep_queue_and_matrix_outputs_do_not_contain_placeholder_todo_text() -> None:
    sweeps_root = REPO_ROOT / "reference" / "system_delta_sweeps"
    generated_files = list(sweeps_root.rglob("queue.yaml")) + list(sweeps_root.rglob("matrix.md"))

    assert generated_files
    for path in generated_files:
        contents = path.read_text(encoding="utf-8")
        assert "TODO:" not in contents, f"placeholder TODO leaked into generated sweep output: {path}"


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
