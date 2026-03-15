from __future__ import annotations

from pathlib import Path

from omegaconf import OmegaConf


REPO_ROOT = Path(__file__).resolve().parents[2]


def test_program_contract_paths_exist() -> None:
    program_path = REPO_ROOT / "program.md"
    sources_path = REPO_ROOT / "reference" / "stage_research_sources.yaml"

    assert program_path.exists()
    assert sources_path.exists()

    contents = program_path.read_text(encoding="utf-8")
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


def test_stage_research_source_manifest_resolves() -> None:
    manifest_path = REPO_ROOT / "reference" / "stage_research_sources.yaml"
    payload = OmegaConf.to_container(OmegaConf.load(manifest_path), resolve=True)

    assert isinstance(payload, dict)
    local_sources = payload.get("local_sources")
    curated_external_sources = payload.get("curated_external_sources")

    assert isinstance(local_sources, list)
    assert local_sources
    assert isinstance(curated_external_sources, list)
    assert curated_external_sources

    for source in local_sources:
        assert isinstance(source, dict)
        path_value = source.get("path")
        assert isinstance(path_value, str) and path_value.strip()
        assert (REPO_ROOT / path_value).resolve().exists()

    for source in curated_external_sources:
        assert isinstance(source, dict)
        title = source.get("title")
        url = source.get("url")
        role = source.get("role")
        assert isinstance(title, str) and title.strip()
        assert isinstance(url, str) and url.startswith("https://")
        assert isinstance(role, str) and role.strip()
