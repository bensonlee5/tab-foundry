from __future__ import annotations

from pathlib import Path

from tab_foundry.repo_paths import normalize_repo_relative_path, resolve_repo_relative_path


def test_normalize_repo_relative_path_uses_repo_relative_form(tmp_path: Path) -> None:
    repo_root = tmp_path / "tab-foundry"
    path = repo_root / "outputs" / "run_001" / "train"

    assert normalize_repo_relative_path(path, root=repo_root) == "outputs/run_001/train"


def test_normalize_repo_relative_path_uses_known_sibling_relative_form(tmp_path: Path) -> None:
    repo_root = tmp_path / "tab-foundry"
    sibling_path = tmp_path / "nanoTabPFN" / "300k_150x5_2.h5"

    normalized = normalize_repo_relative_path(sibling_path, root=repo_root)

    assert normalized == "../nanoTabPFN/300k_150x5_2.h5"
    assert resolve_repo_relative_path(normalized, root=repo_root) == sibling_path.resolve()


def test_normalize_repo_relative_path_keeps_unrelated_absolute_paths(tmp_path: Path) -> None:
    repo_root = tmp_path / "tab-foundry"
    outside_path = tmp_path / "outside" / "artifact.json"

    assert normalize_repo_relative_path(outside_path, root=repo_root) == str(outside_path.resolve())
