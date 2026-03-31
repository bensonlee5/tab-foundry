from __future__ import annotations

import ast
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _module_ast(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def test_cli_modules_do_not_import_private_sibling_helpers() -> None:
    cli_root = REPO_ROOT / "src" / "tab_foundry" / "cli"
    failures: list[str] = []
    for path in sorted(cli_root.glob("*.py")):
        if path.name == "__init__.py":
            continue
        module = _module_ast(path)
        for node in ast.walk(module):
            if not isinstance(node, ast.ImportFrom):
                continue
            if node.level != 1 or node.module is None or node.module == "__init__":
                continue
            for alias in node.names:
                if alias.name.startswith("_"):
                    failures.append(
                        f"{path.relative_to(REPO_ROOT)} imports {alias.name} from sibling CLI module {node.module}"
                    )
    assert failures == []


def test_deleted_compatibility_shims_are_not_reintroduced() -> None:
    deleted_paths = [
        REPO_ROOT / "src" / "tab_foundry" / "data" / "validation.py",
        REPO_ROOT / "src" / "tab_foundry" / "data" / "dagzoo_handoff.py",
        REPO_ROOT / "src" / "tab_foundry" / "training" / "batching.py",
    ]
    for path in deleted_paths:
        assert not path.exists()

    blocked_imports = [
        ".".join(("tab_foundry", "data", "validation")),
        ".".join(("tab_foundry", "data", "dagzoo_handoff")),
        ".".join(("tab_foundry", "training", "batching")),
    ]
    offenders: list[str] = []
    for root in (REPO_ROOT / "src", REPO_ROOT / "tests"):
        for path in sorted(root.rglob("*.py")):
            text = path.read_text(encoding="utf-8")
            for blocked in blocked_imports:
                if blocked in text:
                    offenders.append(f"{path.relative_to(REPO_ROOT)} references {blocked}")
    assert offenders == []


def test_prior_loop_no_longer_uses_dependency_bag_or_mutable_cfg_overlay() -> None:
    loop_path = REPO_ROOT / "src" / "tab_foundry" / "training" / "prior" / "loop.py"
    text = loop_path.read_text(encoding="utf-8")
    assert "PriorTrainingDeps" not in text
    assert "OmegaConf.to_container(cfg, resolve=True)" not in text


def test_materialize_no_longer_owns_corpus_resolution_logic() -> None:
    materialize_path = REPO_ROOT / "src" / "tab_foundry" / "research" / "sweep" / "materialize.py"
    text = materialize_path.read_text(encoding="utf-8")
    assert "load_corpus_record" not in text
    assert "load_corpus_recipe" not in text


def test_inspection_targets_do_not_rebuild_training_surface_records_locally() -> None:
    targets_path = REPO_ROOT / "src" / "tab_foundry" / "research" / "sweep" / "inspection_targets.py"
    text = targets_path.read_text(encoding="utf-8")
    assert "def _anchor_training_surface_record" not in text
    assert "def _anchor_row_payload" not in text
