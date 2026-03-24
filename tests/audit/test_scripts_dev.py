from __future__ import annotations

import os
from pathlib import Path
import stat
import subprocess
import sys


REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_executable(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(contents, encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def _workspace(tmp_path: Path) -> Path:
    repo_root = tmp_path / "repo"
    (repo_root / "scripts").mkdir(parents=True, exist_ok=True)
    (repo_root / "scripts" / "audit").mkdir(parents=True, exist_ok=True)
    (repo_root / "scripts" / "dev").write_text(
        (REPO_ROOT / "scripts" / "dev").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (repo_root / ".python-version").write_text("3.14\n", encoding="utf-8")
    (repo_root / "scripts" / "audit" / "dev_verify.py").write_text("# test placeholder\n", encoding="utf-8")
    (repo_root / "scripts" / "audit" / "tool_roots.py").write_text(
        (REPO_ROOT / "scripts" / "audit" / "tool_roots.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    return repo_root


def _doctor_env(repo_root: Path, *, include_uv: bool = True, include_git: bool = True) -> dict[str, str]:
    fake_bin = repo_root / "fake_bin"
    fake_bin.mkdir(parents=True, exist_ok=True)
    if include_uv:
        _write_executable(fake_bin / "uv", "#!/bin/sh\nexit 0\n")
    if include_git:
        common_git_dir = repo_root / ".git"
        common_git_dir.mkdir(parents=True, exist_ok=True)
        _write_executable(
            fake_bin / "git",
            "#!/bin/sh\n"
            "if [ \"$1\" = \"rev-parse\" ]; then\n"
            "  case \"$3\" in\n"
            "    --show-toplevel)\n"
            f"      printf '%s\\n' '{repo_root}'\n"
            "      ;;\n"
            "    --git-common-dir)\n"
            f"      printf '%s\\n' '{common_git_dir}'\n"
            "      ;;\n"
            "  esac\n"
            "  exit 0\n"
            "fi\n"
            "exit 1\n",
        )
    env = dict(os.environ)
    env["PATH"] = f"{fake_bin}:{env['PATH']}"
    env.pop("TAB_FOUNDRY_PRIMARY_ROOT", None)
    env.pop("TAB_FOUNDRY_WORKTREE_ROOT", None)
    return env


def _write_healthy_doctor_repo(
    repo_root: Path,
    *,
    with_tab_foundry: bool = True,
    hook_python: str | None = None,
) -> None:
    _write_executable(
        repo_root / ".venv" / "bin" / "python",
        "#!/bin/sh\n"
        "if [ \"$1\" = \"-m\" ] && [ \"$2\" = \"pre_commit\" ] && [ \"$3\" = \"--version\" ]; then\n"
        "  exit 0\n"
        "fi\n"
        f'exec "{sys.executable}" "$@"\n',
    )
    if with_tab_foundry:
        _write_executable(repo_root / ".venv" / "bin" / "tab-foundry", "#!/bin/sh\nexit 0\n")
    expected_python = str(repo_root / ".venv" / "bin" / "python")
    install_python = expected_python if hook_python is None else hook_python
    _write_executable(
        repo_root / ".git" / "hooks" / "pre-commit",
        "#!/bin/sh\n"
        f"INSTALL_PYTHON={install_python}\n",
    )


def test_scripts_dev_doctor_succeeds_for_healthy_repo(tmp_path: Path) -> None:
    repo_root = _workspace(tmp_path)
    _write_healthy_doctor_repo(repo_root)

    completed = subprocess.run(
        ["bash", str(repo_root / "scripts" / "dev"), "doctor"],
        cwd=repo_root,
        env=_doctor_env(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert "doctor: ok" in completed.stdout
    assert "source=worktree" in completed.stdout


def test_scripts_dev_doctor_reports_missing_venv(tmp_path: Path) -> None:
    repo_root = _workspace(tmp_path)
    completed = subprocess.run(
        ["bash", str(repo_root / "scripts" / "dev"), "doctor"],
        cwd=repo_root,
        env=_doctor_env(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "[fail] .venv python" in completed.stdout


def test_scripts_dev_doctor_reports_missing_console_script(tmp_path: Path) -> None:
    repo_root = _workspace(tmp_path)
    _write_healthy_doctor_repo(repo_root, with_tab_foundry=False)

    completed = subprocess.run(
        ["bash", str(repo_root / "scripts" / "dev"), "doctor"],
        cwd=repo_root,
        env=_doctor_env(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "[fail] tab-foundry console script" in completed.stdout


def test_scripts_dev_doctor_reports_hook_python_mismatch(tmp_path: Path) -> None:
    repo_root = _workspace(tmp_path)
    _write_healthy_doctor_repo(repo_root, hook_python="/tmp/other-python")

    completed = subprocess.run(
        ["bash", str(repo_root / "scripts" / "dev"), "doctor"],
        cwd=repo_root,
        env=_doctor_env(repo_root),
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "[fail] pre-commit hook python" in completed.stdout


def test_scripts_dev_doctor_prefers_primary_hook_python_when_available(tmp_path: Path) -> None:
    repo_root = _workspace(tmp_path)
    primary_root = tmp_path / "primary"
    _write_healthy_doctor_repo(
        repo_root,
        hook_python=str(primary_root / ".venv" / "bin" / "python"),
    )
    _write_executable(
        primary_root / ".venv" / "bin" / "python",
        "#!/bin/sh\n"
        f'exec "{sys.executable}" "$@"\n',
    )

    env = _doctor_env(repo_root)
    env["TAB_FOUNDRY_PRIMARY_ROOT"] = str(primary_root)

    completed = subprocess.run(
        ["bash", str(repo_root / "scripts" / "dev"), "doctor"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert f"[ok] pre-commit hook python: {primary_root / '.venv' / 'bin' / 'python'}" in completed.stdout


def test_scripts_dev_doctor_reports_primary_tool_root_fallback(tmp_path: Path) -> None:
    repo_root = _workspace(tmp_path)
    primary_root = tmp_path / "primary"
    _write_healthy_doctor_repo(repo_root)
    (repo_root / ".venv" / "bin" / "python").unlink()
    _write_executable(
        primary_root / ".venv" / "bin" / "python",
        "#!/bin/sh\n"
        f'exec "{sys.executable}" "$@"\n',
    )

    env = _doctor_env(repo_root)
    env["TAB_FOUNDRY_PRIMARY_ROOT"] = str(primary_root)

    completed = subprocess.run(
        ["bash", str(repo_root / "scripts" / "dev"), "doctor"],
        cwd=repo_root,
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 1
    assert "source=primary" in completed.stdout


def test_scripts_dev_ready_runs_review_then_verify(tmp_path: Path) -> None:
    repo_root = _workspace(tmp_path)
    log_path = repo_root / "ready.log"
    _write_executable(
        repo_root / ".venv" / "bin" / "python",
        "#!/bin/sh\n"
        f'printf "%s\\n" "$*" >> "{log_path}"\n',
    )

    completed = subprocess.run(
        ["bash", str(repo_root / "scripts" / "dev"), "ready", "--base-ref", "origin/main"],
        cwd=repo_root,
        capture_output=True,
        text=True,
        check=False,
    )

    assert completed.returncode == 0
    assert log_path.read_text(encoding="utf-8").splitlines() == [
        "scripts/audit/dev_verify.py review-base --base-ref origin/main",
        "scripts/audit/dev_verify.py verify affected --base-ref origin/main",
    ]
