"""Benchmark environment bootstrap helpers for external comparator repos."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess
import sys

from tab_foundry.bench.helper_imports import (
    resolve_tab_realdata_hub_root,
    resolve_tab_realdata_hub_src_root,
)


NANOTABPFN_PYPROJECT = """[project]
name = "nanotabpfn-local"
version = "0.1.0"
description = "Local dependency metadata for nanoTabPFN benchmarking"
requires-python = ">=3.10"
dependencies = [
  "numpy",
  "pyarrow",
  "torch",
  "schedulefree",
  "h5py",
  "scikit-learn",
  "openml",
  "pandas",
  "matplotlib",
  "seaborn",
]

[project.optional-dependencies]
experiment = [
  "tabpfn==2.2.1",
]

[tool.uv]
package = false
"""

TAB_REALDATA_HUB_INSTALL_SPEC = "tab-realdata-hub>=0.1.5"
LINUX_TORCH_INSTALL_SPEC = "torch==2.11.0+cu128"
LINUX_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
TAB_REALDATA_HUB_RUNTIME_DEPENDENCIES = (
    "numpy>=2.1",
    "openml>=0.15",
    "pandas>=2.2",
    "pyarrow>=23.0",
    "scikit-learn>=1.6",
)


@dataclass(slots=True)
class BenchmarkEnvConfig:
    """Input configuration for sibling benchmark env bootstrap."""

    nanotabpfn_root: Path
    tabpfn_root: Path
    tabicl_root: Path
    tab_realdata_hub_root: Path | None = None


def ensure_nanotabpfn_pyproject(root: Path) -> Path:
    """Create a minimal nanoTabPFN pyproject if it is missing."""

    pyproject_path = root.expanduser().resolve() / "pyproject.toml"
    if pyproject_path.exists():
        return pyproject_path
    pyproject_path.write_text(NANOTABPFN_PYPROJECT, encoding="utf-8")
    return pyproject_path


def _sync_repo(root: Path) -> None:
    subprocess.run(["uv", "sync"], cwd=root, check=True)


def _validate_import(python_path: Path, module_name: str) -> None:
    subprocess.run(
        [
            str(python_path),
            "-c",
            (
                "import importlib.util, sys; "
                f"sys.exit(0 if importlib.util.find_spec('{module_name}') is not None else 1)"
            ),
        ],
        check=True,
    )


def _install_python_package(
    python_path: Path,
    package_spec: str,
    *,
    index_url: str | None = None,
) -> None:
    command = [
        "uv",
        "pip",
        "install",
        "--python",
        str(python_path),
    ]
    if index_url is not None:
        command.extend(["--index-url", index_url])
    command.append(package_spec)
    subprocess.run(command, check=True)


def _ensure_repo_expected_torch_stack(*python_paths: Path) -> None:
    if sys.platform != "linux":
        return
    for python_path in python_paths:
        _install_python_package(
            python_path,
            LINUX_TORCH_INSTALL_SPEC,
            index_url=LINUX_TORCH_INDEX_URL,
        )


def _python_version_info(python_path: Path) -> tuple[int, int]:
    completed = subprocess.run(
        [
            str(python_path),
            "-c",
            "import sys; print(f'{sys.version_info[0]}.{sys.version_info[1]}')",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    raw = completed.stdout.strip()
    major_str, minor_str = raw.split(".", maxsplit=1)
    return int(major_str), int(minor_str)


def _tab_realdata_hub_install_spec(root: Path | None = None) -> str:
    resolved_root = resolve_tab_realdata_hub_root(tab_realdata_hub_root=root)
    if resolved_root is not None:
        return str(resolved_root)
    return TAB_REALDATA_HUB_INSTALL_SPEC


def _install_tab_realdata_hub_runtime_dependencies(python_path: Path) -> None:
    for dependency in TAB_REALDATA_HUB_RUNTIME_DEPENDENCIES:
        _install_python_package(python_path, dependency)


def _python_site_packages_path(python_path: Path) -> Path:
    completed = subprocess.run(
        [
            str(python_path),
            "-c",
            "import sysconfig; print(sysconfig.get_path('purelib'))",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    resolved = Path(completed.stdout.strip()).expanduser().resolve()
    if not resolved.exists():
        raise RuntimeError(f"python purelib path does not exist: {resolved}")
    return resolved


def _install_explicit_src_root_path(python_path: Path, *, src_root: Path, module_name: str) -> None:
    site_packages_path = _python_site_packages_path(python_path)
    pth_path = site_packages_path / f"{module_name}_explicit_src_root.pth"
    pth_path.write_text(f"{src_root.expanduser().resolve()}\n", encoding="utf-8")


def bootstrap_benchmark_envs(config: BenchmarkEnvConfig) -> dict[str, str]:
    """Create or refresh benchmark envs for sibling repos."""

    nanotabpfn_root = config.nanotabpfn_root.expanduser().resolve()
    tabpfn_root = config.tabpfn_root.expanduser().resolve()
    tabicl_root = config.tabicl_root.expanduser().resolve()

    for root, label in (
        (nanotabpfn_root, "nanoTabPFN"),
        (tabpfn_root, "TabPFN"),
        (tabicl_root, "tabicl"),
    ):
        if not root.exists():
            raise RuntimeError(f"{label} root does not exist: {root}")

    ensure_nanotabpfn_pyproject(nanotabpfn_root)
    _sync_repo(nanotabpfn_root)
    _sync_repo(tabpfn_root)
    _sync_repo(tabicl_root)

    nanotabpfn_python = nanotabpfn_root / ".venv" / "bin" / "python"
    tabpfn_python = tabpfn_root / ".venv" / "bin" / "python"
    tabicl_python = tabicl_root / ".venv" / "bin" / "python"
    _ensure_repo_expected_torch_stack(
        nanotabpfn_python,
        tabpfn_python,
        tabicl_python,
    )
    resolved_tab_realdata_hub_root = resolve_tab_realdata_hub_root(
        tab_realdata_hub_root=config.tab_realdata_hub_root,
    )
    tab_realdata_hub_spec = _tab_realdata_hub_install_spec(
        resolved_tab_realdata_hub_root
    )
    tabicl_python_version = _python_version_info(tabicl_python)
    tabicl_requires_runtime_dependency_bootstrap = tabicl_python_version < (3, 14)
    if (
        tabicl_requires_runtime_dependency_bootstrap
        and resolved_tab_realdata_hub_root is None
    ):
        raise RuntimeError(
            "tabicl benchmark env uses Python "
            f"{tabicl_python_version[0]}.{tabicl_python_version[1]}, but the published "
            "tab-realdata-hub package requires Python >=3.14; pass "
            "--tab-realdata-hub-root to bootstrap against a local checkout"
        )

    _install_python_package(nanotabpfn_python, tab_realdata_hub_spec)
    if tabicl_requires_runtime_dependency_bootstrap:
        _install_tab_realdata_hub_runtime_dependencies(tabicl_python)
        explicit_src_root = resolve_tab_realdata_hub_src_root(
            tab_realdata_hub_root=resolved_tab_realdata_hub_root,
        )
        if explicit_src_root is not None:
            _install_explicit_src_root_path(
                tabicl_python,
                src_root=explicit_src_root,
                module_name="tab_realdata_hub",
            )
    else:
        _install_python_package(tabicl_python, tab_realdata_hub_spec)

    _validate_import(nanotabpfn_python, "h5py")
    _validate_import(nanotabpfn_python, "pyarrow")
    _validate_import(nanotabpfn_python, "schedulefree")
    _validate_import(nanotabpfn_python, "openml")
    _validate_import(nanotabpfn_python, "seaborn")
    _validate_import(nanotabpfn_python, "tab_realdata_hub")
    _validate_import(tabpfn_python, "tabpfn")
    _validate_import(tabicl_python, "pyarrow")
    if not tabicl_requires_runtime_dependency_bootstrap or resolved_tab_realdata_hub_root is not None:
        _validate_import(tabicl_python, "tab_realdata_hub")
    _validate_import(tabicl_python, "tabicl")

    return {
        "nanotabpfn_python": str(nanotabpfn_python),
        "tabpfn_python": str(tabpfn_python),
        "tabicl_python": str(tabicl_python),
    }
