"""Sibling-repo environment bootstrap helpers for benchmark workflows."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import subprocess
from typing import Sequence


NANOTABPFN_PYPROJECT = """[project]
name = "nanotabpfn-local"
version = "0.1.0"
description = "Local dependency metadata for nanoTabPFN benchmarking"
requires-python = ">=3.10"
dependencies = [
  "numpy",
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


@dataclass(slots=True)
class BenchmarkEnvConfig:
    """Input configuration for sibling benchmark env bootstrap."""

    nanotabpfn_root: Path = Path("~/dev/nanoTabPFN")
    tabpfn_root: Path = Path("~/dev/TabPFN")
    tabicl_root: Path = Path("~/dev/tabicl")


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

    _validate_import(nanotabpfn_python, "h5py")
    _validate_import(nanotabpfn_python, "schedulefree")
    _validate_import(nanotabpfn_python, "openml")
    _validate_import(nanotabpfn_python, "seaborn")
    _validate_import(tabpfn_python, "tabpfn")
    _validate_import(tabicl_python, "tabicl")

    return {
        "nanotabpfn_python": str(nanotabpfn_python),
        "tabpfn_python": str(tabpfn_python),
        "tabicl_python": str(tabicl_python),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Bootstrap sibling benchmark environments")
    parser.add_argument("--nanotabpfn-root", default="~/dev/nanoTabPFN", help="Local nanoTabPFN checkout")
    parser.add_argument("--tabpfn-root", default="~/dev/TabPFN", help="Local TabPFN checkout")
    parser.add_argument("--tabicl-root", default="~/dev/tabicl", help="Local tabicl checkout")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    summary = bootstrap_benchmark_envs(
        BenchmarkEnvConfig(
            nanotabpfn_root=Path(str(args.nanotabpfn_root)),
            tabpfn_root=Path(str(args.tabpfn_root)),
            tabicl_root=Path(str(args.tabicl_root)),
        )
    )
    print("Benchmark env bootstrap complete:")
    for key, value in summary.items():
        print(f"  {key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
