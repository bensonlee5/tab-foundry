"""Artifact path and YAML helpers for system-delta execution and promotion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, cast

from omegaconf import OmegaConf

from tab_foundry.control_baseline_registry import default_control_baseline_registry_path

from .paths_io import (
    default_catalog_path,
    default_registry_path,
    default_sweep_index_path,
    default_sweeps_root,
    repo_root as _repo_root,
    write_yaml as _write_yaml_file,
)


@dataclass(frozen=True)
class PromotionPaths:
    index_path: Path
    catalog_path: Path
    sweeps_root: Path
    registry_path: Path
    program_path: Path

    @classmethod
    def default(cls) -> "PromotionPaths":
        repo_root = _repo_root()
        return cls(
            index_path=default_sweep_index_path(),
            catalog_path=default_catalog_path(),
            sweeps_root=default_sweeps_root(),
            registry_path=default_registry_path(),
            program_path=repo_root / "program.md",
        )


@dataclass(frozen=True)
class ExecutionPaths:
    repo_root: Path
    index_path: Path
    catalog_path: Path
    sweeps_root: Path
    registry_path: Path
    program_path: Path
    control_baseline_registry_path: Path

    @classmethod
    def default(cls) -> "ExecutionPaths":
        repo_root = _repo_root()
        return cls(
            repo_root=repo_root,
            index_path=default_sweep_index_path(),
            catalog_path=default_catalog_path(),
            sweeps_root=default_sweeps_root(),
            registry_path=default_registry_path(),
            program_path=repo_root / "program.md",
            control_baseline_registry_path=default_control_baseline_registry_path(),
        )

    def promotion_paths(self) -> PromotionPaths:
        return PromotionPaths(
            index_path=self.index_path,
            catalog_path=self.catalog_path,
            sweeps_root=self.sweeps_root,
            registry_path=self.registry_path,
            program_path=self.program_path,
        )


def read_yaml(path: Path) -> dict[str, Any]:
    payload = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(payload, dict):
        raise RuntimeError(f"expected YAML mapping at {path}")
    return cast(dict[str, Any], payload)


def write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    _write_yaml_file(path, payload)
