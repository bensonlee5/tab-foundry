"""Path and I/O helpers for sweep-aware system-delta tooling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, cast

from omegaconf import OmegaConf
import yaml


class _SystemDeltaYamlDumper(yaml.SafeDumper):
    """Quote ambiguous scalars so queue prose round-trips as strings."""


def _represent_system_delta_str(
    dumper: yaml.SafeDumper,
    value: str,
) -> yaml.ScalarNode:
    style = "'" if ": " in value else None
    return dumper.represent_scalar("tag:yaml.org,2002:str", value, style=style)


_SystemDeltaYamlDumper.add_representer(str, _represent_system_delta_str)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[4]


def default_catalog_path() -> Path:
    return repo_root() / "reference" / "system_delta_catalog.yaml"


def default_sweeps_root() -> Path:
    return repo_root() / "reference" / "system_delta_sweeps"


def default_sweep_index_path() -> Path:
    return default_sweeps_root() / "index.yaml"


def default_queue_path() -> Path:
    return repo_root() / "reference" / "system_delta_queue.yaml"


def default_matrix_path() -> Path:
    return repo_root() / "reference" / "system_delta_matrix.md"


def default_registry_path() -> Path:
    return repo_root() / "src" / "tab_foundry" / "bench" / "benchmark_run_registry_v1.json"


def sweep_dir(sweep_id: str, *, sweeps_root: Path | None = None) -> Path:
    return (sweeps_root or default_sweeps_root()) / str(sweep_id)


def sweep_metadata_path(sweep_id: str, *, sweeps_root: Path | None = None) -> Path:
    return sweep_dir(sweep_id, sweeps_root=sweeps_root) / "sweep.yaml"


def sweep_queue_path(sweep_id: str, *, sweeps_root: Path | None = None) -> Path:
    return sweep_dir(sweep_id, sweeps_root=sweeps_root) / "queue.yaml"


def sweep_matrix_path(sweep_id: str, *, sweeps_root: Path | None = None) -> Path:
    return sweep_dir(sweep_id, sweeps_root=sweeps_root) / "matrix.md"


def _copy_jsonable(payload: Any) -> Any:
    return json.loads(json.dumps(payload))


def _render_path(path: Path) -> str:
    resolved = path.expanduser().resolve()
    root = repo_root()
    try:
        return str(resolved.relative_to(root))
    except ValueError:
        return str(resolved)


def _load_yaml_mapping(path: Path, *, context: str) -> dict[str, Any]:
    payload = OmegaConf.to_container(
        OmegaConf.load(path.expanduser().resolve()),
        resolve=True,
    )
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} must decode to a mapping")
    return cast(dict[str, Any], payload)


def _write_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    text = yaml.dump(
        _copy_jsonable(dict(payload)),
        Dumper=_SystemDeltaYamlDumper,
        sort_keys=False,
        allow_unicode=False,
    )
    resolved.write_text(text, encoding="utf-8")


def _write_text(path: Path, contents: str) -> None:
    resolved = path.expanduser().resolve()
    resolved.parent.mkdir(parents=True, exist_ok=True)
    resolved.write_text(contents, encoding="utf-8")
