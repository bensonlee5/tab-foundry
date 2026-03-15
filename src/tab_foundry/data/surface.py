"""Resolved data-surface settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping


def _mapping_from_any(value: Any, *, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping or null, got {value!r}")
    return {str(key): item for key, item in value.items()}


@dataclass(slots=True, frozen=True)
class DataSurfaceConfig:
    surface_label: str
    source: str
    manifest_path: Path | None
    filter_policy: str | None
    train_row_cap: int | None
    test_row_cap: int | None
    dagzoo_provenance: dict[str, Any] | None
    overrides: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = dict(asdict(self))
        payload["manifest_path"] = None if self.manifest_path is None else str(self.manifest_path)
        return payload


def resolve_data_surface(data_cfg: Mapping[str, Any] | None) -> DataSurfaceConfig:
    """Resolve one data surface with additive overrides."""

    cfg = {} if data_cfg is None else {str(key): value for key, value in data_cfg.items()}
    overrides = _mapping_from_any(cfg.get("surface_overrides"), context="data.surface_overrides")
    raw_source = overrides.get("source")
    if raw_source is None:
        raw_source = cfg.get("source")
    if raw_source is None:
        raw_source = "manifest"
    source = str(raw_source).strip().lower()
    raw_manifest_path = overrides.get("manifest_path")
    if raw_manifest_path is None:
        raw_manifest_path = cfg.get("manifest_path")
    manifest_path = None
    if raw_manifest_path is not None:
        manifest_path = Path(str(raw_manifest_path)).expanduser().resolve()
    filter_policy_raw = overrides.get("filter_policy")
    if filter_policy_raw is None:
        filter_policy_raw = cfg.get("filter_policy")
    filter_policy = None if filter_policy_raw is None else str(filter_policy_raw).strip()
    dagzoo_provenance_raw = overrides.get("dagzoo_provenance")
    if dagzoo_provenance_raw is None:
        dagzoo_provenance_raw = cfg.get("dagzoo_provenance")
    dagzoo_provenance = (
        None
        if dagzoo_provenance_raw is None
        else _mapping_from_any(
            dagzoo_provenance_raw,
            context="data.surface_overrides.dagzoo_provenance",
        )
    )
    train_row_cap_raw = overrides.get("train_row_cap")
    if train_row_cap_raw is None:
        train_row_cap_raw = cfg.get("train_row_cap")
    test_row_cap_raw = overrides.get("test_row_cap")
    if test_row_cap_raw is None:
        test_row_cap_raw = cfg.get("test_row_cap")
    surface_label_raw = cfg.get("surface_label")
    if surface_label_raw is None:
        surface_label_raw = overrides.get("surface_label")
    if surface_label_raw is None:
        surface_label_raw = source
    return DataSurfaceConfig(
        surface_label=str(surface_label_raw).strip(),
        source=source,
        manifest_path=manifest_path,
        filter_policy=filter_policy,
        train_row_cap=None if train_row_cap_raw is None else int(train_row_cap_raw),
        test_row_cap=None if test_row_cap_raw is None else int(test_row_cap_raw),
        dagzoo_provenance=dagzoo_provenance,
        overrides=overrides,
    )
