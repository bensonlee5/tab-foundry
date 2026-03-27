"""Resolved data-surface settings."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

from tab_foundry.repo_paths import repo_root_from_sweeps_root

from .corpus_lookup import load_corpus_record


def _mapping_from_any(value: Any, *, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{context} must be a mapping or null, got {value!r}")
    return {str(key): item for key, item in value.items()}


def _corpus_record_missing_value_policy(corpus_record: Mapping[str, Any]) -> str | None:
    manifest = _mapping_from_any(corpus_record.get("manifest"), context="corpus_record.manifest")
    for nested_context, candidate in (
        (
            "corpus_record.manifest.characteristics.persisted_summary",
            _mapping_from_any(
                _mapping_from_any(
                    manifest.get("characteristics"),
                    context="corpus_record.manifest.characteristics",
                ).get("persisted_summary"),
                context="corpus_record.manifest.characteristics.persisted_summary",
            ).get("missing_value_policy"),
        ),
        (
            "corpus_record.manifest.inspection.persisted_summary",
            _mapping_from_any(
                _mapping_from_any(
                    manifest.get("inspection"),
                    context="corpus_record.manifest.inspection",
                ).get("persisted_summary"),
                context="corpus_record.manifest.inspection.persisted_summary",
            ).get("missing_value_policy"),
        ),
        (
            "corpus_record.manifest.characteristics",
            _mapping_from_any(
                manifest.get("characteristics"),
                context="corpus_record.manifest.characteristics",
            ).get("missing_value_policy"),
        ),
    ):
        if candidate is None:
            continue
        if not isinstance(candidate, str) or not candidate.strip():
            raise ValueError(f"{nested_context}.missing_value_policy must be a non-empty string when present")
        return candidate.strip().lower()
    return None


def _optional_string(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return str(value)


@dataclass(slots=True, frozen=True)
class DataSurfaceConfig:
    surface_label: str
    source: str
    manifest_path: Path | None
    filter_policy: str | None
    allow_missing_values: bool
    train_row_cap: int | None
    test_row_cap: int | None
    dagzoo_provenance: dict[str, Any] | None
    corpus_ref: str | None
    recipe_id: str | None
    corpus_id: str | None
    corpus_record_path: Path | None
    overrides: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        payload = dict(asdict(self))
        payload["manifest_path"] = None if self.manifest_path is None else str(self.manifest_path)
        payload["corpus_record_path"] = (
            None if self.corpus_record_path is None else str(self.corpus_record_path)
        )
        return payload


def resolve_data_surface(
    data_cfg: Mapping[str, Any] | None,
    *,
    allow_unresolved_corpus_ref: bool = False,
    sweep_id: str | None = None,
    sweeps_root: Path | None = None,
) -> DataSurfaceConfig:
    """Resolve one data surface with additive overrides."""

    cfg = {} if data_cfg is None else {str(key): value for key, value in data_cfg.items()}
    overrides = _mapping_from_any(cfg.get("surface_overrides"), context="data.surface_overrides")
    hinted_sweep_id = _optional_string(overrides.get("corpus_lookup_sweep_id"))
    hinted_sweeps_root = _optional_string(overrides.get("corpus_lookup_sweeps_root"))
    resolved_overrides = {
        key: value
        for key, value in overrides.items()
        if key not in {"corpus_lookup_sweep_id", "corpus_lookup_sweeps_root"}
    }
    resolved_sweep_id = sweep_id if sweep_id is not None else hinted_sweep_id
    resolved_sweeps_root = sweeps_root
    if resolved_sweeps_root is None and hinted_sweeps_root is not None:
        resolved_sweeps_root = Path(hinted_sweeps_root).expanduser().resolve()
    resolved_repo_root = repo_root_from_sweeps_root(resolved_sweeps_root)
    raw_source = resolved_overrides.get("source")
    if raw_source is None:
        raw_source = cfg.get("source")
    if raw_source is None:
        raw_source = "manifest"
    source = str(raw_source).strip().lower()
    raw_corpus_ref = resolved_overrides.get("corpus_ref")
    if raw_corpus_ref is None:
        raw_corpus_ref = cfg.get("corpus_ref")
    corpus_record: dict[str, Any] | None = None
    corpus_ref: str | None = None
    recipe_id: str | None = None
    corpus_id: str | None = None
    corpus_record_path: Path | None = None
    unresolved_corpus_ref = False
    if raw_corpus_ref is not None:
        requested_corpus_ref = str(raw_corpus_ref).strip()
        try:
            corpus_record = load_corpus_record(
                requested_corpus_ref,
                repo_root=resolved_repo_root,
                sweep_id=resolved_sweep_id,
                sweeps_root=resolved_sweeps_root,
            )
        except RuntimeError:
            if not allow_unresolved_corpus_ref:
                raise
            corpus_record = None
            unresolved_corpus_ref = True
            corpus_ref = requested_corpus_ref
            recipe_id, separator, parsed_corpus_id = requested_corpus_ref.partition("/")
            recipe_id = recipe_id.strip() or None
            corpus_id = parsed_corpus_id.strip() if separator and parsed_corpus_id.strip() else None
            corpus_record_path = None
        else:
            raw_resolved_corpus_ref = corpus_record.get("corpus_ref")
            raw_recipe_id = corpus_record.get("recipe_id")
            raw_corpus_id = corpus_record.get("corpus_id")
            raw_corpus_record_path = corpus_record.get("corpus_record_path")
            if not isinstance(raw_resolved_corpus_ref, str) or not raw_resolved_corpus_ref.strip():
                raise RuntimeError("corpus record omitted corpus_ref")
            if not isinstance(raw_recipe_id, str) or not raw_recipe_id.strip():
                raise RuntimeError("corpus record omitted recipe_id")
            if not isinstance(raw_corpus_id, str) or not raw_corpus_id.strip():
                raise RuntimeError("corpus record omitted corpus_id")
            if not isinstance(raw_corpus_record_path, str) or not raw_corpus_record_path.strip():
                raise RuntimeError("corpus record omitted corpus_record_path")
            corpus_ref = str(raw_resolved_corpus_ref)
            recipe_id = str(raw_recipe_id)
            corpus_id = str(raw_corpus_id)
            corpus_record_path = Path(raw_corpus_record_path).expanduser().resolve()
        source = "manifest"
    raw_manifest_path = resolved_overrides.get("manifest_path")
    if raw_manifest_path is None:
        raw_manifest_path = cfg.get("manifest_path")
    if corpus_record is not None:
        raw_manifest_path = (
            _mapping_from_any(corpus_record.get("manifest"), context="corpus_record.manifest").get("manifest_path")
        )
    elif unresolved_corpus_ref:
        raw_manifest_path = None
    manifest_path = None
    if raw_manifest_path is not None:
        manifest_path = Path(str(raw_manifest_path)).expanduser().resolve()
    filter_policy_raw = resolved_overrides.get("filter_policy")
    if filter_policy_raw is None:
        filter_policy_raw = cfg.get("filter_policy")
    filter_policy = None if filter_policy_raw is None else str(filter_policy_raw).strip()
    allow_missing_values_raw = resolved_overrides.get("allow_missing_values")
    if allow_missing_values_raw is None:
        allow_missing_values_raw = cfg.get("allow_missing_values")
    if allow_missing_values_raw is None and corpus_record is not None:
        missing_value_policy = _corpus_record_missing_value_policy(corpus_record)
        if missing_value_policy is not None:
            allow_missing_values_raw = missing_value_policy == "allow_any"
    allow_missing_values = bool(allow_missing_values_raw) if allow_missing_values_raw is not None else False
    dagzoo_provenance_raw = resolved_overrides.get("dagzoo_provenance")
    if dagzoo_provenance_raw is None:
        dagzoo_provenance_raw = cfg.get("dagzoo_provenance")
    if corpus_record is not None:
        dagzoo_provenance_raw = corpus_record.get("dagzoo_provenance")
    dagzoo_provenance = (
        None
        if dagzoo_provenance_raw is None
        else _mapping_from_any(
            dagzoo_provenance_raw,
            context="data.surface_overrides.dagzoo_provenance",
        )
    )
    train_row_cap_raw = resolved_overrides.get("train_row_cap")
    if train_row_cap_raw is None:
        train_row_cap_raw = cfg.get("train_row_cap")
    test_row_cap_raw = resolved_overrides.get("test_row_cap")
    if test_row_cap_raw is None:
        test_row_cap_raw = cfg.get("test_row_cap")
    surface_label_raw = cfg.get("surface_label")
    if surface_label_raw is None:
        surface_label_raw = resolved_overrides.get("surface_label")
    if surface_label_raw is None and corpus_record is not None:
        surface_label_raw = corpus_record.get("surface_label")
    if surface_label_raw is None:
        surface_label_raw = source
    return DataSurfaceConfig(
        surface_label=str(surface_label_raw).strip(),
        source=source,
        manifest_path=manifest_path,
        filter_policy=filter_policy,
        allow_missing_values=allow_missing_values,
        train_row_cap=None if train_row_cap_raw is None else int(train_row_cap_raw),
        test_row_cap=None if test_row_cap_raw is None else int(test_row_cap_raw),
        dagzoo_provenance=dagzoo_provenance,
        corpus_ref=corpus_ref,
        recipe_id=recipe_id,
        corpus_id=corpus_id,
        corpus_record_path=corpus_record_path,
        overrides=resolved_overrides,
    )
