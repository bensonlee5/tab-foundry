"""Repo-owned Dagzoo handoff loading helpers."""

from __future__ import annotations

from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, cast

import tab_realdata_hub.manifest as _manifest_module
from tab_realdata_hub.dagzoo_handoff import (
    DAGZOO_HANDOFF_SCHEMA_NAME,
    DagzooGeneratedIdentityAccumulator,
    DagzooHandoffInfo,
    load_dagzoo_handoff_info as _load_upstream_dagzoo_handoff_info,
    stable_dagzoo_generated_corpus_id,
    verify_dagzoo_handoff_matches_generated_corpus,
)


SUPPORTED_DAGZOO_HANDOFF_SCHEMA_VERSIONS = (1, 2, 3, 4, 5)
DAGZOO_HANDOFF_SCHEMA_VERSION = 5

__all__ = [
    "DAGZOO_HANDOFF_SCHEMA_VERSION",
    "SUPPORTED_DAGZOO_HANDOFF_SCHEMA_VERSIONS",
    "DagzooGeneratedIdentityAccumulator",
    "DagzooHandoffInfo",
    "load_dagzoo_handoff_info",
    "stable_dagzoo_generated_corpus_id",
    "verify_dagzoo_handoff_matches_generated_corpus",
]


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _read_json_dict(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise RuntimeError(f"dagzoo handoff manifest must contain a JSON object: path={path}")
    return cast(dict[str, Any], payload)


def _require_mapping(
    payload: Mapping[str, Any],
    key: str,
    *,
    path: Path,
) -> Mapping[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise RuntimeError(f"dagzoo handoff manifest field must be an object: path={path}, key={key}")
    return cast(Mapping[str, Any], value)


def _require_optional_mapping(
    payload: Mapping[str, Any],
    key: str,
    *,
    path: Path,
) -> Mapping[str, Any] | None:
    value = payload.get(key)
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise RuntimeError(f"dagzoo handoff manifest field must be an object: path={path}, key={key}")
    return cast(Mapping[str, Any], value)


def _require_non_empty_string(
    payload: Mapping[str, Any],
    key: str,
    *,
    path: Path,
) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(
            f"dagzoo handoff manifest field must be a non-empty string: path={path}, key={key}"
        )
    return value


def _resolve_relative_path(raw: str, *, path: Path, field_key: str) -> Path:
    relative = Path(raw)
    if relative.is_absolute():
        raise RuntimeError(
            "dagzoo handoff manifest field must be relative: "
            f"path={path}, key={field_key}, value={raw!r}"
        )
    resolved = (path.parent / relative).resolve()
    try:
        _ = resolved.relative_to(path.parent)
    except ValueError as exc:
        raise RuntimeError(
            "dagzoo handoff path escapes the handoff root: "
            f"path={path}, key={field_key}, value={raw!r}"
        ) from exc
    return resolved


def _teacher_summary_from_v1(
    payload: Mapping[str, Any],
    *,
    path: Path,
) -> dict[str, Any] | None:
    provenance = _require_optional_mapping(payload, "provenance", path=path)
    if provenance is None:
        return None
    enabled = provenance.get("teacher_conditional_export")
    if enabled is not True:
        return None
    metric_definition = provenance.get("teacher_conditional_metric_definition")
    target_split = provenance.get("target_split")
    if not isinstance(metric_definition, str) or not metric_definition.strip():
        return None
    if not isinstance(target_split, str) or not target_split.strip():
        target_split = "test"
    return {
        "enabled": True,
        "metric_definition": metric_definition,
        "target_split": target_split,
    }


def _teacher_summary_from_v2(
    payload: Mapping[str, Any],
    *,
    path: Path,
) -> dict[str, Any] | None:
    teacher_conditionals = _require_optional_mapping(payload, "teacher_conditionals", path=path)
    if teacher_conditionals is None:
        return None
    enabled = teacher_conditionals.get("enabled")
    if enabled is not True:
        raise RuntimeError(
            "dagzoo handoff teacher_conditionals.enabled must equal true when present: "
            f"path={path}"
        )
    metric_definition = _require_non_empty_string(
        teacher_conditionals,
        "metric_definition",
        path=path,
    )
    target_split = _require_non_empty_string(
        teacher_conditionals,
        "target_split",
        path=path,
    )
    return {
        "enabled": True,
        "metric_definition": metric_definition,
        "target_split": target_split,
    }


def _normalized_range_mapping(
    payload: Mapping[str, Any],
    key: str,
    *,
    path: Path,
) -> dict[str, Any] | None:
    range_payload = _require_optional_mapping(payload, key, path=path)
    if range_payload is None:
        return None
    normalized: dict[str, Any] = {}
    minimum = range_payload.get("min")
    maximum = range_payload.get("max")
    if minimum is not None:
        if isinstance(minimum, bool) or not isinstance(minimum, (int, float)):
            raise RuntimeError(
                "dagzoo handoff range bound must be numeric when present: "
                f"path={path}, key={key}.min"
            )
        normalized["min"] = float(minimum) if isinstance(minimum, float) else int(minimum)
    if maximum is not None:
        if isinstance(maximum, bool) or not isinstance(maximum, (int, float)):
            raise RuntimeError(
                "dagzoo handoff range bound must be numeric when present: "
                f"path={path}, key={key}.max"
            )
        normalized["max"] = float(maximum) if isinstance(maximum, float) else int(maximum)
    return normalized or None


def _provenance_summary_from_v3(
    payload: Mapping[str, Any],
    *,
    path: Path,
) -> dict[str, Any] | None:
    provenance = _require_optional_mapping(payload, "provenance", path=path)
    if provenance is None:
        return None
    summary: dict[str, Any] = {}
    target_derivation = provenance.get("target_derivation")
    if target_derivation is not None:
        if not isinstance(target_derivation, str) or not target_derivation.strip():
            raise RuntimeError(
                "dagzoo handoff provenance.target_derivation must be a non-empty string "
                f"when present: path={path}"
            )
        summary["target_derivation"] = target_derivation
    for key in (
        "target_relevant_feature_count_range",
        "target_relevant_feature_fraction_range",
    ):
        normalized = _normalized_range_mapping(provenance, key, path=path)
        if normalized is not None:
            summary[key] = normalized
    return summary or None


def load_dagzoo_handoff_info(handoff_manifest_path: Path) -> DagzooHandoffInfo:
    """Load the Dagzoo handoff subset consumed by tab-foundry."""

    path = handoff_manifest_path.expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"dagzoo handoff manifest not found: {path}")
    payload = _read_json_dict(path)
    schema_version = payload.get("schema_version")
    if schema_version != DAGZOO_HANDOFF_SCHEMA_VERSION:
        return _load_upstream_dagzoo_handoff_info(path)

    schema_name = _require_non_empty_string(payload, "schema_name", path=path)
    if schema_name != DAGZOO_HANDOFF_SCHEMA_NAME:
        raise RuntimeError(
            "Unsupported dagzoo handoff schema_name: "
            f"path={path}, value={schema_name!r}, expected={DAGZOO_HANDOFF_SCHEMA_NAME!r}"
        )
    if schema_version not in SUPPORTED_DAGZOO_HANDOFF_SCHEMA_VERSIONS:
        raise RuntimeError(
            "Unsupported dagzoo handoff schema_version: "
            f"path={path}, value={schema_version!r}, expected one of "
            f"{SUPPORTED_DAGZOO_HANDOFF_SCHEMA_VERSIONS}"
        )

    identity = _require_mapping(payload, "identity", path=path)
    source_family = _require_non_empty_string(identity, "source_family", path=path)
    generate_run_id = _require_non_empty_string(identity, "generate_run_id", path=path)
    generated_corpus_id = _require_non_empty_string(identity, "generated_corpus_id", path=path)

    artifacts_relative = _require_mapping(payload, "artifacts_relative", path=path)
    generated_dir_rel = _require_non_empty_string(artifacts_relative, "generated_dir", path=path)
    generated_dir = _resolve_relative_path(
        generated_dir_rel,
        path=path,
        field_key="artifacts_relative.generated_dir",
    )
    curated_dir_raw = artifacts_relative.get("curated_dir")
    curated_dir = None
    if curated_dir_raw is not None:
        if not isinstance(curated_dir_raw, str) or not curated_dir_raw.strip():
            raise RuntimeError(
                "dagzoo handoff manifest field must be a non-empty string when present: "
                f"path={path}, key=artifacts_relative.curated_dir"
            )
        curated_dir = _resolve_relative_path(
            curated_dir_raw,
            path=path,
            field_key="artifacts_relative.curated_dir",
        )

    return DagzooHandoffInfo(
        handoff_manifest_path=path,
        handoff_manifest_sha256=_sha256_path(path),
        source_family=source_family,
        generate_run_id=generate_run_id,
        generated_corpus_id=generated_corpus_id,
        generated_dir=generated_dir,
        curated_dir=curated_dir,
        provenance=_provenance_summary_from_v3(payload, path=path),
        teacher_conditionals=None,
    )


_manifest_module.load_dagzoo_handoff_info = load_dagzoo_handoff_info
