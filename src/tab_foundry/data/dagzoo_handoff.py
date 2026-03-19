"""Helpers for consuming dagzoo handoff manifests."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Mapping, cast


DAGZOO_HANDOFF_SCHEMA_NAME = "dagzoo_generate_handoff_manifest"
DAGZOO_HANDOFF_SCHEMA_VERSION = 1


@dataclass(slots=True, frozen=True)
class DagzooHandoffInfo:
    """Validated subset of one dagzoo handoff manifest."""

    handoff_manifest_path: Path
    handoff_manifest_sha256: str
    source_family: str
    generate_run_id: str
    generated_corpus_id: str
    generated_dir: Path
    recommended_training_corpus: str
    recommended_training_artifact_key: str
    curation_policy: str

    def to_summary_dict(self) -> dict[str, Any]:
        return {
            "handoff_manifest_path": str(self.handoff_manifest_path),
            "handoff_manifest_sha256": self.handoff_manifest_sha256,
            "source_family": self.source_family,
            "generate_run_id": self.generate_run_id,
            "generated_corpus_id": self.generated_corpus_id,
            "generated_dir": str(self.generated_dir),
            "recommended_training_corpus": self.recommended_training_corpus,
            "recommended_training_artifact_key": self.recommended_training_artifact_key,
            "curation_policy": self.curation_policy,
        }


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


def _require_relative_path(
    payload: Mapping[str, Any],
    key: str,
    *,
    path: Path,
) -> str:
    raw = _require_non_empty_string(payload, key, path=path)
    relative = Path(raw)
    if relative.is_absolute():
        raise RuntimeError(
            f"dagzoo handoff manifest field must be relative: path={path}, key={key}, value={raw!r}"
        )
    return raw


def _sha256_path(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def load_dagzoo_handoff_info(handoff_manifest_path: Path) -> DagzooHandoffInfo:
    """Load and validate the dagzoo handoff subset used by tab-foundry."""

    path = handoff_manifest_path.expanduser().resolve()
    if not path.exists():
        raise RuntimeError(f"dagzoo handoff manifest not found: {path}")
    payload = _read_json_dict(path)

    schema_name = _require_non_empty_string(payload, "schema_name", path=path)
    if schema_name != DAGZOO_HANDOFF_SCHEMA_NAME:
        raise RuntimeError(
            "Unsupported dagzoo handoff schema_name: "
            f"path={path}, value={schema_name!r}, expected={DAGZOO_HANDOFF_SCHEMA_NAME!r}"
        )
    schema_version = payload.get("schema_version")
    if schema_version != DAGZOO_HANDOFF_SCHEMA_VERSION:
        raise RuntimeError(
            "Unsupported dagzoo handoff schema_version: "
            f"path={path}, value={schema_version!r}, expected={DAGZOO_HANDOFF_SCHEMA_VERSION}"
        )

    identity = _require_mapping(payload, "identity", path=path)
    source_family = _require_non_empty_string(identity, "source_family", path=path)
    generate_run_id = _require_non_empty_string(identity, "generate_run_id", path=path)
    generated_corpus_id = _require_non_empty_string(identity, "generated_corpus_id", path=path)

    artifacts_relative = _require_mapping(payload, "artifacts_relative", path=path)
    run_root = _require_non_empty_string(artifacts_relative, "run_root", path=path)
    if run_root != ".":
        raise RuntimeError(
            "dagzoo handoff artifacts_relative.run_root must equal '.': "
            f"path={path}, value={run_root!r}"
        )
    generated_dir_rel = _require_relative_path(artifacts_relative, "generated_dir", path=path)
    generated_dir = (path.parent / generated_dir_rel).resolve()
    try:
        _ = generated_dir.relative_to(path.parent)
    except ValueError as exc:
        raise RuntimeError(
            "dagzoo handoff generated_dir escapes the handoff root: "
            f"path={path}, value={generated_dir_rel!r}"
        ) from exc

    defaults = _require_mapping(payload, "defaults", path=path)
    recommended_training_corpus = _require_non_empty_string(
        defaults,
        "recommended_training_corpus",
        path=path,
    )
    recommended_training_artifact_key = _require_non_empty_string(
        defaults,
        "recommended_training_artifact_key",
        path=path,
    )
    curation_policy = _require_non_empty_string(defaults, "curation_policy", path=path)

    return DagzooHandoffInfo(
        handoff_manifest_path=path,
        handoff_manifest_sha256=_sha256_path(path),
        source_family=source_family,
        generate_run_id=generate_run_id,
        generated_corpus_id=generated_corpus_id,
        generated_dir=generated_dir,
        recommended_training_corpus=recommended_training_corpus,
        recommended_training_artifact_key=recommended_training_artifact_key,
        curation_policy=curation_policy,
    )
