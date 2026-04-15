"""Publishable corpus inventory helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, cast

from tab_foundry.repo_paths import normalize_repo_relative_path, repo_root as shared_repo_root
from tab_foundry.timestamps import utc_now


CORPUS_PUBLISH_INVENTORY_SCHEMA = "tab-foundry-corpus-publish-inventory-v1"


def _repo_root(repo_root: Path | None = None) -> Path:
    return (repo_root or shared_repo_root()).expanduser().resolve()


def _ensure_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _ensure_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{context} must be a non-empty string")
    return str(value)


def _ensure_local_path(value: Any, *, context: str, repo_root: Path) -> Path:
    raw_value = _ensure_string(value, context=context)
    path = Path(raw_value).expanduser()
    return path.resolve() if path.is_absolute() else (repo_root / path).resolve()


def _cache_relative_path_for_corpus_file(
    *,
    recipe_id: str,
    corpus_id: str,
    corpus_root: Path,
    file_path: Path,
) -> str:
    return str(
        Path(recipe_id)
        / corpus_id
        / file_path.expanduser().resolve().relative_to(corpus_root.expanduser().resolve())
    )


def _publish_entry(
    *,
    repo_root: Path,
    local_path: Path,
    cache_relative_path: str,
    category: str,
) -> dict[str, Any]:
    resolved_local_path = local_path.expanduser().resolve()
    return {
        "category": str(category),
        "local_repo_relative_path": normalize_repo_relative_path(
            resolved_local_path,
            root=repo_root,
        ),
        "cache_relative_path": str(cache_relative_path),
        "size_bytes": int(resolved_local_path.stat().st_size),
    }


def build_corpus_publish_inventory(
    record: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    resolved_repo_root = _repo_root(repo_root)
    artifacts = _ensure_mapping(record.get("artifacts"), context="corpus_record.artifacts")
    recipe_id = _ensure_string(record.get("recipe_id"), context="corpus_record.recipe_id")
    corpus_id = _ensure_string(record.get("corpus_id"), context="corpus_record.corpus_id")
    corpus_ref = _ensure_string(record.get("corpus_ref"), context="corpus_record.corpus_ref")
    corpus_root = _ensure_local_path(
        artifacts.get("corpus_root"),
        context="corpus_record.artifacts.corpus_root",
        repo_root=resolved_repo_root,
    )
    latest_pointer_path = _ensure_local_path(
        artifacts.get("latest_pointer_path"),
        context="corpus_record.artifacts.latest_pointer_path",
        repo_root=resolved_repo_root,
    )
    inventory_path = corpus_root / "publish_inventory.json"
    entries: list[dict[str, Any]] = []
    for file_path in sorted(path for path in corpus_root.rglob("*") if path.is_file()):
        if file_path.expanduser().resolve() == inventory_path.expanduser().resolve():
            continue
        entries.append(
            _publish_entry(
                repo_root=resolved_repo_root,
                local_path=file_path,
                cache_relative_path=_cache_relative_path_for_corpus_file(
                    recipe_id=recipe_id,
                    corpus_id=corpus_id,
                    corpus_root=corpus_root,
                    file_path=file_path,
                ),
                category="corpus_file",
            )
        )
    entries.append(
        _publish_entry(
            repo_root=resolved_repo_root,
            local_path=latest_pointer_path,
            cache_relative_path=str(Path(recipe_id) / latest_pointer_path.name),
            category="latest_pointer",
        )
    )
    total_bytes = sum(int(entry["size_bytes"]) for entry in entries)
    return {
        "schema": CORPUS_PUBLISH_INVENTORY_SCHEMA,
        "generated_at_utc": utc_now(),
        "recipe_id": recipe_id,
        "corpus_id": corpus_id,
        "corpus_ref": corpus_ref,
        "corpus_root_repo_relative_path": normalize_repo_relative_path(
            corpus_root,
            root=resolved_repo_root,
        ),
        "inventory_repo_relative_path": normalize_repo_relative_path(
            inventory_path,
            root=resolved_repo_root,
        ),
        "inventory_cache_relative_path": str(Path(recipe_id) / corpus_id / inventory_path.name),
        "entries": entries,
        "summary": {
            "entry_count": len(entries),
            "total_bytes": int(total_bytes),
        },
    }


def ensure_corpus_publish_inventory(
    record: dict[str, Any],
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    resolved_repo_root = _repo_root(repo_root)
    artifacts = _ensure_mapping(record.get("artifacts"), context="corpus_record.artifacts")
    corpus_root = _ensure_local_path(
        artifacts.get("corpus_root"),
        context="corpus_record.artifacts.corpus_root",
        repo_root=resolved_repo_root,
    )
    record_path = _ensure_local_path(
        record.get("corpus_record_path"),
        context="corpus_record.corpus_record_path",
        repo_root=resolved_repo_root,
    )
    inventory_path = corpus_root / "publish_inventory.json"
    if inventory_path.exists():
        inventory = cast(
            dict[str, Any],
            json.loads(inventory_path.read_text(encoding="utf-8")),
        )
    else:
        inventory = build_corpus_publish_inventory(record, repo_root=resolved_repo_root)
        inventory_path.write_text(
            json.dumps(inventory, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )

    artifacts["publish_inventory_path"] = str(inventory_path.resolve())
    record["artifacts"] = artifacts
    record["publish_inventory"] = {
        "schema": inventory.get("schema"),
        "inventory_repo_relative_path": inventory.get("inventory_repo_relative_path"),
        "inventory_cache_relative_path": inventory.get("inventory_cache_relative_path"),
        "entry_count": inventory.get("summary", {}).get("entry_count"),
        "total_bytes": inventory.get("summary", {}).get("total_bytes"),
    }
    record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return record
