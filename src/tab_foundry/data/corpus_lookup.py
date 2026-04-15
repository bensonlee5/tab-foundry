"""Corpus record lookup and identity helpers."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .corpus_loading import (
    CorpusRecipe,
    CorpusRecipeStorageContext,
    _ensure_non_empty_string,
    _global_recipe_paths,
    _load_corpus_record_payload,
    _load_latest_pointer,
    _load_recipe_from_path,
    _optional_string,
    _recipe_identity_payload,
    _recipe_storage_context,
    _repo_root,
    _sweep_recipe_paths,
    corpus_outputs_root,
    corpus_record_path,
)
from .manifest_characteristics import (
    compute_manifest_characteristics,
    load_manifest_characteristics_sidecar,
    write_manifest_characteristics_sidecar,
)


def _parse_corpus_ref(corpus_ref: str) -> tuple[str, str | None]:
    normalized = _ensure_non_empty_string(corpus_ref, context="corpus_ref")
    if "/" not in normalized:
        return normalized, None
    recipe_id, corpus_id = normalized.split("/", 1)
    return _ensure_non_empty_string(recipe_id, context="corpus_ref recipe_id"), _ensure_non_empty_string(
        corpus_id,
        context="corpus_ref corpus_id",
    )


def _selected_recipe_for_lookup(
    recipe_id: str,
    *,
    repo_root: Path | None = None,
    sweep_id: str | None = None,
    sweeps_root: Path | None = None,
) -> tuple[CorpusRecipe, CorpusRecipeStorageContext] | None:
    resolved_repo_root = (repo_root or _repo_root()).expanduser().resolve()
    if sweep_id is not None:
        sweep_recipe_paths = _sweep_recipe_paths(
            sweep_id,
            repo_root=resolved_repo_root,
            sweeps_root=sweeps_root,
        )
        if sweep_recipe_paths is not None and recipe_id in sweep_recipe_paths:
            recipe = _load_recipe_from_path(recipe_id, sweep_recipe_paths[recipe_id])
            return recipe, _recipe_storage_context(recipe, repo_root=resolved_repo_root)

    try:
        global_recipe_path = _global_recipe_paths(repo_root=resolved_repo_root).get(recipe_id)
        if global_recipe_path is None:
            return None
        recipe = _load_recipe_from_path(recipe_id, global_recipe_path)
    except RuntimeError:
        return None
    return recipe, _recipe_storage_context(recipe, repo_root=resolved_repo_root)


def _candidate_corpus_record_paths(
    recipe_id: str,
    *,
    repo_root: Path | None = None,
) -> list[Path]:
    recipe_root = corpus_outputs_root(repo_root=repo_root) / recipe_id
    if not recipe_root.exists():
        return []
    return [
        path / "corpus_record.json"
        for path in sorted(recipe_root.iterdir())
        if path.is_dir() and not path.name.startswith(".")
    ]


def _resolved_lookup_repo_root(repo_root: Path | None) -> Path:
    return (repo_root or _repo_root()).expanduser().resolve()


def _path_suffix_after_anchor(path: Path, *, anchor: tuple[str, ...]) -> Path | None:
    parts = path.parts
    anchor_length = len(anchor)
    for index in range(len(parts) - anchor_length + 1):
        if tuple(parts[index : index + anchor_length]) == anchor:
            suffix = parts[index + anchor_length :]
            return Path(*suffix) if suffix else Path()
    return None


def _relocate_repo_anchored_path(
    value: Any,
    *,
    repo_root: Path | None = None,
    allow_missing_parent: bool = False,
) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    raw_path = Path(value).expanduser()
    if not raw_path.is_absolute():
        return str(raw_path)

    resolved_repo_root = _resolved_lookup_repo_root(repo_root)
    relocation_roots = (
        (("outputs", "corpora"), corpus_outputs_root(repo_root=resolved_repo_root)),
        (("reference", "corpus_recipes"), resolved_repo_root / "reference" / "corpus_recipes"),
        (
            ("reference", "system_delta_sweeps"),
            resolved_repo_root / "reference" / "system_delta_sweeps",
        ),
    )
    for anchor, target_root in relocation_roots:
        suffix = _path_suffix_after_anchor(raw_path, anchor=anchor)
        if suffix is None:
            continue
        relocated = (target_root / suffix).resolve()
        if relocated.exists():
            return str(relocated)
        if allow_missing_parent and relocated.parent.exists():
            return str(relocated)
    return str(raw_path.resolve())


def _record_matches_recipe(
    record: Mapping[str, Any],
    recipe: CorpusRecipe,
    *,
    storage: CorpusRecipeStorageContext,
) -> bool:
    expected_recipe_payload = _recipe_identity_payload(recipe)
    recorded_recipe_relative_path = _optional_string(record.get("recipe_relative_path"))
    recorded_recipe_identity = _optional_string(record.get("recipe_identity"))
    raw_recorded_recipe = record.get("recipe")
    recorded_recipe_payload = (
        None
        if not isinstance(raw_recorded_recipe, Mapping)
        else {
            str(key): value
            for key, value in raw_recorded_recipe.items()
            if str(key) != "recipe_path"
        }
    )
    if recorded_recipe_relative_path is not None:
        if storage.recipe_relative_path is None:
            return False
        if recorded_recipe_relative_path != storage.recipe_relative_path:
            return False
        if recorded_recipe_identity == storage.recipe_identity:
            return True
        return recorded_recipe_payload == expected_recipe_payload

    recorded_recipe_path = record.get("recipe_path")
    if not isinstance(recorded_recipe_path, str) or not recorded_recipe_path.strip():
        return False
    recipe_path_matches = (
        Path(recorded_recipe_path).expanduser().resolve() == recipe.recipe_path.expanduser().resolve()
    )
    if not recipe_path_matches:
        return False
    if storage.uses_scoped_identity and recorded_recipe_relative_path is None:
        return False
    if recorded_recipe_identity == storage.recipe_identity:
        return True
    if recorded_recipe_payload == expected_recipe_payload:
        return True
    return (
        not storage.uses_scoped_identity
        and recorded_recipe_identity is None
        and recorded_recipe_relative_path is None
    )


def _load_record_from_latest_pointer(
    recipe: CorpusRecipe,
    storage: CorpusRecipeStorageContext,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any] | None:
    latest = _load_latest_pointer(
        recipe.recipe_id,
        repo_root=repo_root,
        recipe_identity=(
            storage.recipe_identity if storage.uses_scoped_identity else None
        ),
    )
    if latest is None:
        return None
    relocated_latest_record_path = _relocate_repo_anchored_path(
        latest.get("corpus_record_path"),
        repo_root=repo_root,
    )
    if relocated_latest_record_path is None:
        return None
    try:
        record = _load_corpus_record_payload(
            Path(relocated_latest_record_path).expanduser().resolve(),
            context=f"corpus latest pointer record for {recipe.recipe_id!r}",
        )
    except RuntimeError:
        return None
    prepared_record = _relocate_corpus_record_paths(record, repo_root=repo_root)
    return prepared_record if _record_matches_recipe(prepared_record, recipe, storage=storage) else None


def _copy_mapping_payload(record: Mapping[str, Any]) -> dict[str, Any]:
    payload = json.loads(json.dumps(record))
    if not isinstance(payload, Mapping):
        raise RuntimeError("copied corpus record must remain a mapping")
    return {str(key): value for key, value in payload.items()}


def _relocate_corpus_record_paths(
    record: Mapping[str, Any],
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    relocated = _copy_mapping_payload(record)

    relocated_corpus_record_path = _relocate_repo_anchored_path(
        relocated.get("corpus_record_path"),
        repo_root=repo_root,
    )
    if relocated_corpus_record_path is not None:
        relocated["corpus_record_path"] = relocated_corpus_record_path

    relocated_recipe_path = _relocate_repo_anchored_path(
        relocated.get("recipe_path"),
        repo_root=repo_root,
    )
    if relocated_recipe_path is not None:
        relocated["recipe_path"] = relocated_recipe_path

    artifacts = relocated.get("artifacts")
    if isinstance(artifacts, dict):
        for key, allow_missing_parent in (
            ("corpus_root", False),
            ("manifest_path", False),
            ("latest_pointer_path", False),
            ("publish_inventory_path", True),
        ):
            relocated_value = _relocate_repo_anchored_path(
                artifacts.get(key),
                repo_root=repo_root,
                allow_missing_parent=allow_missing_parent,
            )
            if relocated_value is not None:
                artifacts[key] = relocated_value

    manifest = relocated.get("manifest")
    if isinstance(manifest, dict):
        relocated_manifest_path = _relocate_repo_anchored_path(
            manifest.get("manifest_path"),
            repo_root=repo_root,
        )
        if relocated_manifest_path is not None:
            manifest["manifest_path"] = relocated_manifest_path

        characteristics = manifest.get("characteristics")
        if isinstance(characteristics, dict):
            relocated_sidecar_path = _relocate_repo_anchored_path(
                characteristics.get("sidecar_path"),
                repo_root=repo_root,
                allow_missing_parent=True,
            )
            if relocated_sidecar_path is not None:
                characteristics["sidecar_path"] = relocated_sidecar_path

    dagzoo_provenance = relocated.get("dagzoo_provenance")
    if isinstance(dagzoo_provenance, dict):
        for list_key, require_parent in (
            ("filter_manifest_paths", False),
            ("curated_root_lineage", False),
        ):
            values = dagzoo_provenance.get(list_key)
            if isinstance(values, list):
                dagzoo_provenance[list_key] = [
                    _relocate_repo_anchored_path(
                        value,
                        repo_root=repo_root,
                        allow_missing_parent=require_parent,
                    )
                    if isinstance(value, str)
                    else value
                    for value in values
                ]

        invocations = dagzoo_provenance.get("invocations")
        if isinstance(invocations, list):
            for invocation in invocations:
                if not isinstance(invocation, dict):
                    continue
                for key in ("invocation_root", "rendered_config_path"):
                    relocated_value = _relocate_repo_anchored_path(
                        invocation.get(key),
                        repo_root=repo_root,
                    )
                    if relocated_value is not None:
                        invocation[key] = relocated_value
                handoff = invocation.get("handoff")
                if isinstance(handoff, dict):
                    for key in ("handoff_manifest_path", "generated_dir"):
                        relocated_value = _relocate_repo_anchored_path(
                            handoff.get(key),
                            repo_root=repo_root,
                        )
                        if relocated_value is not None:
                            handoff[key] = relocated_value

    return relocated


def hydrate_corpus_record_manifest_characteristics(record: Mapping[str, Any]) -> dict[str, Any]:
    manifest = record.get("manifest")
    if not isinstance(manifest, Mapping):
        return _copy_mapping_payload(record)

    manifest_path_raw = manifest.get("manifest_path")
    if not isinstance(manifest_path_raw, str) or not manifest_path_raw.strip():
        return _copy_mapping_payload(record)
    manifest_path = Path(manifest_path_raw).expanduser().resolve()

    raw_characteristics = manifest.get("characteristics")
    characteristics = raw_characteristics if isinstance(raw_characteristics, Mapping) else {}
    if characteristics.get("record_count") is not None:
        return _copy_mapping_payload(record)

    sidecar_path_raw = characteristics.get("sidecar_path")
    sidecar_path = (
        None
        if not isinstance(sidecar_path_raw, str) or not sidecar_path_raw.strip()
        else Path(sidecar_path_raw).expanduser().resolve()
    )
    full_characteristics = (
        None
        if sidecar_path is None
        else load_manifest_characteristics_sidecar(sidecar_path)
    )
    if full_characteristics is None:
        if sidecar_path is None:
            full_characteristics = compute_manifest_characteristics(manifest_path)
        else:
            full_characteristics = write_manifest_characteristics_sidecar(
                manifest_path=manifest_path,
                sidecar_path=sidecar_path,
            )

    hydrated = _copy_mapping_payload(record)
    hydrated_manifest = hydrated.get("manifest")
    if not isinstance(hydrated_manifest, dict):
        return hydrated
    hydrated_manifest["characteristics"] = full_characteristics
    return hydrated


def _matching_corpus_records_for_recipe(
    recipe: CorpusRecipe,
    storage: CorpusRecipeStorageContext,
    *,
    repo_root: Path | None = None,
) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    for record_path in _candidate_corpus_record_paths(recipe.recipe_id, repo_root=repo_root):
        if not record_path.exists():
            continue
        record = _load_corpus_record_payload(
            record_path,
            context=f"corpus record candidate for {recipe.recipe_id!r}",
        )
        prepared_record = _relocate_corpus_record_paths(record, repo_root=repo_root)
        if _record_matches_recipe(prepared_record, recipe, storage=storage):
            matches.append(prepared_record)
    return matches


def load_corpus_record(
    corpus_ref: str,
    *,
    repo_root: Path | None = None,
    sweep_id: str | None = None,
    sweeps_root: Path | None = None,
    hydrate_characteristics: bool = False,
) -> dict[str, Any]:
    recipe_id, corpus_id = _parse_corpus_ref(corpus_ref)
    if corpus_id is not None:
        record = _relocate_corpus_record_paths(
            _load_corpus_record_payload(
                corpus_record_path(recipe_id=recipe_id, corpus_id=corpus_id, repo_root=repo_root),
                context=f"corpus record {recipe_id}/{corpus_id}",
            ),
            repo_root=repo_root,
        )
        return (
            hydrate_corpus_record_manifest_characteristics(record)
            if hydrate_characteristics
            else record
        )

    selected_recipe = _selected_recipe_for_lookup(
        recipe_id,
        repo_root=repo_root,
        sweep_id=sweep_id,
        sweeps_root=sweeps_root,
    )
    if selected_recipe is not None:
        recipe, storage = selected_recipe
        latest_record = _load_record_from_latest_pointer(recipe, storage, repo_root=repo_root)
        if latest_record is not None:
            return (
                hydrate_corpus_record_manifest_characteristics(latest_record)
                if hydrate_characteristics
                else latest_record
            )
        matches = _matching_corpus_records_for_recipe(recipe, storage, repo_root=repo_root)
        recipe_root = corpus_outputs_root(repo_root=repo_root) / recipe_id
        if len(matches) == 1:
            return (
                hydrate_corpus_record_manifest_characteristics(matches[0])
                if hydrate_characteristics
                else matches[0]
            )
        if not matches:
            raise RuntimeError(
                f"no local corpus materialization found for recipe {recipe_id!r} under {recipe_root}"
            )
        raise RuntimeError(
            f"multiple corpora exist for recipe {recipe_id!r} but no matching latest pointer is present: {recipe_root}"
        )

    latest = _load_latest_pointer(recipe_id, repo_root=repo_root)
    if latest is not None:
        corpus_id = _ensure_non_empty_string(
            latest.get("corpus_id"),
            context=f"latest corpus_id for recipe {recipe_id!r}",
        )
        record = _load_corpus_record_payload(
            corpus_record_path(recipe_id=recipe_id, corpus_id=corpus_id, repo_root=repo_root),
            context=f"corpus record {recipe_id}/{corpus_id}",
        )
        record = _relocate_corpus_record_paths(record, repo_root=repo_root)
        return (
            hydrate_corpus_record_manifest_characteristics(record)
            if hydrate_characteristics
            else record
        )

    recipe_root = corpus_outputs_root(repo_root=repo_root) / recipe_id
    candidates = sorted(
        path.name
        for path in recipe_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ) if recipe_root.exists() else []
    if len(candidates) == 1:
        corpus_id = candidates[0]
        record = _load_corpus_record_payload(
            corpus_record_path(recipe_id=recipe_id, corpus_id=corpus_id, repo_root=repo_root),
            context=f"corpus record {recipe_id}/{corpus_id}",
        )
        record = _relocate_corpus_record_paths(record, repo_root=repo_root)
        return (
            hydrate_corpus_record_manifest_characteristics(record)
            if hydrate_characteristics
            else record
        )
    if not candidates:
        raise RuntimeError(
            f"no local corpus materialization found for recipe {recipe_id!r} under {recipe_root}"
        )
    raise RuntimeError(
        f"multiple corpora exist for recipe {recipe_id!r} but no latest.json pointer is present: {recipe_root}"
    )


def _existing_path(value: Any, *, require_dir: bool) -> Path | None:
    if not isinstance(value, str) or not value.strip():
        return None
    candidate = Path(value).expanduser().resolve()
    if not candidate.exists():
        return None
    if require_dir and not candidate.is_dir():
        return None
    if not require_dir and not candidate.is_file():
        return None
    return candidate


def _corpus_record_is_complete_for_reuse(record: Mapping[str, Any]) -> bool:
    if _existing_path(record.get("corpus_record_path"), require_dir=False) is None:
        return False

    artifacts = record.get("artifacts")
    manifest = record.get("manifest")
    dagzoo_provenance = record.get("dagzoo_provenance")
    if not isinstance(artifacts, Mapping) or not isinstance(manifest, Mapping) or not isinstance(
        dagzoo_provenance,
        Mapping,
    ):
        return False
    if _existing_path(artifacts.get("corpus_root"), require_dir=True) is None:
        return False
    if _existing_path(manifest.get("manifest_path"), require_dir=False) is None:
        return False

    invocations = dagzoo_provenance.get("invocations")
    if not isinstance(invocations, list):
        return False
    for invocation in invocations:
        if not isinstance(invocation, Mapping):
            return False
        if _existing_path(invocation.get("invocation_root"), require_dir=True) is None:
            return False
        rendered_config_path = invocation.get("rendered_config_path")
        if rendered_config_path is not None and _existing_path(rendered_config_path, require_dir=False) is None:
            return False
        handoff = invocation.get("handoff")
        if not isinstance(handoff, Mapping):
            return False
        if _existing_path(handoff.get("handoff_manifest_path"), require_dir=False) is None:
            return False
        if _existing_path(handoff.get("generated_dir"), require_dir=True) is None:
            return False
    return True


def _load_reusable_corpus_record(
    corpus_ref: str,
    *,
    repo_root: Path | None = None,
    sweep_id: str | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any] | None:
    try:
        record = load_corpus_record(
            corpus_ref,
            repo_root=repo_root,
            sweep_id=sweep_id,
            sweeps_root=sweeps_root,
        )
    except RuntimeError:
        return None
    return record if _corpus_record_is_complete_for_reuse(record) else None
