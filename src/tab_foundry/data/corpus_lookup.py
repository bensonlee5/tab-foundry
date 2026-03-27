"""Corpus record lookup and identity helpers."""

from __future__ import annotations

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
    _recipe_storage_context,
    _repo_root,
    _sweep_recipe_paths,
    corpus_outputs_root,
    corpus_record_path,
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


def _record_matches_recipe(
    record: Mapping[str, Any],
    recipe: CorpusRecipe,
    *,
    storage: CorpusRecipeStorageContext,
) -> bool:
    recorded_recipe_relative_path = _optional_string(record.get("recipe_relative_path"))
    if recorded_recipe_relative_path is not None and storage.recipe_relative_path is not None:
        return recorded_recipe_relative_path == storage.recipe_relative_path
    if recorded_recipe_relative_path is not None and storage.recipe_relative_path is None:
        return False

    recorded_recipe_identity = _optional_string(record.get("recipe_identity"))
    recorded_recipe_path = record.get("recipe_path")
    if not isinstance(recorded_recipe_path, str) or not recorded_recipe_path.strip():
        return False
    recipe_path_matches = (
        Path(recorded_recipe_path).expanduser().resolve() == recipe.recipe_path.expanduser().resolve()
    )
    if recorded_recipe_identity is not None:
        if recorded_recipe_identity == storage.recipe_identity:
            return True
        return recipe_path_matches
    if not recipe_path_matches:
        return False
    return not storage.uses_scoped_identity


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
    latest_record_path = latest.get("corpus_record_path")
    if not isinstance(latest_record_path, str) or not latest_record_path.strip():
        return None
    try:
        record = _load_corpus_record_payload(
            Path(latest_record_path).expanduser().resolve(),
            context=f"corpus latest pointer record for {recipe.recipe_id!r}",
        )
    except RuntimeError:
        return None
    return record if _record_matches_recipe(record, recipe, storage=storage) else None


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
        if _record_matches_recipe(record, recipe, storage=storage):
            matches.append(record)
    return matches


def load_corpus_record(
    corpus_ref: str,
    *,
    repo_root: Path | None = None,
    sweep_id: str | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    recipe_id, corpus_id = _parse_corpus_ref(corpus_ref)
    if corpus_id is not None:
        return _load_corpus_record_payload(
            corpus_record_path(recipe_id=recipe_id, corpus_id=corpus_id, repo_root=repo_root),
            context=f"corpus record {recipe_id}/{corpus_id}",
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
            return latest_record
        matches = _matching_corpus_records_for_recipe(recipe, storage, repo_root=repo_root)
        recipe_root = corpus_outputs_root(repo_root=repo_root) / recipe_id
        if len(matches) == 1:
            return matches[0]
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
        return _load_corpus_record_payload(
            corpus_record_path(recipe_id=recipe_id, corpus_id=corpus_id, repo_root=repo_root),
            context=f"corpus record {recipe_id}/{corpus_id}",
        )

    recipe_root = corpus_outputs_root(repo_root=repo_root) / recipe_id
    candidates = sorted(
        path.name
        for path in recipe_root.iterdir()
        if path.is_dir() and not path.name.startswith(".")
    ) if recipe_root.exists() else []
    if len(candidates) == 1:
        corpus_id = candidates[0]
        return _load_corpus_record_payload(
            corpus_record_path(recipe_id=recipe_id, corpus_id=corpus_id, repo_root=repo_root),
            context=f"corpus record {recipe_id}/{corpus_id}",
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
