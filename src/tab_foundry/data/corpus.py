"""First-class corpus recipe, materialization, and inspection helpers."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
import subprocess
from typing import Any, Mapping, cast

import tab_foundry.benchmark_registry as benchmark_registry
import yaml

from tab_foundry.hashing import sha256_path
from tab_foundry.repo_paths import repo_root as shared_repo_root
from tab_foundry.timestamps import utc_now

from .dagzoo_handoff import load_dagzoo_handoff_info
from .dagzoo_workflow import DagzooGenerateConfig, build_dagzoo_generate_argv, run_dagzoo_generate
from .inspection import compare_jsonlike_payloads, inspect_manifest, manifest_characteristics
from .manifest import build_manifest


CORPUS_RECIPE_SCHEMA = "tab-foundry-corpus-recipe-v1"
CORPUS_RECIPE_INDEX_SCHEMA = "tab-foundry-corpus-recipe-index-v1"
CORPUS_RECORD_SCHEMA = "tab-foundry-corpus-record-v1"
CORPUS_LATEST_SCHEMA = "tab-foundry-corpus-latest-v1"
RECIPE_KIND_DAGZOO_SINGLE = "dagzoo_single_invocation"
RECIPE_KIND_DAGZOO_MULTI = "dagzoo_multi_invocation_manifest"
_VALID_RECIPE_KINDS = {
    RECIPE_KIND_DAGZOO_SINGLE,
    RECIPE_KIND_DAGZOO_MULTI,
}


def _repo_root() -> Path:
    return shared_repo_root()


def corpus_recipes_root(*, repo_root: Path | None = None) -> Path:
    resolved_repo_root = (repo_root or _repo_root()).expanduser().resolve()
    return resolved_repo_root / "reference" / "corpus_recipes"


def corpus_recipe_index_path(*, repo_root: Path | None = None) -> Path:
    return corpus_recipes_root(repo_root=repo_root) / "index.yaml"


def corpus_outputs_root(*, repo_root: Path | None = None) -> Path:
    resolved_repo_root = (repo_root or _repo_root()).expanduser().resolve()
    return resolved_repo_root / "outputs" / "corpora"


def _read_json_mapping(path: Path, *, context: str) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} must decode to a JSON object: {path.expanduser().resolve()}")
    return cast(dict[str, Any], payload)


def _load_yaml_mapping(path: Path, *, context: str) -> dict[str, Any]:
    payload = yaml.safe_load(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError(f"{context} must decode to a mapping: {path.expanduser().resolve()}")
    return cast(dict[str, Any], payload)


def _ensure_non_empty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{context} must be a non-empty string")
    return str(value)


def _ensure_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise RuntimeError(f"{context} must be a mapping")
    return {str(key): item for key, item in value.items()}


def _optional_string(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    return str(value)


def _coerce_int(value: Any, *, context: str) -> int:
    if value is None or isinstance(value, bool):
        raise RuntimeError(f"{context} must be an integer-compatible value")
    try:
        return int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{context} must be an integer-compatible value") from exc


def _recipe_path_from_index_entry(
    recipe_id: str,
    entry: Mapping[str, Any],
    *,
    root: Path,
) -> Path:
    raw_path = _ensure_non_empty_string(entry.get("path"), context=f"recipe index entry {recipe_id!r}.path")
    candidate = Path(raw_path)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


@dataclass(slots=True, frozen=True)
class CorpusManifestPolicy:
    train_ratio: float
    val_ratio: float
    filter_policy: str
    missing_value_policy: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "train_ratio": float(self.train_ratio),
            "val_ratio": float(self.val_ratio),
            "filter_policy": str(self.filter_policy),
            "missing_value_policy": str(self.missing_value_policy),
        }


@dataclass(slots=True, frozen=True)
class DagzooInvocationRecipe:
    invocation_id: str
    config_ref: str
    num_datasets: int
    seed: int | None
    rows: str | None
    device: str | None
    hardware_policy: str
    diagnostics: bool
    diagnostics_out_dir: str | None
    missing_rate: float | None
    missing_mechanism: str | None
    missing_mar_observed_fraction: float | None
    missing_mar_logit_scale: float | None
    missing_mnar_logit_scale: float | None

    def to_dict(self) -> dict[str, Any]:
        return {
            "invocation_id": str(self.invocation_id),
            "config_ref": str(self.config_ref),
            "num_datasets": int(self.num_datasets),
            "seed": None if self.seed is None else int(self.seed),
            "rows": self.rows,
            "device": self.device,
            "hardware_policy": str(self.hardware_policy),
            "diagnostics": bool(self.diagnostics),
            "diagnostics_out_dir": self.diagnostics_out_dir,
            "missing_rate": self.missing_rate,
            "missing_mechanism": self.missing_mechanism,
            "missing_mar_observed_fraction": self.missing_mar_observed_fraction,
            "missing_mar_logit_scale": self.missing_mar_logit_scale,
            "missing_mnar_logit_scale": self.missing_mnar_logit_scale,
        }


@dataclass(slots=True, frozen=True)
class CorpusRecipe:
    recipe_id: str
    kind: str
    description: str
    surface_label: str
    manifest_policy: CorpusManifestPolicy
    invocations: tuple[DagzooInvocationRecipe, ...]
    provenance_labels: dict[str, Any]
    recipe_path: Path

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": CORPUS_RECIPE_SCHEMA,
            "recipe_id": str(self.recipe_id),
            "kind": str(self.kind),
            "description": str(self.description),
            "surface_label": str(self.surface_label),
            "manifest": self.manifest_policy.to_dict(),
            "provenance_labels": dict(self.provenance_labels),
            "invocations": [invocation.to_dict() for invocation in self.invocations],
            "recipe_path": str(self.recipe_path),
        }


def _manifest_policy_from_payload(payload: Mapping[str, Any]) -> CorpusManifestPolicy:
    manifest = _ensure_mapping(payload.get("manifest"), context="recipe.manifest")
    return CorpusManifestPolicy(
        train_ratio=float(manifest.get("train_ratio", 0.90)),
        val_ratio=float(manifest.get("val_ratio", 0.05)),
        filter_policy=str(manifest.get("filter_policy", "include_all")),
        missing_value_policy=str(manifest.get("missing_value_policy", "allow_any")),
    )


def _invocation_from_payload(
    payload: Mapping[str, Any],
    *,
    default_invocation_id: str,
) -> DagzooInvocationRecipe:
    raw_num_datasets = payload.get("num_datasets")
    raw_seed = payload.get("seed")
    return DagzooInvocationRecipe(
        invocation_id=_optional_string(payload.get("invocation_id")) or default_invocation_id,
        config_ref=_ensure_non_empty_string(payload.get("config_ref"), context="recipe invocation config_ref"),
        num_datasets=_coerce_int(raw_num_datasets, context="recipe invocation num_datasets"),
        seed=None if raw_seed is None else _coerce_int(raw_seed, context="recipe invocation seed"),
        rows=None if payload.get("rows") is None else str(payload["rows"]),
        device=None if payload.get("device") is None else str(payload["device"]),
        hardware_policy=str(payload.get("hardware_policy", "none")),
        diagnostics=bool(payload.get("diagnostics", False)),
        diagnostics_out_dir=(
            None if payload.get("diagnostics_out_dir") is None else str(payload["diagnostics_out_dir"])
        ),
        missing_rate=None if payload.get("missing_rate") is None else float(payload["missing_rate"]),
        missing_mechanism=(
            None if payload.get("missing_mechanism") is None else str(payload["missing_mechanism"])
        ),
        missing_mar_observed_fraction=(
            None
            if payload.get("missing_mar_observed_fraction") is None
            else float(payload["missing_mar_observed_fraction"])
        ),
        missing_mar_logit_scale=(
            None
            if payload.get("missing_mar_logit_scale") is None
            else float(payload["missing_mar_logit_scale"])
        ),
        missing_mnar_logit_scale=(
            None
            if payload.get("missing_mnar_logit_scale") is None
            else float(payload["missing_mnar_logit_scale"])
        ),
    )


def _recipe_from_payload(payload: Mapping[str, Any], *, recipe_path: Path) -> CorpusRecipe:
    schema = payload.get("schema")
    if schema != CORPUS_RECIPE_SCHEMA:
        raise RuntimeError(
            f"corpus recipe schema must be {CORPUS_RECIPE_SCHEMA!r}, got {schema!r}: {recipe_path}"
        )
    recipe_id = _ensure_non_empty_string(payload.get("recipe_id"), context="recipe.recipe_id")
    kind = _ensure_non_empty_string(payload.get("kind"), context=f"recipe {recipe_id!r}.kind")
    if kind not in _VALID_RECIPE_KINDS:
        raise RuntimeError(f"unsupported corpus recipe kind {kind!r}: {recipe_path}")
    description = _ensure_non_empty_string(
        payload.get("description"),
        context=f"recipe {recipe_id!r}.description",
    )
    surface_label = _ensure_non_empty_string(
        payload.get("surface_label"),
        context=f"recipe {recipe_id!r}.surface_label",
    )
    manifest_policy = _manifest_policy_from_payload(payload)
    provenance_labels = _ensure_mapping(
        payload.get("provenance_labels"),
        context=f"recipe {recipe_id!r}.provenance_labels",
    )
    invocations: tuple[DagzooInvocationRecipe, ...]
    if kind == RECIPE_KIND_DAGZOO_SINGLE:
        dagzoo_payload = _ensure_mapping(
            payload.get("dagzoo"),
            context=f"recipe {recipe_id!r}.dagzoo",
        )
        invocations = (_invocation_from_payload(dagzoo_payload, default_invocation_id="default"),)
    else:
        raw_invocations = payload.get("invocations")
        if not isinstance(raw_invocations, list) or not raw_invocations:
            raise RuntimeError(f"recipe {recipe_id!r}.invocations must be a non-empty list")
        invocations = tuple(
            _invocation_from_payload(_ensure_mapping(item, context=f"recipe {recipe_id!r}.invocations[{index}]"), default_invocation_id=f"invocation_{index + 1}")
            for index, item in enumerate(raw_invocations)
        )
    return CorpusRecipe(
        recipe_id=recipe_id,
        kind=kind,
        description=description,
        surface_label=surface_label,
        manifest_policy=manifest_policy,
        invocations=invocations,
        provenance_labels=provenance_labels,
        recipe_path=recipe_path.expanduser().resolve(),
    )


def load_corpus_recipe(
    recipe_id: str,
    *,
    repo_root: Path | None = None,
) -> CorpusRecipe:
    root = corpus_recipes_root(repo_root=repo_root)
    index = _load_yaml_mapping(corpus_recipe_index_path(repo_root=repo_root), context="corpus recipe index")
    if index.get("schema") != CORPUS_RECIPE_INDEX_SCHEMA:
        raise RuntimeError(
            f"corpus recipe index schema must be {CORPUS_RECIPE_INDEX_SCHEMA!r}, got {index.get('schema')!r}"
        )
    recipes = _ensure_mapping(index.get("recipes"), context="corpus recipe index recipes")
    entry = recipes.get(recipe_id)
    if not isinstance(entry, Mapping):
        raise RuntimeError(f"unknown corpus recipe: {recipe_id!r}")
    recipe_path = _recipe_path_from_index_entry(recipe_id, entry, root=root)
    return _recipe_from_payload(
        _load_yaml_mapping(recipe_path, context=f"corpus recipe {recipe_id!r}"),
        recipe_path=recipe_path,
    )


def list_corpus_recipes(*, repo_root: Path | None = None) -> list[CorpusRecipe]:
    index = _load_yaml_mapping(corpus_recipe_index_path(repo_root=repo_root), context="corpus recipe index")
    if index.get("schema") != CORPUS_RECIPE_INDEX_SCHEMA:
        raise RuntimeError(
            f"corpus recipe index schema must be {CORPUS_RECIPE_INDEX_SCHEMA!r}, got {index.get('schema')!r}"
        )
    recipes = _ensure_mapping(index.get("recipes"), context="corpus recipe index recipes")
    return [
        load_corpus_recipe(recipe_id, repo_root=repo_root)
        for recipe_id in sorted(recipes)
    ]


def corpus_id_for_manifest(*, recipe_id: str, manifest_sha256: str) -> str:
    return f"{recipe_id}__{manifest_sha256[:12]}"


def corpus_record_path(
    *,
    recipe_id: str,
    corpus_id: str,
    repo_root: Path | None = None,
) -> Path:
    return corpus_outputs_root(repo_root=repo_root) / recipe_id / corpus_id / "corpus_record.json"


def _latest_pointer_path(*, recipe_id: str, repo_root: Path | None = None) -> Path:
    return corpus_outputs_root(repo_root=repo_root) / recipe_id / "latest.json"


def _write_latest_pointer(
    *,
    recipe_id: str,
    corpus_id: str,
    corpus_ref: str,
    record_path: Path,
    repo_root: Path | None = None,
) -> Path:
    payload = {
        "schema": CORPUS_LATEST_SCHEMA,
        "generated_at_utc": utc_now(),
        "recipe_id": str(recipe_id),
        "corpus_id": str(corpus_id),
        "corpus_ref": str(corpus_ref),
        "corpus_record_path": str(record_path.expanduser().resolve()),
    }
    latest_path = _latest_pointer_path(recipe_id=recipe_id, repo_root=repo_root)
    latest_path.parent.mkdir(parents=True, exist_ok=True)
    latest_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return latest_path


def _load_latest_pointer(recipe_id: str, *, repo_root: Path | None = None) -> dict[str, Any] | None:
    latest_path = _latest_pointer_path(recipe_id=recipe_id, repo_root=repo_root)
    if not latest_path.exists():
        return None
    payload = _read_json_mapping(latest_path, context=f"corpus latest pointer for {recipe_id!r}")
    if payload.get("schema") != CORPUS_LATEST_SCHEMA:
        raise RuntimeError(
            f"corpus latest pointer schema must be {CORPUS_LATEST_SCHEMA!r}, got {payload.get('schema')!r}: {latest_path}"
        )
    return payload


def _parse_corpus_ref(corpus_ref: str) -> tuple[str, str | None]:
    normalized = _ensure_non_empty_string(corpus_ref, context="corpus_ref")
    if "/" not in normalized:
        return normalized, None
    recipe_id, corpus_id = normalized.split("/", 1)
    return _ensure_non_empty_string(recipe_id, context="corpus_ref recipe_id"), _ensure_non_empty_string(
        corpus_id,
        context="corpus_ref corpus_id",
    )


def load_corpus_record(
    corpus_ref: str,
    *,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    recipe_id, corpus_id = _parse_corpus_ref(corpus_ref)
    if corpus_id is None:
        latest = _load_latest_pointer(recipe_id, repo_root=repo_root)
        if latest is not None:
            corpus_id = _ensure_non_empty_string(
                latest.get("corpus_id"),
                context=f"latest corpus_id for recipe {recipe_id!r}",
            )
        else:
            recipe_root = corpus_outputs_root(repo_root=repo_root) / recipe_id
            candidates = sorted(
                path.name
                for path in recipe_root.iterdir()
                if path.is_dir() and not path.name.startswith(".")
            ) if recipe_root.exists() else []
            if len(candidates) == 1:
                corpus_id = candidates[0]
            elif not candidates:
                raise RuntimeError(
                    f"no local corpus materialization found for recipe {recipe_id!r} under {recipe_root}"
                )
            else:
                raise RuntimeError(
                    f"multiple corpora exist for recipe {recipe_id!r} but no latest.json pointer is present: {recipe_root}"
                )
    record_path = corpus_record_path(recipe_id=recipe_id, corpus_id=corpus_id, repo_root=repo_root)
    if not record_path.exists():
        raise RuntimeError(f"corpus record does not exist: {record_path}")
    payload = _read_json_mapping(record_path, context=f"corpus record {recipe_id}/{corpus_id}")
    if payload.get("schema") != CORPUS_RECORD_SCHEMA:
        raise RuntimeError(
            f"corpus record schema must be {CORPUS_RECORD_SCHEMA!r}, got {payload.get('schema')!r}: {record_path}"
        )
    return payload


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
) -> dict[str, Any] | None:
    try:
        record = load_corpus_record(corpus_ref, repo_root=repo_root)
    except RuntimeError:
        return None
    return record if _corpus_record_is_complete_for_reuse(record) else None


def _git_info(root: Path) -> dict[str, Any] | None:
    if not root.exists():
        return None

    def _capture(*argv: str) -> str | None:
        completed = subprocess.run(
            list(argv),
            cwd=root,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            return None
        value = completed.stdout.strip()
        return value or None

    head = _capture("git", "rev-parse", "HEAD")
    if head is None:
        return None
    describe = _capture("git", "describe", "--always", "--dirty", "--tags")
    status = _capture("git", "status", "--short")
    return {
        "head": head,
        "describe": describe,
        "dirty": bool(status),
    }


def _drop_none_values(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        str(key): value
        for key, value in payload.items()
        if value is not None
    }


def _invocation_paths(*, corpus_root: Path, invocation_id: str) -> tuple[Path, Path]:
    invocation_root = corpus_root / "invocations" / invocation_id
    return invocation_root, invocation_root / "handoff_manifest.json"


def _materialize_invocation(
    *,
    dagzoo_root: Path,
    corpus_root: Path,
    spec: DagzooInvocationRecipe,
) -> None:
    invocation_root, _handoff_manifest_path = _invocation_paths(
        corpus_root=corpus_root,
        invocation_id=spec.invocation_id,
    )
    invocation_root.parent.mkdir(parents=True, exist_ok=True)
    run_dagzoo_generate(
        DagzooGenerateConfig(
            dagzoo_root=dagzoo_root,
            dagzoo_config=Path(str(spec.config_ref)),
            handoff_root=invocation_root,
            num_datasets=int(spec.num_datasets),
            seed=spec.seed,
            rows=spec.rows,
            device=spec.device,
            hardware_policy=str(spec.hardware_policy),
            diagnostics=bool(spec.diagnostics),
            diagnostics_out_dir=(
                None if spec.diagnostics_out_dir is None else Path(str(spec.diagnostics_out_dir))
            ),
            missing_rate=spec.missing_rate,
            missing_mechanism=spec.missing_mechanism,
            missing_mar_observed_fraction=spec.missing_mar_observed_fraction,
            missing_mar_logit_scale=spec.missing_mar_logit_scale,
            missing_mnar_logit_scale=spec.missing_mnar_logit_scale,
        )
    )


def _invocation_record_payload(
    *,
    dagzoo_root: Path,
    corpus_root: Path,
    spec: DagzooInvocationRecipe,
) -> dict[str, Any]:
    invocation_root, handoff_manifest_path = _invocation_paths(
        corpus_root=corpus_root,
        invocation_id=spec.invocation_id,
    )
    handoff = load_dagzoo_handoff_info(handoff_manifest_path)
    command = build_dagzoo_generate_argv(
        DagzooGenerateConfig(
            dagzoo_root=dagzoo_root,
            dagzoo_config=Path(str(spec.config_ref)),
            handoff_root=invocation_root,
            num_datasets=int(spec.num_datasets),
            seed=spec.seed,
            rows=spec.rows,
            device=spec.device,
            hardware_policy=str(spec.hardware_policy),
            diagnostics=bool(spec.diagnostics),
            diagnostics_out_dir=(
                None if spec.diagnostics_out_dir is None else Path(str(spec.diagnostics_out_dir))
            ),
            missing_rate=spec.missing_rate,
            missing_mechanism=spec.missing_mechanism,
            missing_mar_observed_fraction=spec.missing_mar_observed_fraction,
            missing_mar_logit_scale=spec.missing_mar_logit_scale,
            missing_mnar_logit_scale=spec.missing_mnar_logit_scale,
        )
    )
    return {
        "invocation_id": str(spec.invocation_id),
        "config_ref": str(spec.config_ref),
        "num_datasets": int(spec.num_datasets),
        "seed": None if spec.seed is None else int(spec.seed),
        "rows": spec.rows,
        "device": spec.device,
        "hardware_policy": str(spec.hardware_policy),
        "command": " ".join(command),
        "invocation_root": str(invocation_root.resolve()),
        "handoff": handoff.to_summary_dict(),
    }


def materialize_corpus_recipe(
    *,
    recipe_id: str,
    dagzoo_root: Path,
    force: bool = False,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    resolved_repo_root = (repo_root or _repo_root()).expanduser().resolve()
    resolved_dagzoo_root = dagzoo_root.expanduser().resolve()
    if not force:
        existing_record = _load_reusable_corpus_record(recipe_id, repo_root=resolved_repo_root)
        if existing_record is not None:
            return existing_record

    recipe = load_corpus_recipe(recipe_id, repo_root=resolved_repo_root)
    recipe_root = corpus_outputs_root(repo_root=resolved_repo_root) / recipe.recipe_id
    stage_root = recipe_root / ".staging"
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_root.mkdir(parents=True, exist_ok=True)

    try:
        for spec in recipe.invocations:
            _materialize_invocation(
                dagzoo_root=resolved_dagzoo_root,
                corpus_root=stage_root,
                spec=spec,
            )
        generated_roots = [
            load_dagzoo_handoff_info(
                _invocation_paths(corpus_root=stage_root, invocation_id=spec.invocation_id)[1]
            ).generated_dir
            for spec in recipe.invocations
        ]
        manifest_path = stage_root / "manifest.parquet"
        _ = build_manifest(
            data_roots=generated_roots,
            out_path=manifest_path,
            train_ratio=float(recipe.manifest_policy.train_ratio),
            val_ratio=float(recipe.manifest_policy.val_ratio),
            filter_policy=str(recipe.manifest_policy.filter_policy),
            missing_value_policy=str(recipe.manifest_policy.missing_value_policy),
        )
        manifest_sha256 = sha256_path(manifest_path)
        corpus_id = corpus_id_for_manifest(recipe_id=recipe.recipe_id, manifest_sha256=manifest_sha256)
        corpus_ref = f"{recipe.recipe_id}/{corpus_id}"
        final_root = recipe_root / corpus_id
        if final_root.exists():
            if force:
                shutil.rmtree(final_root)
            else:
                existing_record = _load_reusable_corpus_record(corpus_ref, repo_root=resolved_repo_root)
                if existing_record is not None:
                    shutil.rmtree(stage_root)
                    return existing_record
                shutil.rmtree(final_root)
        shutil.move(str(stage_root), str(final_root))
        resolved_manifest_path = final_root / "manifest.parquet"
        invocation_payloads = [
            _invocation_record_payload(
                dagzoo_root=resolved_dagzoo_root,
                corpus_root=final_root,
                spec=spec,
            )
            for spec in recipe.invocations
        ]
        dagzoo_provenance = _drop_none_values(
            {
                "corpus_ref": corpus_ref,
                "recipe_id": recipe.recipe_id,
                "corpus_id": corpus_id,
                "recipe_kind": recipe.kind,
                "corpus_variant": recipe.provenance_labels.get("corpus_variant", recipe.surface_label),
                "comparator_role": recipe.provenance_labels.get("comparator_role"),
                "config_refs": sorted({invocation.config_ref for invocation in recipe.invocations}),
                "commands": [payload["command"] for payload in invocation_payloads],
                "curated_root_lineage": [],
                "invocations": invocation_payloads,
                "dagzoo_git": _git_info(resolved_dagzoo_root),
            }
        )
        manifest_inspection = inspect_manifest(resolved_manifest_path)
        record = {
            "schema": CORPUS_RECORD_SCHEMA,
            "generated_at_utc": utc_now(),
            "recipe_id": recipe.recipe_id,
            "corpus_id": corpus_id,
            "corpus_ref": corpus_ref,
            "recipe_path": str(recipe.recipe_path),
            "surface_label": recipe.surface_label,
            "surface_label_recommendation": recipe.surface_label,
            "recipe": recipe.to_dict(),
            "artifacts": {
                "corpus_root": str(final_root.resolve()),
                "manifest_path": str(resolved_manifest_path.resolve()),
                "latest_pointer_path": str(_latest_pointer_path(recipe_id=recipe.recipe_id, repo_root=resolved_repo_root)),
            },
            "manifest": {
                "manifest_path": str(resolved_manifest_path.resolve()),
                "manifest_sha256": manifest_sha256,
                "inspection": manifest_inspection,
                "characteristics": manifest_characteristics(resolved_manifest_path),
            },
            "dagzoo_provenance": dagzoo_provenance,
        }
        record_path = final_root / "corpus_record.json"
        record["corpus_record_path"] = str(record_path.resolve())
        record_path.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        _write_latest_pointer(
            recipe_id=recipe.recipe_id,
            corpus_id=corpus_id,
            corpus_ref=corpus_ref,
            record_path=record_path,
            repo_root=resolved_repo_root,
        )
        return record
    finally:
        if stage_root.exists():
            shutil.rmtree(stage_root)


def corpus_compare_payload(
    *,
    left: str,
    right: str,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    left_record = load_corpus_record(left, repo_root=repo_root)
    right_record = load_corpus_record(right, repo_root=repo_root)
    left_manifest = cast(Mapping[str, Any], left_record["manifest"])
    right_manifest = cast(Mapping[str, Any], right_record["manifest"])
    left_payload = {
        "recipe_id": left_record.get("recipe_id"),
        "corpus_id": left_record.get("corpus_id"),
        "surface_label": left_record.get("surface_label"),
        "inspection": left_manifest.get("inspection"),
        "characteristics": left_manifest.get("characteristics"),
    }
    right_payload = {
        "recipe_id": right_record.get("recipe_id"),
        "corpus_id": right_record.get("corpus_id"),
        "surface_label": right_record.get("surface_label"),
        "inspection": right_manifest.get("inspection"),
        "characteristics": right_manifest.get("characteristics"),
    }
    differences = compare_jsonlike_payloads(left_payload, right_payload)
    return {
        "left": left_record,
        "right": right_record,
        "difference_count": len(differences),
        "differences": differences,
    }


def corpus_results_payload(
    *,
    corpus_ref: str,
    registry_path: Path | None = None,
    repo_root: Path | None = None,
) -> dict[str, Any]:
    record = load_corpus_record(corpus_ref, repo_root=repo_root)
    normalized_corpus_ref = _ensure_non_empty_string(record.get("corpus_ref"), context="corpus record corpus_ref")
    registry = benchmark_registry.load_benchmark_run_registry(
        registry_path or benchmark_registry.default_benchmark_run_registry_path()
    )
    runs = _ensure_mapping(registry.get("runs"), context="benchmark run registry runs")
    matched_runs: list[dict[str, Any]] = []
    for run_id in sorted(runs):
        entry = runs.get(run_id)
        if not isinstance(entry, Mapping):
            continue
        artifacts = entry.get("artifacts")
        if not isinstance(artifacts, Mapping):
            continue
        training_surface_record_path = artifacts.get("training_surface_record_path")
        if not isinstance(training_surface_record_path, str) or not training_surface_record_path.strip():
            continue
        resolved_surface_path = benchmark_registry.resolve_registry_path_value(training_surface_record_path)
        if not resolved_surface_path.exists():
            continue
        training_surface_record = _read_json_mapping(
            resolved_surface_path,
            context=f"training surface record for run {run_id!r}",
        )
        data_payload = training_surface_record.get("data")
        if not isinstance(data_payload, Mapping):
            continue
        if data_payload.get("corpus_ref") != normalized_corpus_ref:
            continue
        sweep_payload = entry.get("sweep")
        metrics = entry.get("tab_foundry_metrics")
        matched_runs.append(
            {
                "run_id": str(run_id),
                "experiment": entry.get("experiment"),
                "config_profile": entry.get("config_profile"),
                "decision": entry.get("decision"),
                "surface_labels": entry.get("surface_labels"),
                "sweep": {
                    "sweep_id": None if not isinstance(sweep_payload, Mapping) else sweep_payload.get("sweep_id"),
                    "delta_id": None if not isinstance(sweep_payload, Mapping) else sweep_payload.get("delta_id"),
                    "queue_order": None
                    if not isinstance(sweep_payload, Mapping)
                    else sweep_payload.get("queue_order"),
                },
                "headline_metrics": (
                    None
                    if not isinstance(metrics, Mapping)
                    else {
                        key: metrics.get(key)
                        for key in (
                            "best_roc_auc",
                            "final_roc_auc",
                            "best_log_loss",
                            "final_log_loss",
                            "best_brier_score",
                            "final_brier_score",
                            "best_step",
                        )
                        if key in metrics
                    }
                ),
                "training_surface_record_path": str(resolved_surface_path),
            }
        )
    return {
        "corpus_ref": normalized_corpus_ref,
        "recipe_id": record.get("recipe_id"),
        "corpus_id": record.get("corpus_id"),
        "run_count": len(matched_runs),
        "runs": matched_runs,
    }
