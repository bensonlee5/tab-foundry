"""Sweep-aware helpers for the anchor-only system-delta workflow."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

from omegaconf import OmegaConf
import yaml  # type: ignore[import-untyped]

from tab_foundry.bench.benchmark_run_registry import (
    load_benchmark_run_registry,
    resolve_registry_path_value,
)
from tab_foundry.bench.nanotabpfn import load_benchmark_bundle
from tab_foundry.model.architectures.tabfoundry_staged.resolved import resolve_staged_surface
from tab_foundry.model.spec import ModelBuildSpec


CATALOG_SCHEMA = "tab-foundry-system-delta-catalog-v1"
SWEEP_INDEX_SCHEMA = "tab-foundry-system-delta-sweep-index-v1"
SWEEP_SCHEMA = "tab-foundry-system-delta-sweep-v1"
SWEEP_QUEUE_SCHEMA = "tab-foundry-system-delta-sweep-queue-v1"
MATERIALIZED_QUEUE_SCHEMA = "tab-foundry-system-delta-queue-v1"
DEFAULT_SWEEP_STATUS = "draft"
LEGACY_PRIOR_CONSTANT_LR_LABEL = "prior_constant_lr"
UNAVAILABLE_TRAINING_LABEL = "training surface label unavailable"
_QUEUE_PROSE_FIELDS = (
    "notes",
    "confounders",
    "parameter_adequacy_plan",
    "adequacy_knobs",
)
_LEGACY_PRIOR_CONFIG_PROFILE = "cls_benchmark_staged_prior"


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
    return Path(__file__).resolve().parents[3]


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


def _ensure_non_empty_string(value: Any, *, context: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RuntimeError(f"{context} must be a non-empty string")
    return str(value)


def _ensure_mapping(value: Any, *, context: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RuntimeError(f"{context} must be a mapping")
    return cast(dict[str, Any], value)


def _ensure_rows(value: Any, *, context: str) -> list[dict[str, Any]]:
    if not isinstance(value, list) or not value:
        raise RuntimeError(f"{context} must be a non-empty list")
    if not all(isinstance(item, dict) for item in value):
        raise RuntimeError(f"{context} must contain only mappings")
    return cast(list[dict[str, Any]], value)


def _ensure_string_list(value: Any, *, context: str) -> list[str]:
    if not isinstance(value, list):
        raise RuntimeError(f"{context} must be a list")
    normalized: list[str] = []
    for index, item in enumerate(value):
        if not isinstance(item, str) or not item.strip():
            raise RuntimeError(f"{context}[{index}] must be a non-empty string")
        normalized.append(str(item))
    return normalized


def _validate_prose_fields(
    payload: Mapping[str, Any],
    *,
    context: str,
    field_names: tuple[str, ...] = _QUEUE_PROSE_FIELDS,
) -> None:
    for field_name in field_names:
        _ = _ensure_string_list(payload.get(field_name, []), context=f"{context}.{field_name}")


def load_system_delta_catalog(path: Path | None = None) -> dict[str, Any]:
    catalog = _load_yaml_mapping(path or default_catalog_path(), context="system delta catalog")
    if catalog.get("schema") != CATALOG_SCHEMA:
        raise RuntimeError(
            f"system delta catalog schema must be {CATALOG_SCHEMA!r}, got {catalog.get('schema')!r}"
        )
    deltas = catalog.get("deltas")
    if not isinstance(deltas, dict) or not deltas:
        raise RuntimeError("system delta catalog must include a non-empty deltas mapping")
    return catalog


def load_system_delta_index(path: Path | None = None) -> dict[str, Any]:
    index = _load_yaml_mapping(path or default_sweep_index_path(), context="system delta sweep index")
    if index.get("schema") != SWEEP_INDEX_SCHEMA:
        raise RuntimeError(
            f"system delta sweep index schema must be {SWEEP_INDEX_SCHEMA!r}, got {index.get('schema')!r}"
        )
    _ensure_non_empty_string(index.get("active_sweep_id"), context="system delta sweep index.active_sweep_id")
    sweeps = index.get("sweeps")
    if not isinstance(sweeps, dict) or not sweeps:
        raise RuntimeError("system delta sweep index must include a non-empty sweeps mapping")
    return index


def _resolve_selected_sweep_id(
    sweep_id: str | None,
    *,
    index_path: Path | None = None,
) -> str:
    if sweep_id is not None:
        return _ensure_non_empty_string(sweep_id, context="sweep_id")
    index = load_system_delta_index(index_path)
    return _ensure_non_empty_string(index.get("active_sweep_id"), context="active_sweep_id")


def load_system_delta_sweep(
    sweep_id: str | None = None,
    *,
    index_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    resolved_sweep_id = _resolve_selected_sweep_id(sweep_id, index_path=index_path)
    sweep = _load_yaml_mapping(
        sweep_metadata_path(resolved_sweep_id, sweeps_root=sweeps_root),
        context=f"system delta sweep {resolved_sweep_id!r}",
    )
    if sweep.get("schema") != SWEEP_SCHEMA:
        raise RuntimeError(
            f"system delta sweep schema must be {SWEEP_SCHEMA!r}, got {sweep.get('schema')!r}"
        )
    if _ensure_non_empty_string(sweep.get("sweep_id"), context="sweep.sweep_id") != resolved_sweep_id:
        raise RuntimeError(
            f"system delta sweep id mismatch: expected {resolved_sweep_id!r}, got {sweep.get('sweep_id')!r}"
        )
    return sweep


def load_system_delta_queue_instance(
    sweep_id: str | None = None,
    *,
    index_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    resolved_sweep_id = _resolve_selected_sweep_id(sweep_id, index_path=index_path)
    queue = _load_yaml_mapping(
        sweep_queue_path(resolved_sweep_id, sweeps_root=sweeps_root),
        context=f"system delta queue instance {resolved_sweep_id!r}",
    )
    if queue.get("schema") != SWEEP_QUEUE_SCHEMA:
        raise RuntimeError(
            f"system delta queue instance schema must be {SWEEP_QUEUE_SCHEMA!r}, got {queue.get('schema')!r}"
        )
    if _ensure_non_empty_string(queue.get("sweep_id"), context="queue.sweep_id") != resolved_sweep_id:
        raise RuntimeError(
            f"system delta queue sweep id mismatch: expected {resolved_sweep_id!r}, got {queue.get('sweep_id')!r}"
        )
    rows = _ensure_rows(queue.get("rows"), context="system delta queue instance rows")
    for index, row in enumerate(rows):
        _validate_prose_fields(
            row,
            context=f"system delta queue instance rows[{index}]",
            field_names=("notes", "confounders", "parameter_adequacy_plan"),
        )
    return queue


def _staged_module_selection_from_run_model(model_payload: Mapping[str, Any]) -> dict[str, Any] | None:
    module_selection = model_payload.get("module_selection")
    if isinstance(module_selection, dict) and module_selection:
        return cast(dict[str, Any], _copy_jsonable(module_selection))
    if str(model_payload.get("arch")) != "tabfoundry_staged":
        return None
    stage_raw = model_payload.get("stage")
    if not isinstance(stage_raw, str) or not stage_raw.strip():
        return None
    stage_label = model_payload.get("stage_label")
    spec = ModelBuildSpec(
        task="classification",
        arch="tabfoundry_staged",
        stage=str(stage_raw),
        stage_label=str(stage_label) if isinstance(stage_label, str) and stage_label.strip() else None,
        d_icl=int(model_payload.get("d_icl", 512)),
        input_normalization=str(model_payload.get("input_normalization", "none")),
        many_class_base=int(model_payload.get("many_class_base", 10)),
        tficl_n_heads=int(model_payload.get("tficl_n_heads", 8)),
        tficl_n_layers=int(model_payload.get("tficl_n_layers", 12)),
        head_hidden_dim=int(model_payload.get("head_hidden_dim", 1024)),
    )
    return resolve_staged_surface(spec).module_selection()


def _anchor_context_from_registry_run(
    *,
    anchor_run_id: str,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    registry = load_benchmark_run_registry(registry_path or default_registry_path())
    runs = _ensure_mapping(registry.get("runs"), context="benchmark registry runs")
    run = runs.get(anchor_run_id)
    if not isinstance(run, dict):
        raise RuntimeError(f"anchor_run_id {anchor_run_id!r} is missing from the benchmark registry")
    model = _ensure_mapping(run.get("model"), context=f"benchmark registry run {anchor_run_id}.model")
    surface_labels_raw = run.get("surface_labels")
    surface_labels = (
        None
        if not isinstance(surface_labels_raw, dict)
        else cast(dict[str, Any], _copy_jsonable(surface_labels_raw))
    )
    return {
        "run_id": anchor_run_id,
        "experiment": run.get("experiment"),
        "config_profile": run.get("config_profile"),
        "model": {
            "arch": model.get("arch"),
            "benchmark_profile": model.get("benchmark_profile"),
            "stage": model.get("stage"),
            "stage_label": model.get("stage_label"),
            "module_selection": _staged_module_selection_from_run_model(model),
        },
        "surface_labels": surface_labels,
    }


def _surface_label_from_anchor_context(
    anchor_context: Mapping[str, Any],
    *,
    key: str,
    fallback: str,
) -> str:
    surface_labels = anchor_context.get("surface_labels")
    if isinstance(surface_labels, dict):
        value = surface_labels.get(key)
        if isinstance(value, str) and value.strip():
            return str(value)
    if key == "model":
        model = cast(dict[str, Any], anchor_context.get("model", {}))
        for candidate in ("stage_label", "stage", "benchmark_profile"):
            value = model.get(candidate)
            if isinstance(value, str) and value.strip():
                return str(value)
    return fallback


def _anchor_training_surface_label(anchor_context: Mapping[str, Any]) -> str:
    surface_labels = anchor_context.get("surface_labels")
    if isinstance(surface_labels, dict):
        value = surface_labels.get("training")
        if isinstance(value, str) and value.strip():
            return str(value)
    experiment = anchor_context.get("experiment")
    config_profile = anchor_context.get("config_profile")
    if experiment == _LEGACY_PRIOR_CONFIG_PROFILE or config_profile == _LEGACY_PRIOR_CONFIG_PROFILE:
        return LEGACY_PRIOR_CONSTANT_LR_LABEL
    return UNAVAILABLE_TRAINING_LABEL


def _anchor_module_selection(anchor_context: Mapping[str, Any]) -> Mapping[str, Any]:
    model = anchor_context.get("model")
    if not isinstance(model, dict):
        return {}
    module_selection = model.get("module_selection")
    if isinstance(module_selection, dict):
        return cast(dict[str, Any], module_selection)
    resolved = _staged_module_selection_from_run_model(cast(dict[str, Any], model))
    if resolved is None:
        return {}
    return resolved


def _module_choice(module_selection: Mapping[str, Any], key: str, *, fallback: str = "unknown") -> str:
    value = module_selection.get(key)
    if isinstance(value, str) and value.strip():
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return fallback
    return str(value)


def _describe_feature_encoder(module_selection: Mapping[str, Any]) -> tuple[str, str]:
    feature_encoder = _module_choice(module_selection, "feature_encoder")
    if feature_encoder == "nano":
        return (
            "Same nano feature encoder path with internal benchmark normalization.",
            "Feature encoding remains close to upstream parity; later deltas should be attributed elsewhere.",
        )
    if feature_encoder == "shared":
        return (
            "Shared feature encoder path with benchmark-external normalization.",
            "Feature encoder swaps change both the representation path and where normalization lives.",
        )
    return (
        f"Staged feature encoder `{feature_encoder}` from the benchmark registry surface.",
        "Feature encoder changes alter the per-cell representation and should be interpreted explicitly.",
    )


def _describe_target_conditioner(module_selection: Mapping[str, Any]) -> tuple[str, str]:
    target_conditioner = _module_choice(module_selection, "target_conditioner")
    if target_conditioner == "mean_padded_linear":
        return (
            "Same mean-padded linear target conditioner.",
            "The anchor preserves the upstream label-conditioning mechanism.",
        )
    if target_conditioner == "label_token":
        return (
            "Label-token target conditioning.",
            "Target-conditioning swaps change how labels enter the model and need their own attribution.",
        )
    return (
        f"Target conditioner `{target_conditioner}` from the staged surface.",
        "Target-conditioning changes should be interpreted separately from encoder or context changes.",
    )


def _describe_table_block(module_selection: Mapping[str, Any]) -> tuple[str, str]:
    table_block_style = _module_choice(module_selection, "table_block_style")
    allow_test_self_attention = _module_choice(
        module_selection,
        "allow_test_self_attention",
        fallback="false",
    )
    if table_block_style == "nano_postnorm":
        return (
            "Same nano post-norm cell transformer block.",
            "This keeps the strongest structural tie to upstream nanoTabPFN.",
        )
    if table_block_style == "prenorm":
        if allow_test_self_attention == "true":
            return (
                "Pre-norm cell transformer block with test-self attention enabled.",
                "Block-style changes alter attention flow and should not be conflated with tokenizer or readout deltas.",
            )
        return (
            "Pre-norm cell transformer block without test-self attention.",
            "Block-style changes alter attention flow and should not be conflated with tokenizer or readout deltas.",
        )
    return (
        f"Cell transformer block `{table_block_style}` from the staged surface.",
        "Cell-block changes affect the core table computation and should be isolated carefully.",
    )


def _describe_tokenizer(module_selection: Mapping[str, Any]) -> tuple[str, str]:
    tokenizer = _module_choice(module_selection, "tokenizer")
    if tokenizer == "scalar_per_feature":
        return (
            "Same scalar-per-feature tokenizer.",
            "Tokenization remains aligned with upstream parity.",
        )
    if tokenizer == "shifted_grouped":
        return (
            "Shifted grouped tokenizer.",
            "Tokenizer changes reshape the effective table sequence and need their own adequacy commentary.",
        )
    return (
        f"Tokenizer `{tokenizer}` from the staged surface.",
        "Tokenizer changes alter the token sequence presented to the transformer stack.",
    )


def _describe_column_encoder(module_selection: Mapping[str, Any]) -> tuple[str, str]:
    column_encoder = _module_choice(module_selection, "column_encoder")
    if column_encoder == "none":
        return (
            "No column-set encoder on the anchor path.",
            "Column-set modeling remains absent and should not explain anchor behavior.",
        )
    if column_encoder == "tfcol":
        return (
            "Transformer column-set encoder (`tfcol`).",
            "Column-set encoding changes how feature interactions are aggregated before row reasoning.",
        )
    return (
        f"Column encoder `{column_encoder}` from the staged surface.",
        "Column-encoder changes should be read separately from row pooling or context changes.",
    )


def _describe_row_pool(module_selection: Mapping[str, Any]) -> tuple[str, str]:
    row_pool = _module_choice(module_selection, "row_pool")
    if row_pool == "target_column":
        return (
            "Same target-column row pool.",
            "Readout remains on the direct upstream-style path.",
        )
    if row_pool == "row_cls":
        return (
            "Row-CLS pooling path.",
            "Row-pool changes alter how the table summary is extracted and should be isolated from context changes.",
        )
    return (
        f"Row pool `{row_pool}` from the staged surface.",
        "Row-pool changes alter the readout contract and require their own interpretation.",
    )


def _describe_context_encoder(module_selection: Mapping[str, Any]) -> tuple[str, str]:
    context_encoder = _module_choice(module_selection, "context_encoder")
    if context_encoder == "none":
        return (
            "None on the anchor path.",
            "Context encoding remains absent; later context rows will change both depth and label-flow semantics.",
        )
    if context_encoder == "plain":
        return (
            "Plain context encoder.",
            "Context encoding adds extra sequence processing that must be interpreted separately from readout changes.",
        )
    if context_encoder == "qass":
        return (
            "QASS context encoder.",
            "QASS changes both compute graph depth and label-context semantics and needs explicit adequacy notes.",
        )
    return (
        f"Context encoder `{context_encoder}` from the staged surface.",
        "Context-encoder changes alter how training rows condition test rows.",
    )


def _describe_head(module_selection: Mapping[str, Any]) -> tuple[str, str]:
    head = _module_choice(module_selection, "head")
    if head == "binary_direct":
        return (
            "Direct binary logits head.",
            "The prediction head remains on the narrow upstream-style binary path.",
        )
    if head == "small_class":
        return (
            "Small-class direct head.",
            "Head changes alter the task contract and should be interpreted separately from shared trunk changes.",
        )
    if head == "many_class":
        return (
            "Many-class head.",
            "Many-class support changes both the task contract and the downstream label path.",
        )
    return (
        f"Prediction head `{head}` from the staged surface.",
        "Head changes alter the task contract and output semantics.",
    )


def _build_anchor_surface(
    *,
    anchor_run_id: str,
    benchmark_bundle_path: str,
    anchor_context: Mapping[str, Any],
) -> dict[str, Any]:
    bundle_path = resolve_registry_path_value(benchmark_bundle_path)
    bundle = load_benchmark_bundle(bundle_path)
    bundle_name = _ensure_non_empty_string(bundle.get("name"), context="benchmark bundle name")
    task_ids = cast(list[Any], bundle.get("task_ids", []))
    task_count = int(len(task_ids))
    module_selection = _anchor_module_selection(anchor_context)
    model_label = _surface_label_from_anchor_context(
        anchor_context,
        key="model",
        fallback="registry surface label unavailable",
    )
    data_label = _surface_label_from_anchor_context(
        anchor_context,
        key="data",
        fallback="registry surface label unavailable",
    )
    preprocessing_label = _surface_label_from_anchor_context(
        anchor_context,
        key="preprocessing",
        fallback="registry surface label unavailable",
    )
    training_label = _anchor_training_surface_label(anchor_context)
    feature_encoder, feature_encoder_interpretation = _describe_feature_encoder(module_selection)
    target_conditioner, target_conditioner_interpretation = _describe_target_conditioner(
        module_selection
    )
    table_block, table_block_interpretation = _describe_table_block(module_selection)
    tokenizer, tokenizer_interpretation = _describe_tokenizer(module_selection)
    column_encoder, column_encoder_interpretation = _describe_column_encoder(module_selection)
    row_pool, row_pool_interpretation = _describe_row_pool(module_selection)
    context_encoder, context_encoder_interpretation = _describe_context_encoder(module_selection)
    head, head_interpretation = _describe_head(module_selection)
    return {
        "notes": [
            f"The locked anchor is benchmark registry run `{anchor_run_id}` on bundle `{bundle_name}` ({task_count} tasks).",
            f"The anchor model surface is taken from the registry-resolved staged selection labeled `{model_label}`.",
            "Data and preprocessing remain part of the comparison surface and must stay fixed unless the queue row declares that exact dimension.",
        ],
        "dimension_table": [
            {
                "dimension": "feature encoder",
                "upstream": "Scalar feature linear encoder with internal train/test z-score+clip handling.",
                "anchor": feature_encoder,
                "interpretation": feature_encoder_interpretation,
            },
            {
                "dimension": "target conditioning",
                "upstream": "Mean-padded linear target encoder on the direct binary path.",
                "anchor": target_conditioner,
                "interpretation": target_conditioner_interpretation,
            },
            {
                "dimension": "cell transformer block",
                "upstream": "Post-norm nanoTabPFN block with feature attention then row attention.",
                "anchor": table_block,
                "interpretation": table_block_interpretation,
            },
            {
                "dimension": "tokenizer",
                "upstream": "One scalar token per feature.",
                "anchor": tokenizer,
                "interpretation": tokenizer_interpretation,
            },
            {
                "dimension": "column encoder",
                "upstream": "None on the upstream direct path.",
                "anchor": column_encoder,
                "interpretation": column_encoder_interpretation,
            },
            {
                "dimension": "row readout",
                "upstream": "Target-column readout from the final cell tensor.",
                "anchor": row_pool,
                "interpretation": row_pool_interpretation,
            },
            {
                "dimension": "context encoder",
                "upstream": "None on the upstream direct path.",
                "anchor": context_encoder,
                "interpretation": context_encoder_interpretation,
            },
            {
                "dimension": "prediction head",
                "upstream": "Direct binary logits head.",
                "anchor": head,
                "interpretation": head_interpretation,
            },
            {
                "dimension": "training data surface",
                "upstream": "OpenML notebook tasks only for benchmarking; no repo-local prior-training manifest contract.",
                "anchor": f"Benchmark bundle `{bundle_name}` ({task_count} tasks) with data surface label `{data_label}`.",
                "interpretation": "Bundle and training-data changes are first-class sweep rows and should not be inherited from parent sweep prose.",
            },
            {
                "dimension": "preprocessing",
                "upstream": "Notebook preprocessing inside the benchmark helper.",
                "anchor": f"Benchmark preprocessing surface label `{preprocessing_label}`.",
                "interpretation": "Preprocessing changes can alter the effective task definition and must be tracked explicitly.",
            },
            {
                "dimension": "training recipe",
                "upstream": "No repo-local prior-dump training-surface contract.",
                "anchor": f"Training surface label `{training_label}`.",
                "interpretation": "Optimizer and schedule changes are first-class sweep rows, not background recipe assumptions.",
            },
        ],
    }


def _evaluate_applicability_guard(
    guard: Mapping[str, Any],
    *,
    anchor_context: Mapping[str, Any],
) -> tuple[bool, str | None]:
    kind = _ensure_non_empty_string(guard.get("kind"), context="applicability guard kind")
    if kind != "requires_anchor_model_selection":
        raise RuntimeError(f"Unsupported applicability guard kind: {kind!r}")
    key = _ensure_non_empty_string(guard.get("key"), context="applicability guard key")
    any_of_raw = guard.get("any_of")
    if not isinstance(any_of_raw, list) or not any_of_raw:
        raise RuntimeError("applicability guard any_of must be a non-empty list")
    any_of = {str(item) for item in any_of_raw}
    anchor_model = cast(dict[str, Any], anchor_context.get("model", {}))
    module_selection = anchor_model.get("module_selection")
    if not isinstance(module_selection, dict):
        return False, None
    current_value = module_selection.get(key)
    if current_value is None:
        return False, None
    current_value_str = str(current_value)
    return current_value_str in any_of, current_value_str


def _guarded_initial_state(
    *,
    delta_entry: Mapping[str, Any],
    anchor_context: Mapping[str, Any],
) -> tuple[str, str, str | None]:
    status = str(delta_entry.get("default_initial_status", "ready"))
    interpretation_status = str(delta_entry.get("default_initial_interpretation_status", "pending"))
    next_action_override: str | None = None
    guards = delta_entry.get("applicability_guards")
    if not isinstance(guards, list):
        return status, interpretation_status, next_action_override
    for raw_guard in guards:
        if not isinstance(raw_guard, dict):
            raise RuntimeError("applicability_guards entries must be mappings")
        matched, _value = _evaluate_applicability_guard(raw_guard, anchor_context=anchor_context)
        if matched:
            continue
        status = str(raw_guard.get("failure_status", status))
        interpretation_status = str(
            raw_guard.get("failure_interpretation_status", interpretation_status)
        )
        failure_next_action = raw_guard.get("failure_next_action")
        if isinstance(failure_next_action, str) and failure_next_action.strip():
            next_action_override = str(failure_next_action)
        break
    return status, interpretation_status, next_action_override


def _materialize_row(
    *,
    queue_row: Mapping[str, Any],
    delta_entry: Mapping[str, Any],
    anchor_context: Mapping[str, Any],
) -> dict[str, Any]:
    default_effective_surface = cast(
        dict[str, Any],
        _copy_jsonable(cast(dict[str, Any], delta_entry.get("default_effective_surface", {}))),
    )
    parameter_policy = cast(
        dict[str, Any],
        _copy_jsonable(cast(dict[str, Any], delta_entry.get("parameter_adequacy_policy", {}))),
    )
    _validate_prose_fields(
        queue_row,
        context=f"queue row {queue_row.get('delta_ref', '<missing>')!r}",
        field_names=("notes", "confounders", "parameter_adequacy_plan"),
    )
    _validate_prose_fields(
        delta_entry,
        context=f"delta entry {queue_row.get('delta_ref', '<missing>')!r}",
        field_names=("adequacy_knobs",),
    )
    parameter_plan = queue_row.get("parameter_adequacy_plan")
    if not isinstance(parameter_plan, list):
        parameter_plan = parameter_policy.get("default_plan", [])
    _ = _ensure_string_list(
        parameter_plan,
        context=f"queue row {queue_row.get('delta_ref', '<missing>')!r}.parameter_adequacy_plan",
    )
    return {
        "order": int(queue_row["order"]),
        "delta_id": _ensure_non_empty_string(queue_row.get("delta_ref"), context="queue row delta_ref"),
        "status": str(queue_row["status"]),
        "dimension_family": str(delta_entry["dimension_family"]),
        "family": str(delta_entry["family"]),
        "binary_applicable": bool(delta_entry.get("binary_applicable", False)),
        "description": str(delta_entry["description"]),
        "rationale": str(queue_row.get("rationale", "")),
        "hypothesis": str(queue_row.get("hypothesis", "")),
        "upstream_delta": str(delta_entry["upstream_delta"]),
        "anchor_delta": str(queue_row.get("anchor_delta", "")),
        "entangled_legacy_stage": str(delta_entry.get("legacy_stage_alias", "none")),
        "expected_effect": str(delta_entry["expected_effect"]),
        "adequacy_knobs": cast(list[Any], _copy_jsonable(delta_entry.get("adequacy_knobs", []))),
        "parameter_adequacy_policy": parameter_policy,
        "applicability_guards": cast(
            list[Any],
            _copy_jsonable(delta_entry.get("applicability_guards", [])),
        ),
        "model": cast(
            dict[str, Any],
            _copy_jsonable(queue_row.get("model", default_effective_surface.get("model", {}))),
        ),
        "data": cast(
            dict[str, Any],
            _copy_jsonable(queue_row.get("data", default_effective_surface.get("data", {}))),
        ),
        "preprocessing": cast(
            dict[str, Any],
            _copy_jsonable(
                queue_row.get("preprocessing", default_effective_surface.get("preprocessing", {}))
            ),
        ),
        "training": cast(
            dict[str, Any],
            _copy_jsonable(
                queue_row.get(
                    "training",
                    default_effective_surface.get(
                        "training",
                        {
                            "surface_label": _anchor_training_surface_label(anchor_context),
                            "overrides": {},
                        },
                    ),
                )
            ),
        ),
        "parameter_adequacy_plan": cast(list[Any], _copy_jsonable(parameter_plan)),
        "run_id": queue_row.get("run_id"),
        "followup_run_ids": cast(list[Any], _copy_jsonable(queue_row.get("followup_run_ids", []))),
        "decision": queue_row.get("decision"),
        "interpretation_status": str(queue_row.get("interpretation_status", "pending")),
        "confounders": cast(list[Any], _copy_jsonable(queue_row.get("confounders", []))),
        "next_action": str(queue_row.get("next_action", "")),
        "notes": cast(list[Any], _copy_jsonable(queue_row.get("notes", []))),
        "benchmark_metrics": cast(
            dict[str, Any] | None,
            _copy_jsonable(queue_row.get("benchmark_metrics")) if queue_row.get("benchmark_metrics") else None,
        ),
    }


def materialize_system_delta_queue(
    *,
    catalog: Mapping[str, Any],
    sweep: Mapping[str, Any],
    queue_instance: Mapping[str, Any],
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    deltas = _ensure_mapping(catalog.get("deltas"), context="catalog deltas")
    sweep_id = _ensure_non_empty_string(sweep.get("sweep_id"), context="sweep.sweep_id")
    rows_payload = _ensure_rows(queue_instance.get("rows"), context="queue rows")
    rows: list[dict[str, Any]] = []
    for queue_row in sorted(rows_payload, key=lambda row: (int(row["order"]), str(row["delta_ref"]))):
        delta_ref = _ensure_non_empty_string(queue_row.get("delta_ref"), context="queue row delta_ref")
        delta_entry = deltas.get(delta_ref)
        if not isinstance(delta_entry, dict):
            raise RuntimeError(f"unknown delta_ref {delta_ref!r} in sweep {sweep_id!r}")
        rows.append(
            _materialize_row(
                queue_row=queue_row,
                delta_entry=delta_entry,
                anchor_context=cast(dict[str, Any], sweep.get("anchor_context", {})),
            )
        )
    for index, row in enumerate(rows):
        _validate_prose_fields(row, context=f"materialized queue rows[{index}]")
    resolved_sweeps_root = sweeps_root or default_sweeps_root()
    return {
        "schema": MATERIALIZED_QUEUE_SCHEMA,
        "generated_from_sweep_id": sweep_id,
        "catalog_path": _render_path(catalog_path or default_catalog_path()),
        "canonical_sweep_path": _render_path(sweep_metadata_path(sweep_id, sweeps_root=resolved_sweeps_root)),
        "canonical_queue_path": _render_path(sweep_queue_path(sweep_id, sweeps_root=resolved_sweeps_root)),
        "canonical_matrix_path": _render_path(sweep_matrix_path(sweep_id, sweeps_root=resolved_sweeps_root)),
        "sweep_id": sweep_id,
        "parent_sweep_id": sweep.get("parent_sweep_id"),
        "sweep_status": sweep.get("status"),
        "complexity_level": sweep.get("complexity_level"),
        "anchor_run_id": sweep["anchor_run_id"],
        "benchmark_bundle_path": sweep["benchmark_bundle_path"],
        "control_baseline_id": sweep["control_baseline_id"],
        "comparison_policy": sweep["comparison_policy"],
        "upstream_reference": cast(dict[str, Any], _copy_jsonable(sweep["upstream_reference"])),
        "anchor_surface": cast(dict[str, Any], _copy_jsonable(sweep["anchor_surface"])),
        "anchor_context": cast(dict[str, Any], _copy_jsonable(sweep.get("anchor_context", {}))),
        "rows": rows,
    }


def load_system_delta_queue(
    path: Path | None = None,
    *,
    sweep_id: str | None = None,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, Any]:
    if path is None:
        catalog = load_system_delta_catalog(catalog_path)
        sweep = load_system_delta_sweep(sweep_id, index_path=index_path, sweeps_root=sweeps_root)
        queue_instance = load_system_delta_queue_instance(
            sweep_id or str(sweep["sweep_id"]),
            index_path=index_path,
            sweeps_root=sweeps_root,
        )
        return materialize_system_delta_queue(
            catalog=catalog,
            sweep=sweep,
            queue_instance=queue_instance,
            catalog_path=catalog_path,
            sweeps_root=sweeps_root,
        )

    payload = _load_yaml_mapping(path, context="system delta queue")
    schema = payload.get("schema")
    if schema == SWEEP_QUEUE_SCHEMA:
        queue_instance = payload
        resolved_sweep_id = _ensure_non_empty_string(
            queue_instance.get("sweep_id"),
            context="system delta queue instance sweep_id",
        )
        catalog = load_system_delta_catalog(catalog_path)
        sweep = load_system_delta_sweep(
            resolved_sweep_id,
            index_path=index_path,
            sweeps_root=sweeps_root,
        )
        return materialize_system_delta_queue(
            catalog=catalog,
            sweep=sweep,
            queue_instance=queue_instance,
            catalog_path=catalog_path,
            sweeps_root=sweeps_root,
        )
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise RuntimeError("materialized system delta queue must include rows")
    for index, row in enumerate(rows):
        if not isinstance(row, dict):
            raise RuntimeError(f"materialized system delta queue rows[{index}] must be mappings")
        _validate_prose_fields(row, context=f"materialized system delta queue rows[{index}]")
    return payload


def ordered_rows(queue: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = cast(list[dict[str, Any]], queue["rows"])
    return sorted(rows, key=lambda row: (int(row["order"]), str(row["delta_id"])))


def _render_model_change_payload(model_payload: Mapping[str, Any]) -> dict[str, Any]:
    rendered: dict[str, Any] = {}
    module_overrides = model_payload.get("module_overrides")
    if isinstance(module_overrides, dict) and module_overrides:
        rendered["module_overrides"] = module_overrides
    for key, value in model_payload.items():
        if key in {"stage_label", "module_overrides"}:
            continue
        if value in (None, {}, []):
            continue
        rendered[str(key)] = value
    return rendered


def next_ready_row(queue: Mapping[str, Any]) -> dict[str, Any] | None:
    for row in ordered_rows(queue):
        if str(row.get("status", "")).strip().lower() == "ready":
            return row
    return None


def _metric_summary(run: dict[str, Any], anchor: dict[str, Any]) -> dict[str, str]:
    def _optional_float(value: Any) -> float | None:
        if value is None:
            return None
        return float(value)

    def _format(value: float | None, *, suffix: str = "", signed: bool = False) -> str:
        if value is None:
            return "n/a"
        return f"{value:+.4f}{suffix}" if signed else f"{value:.4f}{suffix}"

    metrics = cast(dict[str, Any], run["tab_foundry_metrics"])
    anchor_metrics = cast(dict[str, Any], anchor["tab_foundry_metrics"])
    best = _optional_float(metrics.get("best_roc_auc"))
    final = _optional_float(metrics.get("final_roc_auc"))
    final_log_loss = _optional_float(metrics.get("final_log_loss"))
    anchor_final_log_loss = _optional_float(anchor_metrics.get("final_log_loss"))
    final_brier_score = _optional_float(metrics.get("final_brier_score"))
    anchor_final_brier_score = _optional_float(anchor_metrics.get("final_brier_score"))
    final_crps = _optional_float(metrics.get("final_crps"))
    anchor_final_crps = _optional_float(anchor_metrics.get("final_crps"))
    final_avg_pinball_loss = _optional_float(metrics.get("final_avg_pinball_loss"))
    anchor_final_avg_pinball_loss = _optional_float(anchor_metrics.get("final_avg_pinball_loss"))
    final_picp_90 = _optional_float(metrics.get("final_picp_90"))
    anchor_final_picp_90 = _optional_float(anchor_metrics.get("final_picp_90"))
    best_time = float(metrics["best_training_time"])
    final_time = float(metrics["final_training_time"])
    anchor_best = _optional_float(anchor_metrics.get("best_roc_auc"))
    anchor_final = _optional_float(anchor_metrics.get("final_roc_auc"))
    anchor_best_time = float(anchor_metrics["best_training_time"])
    anchor_final_time = float(anchor_metrics["final_training_time"])
    drift = None if best is None or final is None else final - best
    anchor_drift = (
        None if anchor_best is None or anchor_final is None else anchor_final - anchor_best
    )
    return {
        "best_roc_auc": _format(best),
        "final_roc_auc": _format(final),
        "final_minus_best": _format(drift, signed=True),
        "delta_best_roc_auc": (
            "n/a"
            if best is None or anchor_best is None
            else f"{best - anchor_best:+.4f}"
        ),
        "delta_final_roc_auc": (
            "n/a"
            if final is None or anchor_final is None
            else f"{final - anchor_final:+.4f}"
        ),
        "delta_drift": (
            "n/a"
            if drift is None or anchor_drift is None
            else f"{drift - anchor_drift:+.4f}"
        ),
        "delta_training_time": f"{final_time - anchor_final_time:+.1f}s",
        "final_training_time": f"{final_time:.1f}s",
        "best_training_time": f"{best_time:.1f}s",
        "delta_best_training_time": f"{best_time - anchor_best_time:+.1f}s",
        "final_log_loss": _format(final_log_loss),
        "delta_final_log_loss": (
            "n/a"
            if final_log_loss is None or anchor_final_log_loss is None
            else f"{final_log_loss - anchor_final_log_loss:+.4f}"
        ),
        "final_brier_score": _format(final_brier_score),
        "delta_final_brier_score": (
            "n/a"
            if final_brier_score is None or anchor_final_brier_score is None
            else f"{final_brier_score - anchor_final_brier_score:+.4f}"
        ),
        "final_crps": _format(final_crps),
        "delta_final_crps": (
            "n/a"
            if final_crps is None or anchor_final_crps is None
            else f"{final_crps - anchor_final_crps:+.4f}"
        ),
        "final_avg_pinball_loss": _format(final_avg_pinball_loss),
        "delta_final_avg_pinball_loss": (
            "n/a"
            if final_avg_pinball_loss is None or anchor_final_avg_pinball_loss is None
            else f"{final_avg_pinball_loss - anchor_final_avg_pinball_loss:+.4f}"
        ),
        "final_picp_90": _format(final_picp_90),
        "delta_final_picp_90": (
            "n/a"
            if final_picp_90 is None or anchor_final_picp_90 is None
            else f"{final_picp_90 - anchor_final_picp_90:+.4f}"
        ),
    }


def _result_card_path(*, sweep_id: str, delta_id: str) -> Path:
    return (
        repo_root()
        / "outputs"
        / "staged_ladder"
        / "research"
        / sweep_id
        / delta_id
        / "result_card.md"
    )


def validate_system_delta_queue(
    queue: Mapping[str, Any],
    *,
    registry_path: Path | None = None,
) -> list[str]:
    issues: list[str] = []
    registry = load_benchmark_run_registry(registry_path or default_registry_path())
    runs = cast(dict[str, dict[str, Any]], registry["runs"])
    sweep_id = _ensure_non_empty_string(queue.get("sweep_id"), context="materialized queue sweep_id")
    for row in ordered_rows(queue):
        status = str(row.get("status", "")).strip().lower()
        if status != "completed":
            continue
        delta_id = str(row["delta_id"])
        run_id = row.get("run_id")
        if not isinstance(run_id, str) or not run_id.strip():
            issues.append(f"{delta_id}: completed rows must include run_id")
            continue
        run = runs.get(run_id)
        if run is None:
            issues.append(f"{delta_id}: run_id {run_id!r} is missing from the benchmark registry")
            continue
        result_card_path = _result_card_path(sweep_id=sweep_id, delta_id=delta_id)
        if not result_card_path.exists():
            issues.append(f"{delta_id}: missing result card at {result_card_path}")
        training_surface_record_path = cast(dict[str, Any], run["artifacts"]).get(
            "training_surface_record_path"
        )
        if not isinstance(training_surface_record_path, str) or not training_surface_record_path.strip():
            issues.append(f"{delta_id}: run {run_id!r} is missing artifacts.training_surface_record_path")
        else:
            resolved = resolve_registry_path_value(training_surface_record_path)
            if not resolved.exists():
                issues.append(
                    f"{delta_id}: training surface artifact does not exist at {resolved}"
                )
    return issues


def render_system_delta_matrix(
    queue: Mapping[str, Any],
    *,
    registry_path: Path | None = None,
) -> str:
    registry = load_benchmark_run_registry(registry_path or default_registry_path())
    runs = cast(dict[str, dict[str, Any]], registry["runs"])
    sweep_id = _ensure_non_empty_string(queue.get("sweep_id"), context="materialized queue sweep_id")
    anchor_run_id = str(queue["anchor_run_id"])
    anchor = runs.get(anchor_run_id)
    if anchor is None:
        raise RuntimeError(f"anchor_run_id {anchor_run_id!r} is missing from the benchmark registry")
    anchor_metrics = cast(dict[str, Any], anchor["tab_foundry_metrics"])
    upstream = cast(dict[str, Any], queue["upstream_reference"])
    anchor_surface = cast(dict[str, Any], queue["anchor_surface"])
    catalog_path = str(queue.get("catalog_path", _render_path(default_catalog_path())))
    canonical_queue_path = str(
        queue.get("canonical_queue_path", _render_path(sweep_queue_path(sweep_id)))
    )

    lines: list[str] = []
    lines.append("# System Delta Matrix")
    lines.append("")
    lines.append(
        f"This file is rendered from `{canonical_queue_path}` plus `{catalog_path}` and the canonical benchmark registry."
    )
    lines.append("")
    lines.append("## Sweep")
    lines.append("")
    lines.append(f"- Sweep id: `{sweep_id}`")
    lines.append(f"- Sweep status: `{queue.get('sweep_status')}`")
    lines.append(f"- Parent sweep id: `{queue.get('parent_sweep_id')}`")
    lines.append(f"- Complexity level: `{queue.get('complexity_level')}`")
    lines.append("")
    lines.append("## Locked Surface")
    lines.append("")
    lines.append(f"- Anchor run id: `{anchor_run_id}`")
    lines.append(f"- Benchmark bundle: `{queue['benchmark_bundle_path']}`")
    lines.append(f"- Control baseline id: `{queue['control_baseline_id']}`")
    lines.append(f"- Comparison policy: `{queue['comparison_policy']}`")
    anchor_metric_parts: list[str] = []
    for label, key in (
        ("final log loss", "final_log_loss"),
        ("final Brier score", "final_brier_score"),
        ("best ROC AUC", "best_roc_auc"),
        ("final ROC AUC", "final_roc_auc"),
        ("final CRPS", "final_crps"),
        ("final avg pinball loss", "final_avg_pinball_loss"),
        ("final PICP 90", "final_picp_90"),
    ):
        raw_value = anchor_metrics.get(key)
        if raw_value is not None:
            anchor_metric_parts.append(f"{label} `{float(raw_value):.4f}`")
    anchor_metric_parts.append(f"final training time `{float(anchor_metrics['final_training_time']):.1f}s`")
    lines.append(f"- Anchor metrics: {', '.join(anchor_metric_parts)}")
    lines.append("")
    lines.append("## Anchor Comparison")
    lines.append("")
    lines.append(f"Upstream reference: `{upstream['name']}` from `{upstream['model_source']}`.")
    lines.append("")
    lines.append("| Dimension | Upstream nanoTabPFN | Locked anchor | Interpretation |")
    lines.append("| --- | --- | --- | --- |")
    for dimension_row in cast(list[dict[str, Any]], anchor_surface["dimension_table"]):
        lines.append(
            f"| {dimension_row['dimension']} | {dimension_row['upstream']} | "
            f"{dimension_row['anchor']} | {dimension_row['interpretation']} |"
        )
    lines.append("")
    lines.append("## Queue Summary")
    lines.append("")
    lines.append("| Order | Delta | Family | Binary | Status | Legacy stage alias | Effective change | Next action |")
    lines.append("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for queue_row in ordered_rows(queue):
        lines.append(
            f"| {queue_row['order']} | `{queue_row['delta_id']}` | {queue_row['family']} | "
            f"{'yes' if queue_row.get('binary_applicable', False) else 'no'} | {queue_row['status']} | "
            f"{queue_row.get('entangled_legacy_stage', 'none')} | "
            f"{queue_row['description']} | {queue_row['next_action']} |"
        )
    lines.append("")
    lines.append("## Detailed Rows")
    lines.append("")
    for queue_row in ordered_rows(queue):
        delta_id = str(queue_row["delta_id"])
        run_id = queue_row.get("run_id")
        run = runs.get(run_id) if isinstance(run_id, str) else None
        lines.append(f"### {queue_row['order']}. `{delta_id}`")
        lines.append("")
        lines.append(f"- Dimension family: `{queue_row['dimension_family']}`")
        lines.append(f"- Status: `{queue_row['status']}`")
        lines.append(f"- Binary applicable: `{queue_row.get('binary_applicable', False)}`")
        lines.append(
            f"- Legacy stage alias: `{queue_row.get('entangled_legacy_stage', 'none')}`"
        )
        lines.append(f"- Description: {queue_row['description']}")
        lines.append(f"- Rationale: {queue_row['rationale']}")
        lines.append(f"- Hypothesis: {queue_row['hypothesis']}")
        lines.append(f"- Upstream delta: {queue_row['upstream_delta']}")
        lines.append(f"- Anchor delta: {queue_row['anchor_delta']}")
        lines.append(f"- Expected effect: {queue_row['expected_effect']}")
        lines.append(
            f"- Effective labels: model=`{queue_row['model']['stage_label']}`, "
            f"data=`{queue_row['data']['surface_label']}`, "
            f"preprocessing=`{queue_row['preprocessing']['surface_label']}`, "
            f"training=`{queue_row['training']['surface_label']}`"
        )
        if queue_row["dimension_family"] == "model":
            lines.append(
                f"- Model overrides: `{_render_model_change_payload(cast(Mapping[str, Any], queue_row['model']))}`"
            )
        elif queue_row["dimension_family"] == "data":
            lines.append(f"- Data overrides: `{queue_row['data'].get('surface_overrides', {})}`")
        elif queue_row["dimension_family"] == "training":
            lines.append(f"- Training overrides: `{queue_row['training'].get('overrides', {})}`")
        else:
            lines.append(
                f"- Preprocessing overrides: `{queue_row['preprocessing'].get('overrides', {})}`"
            )
        lines.append("- Parameter adequacy plan:")
        for plan_item in cast(list[str], queue_row.get("parameter_adequacy_plan", [])):
            lines.append(f"  - {plan_item}")
        if cast(list[str], queue_row.get("adequacy_knobs", [])):
            lines.append("- Adequacy knobs to dimension explicitly:")
            for adequacy_knob in cast(list[str], queue_row["adequacy_knobs"]):
                lines.append(f"  - {adequacy_knob}")
        lines.append(
            f"- Interpretation status: `{queue_row.get('interpretation_status', 'pending')}`"
        )
        lines.append(f"- Decision: `{queue_row.get('decision')}`")
        if cast(list[str], queue_row.get("confounders", [])):
            lines.append("- Confounders:")
            for confounder in cast(list[str], queue_row["confounders"]):
                lines.append(f"  - {confounder}")
        if cast(list[str], queue_row.get("notes", [])):
            lines.append("- Notes:")
            for note in cast(list[str], queue_row["notes"]):
                lines.append(f"  - {note}")
        lines.append(f"- Follow-up run ids: `{queue_row.get('followup_run_ids', [])}`")
        lines.append(
            f"- Result card path: `{_render_path(_result_card_path(sweep_id=sweep_id, delta_id=delta_id))}`"
        )
        if run is None:
            inline_metrics = queue_row.get("benchmark_metrics")
            if inline_metrics:
                best = float(inline_metrics["best_roc_auc"])
                step = inline_metrics.get("best_step", "?")
                final = float(inline_metrics["final_roc_auc"])
                drift = float(inline_metrics["drift"])
                lines.append("- Benchmark metrics:")
                lines.append(f"  - Best ROC AUC: `{best:.4f}` (step {step})")
                lines.append(f"  - Final ROC AUC: `{final:.4f}`")
                lines.append(f"  - Drift (final − best): `{drift:.4f}`")
                if "nanotabpfn_best" in inline_metrics:
                    lines.append(f"  - NanoTabPFN control: `{float(inline_metrics['nanotabpfn_best']):.4f}`")
                if "max_grad_norm" in inline_metrics:
                    lines.append(f"  - max_grad_norm: `{float(inline_metrics['max_grad_norm']):.3f}`")
            else:
                lines.append("- Benchmark metrics: pending")
        else:
            metrics = _metric_summary(run, anchor)
            metric_parts = [
                f"final log loss `{metrics['final_log_loss']}`",
                f"delta final log loss `{metrics['delta_final_log_loss']}`",
                f"final Brier score `{metrics['final_brier_score']}`",
                f"delta final Brier score `{metrics['delta_final_brier_score']}`",
                f"best ROC AUC `{metrics['best_roc_auc']}`",
                f"final ROC AUC `{metrics['final_roc_auc']}`",
                f"final-minus-best `{metrics['final_minus_best']}`",
                f"delta final ROC AUC `{metrics['delta_final_roc_auc']}`",
                f"delta drift `{metrics['delta_drift']}`",
                f"final CRPS `{metrics['final_crps']}`",
                f"delta final CRPS `{metrics['delta_final_crps']}`",
                f"final avg pinball loss `{metrics['final_avg_pinball_loss']}`",
                f"delta final avg pinball loss `{metrics['delta_final_avg_pinball_loss']}`",
                f"final PICP 90 `{metrics['final_picp_90']}`",
                f"delta final PICP 90 `{metrics['delta_final_picp_90']}`",
                f"delta final training time `{metrics['delta_training_time']}`",
            ]
            filtered_metric_parts = [part for part in metric_parts if not part.endswith("`n/a`")]
            lines.append(f"- Registered run: `{run_id}` with {', '.join(filtered_metric_parts)}")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _instantiate_queue_row(
    *,
    sweep_id: str,
    anchor_run_id: str,
    order: int,
    delta_id: str,
    delta_entry: Mapping[str, Any],
    anchor_context: Mapping[str, Any],
) -> dict[str, Any]:
    status, interpretation_status, next_action_override = _guarded_initial_state(
        delta_entry=delta_entry,
        anchor_context=anchor_context,
    )
    default_effective_surface = cast(
        dict[str, Any],
        _copy_jsonable(cast(dict[str, Any], delta_entry.get("default_effective_surface", {}))),
    )
    parameter_policy = cast(dict[str, Any], delta_entry.get("parameter_adequacy_policy", {}))
    return {
        "order": int(order),
        "delta_ref": str(delta_id),
        "status": status,
        "rationale": (
            f"Contextualize `{delta_id}` against anchor `{anchor_run_id}` for sweep `{sweep_id}`."
        ),
        "hypothesis": "",
        "anchor_delta": (
            f"TODO: describe how `{delta_id}` differs from the locked anchor `{anchor_run_id}`."
        ),
        "model": cast(dict[str, Any], _copy_jsonable(default_effective_surface.get("model", {}))),
        "data": cast(dict[str, Any], _copy_jsonable(default_effective_surface.get("data", {}))),
        "preprocessing": cast(
            dict[str, Any],
            _copy_jsonable(default_effective_surface.get("preprocessing", {})),
        ),
        "training": cast(
            dict[str, Any],
            _copy_jsonable(
                default_effective_surface.get(
                    "training",
                    {
                        "surface_label": _anchor_training_surface_label(anchor_context),
                        "overrides": {},
                    },
                )
            ),
        ),
        "parameter_adequacy_plan": cast(
            list[Any],
            _copy_jsonable(parameter_policy.get("default_plan", [])),
        ),
        "run_id": None,
        "followup_run_ids": [],
        "decision": None,
        "interpretation_status": interpretation_status,
        "confounders": [],
        "next_action": str(next_action_override or delta_entry.get("default_next_action", "")),
        "notes": [],
    }


def create_sweep(
    *,
    sweep_id: str,
    anchor_run_id: str,
    parent_sweep_id: str | None,
    complexity_level: str,
    benchmark_bundle_path: str,
    control_baseline_id: str,
    delta_refs: Sequence[str] | None = None,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    registry_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, str]:
    normalized_sweep_id = _ensure_non_empty_string(sweep_id, context="sweep_id")
    normalized_anchor_run_id = _ensure_non_empty_string(anchor_run_id, context="anchor_run_id")
    normalized_complexity_level = _ensure_non_empty_string(
        complexity_level,
        context="complexity_level",
    )
    normalized_benchmark_bundle_path = _ensure_non_empty_string(
        benchmark_bundle_path,
        context="benchmark_bundle_path",
    )
    normalized_control_baseline_id = _ensure_non_empty_string(
        control_baseline_id,
        context="control_baseline_id",
    )
    resolved_index_path = (index_path or default_sweep_index_path()).expanduser().resolve()
    resolved_sweeps_root = sweeps_root or resolved_index_path.parent
    index = load_system_delta_index(resolved_index_path)
    sweeps = _ensure_mapping(index.get("sweeps"), context="sweep index sweeps")
    if normalized_sweep_id in sweeps:
        raise RuntimeError(f"sweep_id {normalized_sweep_id!r} already exists")

    catalog = load_system_delta_catalog(catalog_path)
    active_sweep_id = _ensure_non_empty_string(index.get("active_sweep_id"), context="active_sweep_id")
    template_sweep = load_system_delta_sweep(
        parent_sweep_id or active_sweep_id,
        index_path=resolved_index_path,
        sweeps_root=resolved_sweeps_root,
    )
    anchor_context = _anchor_context_from_registry_run(
        anchor_run_id=normalized_anchor_run_id,
        registry_path=registry_path,
    )
    sweep_status = DEFAULT_SWEEP_STATUS
    if not sweeps:
        sweep_status = "active"
        index["active_sweep_id"] = normalized_sweep_id

    sweep_payload = {
        "schema": SWEEP_SCHEMA,
        "sweep_id": normalized_sweep_id,
        "parent_sweep_id": None if parent_sweep_id is None else str(parent_sweep_id),
        "status": sweep_status,
        "complexity_level": normalized_complexity_level,
        "anchor_run_id": normalized_anchor_run_id,
        "benchmark_bundle_path": normalized_benchmark_bundle_path,
        "control_baseline_id": normalized_control_baseline_id,
        "comparison_policy": str(template_sweep.get("comparison_policy", "anchor_only")),
        "upstream_reference": cast(
            dict[str, Any],
            _copy_jsonable(cast(dict[str, Any], template_sweep.get("upstream_reference", {}))),
        ),
        "anchor_surface": _build_anchor_surface(
            anchor_run_id=normalized_anchor_run_id,
            benchmark_bundle_path=normalized_benchmark_bundle_path,
            anchor_context=anchor_context,
        ),
        "anchor_context": anchor_context,
    }

    deltas = _ensure_mapping(catalog.get("deltas"), context="catalog deltas")
    selected_delta_ids: list[str]
    if delta_refs is None:
        selected_delta_ids = list(deltas)
    else:
        selected_delta_ids = [
            _ensure_non_empty_string(delta_ref, context="delta_refs[]") for delta_ref in delta_refs
        ]
        if not selected_delta_ids:
            raise RuntimeError("delta_refs must include at least one delta id when provided")
        if len(set(selected_delta_ids)) != len(selected_delta_ids):
            raise RuntimeError("delta_refs must not contain duplicates")
        unknown_delta_ids = [delta_id for delta_id in selected_delta_ids if delta_id not in deltas]
        if unknown_delta_ids:
            raise RuntimeError(f"unknown delta_refs for sweep {normalized_sweep_id!r}: {unknown_delta_ids}")
    queue_rows = [
        _instantiate_queue_row(
            sweep_id=normalized_sweep_id,
            anchor_run_id=normalized_anchor_run_id,
            order=order,
            delta_id=delta_id,
            delta_entry=cast(dict[str, Any], deltas[delta_id]),
            anchor_context=anchor_context,
        )
        for order, delta_id in enumerate(selected_delta_ids, start=1)
    ]
    queue_payload = {
        "schema": SWEEP_QUEUE_SCHEMA,
        "sweep_id": normalized_sweep_id,
        "rows": queue_rows,
    }

    sweep_info = {
        "parent_sweep_id": None if parent_sweep_id is None else str(parent_sweep_id),
        "status": sweep_status,
        "anchor_run_id": normalized_anchor_run_id,
        "complexity_level": normalized_complexity_level,
        "benchmark_bundle_path": normalized_benchmark_bundle_path,
        "control_baseline_id": normalized_control_baseline_id,
    }
    sweeps[normalized_sweep_id] = sweep_info

    _write_yaml(sweep_metadata_path(normalized_sweep_id, sweeps_root=resolved_sweeps_root), sweep_payload)
    _write_yaml(sweep_queue_path(normalized_sweep_id, sweeps_root=resolved_sweeps_root), queue_payload)
    _write_yaml(resolved_index_path, index)

    queue = materialize_system_delta_queue(
        catalog=catalog,
        sweep=sweep_payload,
        queue_instance=queue_payload,
        catalog_path=catalog_path,
        sweeps_root=resolved_sweeps_root,
    )
    matrix_contents = render_system_delta_matrix(queue, registry_path=registry_path)
    _write_text(sweep_matrix_path(normalized_sweep_id, sweeps_root=resolved_sweeps_root), matrix_contents)
    if _ensure_non_empty_string(index.get("active_sweep_id"), context="active_sweep_id") == normalized_sweep_id:
        sync_active_sweep_aliases(
            sweep_id=normalized_sweep_id,
            index_path=resolved_index_path,
            catalog_path=catalog_path,
            registry_path=registry_path,
            sweeps_root=resolved_sweeps_root,
        )

    return {
        "sweep_path": str(sweep_metadata_path(normalized_sweep_id, sweeps_root=resolved_sweeps_root).resolve()),
        "queue_path": str(sweep_queue_path(normalized_sweep_id, sweeps_root=resolved_sweeps_root).resolve()),
        "matrix_path": str(sweep_matrix_path(normalized_sweep_id, sweeps_root=resolved_sweeps_root).resolve()),
        "index_path": str(resolved_index_path),
    }


def set_active_sweep(
    sweep_id: str,
    *,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    registry_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, str]:
    normalized_sweep_id = _ensure_non_empty_string(sweep_id, context="sweep_id")
    resolved_index_path = (index_path or default_sweep_index_path()).expanduser().resolve()
    index = load_system_delta_index(resolved_index_path)
    sweeps = _ensure_mapping(index.get("sweeps"), context="sweep index sweeps")
    if normalized_sweep_id not in sweeps:
        raise RuntimeError(f"unknown sweep_id: {normalized_sweep_id}")
    index["active_sweep_id"] = normalized_sweep_id
    _write_yaml(resolved_index_path, index)
    return sync_active_sweep_aliases(
        sweep_id=normalized_sweep_id,
        index_path=resolved_index_path,
        catalog_path=catalog_path,
        registry_path=registry_path,
        sweeps_root=sweeps_root,
    )


def sync_active_sweep_aliases(
    *,
    sweep_id: str | None = None,
    index_path: Path | None = None,
    catalog_path: Path | None = None,
    registry_path: Path | None = None,
    sweeps_root: Path | None = None,
) -> dict[str, str]:
    queue = load_system_delta_queue(
        sweep_id=sweep_id,
        index_path=index_path,
        catalog_path=catalog_path,
        sweeps_root=sweeps_root,
    )
    matrix_contents = render_system_delta_matrix(queue, registry_path=registry_path)
    alias_queue_path = default_queue_path()
    alias_matrix_path = default_matrix_path()
    _write_yaml(alias_queue_path, queue)
    _write_text(alias_matrix_path, matrix_contents)
    return {
        "queue_alias_path": str(alias_queue_path.resolve()),
        "matrix_alias_path": str(alias_matrix_path.resolve()),
    }


def list_sweeps(*, index_path: Path | None = None) -> list[dict[str, Any]]:
    index = load_system_delta_index(index_path)
    active_sweep_id = _ensure_non_empty_string(index.get("active_sweep_id"), context="active_sweep_id")
    sweeps = _ensure_mapping(index.get("sweeps"), context="sweep index sweeps")
    ordered = sorted(sweeps.items(), key=lambda item: str(item[0]))
    return [
        {
            "sweep_id": sweep_id,
            "is_active": sweep_id == active_sweep_id,
            **cast(dict[str, Any], _copy_jsonable(sweep_info)),
        }
        for sweep_id, sweep_info in ordered
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Manage sweep-aware system-delta queues")
    parser.add_argument(
        "--catalog-path",
        default=str(default_catalog_path()),
        help="Path to reference/system_delta_catalog.yaml",
    )
    parser.add_argument(
        "--index-path",
        default=str(default_sweep_index_path()),
        help="Path to reference/system_delta_sweeps/index.yaml",
    )
    parser.add_argument(
        "--registry-path",
        default=str(default_registry_path()),
        help="Path to benchmark_run_registry_v1.json",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list-sweeps", help="List known sweeps")
    subparsers.add_parser("show-active", help="Print the active sweep id")

    set_active_parser = subparsers.add_parser("set-active", help="Set the active sweep and regenerate aliases")
    set_active_parser.add_argument("--sweep-id", required=True, help="Sweep id to activate")

    for command_name, help_text in (
        ("list", "List queue rows in order"),
        ("next", "Print the next ready row"),
        ("render", "Render the selected sweep matrix"),
        ("validate", "Validate completed rows for the selected sweep"),
    ):
        command_parser = subparsers.add_parser(command_name, help=help_text)
        command_parser.add_argument(
            "--sweep-id",
            default=None,
            help="Optional sweep id; defaults to the active sweep",
        )
        if command_name == "render":
            command_parser.add_argument(
                "--out-path",
                default=None,
                help="Optional alternate markdown output path",
            )

    create_parser = subparsers.add_parser("create-sweep", help="Bootstrap a new sweep from the delta catalog")
    create_parser.add_argument("--sweep-id", required=True, help="New sweep id")
    create_parser.add_argument("--anchor-run-id", required=True, help="Anchor benchmark registry run id")
    create_parser.add_argument("--parent-sweep-id", default=None, help="Optional parent sweep id")
    create_parser.add_argument("--complexity-level", required=True, help="Complexity level label")
    create_parser.add_argument(
        "--benchmark-bundle-path",
        required=True,
        help="Benchmark bundle path for the new sweep",
    )
    create_parser.add_argument(
        "--control-baseline-id",
        required=True,
        help="Control baseline id for the new sweep",
    )
    create_parser.add_argument(
        "--delta-ref",
        action="append",
        dest="delta_refs",
        default=None,
        help="Optional ordered delta id to include; repeat to build a curated subset",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    catalog_path = Path(str(args.catalog_path))
    index_path = Path(str(args.index_path))
    registry_path = Path(str(args.registry_path))

    if args.command == "list-sweeps":
        for sweep_info in list_sweeps(index_path=index_path):
            marker = "*" if sweep_info["is_active"] else " "
            print(
                f"{marker} {sweep_info['sweep_id']}  {sweep_info['status']:<8}  "
                f"{sweep_info['complexity_level']:<16}  anchor={sweep_info['anchor_run_id']}"
            )
        return 0

    if args.command == "show-active":
        index = load_system_delta_index(index_path)
        print(_ensure_non_empty_string(index.get("active_sweep_id"), context="active_sweep_id"))
        return 0

    if args.command == "set-active":
        result = set_active_sweep(
            str(args.sweep_id),
            index_path=index_path,
            catalog_path=catalog_path,
            registry_path=registry_path,
        )
        print(f"Active sweep set to {args.sweep_id}")
        print(f"  queue_alias_path={result['queue_alias_path']}")
        print(f"  matrix_alias_path={result['matrix_alias_path']}")
        return 0

    if args.command == "create-sweep":
        result = create_sweep(
            sweep_id=str(args.sweep_id),
            anchor_run_id=str(args.anchor_run_id),
            parent_sweep_id=None if args.parent_sweep_id is None else str(args.parent_sweep_id),
            complexity_level=str(args.complexity_level),
            benchmark_bundle_path=str(args.benchmark_bundle_path),
            control_baseline_id=str(args.control_baseline_id),
            delta_refs=None if args.delta_refs is None else [str(value) for value in args.delta_refs],
            index_path=index_path,
            catalog_path=catalog_path,
            registry_path=registry_path,
        )
        print("Sweep created:")
        print(f"  sweep_path={result['sweep_path']}")
        print(f"  queue_path={result['queue_path']}")
        print(f"  matrix_path={result['matrix_path']}")
        print(f"  index_path={result['index_path']}")
        return 0

    selected_sweep_id = None if getattr(args, "sweep_id", None) is None else str(args.sweep_id)
    queue = load_system_delta_queue(
        sweep_id=selected_sweep_id,
        index_path=index_path,
        catalog_path=catalog_path,
    )
    if args.command == "list":
        for row in ordered_rows(queue):
            print(
                f"{int(row['order']):02d}  {row['status']:<28}  "
                f"{row['dimension_family']:<13}  {row['delta_id']}"
            )
        return 0
    if args.command == "next":
        next_row = next_ready_row(queue)
        if next_row is None:
            print("No ready rows.")
            return 0
        print(OmegaConf.to_yaml(next_row, resolve=True).strip())
        return 0
    if args.command == "render":
        contents = render_system_delta_matrix(queue, registry_path=registry_path)
        resolved_out_path = (
            sweep_matrix_path(str(queue["sweep_id"]))
            if args.out_path is None
            else Path(str(args.out_path))
        )
        _write_text(resolved_out_path, contents)
        active_sweep_id = _ensure_non_empty_string(
            load_system_delta_index(index_path).get("active_sweep_id"),
            context="active_sweep_id",
        )
        if str(queue["sweep_id"]) == active_sweep_id:
            sync_active_sweep_aliases(
                sweep_id=str(queue["sweep_id"]),
                index_path=index_path,
                catalog_path=catalog_path,
                registry_path=registry_path,
            )
        print(f"Rendered system delta matrix to {resolved_out_path.expanduser().resolve()}")
        return 0
    if args.command == "validate":
        issues = validate_system_delta_queue(queue, registry_path=registry_path)
        if not issues:
            print("System delta queue validation passed.")
            return 0
        for issue in issues:
            print(issue)
        return 1
    raise RuntimeError(f"Unsupported command: {args.command!r}")
