"""Anchor-context helpers for system-delta sweeps."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, cast

from tab_foundry.bench.benchmark_run_registry import load_benchmark_run_registry, resolve_registry_path_value
from tab_foundry.bench.nanotabpfn import load_benchmark_bundle
from tab_foundry.model.architectures.tabfoundry_staged.resolved import resolve_staged_surface
from tab_foundry.model.spec import ModelBuildSpec

from .paths_io import _copy_jsonable, default_registry_path
from .validation import ensure_mapping, ensure_non_empty_string


LEGACY_PRIOR_CONSTANT_LR_LABEL = "prior_constant_lr"
UNAVAILABLE_TRAINING_LABEL = "training surface label unavailable"
_LEGACY_PRIOR_CONFIG_PROFILE = "cls_benchmark_staged_prior"


def staged_module_selection_from_run_model(model_payload: Mapping[str, Any]) -> dict[str, Any] | None:
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


def anchor_context_from_registry_run(
    *,
    anchor_run_id: str,
    registry_path: Path | None = None,
) -> dict[str, Any]:
    registry = load_benchmark_run_registry(registry_path or default_registry_path())
    runs = ensure_mapping(registry.get("runs"), context="benchmark registry runs")
    run = runs.get(anchor_run_id)
    if not isinstance(run, dict):
        raise RuntimeError(f"anchor_run_id {anchor_run_id!r} is missing from the benchmark registry")
    model = ensure_mapping(run.get("model"), context=f"benchmark registry run {anchor_run_id}.model")
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
            "module_selection": staged_module_selection_from_run_model(model),
        },
        "surface_labels": surface_labels,
    }


def surface_label_from_anchor_context(
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


def anchor_training_surface_label(anchor_context: Mapping[str, Any]) -> str:
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


def anchor_module_selection(anchor_context: Mapping[str, Any]) -> Mapping[str, Any]:
    model = anchor_context.get("model")
    if not isinstance(model, dict):
        return {}
    module_selection = model.get("module_selection")
    if isinstance(module_selection, dict):
        return cast(dict[str, Any], module_selection)
    resolved = staged_module_selection_from_run_model(cast(dict[str, Any], model))
    if resolved is None:
        return {}
    return resolved


def module_choice(module_selection: Mapping[str, Any], key: str, *, fallback: str = "unknown") -> str:
    value = module_selection.get(key)
    if isinstance(value, str) and value.strip():
        return str(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return fallback
    return str(value)


def describe_feature_encoder(module_selection: Mapping[str, Any]) -> tuple[str, str]:
    feature_encoder = module_choice(module_selection, "feature_encoder")
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


def describe_target_conditioner(module_selection: Mapping[str, Any]) -> tuple[str, str]:
    target_conditioner = module_choice(module_selection, "target_conditioner")
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


def describe_table_block(module_selection: Mapping[str, Any]) -> tuple[str, str]:
    table_block_style = module_choice(module_selection, "table_block_style")
    allow_test_self_attention = module_choice(
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


def describe_tokenizer(module_selection: Mapping[str, Any]) -> tuple[str, str]:
    tokenizer = module_choice(module_selection, "tokenizer")
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


def describe_column_encoder(module_selection: Mapping[str, Any]) -> tuple[str, str]:
    column_encoder = module_choice(module_selection, "column_encoder")
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


def describe_row_pool(module_selection: Mapping[str, Any]) -> tuple[str, str]:
    row_pool = module_choice(module_selection, "row_pool")
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


def describe_context_encoder(module_selection: Mapping[str, Any]) -> tuple[str, str]:
    context_encoder = module_choice(module_selection, "context_encoder")
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


def describe_head(module_selection: Mapping[str, Any]) -> tuple[str, str]:
    head = module_choice(module_selection, "head")
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


def build_anchor_surface(
    *,
    anchor_run_id: str,
    benchmark_bundle_path: str,
    anchor_context: Mapping[str, Any],
) -> dict[str, Any]:
    bundle_path = resolve_registry_path_value(benchmark_bundle_path)
    bundle = load_benchmark_bundle(bundle_path)
    bundle_name = ensure_non_empty_string(bundle.get("name"), context="benchmark bundle name")
    task_ids = cast(list[Any], bundle.get("task_ids", []))
    task_count = int(len(task_ids))
    module_selection = anchor_module_selection(anchor_context)
    model_label = surface_label_from_anchor_context(
        anchor_context,
        key="model",
        fallback="registry surface label unavailable",
    )
    data_label = surface_label_from_anchor_context(
        anchor_context,
        key="data",
        fallback="registry surface label unavailable",
    )
    preprocessing_label = surface_label_from_anchor_context(
        anchor_context,
        key="preprocessing",
        fallback="registry surface label unavailable",
    )
    training_label = anchor_training_surface_label(anchor_context)
    feature_encoder, feature_encoder_interpretation = describe_feature_encoder(module_selection)
    target_conditioner, target_conditioner_interpretation = describe_target_conditioner(
        module_selection
    )
    table_block, table_block_interpretation = describe_table_block(module_selection)
    tokenizer, tokenizer_interpretation = describe_tokenizer(module_selection)
    column_encoder, column_encoder_interpretation = describe_column_encoder(module_selection)
    row_pool, row_pool_interpretation = describe_row_pool(module_selection)
    context_encoder, context_encoder_interpretation = describe_context_encoder(module_selection)
    head, head_interpretation = describe_head(module_selection)
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
