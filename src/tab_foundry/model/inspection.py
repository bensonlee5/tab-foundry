"""Shared model inspection helpers for developer tooling."""

from __future__ import annotations

from dataclasses import asdict
from dataclasses import dataclass
from typing import Any, Mapping

import torch

from tab_foundry.feature_types import DEFAULT_FEATURE_TYPE
from tab_foundry.types import TaskBatch

from .accounting import compute_accounting_from_model, parameter_accounting_from_model
from .architectures.tabfoundry_staged.resolved import resolve_staged_surface
from .factory import build_model_from_spec
from .spec import (
    GRID_SANDWICH_MODEL_ARCH,
    ModelBuildSpec,
    ROUTED_SANDWICH_MODEL_ARCH,
    SANDWICH_FAMILY_MODEL_ARCHES,
    SANDWICH_MODEL_ARCH,
)


@dataclass(slots=True, frozen=True)
class SyntheticForwardBatch:
    """Deterministic synthetic inputs for model construction checks."""

    task_batch: TaskBatch
    x_all: torch.Tensor
    y_train_batched: torch.Tensor
    train_test_split_index: int
    expected_output_kind: str
    expected_num_classes: int
    expected_test_rows: int


@dataclass(slots=True, frozen=True)
class SyntheticReferenceArrays:
    """Deterministic runtime arrays for export/reference smoke checks."""

    x_train: torch.Tensor
    y_train: torch.Tensor
    x_test: torch.Tensor
    feature_types: list[str] | None
    expected_num_classes: int


def parameter_counts_from_model_spec(spec: ModelBuildSpec) -> dict[str, int]:
    """Return total and trainable parameter counts for one resolved model spec."""

    model = build_model_from_spec(spec)
    total_params = sum(int(parameter.numel()) for parameter in model.parameters())
    trainable_params = sum(
        int(parameter.numel()) for parameter in model.parameters() if parameter.requires_grad
    )
    return {
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
    }


def parameter_accounting_from_model_spec(spec: ModelBuildSpec) -> dict[str, Any]:
    """Return inspected parameter accounting for one resolved model spec."""

    model = build_model_from_spec(spec)
    return parameter_accounting_from_model(model)


def compute_accounting_from_model_spec(
    spec: ModelBuildSpec,
    *,
    training_shape_summary: Mapping[str, Any] | None,
    tokens_seen: int | None,
    tokens_per_step: float | None,
) -> dict[str, Any]:
    """Return inspected analytic compute accounting for one resolved model spec."""

    model = build_model_from_spec(spec)
    return compute_accounting_from_model(
        model,
        training_shape_summary=training_shape_summary,
        tokens_seen=tokens_seen,
        tokens_per_step=tokens_per_step,
    )


def model_surface_payload(spec: ModelBuildSpec) -> dict[str, Any]:
    """Render the resolved model surface for CLI and artifact summaries."""

    payload: dict[str, Any] = {
        "arch": str(spec.arch),
        "stage": None if spec.stage is None else str(spec.stage),
        "stage_label": None if spec.stage_label is None else str(spec.stage_label),
        "input_normalization": str(spec.input_normalization),
        "feature_group_size": int(spec.feature_group_size),
        "many_class_base": int(spec.many_class_base),
        "build_spec": spec.to_dict(),
    }
    if spec.arch != "tabfoundry_staged":
        if spec.arch == SANDWICH_MODEL_ARCH:
            payload["architecture"] = {
                "initial_input_tokens": "full_cell_plus_row_col_summary_stream",
                "initial_input_token_count": "R_times_C_plus_K_times_(R_plus_C)",
                "repeated_input_tokens": "row_col_summary_stream",
                "repeated_input_token_count": "K_times_(R_plus_C)",
                "summary_tokens_per_axis": int(spec.sandwich_summary_tokens_per_axis),
                "pre_perceiver_cell_mixer": "row_feature_self_attention_then_column_row_isab",
                "pre_row_attention_layers": int(spec.sandwich_pre_row_attention_layers),
                "pre_column_attention_layers": int(spec.sandwich_pre_column_attention_layers),
                "pre_column_inducing_tokens": int(spec.sandwich_pre_column_inducing_tokens),
                "label_injection": "fused_into_row_summaries_and_feature_cells",
                "summary_builder": "summary_query_attention",
                "position_encoding": "shared_fourier_row_col",
                "feature_type_encoding": str(spec.feature_type_conditioning),
                "floating_likelihood": str(spec.floating_likelihood),
                "integer_likelihood": str(spec.integer_likelihood),
                "sandwich_activation": str(spec.sandwich_activation),
                "sandwich_block_norm": str(spec.sandwich_block_norm),
                "sandwich_packed_attention": bool(spec.sandwich_packed_attention),
                "latent_core": "stage0_full_cell_plus_summary_then_summary_repeated_cross_self_stages",
                "layer_semantics": "stage0_hybrid_then_summary_repeated_stages",
                "readout": "latent_then_full_cell_cross_attention_then_latent_conditioned_query_pool",
                "latents": int(spec.sandwich_latents),
                "layers": int(spec.sandwich_layers),
                "heads": int(spec.sandwich_heads),
                "ff_expansion": int(spec.sandwich_ff_expansion),
                "self_attention_per_cross": int(spec.sandwich_self_attention_per_cross),
            }
        elif spec.arch == ROUTED_SANDWICH_MODEL_ARCH:
            if bool(spec.routed_direct_cell_bypass):
                initial_input_tokens = "full_cell_plus_row_col_summary_plus_evidence_bank"
                initial_input_token_count = (
                    "R_times_C_plus_K_row_times_R_plus_K_col_times_C_plus_K_evidence"
                )
                readout = "latent_then_full_cell_routed_cross_attention_then_latent_conditioned_query_pool"
            else:
                initial_input_tokens = "row_col_summary_plus_evidence_bank"
                initial_input_token_count = "K_row_times_R_plus_K_col_times_C_plus_K_evidence"
                readout = "latent_conditioned_routed_query_pool"
            payload["architecture"] = {
                "initial_input_tokens": initial_input_tokens,
                "initial_input_token_count": initial_input_token_count,
                "repeated_input_tokens": "row_col_summary_plus_evidence_bank",
                "repeated_input_token_count": "K_row_times_R_plus_K_col_times_C_plus_K_evidence",
                "summary_tokens_per_row": int(spec.routed_row_summary_tokens),
                "summary_tokens_per_column": int(spec.routed_column_summary_tokens),
                "evidence_tokens": int(spec.routed_evidence_tokens),
                "pre_perceiver_cell_mixer": "row_feature_self_attention_then_column_row_isab",
                "pre_row_attention_layers": int(spec.sandwich_pre_row_attention_layers),
                "pre_column_attention_layers": int(spec.sandwich_pre_column_attention_layers),
                "pre_column_inducing_tokens": int(spec.sandwich_pre_column_inducing_tokens),
                "label_injection": "fused_into_row_summaries_and_evidence_builder",
                "summary_builder": "summary_query_attention",
                "evidence_builder": "learned_evidence_query_attention",
                "position_encoding": "shared_fourier_row_col",
                "feature_type_encoding": str(spec.feature_type_conditioning),
                "sandwich_activation": str(spec.sandwich_activation),
                "sandwich_block_norm": str(spec.sandwich_block_norm),
                "sandwich_packed_attention": bool(spec.sandwich_packed_attention),
                "latent_core": "routed_cross_self_stages_over_latent_streams",
                "residual_routing": str(spec.routed_residual_mode),
                "residual_streams": int(spec.routed_residual_streams),
                "residual_scaling": str(spec.routed_residual_scale),
                "readout": readout,
                "direct_cell_bypass": bool(spec.routed_direct_cell_bypass),
                "latents": int(spec.sandwich_latents),
                "layers": int(spec.sandwich_layers),
                "heads": int(spec.sandwich_heads),
                "ff_expansion": int(spec.sandwich_ff_expansion),
                "self_attention_per_cross": int(spec.sandwich_self_attention_per_cross),
            }
        elif spec.arch == GRID_SANDWICH_MODEL_ARCH:
            payload["architecture"] = {
                "input_tokens": "row_feature_cell_grid",
                "input_token_count": "R_times_C",
                "grid_core": "alternating_row_self_attention_and_column_row_isab",
                "pre_perceiver_cell_mixer": "row_feature_self_attention_then_column_row_isab",
                "pre_row_attention_layers": int(spec.sandwich_pre_row_attention_layers),
                "pre_column_attention_layers": int(spec.sandwich_pre_column_attention_layers),
                "grid_preservation": "explicit_row_feature_grid_through_core",
                "label_injection": "train_row_feature_tokens_only",
                "position_encoding": "shared_fourier_row_col",
                "feature_type_encoding": str(spec.feature_type_conditioning),
                "sandwich_activation": str(spec.sandwich_activation),
                "sandwich_block_norm": str(spec.sandwich_block_norm),
                "sandwich_packed_attention": bool(spec.sandwich_packed_attention),
                "grid_residual_mode": str(spec.grid_residual_mode),
                "grid_attention_mode": str(spec.grid_attention_mode),
                "grid_ffn_mode": str(spec.grid_ffn_mode),
                "grid_recurrence_steps": None
                if spec.grid_recurrence_steps is None
                else int(spec.grid_recurrence_steps),
                "grid_recurrence_unique_layers": None
                if spec.grid_recurrence_unique_layers is None
                else int(spec.grid_recurrence_unique_layers),
                "classification_logit_softcap": None
                if spec.classification_logit_softcap is None
                else float(spec.classification_logit_softcap),
                "attention_qk_norm": bool(spec.attention_qk_norm),
                "grid_moe_scope": str(spec.grid_moe_scope),
                "grid_moe_num_experts": int(spec.grid_moe_num_experts),
                "grid_moe_top_k": int(spec.grid_moe_top_k),
                "grid_moe_router_init_std": float(spec.grid_moe_router_init_std),
                "grid_moe_normalize_top_k": bool(spec.grid_moe_normalize_top_k),
                "grid_core_iterations": int(spec.grid_recurrence_steps or spec.sandwich_layers),
                "grid_core_unique_layers": int(
                    spec.grid_recurrence_unique_layers
                    or (1 if spec.grid_recurrence_steps is not None else spec.sandwich_layers)
                ),
                "layers": int(spec.sandwich_layers),
                "heads": int(spec.sandwich_heads),
                "ff_expansion": int(spec.sandwich_ff_expansion),
                "column_inducing_tokens": int(spec.sandwich_pre_column_inducing_tokens),
                "readout": "per_test_row_feature_bundle_pool",
            }
        return payload

    surface = resolve_staged_surface(spec)
    payload.update(
        {
            "benchmark_profile": str(surface.benchmark_profile),
            "module_selection": surface.module_selection(),
            "module_hyperparameters": surface.component_hyperparameters(),
            "task_contract": dict(asdict(surface.task_contract)),
        }
    )
    return payload


def synthetic_forward_batch(spec: ModelBuildSpec) -> SyntheticForwardBatch:
    """Build one deterministic synthetic task batch for forward-only checks."""

    train_rows = 3
    test_rows = 2
    feature_count = 4
    total_rows = train_rows + test_rows
    x_all = torch.arange(total_rows * feature_count, dtype=torch.float32).reshape(total_rows, feature_count)
    x_all = (x_all / float(feature_count)) - 1.0
    num_classes, output_kind = _resolved_forward_shape(spec)
    y_all = torch.arange(total_rows, dtype=torch.int64).remainder(num_classes)
    x_train = x_all[:train_rows].clone()
    x_test = x_all[train_rows:].clone()
    y_train = y_all[:train_rows].clone()
    y_test = y_all[train_rows:].clone()
    task_batch = TaskBatch(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        metadata={
            "source": "synthetic_forward_check",
            "feature_types": [DEFAULT_FEATURE_TYPE] * feature_count,
        },
        num_classes=int(num_classes),
    )
    return SyntheticForwardBatch(
        task_batch=task_batch,
        x_all=x_all.unsqueeze(0),
        y_train_batched=y_train.unsqueeze(0),
        train_test_split_index=int(train_rows),
        expected_output_kind=output_kind,
        expected_num_classes=int(num_classes),
        expected_test_rows=int(test_rows),
    )


def synthetic_reference_arrays(
    spec: ModelBuildSpec,
    *,
    include_missing_inputs: bool,
) -> SyntheticReferenceArrays:
    """Build deterministic runtime arrays for reference-consumer smoke checks."""

    num_classes, _ = _resolved_forward_shape(spec)
    train_rows = max(int(num_classes) + 1, 4)
    test_rows = 2
    feature_count = 3
    total_rows = train_rows + test_rows

    x_all = torch.arange(total_rows * feature_count, dtype=torch.float32).reshape(total_rows, feature_count)
    x_all = (x_all / float(feature_count)) + 1.0
    x_train = x_all[:train_rows].clone()
    x_test = x_all[train_rows:].clone()
    if include_missing_inputs:
        x_train[0, 1] = float("nan")
        x_test[0, 0] = float("nan")

    y_train = torch.arange(train_rows, dtype=torch.int64).remainder(int(num_classes))
    y_train = y_train + 100
    feature_types = (
        [DEFAULT_FEATURE_TYPE] * feature_count
        if str(spec.arch).strip().lower() in SANDWICH_FAMILY_MODEL_ARCHES
        else None
    )
    return SyntheticReferenceArrays(
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        feature_types=feature_types,
        expected_num_classes=int(num_classes),
    )


def _resolved_forward_shape(spec: ModelBuildSpec) -> tuple[int, str]:
    if spec.arch == "tabfoundry_simple":
        return 2, "logits"
    if spec.arch in SANDWICH_FAMILY_MODEL_ARCHES:
        return int(spec.many_class_base), "logits"

    surface = resolve_staged_surface(spec)
    if surface.head == "many_class":
        return max(int(spec.many_class_base) + 1, int(surface.task_contract.min_classes)), "class_probs"
    if surface.head == "small_class":
        return int(spec.many_class_base), "logits"

    target_classes = 2 if surface.task_contract.max_classes == 2 else 3
    if surface.task_contract.max_classes is not None:
        target_classes = min(target_classes, int(surface.task_contract.max_classes))
    target_classes = max(target_classes, int(surface.task_contract.min_classes))
    return int(target_classes), "logits"
