"""Cell-likelihood helpers for the sandwich architecture family."""

from __future__ import annotations

from typing import Any, cast

import torch

from tab_foundry.feature_types import (
    FEATURE_TYPE_BOOL,
    FEATURE_TYPE_FLOATING,
    FEATURE_TYPE_INTEGER,
    FEATURE_TYPE_VOCAB,
)
from tab_foundry.likelihoods import cross_entropy_bits, gaussian_nll_bits, mixture_bits
from tab_foundry.model.components.attention import attention_bias_from_allowed_mask
from tab_foundry.model.outputs import (
    CategoricalCellPrediction,
    CellLikelihoodOutput,
    FloatingCellPrediction,
    IntegerHybridCellPrediction,
)

from .blocks import _SelfAttentionBlock
from .classification_flow import build_classification_state, forward_logits, full_cell_tokens
from .states import SandwichFeatureState


def cell_decoder_hidden(
    model: Any,
    *,
    feature_cells: torch.Tensor,
    y_train: torch.Tensor,
) -> torch.Tensor:
    """Run the autoregressive cell decoder over the feature-cell stream."""

    batch_size = int(feature_cells.shape[0])
    num_rows = int(feature_cells.shape[1])
    num_features = int(feature_cells.shape[2])
    cell_stream = full_cell_tokens(model, feature_cells, y_train=y_train)
    shifted = torch.cat(
        [
            model.cell_bos.expand(batch_size, -1, -1).to(
                device=cell_stream.device,
                dtype=cell_stream.dtype,
            ),
            cell_stream[:, :-1, :],
        ],
        dim=1,
    )
    seq_len = int(shifted.shape[1])
    allowed = torch.tril(
        torch.ones((seq_len, seq_len), device=shifted.device, dtype=torch.bool)
    )
    attn_bias = attention_bias_from_allowed_mask(allowed, dtype=shifted.dtype)
    hidden = shifted
    for index, block in enumerate(model.cell_decoder_blocks):
        block = cast(_SelfAttentionBlock, block)
        hidden = model._self_block(block, hidden, attn_bias=attn_bias)
        model.trace_activation(f"post_cell_decoder_{index}", hidden)
    return hidden.reshape(batch_size, num_rows, num_features, model.d_icl)


def support_values_for_feature(
    train_values: torch.Tensor,
    *,
    feature_type: str,
) -> torch.Tensor:
    """Resolve the discrete support values for one feature column."""

    if feature_type == FEATURE_TYPE_BOOL:
        return torch.tensor([0.0, 1.0], device=train_values.device, dtype=train_values.dtype)
    finite_values = train_values[torch.isfinite(train_values)]
    if int(finite_values.numel()) <= 0:
        return torch.empty((0,), device=train_values.device, dtype=train_values.dtype)
    return torch.unique(finite_values, sorted=True)


def support_embeddings(
    model: Any,
    support_values: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Encode discrete support values with the sandwich feature encoder."""

    if int(support_values.numel()) <= 0:
        return torch.empty((0, model.d_icl), device=device, dtype=dtype)
    tokenized, _ = model.tokenizer(support_values.to(torch.float32))
    return model.feature_encoder(tokenized.to(device=device)).to(dtype=dtype)


def dynamic_discrete_logits(
    model: Any,
    feature_hidden: torch.Tensor,
    *,
    support_embeddings: torch.Tensor,
) -> torch.Tensor:
    """Project feature hidden states onto the dynamic discrete support."""

    query = model.discrete_query(feature_hidden)
    if int(support_embeddings.shape[0]) <= 0:
        support_logits = torch.empty(
            (int(feature_hidden.shape[0]), 0),
            device=feature_hidden.device,
            dtype=feature_hidden.dtype,
        )
    else:
        support_logits = query @ support_embeddings.transpose(0, 1)
    oov_logits = model.discrete_oov(feature_hidden)
    return torch.cat([support_logits, oov_logits], dim=-1)


def discrete_target_indices(
    values: torch.Tensor,
    *,
    support_values: torch.Tensor,
) -> torch.Tensor:
    """Resolve the categorical target index for each feature value."""

    oov_index = int(support_values.shape[0])
    if oov_index <= 0:
        return torch.full(
            (int(values.shape[0]),),
            oov_index,
            device=values.device,
            dtype=torch.int64,
        )
    equality = values.unsqueeze(-1) == support_values.unsqueeze(0)
    any_match = equality.any(dim=-1)
    first_match = equality.to(torch.int64).argmax(dim=-1)
    target_indices = torch.full(
        (int(values.shape[0]),),
        oov_index,
        device=values.device,
        dtype=torch.int64,
    )
    target_indices[any_match] = first_match[any_match]
    target_indices[~torch.isfinite(values)] = oov_index
    return target_indices


def integer_gate_logit(
    model: Any,
    support_embeddings: torch.Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Resolve the integer hybrid gate logit from support embeddings."""

    if int(support_embeddings.shape[0]) <= 0:
        summary = torch.zeros((model.d_icl,), device=device, dtype=dtype)
    else:
        summary = support_embeddings.mean(dim=0)
    return model.integer_gate(summary).reshape(())


def gaussian_params(model: Any, feature_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Resolve the Gaussian mean and log-variance parameters."""

    params = model.gaussian_head(feature_hidden)
    mean = params[:, 0]
    log_variance = params[:, 1].clamp(min=-10.0, max=10.0)
    return mean, log_variance


def feature_mean_bits(per_cell_bits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute feature-level mean bits and the valid feature mask."""

    finite_mask = torch.isfinite(per_cell_bits)
    feature_counts = finite_mask.sum(dim=1)
    feature_sums = torch.where(finite_mask, per_cell_bits, torch.zeros_like(per_cell_bits)).sum(
        dim=1
    )
    current_feature_mean_bits = torch.full(
        (int(per_cell_bits.shape[0]), int(per_cell_bits.shape[2])),
        float("nan"),
        device=per_cell_bits.device,
        dtype=per_cell_bits.dtype,
    )
    valid_feature_mask = feature_counts > 0
    current_feature_mean_bits[valid_feature_mask] = feature_sums[valid_feature_mask] / feature_counts[
        valid_feature_mask
    ].to(dtype=per_cell_bits.dtype)
    return current_feature_mean_bits, valid_feature_mask


def forward_cell_likelihood(
    model: Any,
    feature_state: SandwichFeatureState,
) -> CellLikelihoodOutput:
    """Run the sandwich cell-likelihood path from prepared feature state."""

    raw_state = feature_state.raw_state
    decoder_hidden = cell_decoder_hidden(
        model,
        feature_cells=feature_state.feature_cells,
        y_train=raw_state.y_train,
    )
    batch_size = int(raw_state.x_all.shape[0])
    num_rows = int(raw_state.x_all.shape[1])
    num_features = int(raw_state.x_all.shape[2])
    targets = raw_state.x_all.to(torch.float32)
    per_cell_bits = torch.full(
        (batch_size, num_rows, num_features),
        float("nan"),
        device=raw_state.x_all.device,
        dtype=torch.float32,
    )
    floating_predictions: list[FloatingCellPrediction] = []
    categorical_predictions: list[CategoricalCellPrediction] = []
    integer_predictions: list[IntegerHybridCellPrediction] = []
    for task_index in range(batch_size):
        for feature_index in range(num_features):
            feature_type = FEATURE_TYPE_VOCAB[
                int(raw_state.feature_type_ids[task_index, feature_index].item())
            ]
            feature_hidden = decoder_hidden[task_index, :, feature_index, :]
            feature_targets = targets[task_index, :, feature_index]
            valid_target_mask = torch.isfinite(feature_targets)
            sanitized_targets = torch.where(
                valid_target_mask,
                feature_targets,
                torch.zeros_like(feature_targets),
            )
            excluded_bits = torch.full_like(feature_targets, float("nan"))
            train_values = targets[task_index, : raw_state.train_test_split_index, feature_index]
            current_support_values = support_values_for_feature(
                train_values,
                feature_type=feature_type,
            )
            current_support_embeddings = support_embeddings(
                model,
                current_support_values,
                device=feature_hidden.device,
                dtype=feature_hidden.dtype,
            )
            if feature_type == FEATURE_TYPE_FLOATING:
                mean, log_variance = gaussian_params(model, feature_hidden)
                per_cell_bits[task_index, :, feature_index] = torch.where(
                    valid_target_mask,
                    gaussian_nll_bits(
                        mean,
                        log_variance,
                        sanitized_targets,
                    ),
                    excluded_bits,
                )
                floating_predictions.append(
                    FloatingCellPrediction(
                        task_index=task_index,
                        feature_index=feature_index,
                        mean=mean,
                        log_variance=log_variance,
                    )
                )
                continue
            if feature_type == FEATURE_TYPE_INTEGER and model.integer_likelihood == "hybrid_mixture":
                gate_logit = integer_gate_logit(
                    model,
                    current_support_embeddings,
                    device=feature_hidden.device,
                    dtype=feature_hidden.dtype,
                )
                discrete_logits = dynamic_discrete_logits(
                    model,
                    feature_hidden,
                    support_embeddings=current_support_embeddings,
                )
                target_indices = discrete_target_indices(
                    feature_targets,
                    support_values=current_support_values,
                )
                discrete_bits = cross_entropy_bits(discrete_logits, target_indices)
                mean, log_variance = gaussian_params(model, feature_hidden)
                continuous_bits = gaussian_nll_bits(mean, log_variance, sanitized_targets)
                per_cell_bits[task_index, :, feature_index] = torch.where(
                    valid_target_mask,
                    mixture_bits(
                        gate_logit=gate_logit,
                        discrete_bits=discrete_bits,
                        continuous_bits=continuous_bits,
                    ),
                    excluded_bits,
                )
                integer_predictions.append(
                    IntegerHybridCellPrediction(
                        task_index=task_index,
                        feature_index=feature_index,
                        support_values=current_support_values,
                        gate_logit=gate_logit,
                        discrete_logits=discrete_logits,
                        mean=mean,
                        log_variance=log_variance,
                    )
                )
                continue
            discrete_logits = dynamic_discrete_logits(
                model,
                feature_hidden,
                support_embeddings=current_support_embeddings,
            )
            target_indices = discrete_target_indices(
                feature_targets,
                support_values=current_support_values,
            )
            per_cell_bits[task_index, :, feature_index] = torch.where(
                valid_target_mask,
                cross_entropy_bits(
                    discrete_logits,
                    target_indices,
                ),
                excluded_bits,
            )
            categorical_predictions.append(
                CategoricalCellPrediction(
                    task_index=task_index,
                    feature_index=feature_index,
                    feature_type=feature_type,
                    support_values=current_support_values,
                    logits=discrete_logits,
                )
            )

    finite_cell_mask = torch.isfinite(per_cell_bits)
    bpc_cell_count = int(finite_cell_mask.sum().item())
    if bpc_cell_count <= 0:
        raise RuntimeError(
            "tabfoundry_sandwich cell_bpc requires at least one finite target cell "
            "after excluding non-finite values"
        )
    current_feature_mean_bits, valid_feature_mask = feature_mean_bits(per_cell_bits)
    bpf_feature_count = int(valid_feature_mask.sum().item())
    if bpf_feature_count <= 0:
        raise RuntimeError(
            "tabfoundry_sandwich cell_bpc requires at least one feature with a finite "
            "target cell after excluding non-finite values"
        )
    bpc = torch.where(
        finite_cell_mask, per_cell_bits, torch.zeros_like(per_cell_bits)
    ).sum() / finite_cell_mask.sum().to(dtype=per_cell_bits.dtype)
    bpf = torch.where(
        valid_feature_mask, current_feature_mean_bits, torch.zeros_like(current_feature_mean_bits)
    ).sum() / valid_feature_mask.sum().to(dtype=current_feature_mean_bits.dtype)
    excluded_non_finite_cell_count = int((~torch.isfinite(targets)).sum().item())
    aux_metrics = {
        "bpc": float(bpc.detach().item()),
        "bpf": float(bpf.detach().item()),
        "bpc_cell_count": float(bpc_cell_count),
        "bpf_feature_count": float(bpf_feature_count),
        "excluded_non_finite_cell_count": float(excluded_non_finite_cell_count),
    }
    y_test = raw_state.y_test
    if y_test is not None and int(y_test.numel()) > 0:
        classification_state = build_classification_state(model, feature_state)
        logits = forward_logits(model, classification_state)
        num_classes = int(raw_state.num_classes)
        logits = logits[:, :, :num_classes]
        targets_for_accuracy = y_test.to(torch.int64)
        expected_shape = (int(logits.shape[0]), int(logits.shape[1]))
        if tuple(int(dim) for dim in targets_for_accuracy.shape) != expected_shape:
            raise RuntimeError(
                "tabfoundry_sandwich cell_bpc accuracy requires y_test shape "
                f"{expected_shape}, got {tuple(int(dim) for dim in targets_for_accuracy.shape)}"
            )
        aux_metrics["acc"] = float(
            (logits.argmax(dim=-1) == targets_for_accuracy).float().mean().item()
        )
    return CellLikelihoodOutput(
        per_cell_bits=per_cell_bits,
        bpc=bpc,
        bpf=bpf,
        floating_predictions=floating_predictions or None,
        categorical_predictions=categorical_predictions or None,
        integer_predictions=integer_predictions or None,
        aux_metrics=aux_metrics,
    )
