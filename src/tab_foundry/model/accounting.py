"""Inspected parameter and compute accounting for model surfaces."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import math
from typing import Any, Mapping, cast

import torch
from torch import nn

from tab_foundry.feature_types import DEFAULT_FEATURE_TYPE
from tab_foundry.model.components.rational import RationalActivation
from tab_foundry.task_batching import parse_task_batch_signature_text
from tab_foundry.types import TaskBatch


ACCOUNTING_ARTIFACT_SCHEMA = "tab-foundry-model-accounting-v1"
PARAMETER_ACCOUNTING_METHOD = "inspected_parameter_partition_v1"
COMPUTE_ACCOUNTING_METHOD = "inspected_analytic_v1"
TRAINING_FLOP_MULTIPLIER = 3.0
_DEFAULT_TRAINING_SHAPE_SIGNATURE = ("3x2x4x2", 1)
_LAYER_NORM_FLOPS_PER_ELEMENT = 7.0
_GELU_FLOPS_PER_ELEMENT = 8.0
_RATIONAL_5_4_FLOPS_PER_ELEMENT = 19.0
_SOFTMAX_FLOPS_PER_ELEMENT = 5.0
_EPSILON = 1.0e-12
_MHA_TENSOR_INPUT_COUNT = 3


@dataclass(frozen=True, slots=True)
class _ParameterEntry:
    name: str
    owner_module: str
    owner_type: str
    numel: int
    requires_grad: bool
    strict_partition: str
    strict_reason: str
    expanded_partition: str
    expanded_reason: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "owner_module": self.owner_module,
            "owner_type": self.owner_type,
            "numel": int(self.numel),
            "requires_grad": bool(self.requires_grad),
            "strict_partition": self.strict_partition,
            "strict_reason": self.strict_reason,
            "expanded_partition": self.expanded_partition,
            "expanded_reason": self.expanded_reason,
        }


def _parameter_owner(
    modules_by_name: Mapping[str, nn.Module],
    parameter_name: str,
) -> tuple[str, nn.Module, str]:
    owner_module_name, separator, local_name = parameter_name.rpartition(".")
    owner_name = owner_module_name if separator else ""
    try:
        owner_module = modules_by_name[owner_name]
    except KeyError as exc:  # pragma: no cover - defensive consistency guard
        raise RuntimeError(f"unable to resolve owner module for parameter {parameter_name!r}") from exc
    return owner_name, owner_module, local_name if separator else parameter_name


def _expanded_embedding_like_reason(
    *,
    parameter_name: str,
    local_name: str,
) -> str | None:
    if local_name == "test_token":
        return "learned_test_token"
    if local_name in {
        "row_summary_query",
        "column_summary_query",
        "test_row_pool_query",
        "latent_seed",
        "cell_bos",
    }:
        return "learned_query_or_seed"
    if local_name in {"inducing_seed", "inducing"}:
        return "learned_inducing_seed"
    if local_name == "cls":
        return "learned_cls_token"
    if parameter_name.endswith(".cls"):
        return "learned_cls_token"
    return None


def parameter_accounting_from_model(model: nn.Module) -> dict[str, Any]:
    """Partition parameters using explicit inspected module ownership."""

    modules_by_name = dict(model.named_modules())
    entries: list[_ParameterEntry] = []
    module_totals: dict[str, dict[str, Any]] = {}
    total_params = 0
    trainable_params = 0
    strict_embedding_params = 0
    expanded_embedding_like_params = 0

    for parameter_name, parameter in model.named_parameters():
        owner_name, owner_module, local_name = _parameter_owner(modules_by_name, parameter_name)
        owner_type = type(owner_module).__name__
        numel = int(parameter.numel())
        strict_partition: str
        strict_reason: str
        expanded_partition: str
        expanded_reason: str | None
        total_params += numel
        if parameter.requires_grad:
            trainable_params += numel
        if isinstance(owner_module, nn.Embedding):
            strict_partition = "embedding"
            strict_reason = "explicit_nn_embedding"
            expanded_partition = "embedding_like"
            expanded_reason = "explicit_nn_embedding"
            strict_embedding_params += numel
            expanded_embedding_like_params += numel
        else:
            strict_partition = "non_embedding"
            strict_reason = "non_embedding_default"
            expanded_reason = _expanded_embedding_like_reason(
                parameter_name=parameter_name,
                local_name=local_name,
            )
            if expanded_reason is None:
                expanded_partition = "non_embedding"
            else:
                expanded_partition = "embedding_like"
                expanded_embedding_like_params += numel
        entry = _ParameterEntry(
            name=str(parameter_name),
            owner_module=str(owner_name),
            owner_type=owner_type,
            numel=numel,
            requires_grad=bool(parameter.requires_grad),
            strict_partition=strict_partition,
            strict_reason=strict_reason,
            expanded_partition=expanded_partition,
            expanded_reason=expanded_reason or "non_embedding_default",
        )
        entries.append(entry)
        module_key = str(owner_name)
        module_summary = module_totals.get(module_key)
        if module_summary is None:
            module_summary = {
                "owner_module": module_key,
                "owner_type": owner_type,
                "total_params": 0,
                "trainable_params": 0,
                "strict_embedding_params": 0,
                "expanded_embedding_like_params": 0,
                "parameter_names": [],
            }
            module_totals[module_key] = module_summary
        module_summary["total_params"] = int(module_summary["total_params"]) + numel
        if parameter.requires_grad:
            module_summary["trainable_params"] = int(module_summary["trainable_params"]) + numel
        if strict_partition == "embedding":
            module_summary["strict_embedding_params"] = (
                int(module_summary["strict_embedding_params"]) + numel
            )
        if expanded_partition == "embedding_like":
            module_summary["expanded_embedding_like_params"] = (
                int(module_summary["expanded_embedding_like_params"]) + numel
            )
        module_parameter_names = module_summary["parameter_names"]
        assert isinstance(module_parameter_names, list)
        module_parameter_names.append(str(parameter_name))

    strict_non_embedding_params = total_params - strict_embedding_params
    expanded_non_embedding_params = total_params - expanded_embedding_like_params
    return {
        "schema": ACCOUNTING_ARTIFACT_SCHEMA,
        "method": PARAMETER_ACCOUNTING_METHOD,
        "total_params": int(total_params),
        "trainable_params": int(trainable_params),
        "strict": {
            "embedding_params": int(strict_embedding_params),
            "non_embedding_params": int(strict_non_embedding_params),
        },
        "expanded": {
            "embedding_like_params": int(expanded_embedding_like_params),
            "non_embedding_params": int(expanded_non_embedding_params),
        },
        "canonical_non_embedding_params": int(strict_non_embedding_params),
        "module_breakdown": [
            {
                **summary,
                "parameter_names": sorted(cast(list[str], summary["parameter_names"])),
            }
            for _, summary in sorted(module_totals.items(), key=lambda item: item[0])
        ],
        "parameter_breakdown": [entry.as_dict() for entry in sorted(entries, key=lambda entry: entry.name)],
    }


def _reference_batch_from_signature(
    *,
    n_train: int,
    n_test: int,
    n_features: int,
    num_classes: int | None,
) -> TaskBatch:
    total_rows = int(n_train) + int(n_test)
    x_all = torch.arange(total_rows * int(n_features), dtype=torch.float32).reshape(total_rows, int(n_features))
    x_all = (x_all / float(max(1, int(n_features)))) - 0.5
    resolved_num_classes = 2 if num_classes is None else max(2, int(num_classes))
    y_all = torch.arange(total_rows, dtype=torch.int64).remainder(resolved_num_classes)
    return TaskBatch(
        x_train=x_all[: int(n_train)].clone(),
        y_train=y_all[: int(n_train)].clone(),
        x_test=x_all[int(n_train) :].clone(),
        y_test=y_all[int(n_train) :].clone(),
        metadata={
            "feature_types": [DEFAULT_FEATURE_TYPE] * int(n_features),
        },
        num_classes=resolved_num_classes,
    )


def _module_forward_flops(
    module: nn.Module,
    *,
    inputs: tuple[Any, ...],
    output: Any,
    skip_linear_ids: set[int],
) -> tuple[float, str | None]:
    if isinstance(module, nn.Linear):
        if id(module) in skip_linear_ids:
            return 0.0, None
        tensor = inputs[0]
        if not isinstance(tensor, torch.Tensor) or tensor.ndim <= 0:
            return 0.0, None
        batch_elements = int(tensor.numel() // max(1, module.in_features))
        flops = 2.0 * float(batch_elements) * float(module.in_features) * float(module.out_features)
        if module.bias is not None:
            flops += float(batch_elements) * float(module.out_features)
        return flops, "linear"
    if isinstance(module, nn.MultiheadAttention):
        if len(inputs) < _MHA_TENSOR_INPUT_COUNT:
            return 0.0, None
        query, key, value = inputs[:3]
        if not all(isinstance(item, torch.Tensor) for item in (query, key, value)):
            return 0.0, None
        query_t = query
        key_t = key
        batch_size = int(query_t.shape[0])
        query_len = int(query_t.shape[1])
        key_len = int(key_t.shape[1])
        embed_dim = int(module.embed_dim)
        head_dim = int(embed_dim // max(1, int(module.num_heads)))
        projection_flops = 2.0 * float(batch_size) * float(embed_dim) * (
            float(query_len) * float(embed_dim)
            + float(key_len) * float(module.kdim or embed_dim)
            + float(key_len) * float(module.vdim or embed_dim)
            + float(query_len) * float(embed_dim)
        )
        attention_scores = (
            2.0
            * float(batch_size)
            * float(module.num_heads)
            * float(query_len)
            * float(key_len)
            * float(head_dim)
        )
        softmax_flops = (
            _SOFTMAX_FLOPS_PER_ELEMENT
            * float(batch_size)
            * float(module.num_heads)
            * float(query_len)
            * float(key_len)
        )
        attention_values = (
            2.0
            * float(batch_size)
            * float(module.num_heads)
            * float(query_len)
            * float(key_len)
            * float(head_dim)
        )
        return projection_flops + attention_scores + softmax_flops + attention_values, "multihead_attention"
    if isinstance(module, nn.LayerNorm):
        normalized = inputs[0]
        if not isinstance(normalized, torch.Tensor):
            return 0.0, None
        flops = _LAYER_NORM_FLOPS_PER_ELEMENT * float(normalized.numel())
        if module.elementwise_affine:
            flops += 2.0 * float(normalized.numel())
        return flops, "layer_norm"
    if isinstance(module, nn.GELU):
        activated = output if isinstance(output, torch.Tensor) else None
        if activated is None:
            return 0.0, None
        return _GELU_FLOPS_PER_ELEMENT * float(activated.numel()), "gelu"
    if isinstance(module, RationalActivation):
        activated = output if isinstance(output, torch.Tensor) else None
        if activated is None:
            return 0.0, None
        return _RATIONAL_5_4_FLOPS_PER_ELEMENT * float(activated.numel()), "rational_5_4"
    return 0.0, None


def _forward_compute_contributions(
    model: nn.Module,
    *,
    batch: TaskBatch,
) -> list[dict[str, Any]]:
    modules_by_name = dict(model.named_modules())
    module_names_by_id = {id(module): name for name, module in modules_by_name.items()}
    skip_linear_ids = {
        id(module.out_proj)
        for module in modules_by_name.values()
        if isinstance(module, nn.MultiheadAttention)
    }
    contributions: list[dict[str, Any]] = []
    handles: list[Any] = []

    def _hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        flops, op_kind = _module_forward_flops(
            module,
            inputs=inputs,
            output=output,
            skip_linear_ids=skip_linear_ids,
        )
        if flops <= 0.0 or op_kind is None:
            return
        module_name = module_names_by_id.get(id(module), "")
        contributions.append(
            {
                "module": str(module_name),
                "module_type": type(module).__name__,
                "op_kind": op_kind,
                "forward_flops": float(flops),
            }
        )

    tracked_types = (nn.Linear, nn.MultiheadAttention, nn.LayerNorm, nn.GELU, RationalActivation)
    for module in modules_by_name.values():
        if not isinstance(module, tracked_types):
            continue
        handles.append(module.register_forward_hook(_hook))

    was_training = model.training
    try:
        model.eval()
        with torch.no_grad():
            _ = model(batch)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)
    return contributions


def _training_shape_signatures(training_shape_summary: Mapping[str, Any] | None) -> list[tuple[str, int]]:
    # This fallback keeps generic accounting artifacts available when legacy
    # telemetry lacks shape data. Research scaling C-axis fits reject fallback
    # rows by requiring an observed training_shape_summary in the registry.
    if not isinstance(training_shape_summary, Mapping):
        return [_DEFAULT_TRAINING_SHAPE_SIGNATURE]
    raw_signature_task_counts = training_shape_summary.get("signature_task_counts")
    if not isinstance(raw_signature_task_counts, Mapping) or not raw_signature_task_counts:
        return [_DEFAULT_TRAINING_SHAPE_SIGNATURE]
    resolved: list[tuple[str, int]] = []
    for signature_text, task_count in raw_signature_task_counts.items():
        if not isinstance(signature_text, str):
            continue
        resolved_task_count = int(task_count)
        if resolved_task_count <= 0:
            continue
        resolved.append((signature_text, resolved_task_count))
    return sorted(resolved, key=lambda item: item[0]) or [_DEFAULT_TRAINING_SHAPE_SIGNATURE]


def compute_accounting_from_model(
    model: nn.Module,
    *,
    training_shape_summary: Mapping[str, Any] | None,
    tokens_seen: int | None,
    tokens_per_step: float | None,
) -> dict[str, Any]:
    """Estimate training FLOPs analytically from inspected modules and observed shapes."""

    weighted_forward_flops_per_token_sum = 0.0
    total_task_weight = 0
    signature_breakdown: list[dict[str, Any]] = []
    module_totals: dict[str, float] = defaultdict(float)

    for signature_text, task_count in _training_shape_signatures(training_shape_summary):
        n_train, n_test, n_features, num_classes = parse_task_batch_signature_text(signature_text)
        reference_batch = _reference_batch_from_signature(
            n_train=n_train,
            n_test=n_test,
            n_features=n_features,
            num_classes=num_classes,
        )
        contributions = _forward_compute_contributions(model, batch=reference_batch)
        forward_flops = float(sum(float(item["forward_flops"]) for item in contributions))
        tokens_per_task = max(1, (int(n_train) + int(n_test)) * int(n_features))
        forward_flops_per_token = float(forward_flops) / float(tokens_per_task)
        for item in contributions:
            module_key = f"{item['module']}::{item['op_kind']}"
            module_totals[module_key] += float(item["forward_flops"])
        signature_breakdown.append(
            {
                "signature": signature_text,
                "task_count": int(task_count),
                "n_train": int(n_train),
                "n_test": int(n_test),
                "n_features": int(n_features),
                "num_classes": None if num_classes is None else int(num_classes),
                "tokens_per_task": int(tokens_per_task),
                "forward_flops_per_task": float(forward_flops),
                "forward_flops_per_token": float(forward_flops_per_token),
                "contributions": contributions,
            }
        )
        weighted_forward_flops_per_token_sum += float(task_count) * float(forward_flops_per_token)
        total_task_weight += int(task_count)

    mean_forward_flops_per_token = (
        weighted_forward_flops_per_token_sum / float(total_task_weight)
        if total_task_weight > 0
        else 0.0
    )
    train_flops_per_token = float(mean_forward_flops_per_token * TRAINING_FLOP_MULTIPLIER)
    resolved_tokens_per_step = None
    if tokens_per_step is not None and math.isfinite(float(tokens_per_step)) and float(tokens_per_step) > 0.0:
        resolved_tokens_per_step = float(tokens_per_step)
    resolved_tokens_seen = None
    if tokens_seen is not None and int(tokens_seen) >= 0:
        resolved_tokens_seen = int(tokens_seen)
    return {
        "schema": ACCOUNTING_ARTIFACT_SCHEMA,
        "method": COMPUTE_ACCOUNTING_METHOD,
        "training_multiplier": float(TRAINING_FLOP_MULTIPLIER),
        "forward_flops_per_token": float(mean_forward_flops_per_token),
        "train_flops_per_token": float(train_flops_per_token),
        "train_flops_per_step": (
            None
            if resolved_tokens_per_step is None
            else float(train_flops_per_token * resolved_tokens_per_step)
        ),
        "total_train_flops": (
            None
            if resolved_tokens_seen is None
            else float(train_flops_per_token * float(resolved_tokens_seen))
        ),
        "tokens_seen": resolved_tokens_seen,
        "tokens_per_step": resolved_tokens_per_step,
        "training_shape_summary": None if training_shape_summary is None else dict(training_shape_summary),
        "signature_breakdown": signature_breakdown,
        "module_forward_flop_totals": [
            {"module_op": key, "forward_flops": float(value)}
            for key, value in sorted(module_totals.items(), key=lambda item: item[0])
        ],
    }


def write_accounting_artifact_payload(
    *,
    parameter_accounting: Mapping[str, Any],
    compute_accounting: Mapping[str, Any] | None,
    training_shape_summary: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Build the canonical per-run accounting artifact payload."""

    return {
        "schema": ACCOUNTING_ARTIFACT_SCHEMA,
        "parameter_accounting": dict(parameter_accounting),
        "compute_accounting": None if compute_accounting is None else dict(compute_accounting),
        "training_shape_summary": None if training_shape_summary is None else dict(training_shape_summary),
    }
