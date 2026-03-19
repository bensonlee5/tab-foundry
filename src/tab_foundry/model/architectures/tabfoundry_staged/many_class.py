"""Many-class staged forward helpers."""

from __future__ import annotations

from typing import Any

import torch

from tab_foundry.model.components.many_class import (
    HierNode,
    balanced_bases,
    cached_build_balanced_class_tree,
    encode_mixed_radix,
    map_labels_to_child_groups,
)
from tab_foundry.model.outputs import ClassificationOutput

from .forward_common import context_train_embeddings, encode_to_cell_state, pool_rows
from .states import RawInputState, RowState


def node_train_indices(model: Any, *, node: HierNode, y_train: torch.Tensor) -> torch.Tensor:
    del model
    node_classes = node.node_classes_tensor(y_train.device)
    train_mask = torch.isin(y_train.to(torch.int64), node_classes)
    return torch.nonzero(train_mask, as_tuple=False).squeeze(-1)


def encode_rows_with_targets(
    model: Any,
    rows: torch.Tensor,
    *,
    train_targets: torch.Tensor,
    train_test_split_index: int,
) -> torch.Tensor:
    if model.context_encoder is None:
        raise RuntimeError("context_encoder must be initialized for many-class conditioning")
    conditioned = model.context_encoder(
        rows.unsqueeze(0),
        train_target_embeddings=train_targets.unsqueeze(0),
        train_test_split_index=train_test_split_index,
    )
    model.trace_activation("post_context_encoder", conditioned)
    return conditioned[0]


def digit_conditioned_rows(model: Any, row_state: RowState, y_train: torch.Tensor) -> torch.Tensor:
    digits = encode_mixed_radix(
        y_train,
        bases=balanced_bases(
            num_classes=row_state.num_classes,
            max_base=model.many_class_base,
        ),
    )
    if int(digits.shape[0]) > model.max_mixed_radix_digits:
        raise RuntimeError(
            "mixed-radix depth exceeds model.max_mixed_radix_digits; "
            f"got {int(digits.shape[0])} > {model.max_mixed_radix_digits}"
        )
    accum = torch.zeros_like(row_state.rows[0])
    for view in range(int(digits.shape[0])):
        train_targets = context_train_embeddings(model, digits[view].unsqueeze(0))[0]
        if model.digit_position_embed is not None:
            pos = model.digit_position_embed(
                torch.tensor([view], device=row_state.rows.device, dtype=torch.int64)
            )[0]
            train_targets = train_targets + pos[None, :]
        conditioned = encode_rows_with_targets(
            model,
            row_state.rows[0],
            train_targets=train_targets,
            train_test_split_index=row_state.train_test_split_index,
        )
        accum = accum + conditioned
    return accum / float(digits.shape[0])


def hierarchical_probs(
    model: Any,
    row_embeddings: torch.Tensor,
    y_train: torch.Tensor,
    tree: HierNode,
    *,
    n_train: int,
    num_classes: int,
) -> tuple[torch.Tensor, int, int]:
    n_test = row_embeddings.shape[0] - n_train
    test_embeddings = row_embeddings[n_train:]
    class_probs = torch.zeros((n_test, num_classes), device=row_embeddings.device)
    nodes_visited = 0
    empty_nodes = 0

    def _recurse(node: HierNode, parent_prob: torch.Tensor) -> None:
        nonlocal nodes_visited, empty_nodes
        nodes_visited += 1
        idx = node_train_indices(model, node=node, y_train=y_train)
        if idx.numel() == 0:
            empty_nodes += 1
            n_choices = len(node.classes) if node.is_leaf else len(node.children)
            uniform_prob = parent_prob / float(n_choices)
            if node.is_leaf:
                for cls in node.classes:
                    class_probs[:, cls] = class_probs[:, cls] + uniform_prob
                return
            for child in node.children:
                _recurse(child, uniform_prob)
            return

        node_train_embed = row_embeddings[idx]
        mapped = (
            map_labels_to_child_groups(y_train[idx], node)
            .clamp(max=model.many_class_base - 1)
            .to(torch.int64)
        )
        seq = torch.cat([node_train_embed, test_embeddings], dim=0)
        conditioned = encode_rows_with_targets(
            model,
            seq,
            train_targets=context_train_embeddings(model, mapped.unsqueeze(0))[0],
            train_test_split_index=int(node_train_embed.shape[0]),
        )
        test_out = conditioned[int(node_train_embed.shape[0]) :]
        if node.is_leaf:
            logits = model.direct_head(test_out)[:, : len(node.classes)]
            probs = torch.softmax(logits, dim=-1)
            for local_idx, cls in enumerate(node.classes):
                class_probs[:, cls] = class_probs[:, cls] + parent_prob * probs[:, local_idx]
            return

        logits = model.direct_head(test_out)[:, : len(node.children)]
        probs = torch.softmax(logits, dim=-1)
        for child_idx, child in enumerate(node.children):
            _recurse(child, parent_prob * probs[:, child_idx])

    _recurse(tree, torch.ones((n_test,), device=row_embeddings.device))
    denom = class_probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
    return class_probs / denom, nodes_visited, empty_nodes


def hierarchical_path_terms(
    model: Any,
    row_embeddings: torch.Tensor,
    y_train: torch.Tensor,
    y_test: torch.Tensor,
    tree: HierNode,
    *,
    n_train: int,
) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int], dict[str, float]]:
    n_test = row_embeddings.shape[0] - n_train
    test_embeddings = row_embeddings[n_train:]

    logits_terms: list[torch.Tensor] = []
    target_terms: list[torch.Tensor] = []
    sample_counts: list[int] = []
    nodes_visited = 0
    total_path_steps = 0
    empty_nodes = 0

    def _recurse(node: HierNode, sample_idx: torch.Tensor) -> None:
        nonlocal nodes_visited, total_path_steps, empty_nodes
        if sample_idx.numel() == 0:
            return
        nodes_visited += 1
        total_path_steps += int(sample_idx.numel())

        mapped_test = (
            map_labels_to_child_groups(y_test[sample_idx].to(torch.int64), node)
            .clamp(max=model.many_class_base - 1)
            .to(torch.int64)
        )
        n_choices = len(node.classes) if node.is_leaf else len(node.children)
        idx = node_train_indices(model, node=node, y_train=y_train)
        if idx.numel() == 0:
            empty_nodes += 1
            logits = torch.zeros(
                (int(sample_idx.numel()), n_choices),
                device=row_embeddings.device,
                dtype=row_embeddings.dtype,
            )
        else:
            node_train_embed = row_embeddings[idx]
            mapped_train = (
                map_labels_to_child_groups(y_train[idx], node)
                .clamp(max=model.many_class_base - 1)
                .to(torch.int64)
            )
            seq = torch.cat([node_train_embed, test_embeddings[sample_idx]], dim=0)
            conditioned = encode_rows_with_targets(
                model,
                seq,
                train_targets=context_train_embeddings(model, mapped_train.unsqueeze(0))[0],
                train_test_split_index=int(node_train_embed.shape[0]),
            )
            logits = model.direct_head(conditioned[int(node_train_embed.shape[0]) :])[:, :n_choices]
        logits_terms.append(logits)
        target_terms.append(mapped_test)
        sample_counts.append(int(sample_idx.numel()))

        if node.is_leaf:
            return
        for child_idx, child in enumerate(node.children):
            child_samples = sample_idx[mapped_test == int(child_idx)]
            _recurse(child, child_samples)

    _recurse(tree, torch.arange(n_test, device=row_embeddings.device))
    avg_path_depth = float(total_path_steps) / float(max(1, n_test))
    return (
        logits_terms,
        target_terms,
        sample_counts,
        {
            "many_class_nodes_visited": float(nodes_visited),
            "many_class_avg_path_depth": float(avg_path_depth),
            "many_class_empty_nodes": float(empty_nodes),
        },
    )


def forward_many_class(model: Any, raw_state: RawInputState) -> ClassificationOutput:
    if raw_state.y_test is None:
        raise RuntimeError("forward_many_class requires y_test in raw input state")
    if int(raw_state.x_all.shape[0]) != 1:
        raise RuntimeError(
            "staged many_class currently requires a single task (batch dimension 1); "
            "task-level training already uses batch_size=1 and tensor-batched many_class "
            "execution is not implemented"
        )
    cell_state = encode_to_cell_state(model, raw_state)
    row_state = pool_rows(model, cell_state)
    conditioned_rows = digit_conditioned_rows(model, row_state, raw_state.y_train[0])
    tree = cached_build_balanced_class_tree(
        raw_state.num_classes,
        max_branch=model.many_class_base,
    )
    if model.training and model.many_class_train_mode == "path_nll":
        path_logits, path_targets, path_sample_counts, path_metrics = hierarchical_path_terms(
            model,
            conditioned_rows,
            raw_state.y_train[0],
            raw_state.y_test[0],
            tree,
            n_train=row_state.train_test_split_index,
        )
        return ClassificationOutput(
            logits=None,
            num_classes=raw_state.num_classes,
            class_probs=None,
            path_logits=path_logits,
            path_targets=path_targets,
            path_sample_counts=path_sample_counts,
            aux_metrics=path_metrics,
        )

    class_probs, nodes_visited, empty_nodes = hierarchical_probs(
        model,
        conditioned_rows,
        raw_state.y_train[0],
        tree,
        n_train=row_state.train_test_split_index,
        num_classes=raw_state.num_classes,
    )
    return ClassificationOutput(
        logits=None,
        num_classes=raw_state.num_classes,
        class_probs=class_probs,
        aux_metrics={
            "many_class_nodes_visited": float(nodes_visited),
            "many_class_empty_nodes": float(empty_nodes),
        },
    )


def validate_num_classes(model: Any, num_classes: int) -> None:
    if num_classes < 2:
        raise RuntimeError("tabfoundry_staged requires at least 2 classes")
    if not model.surface.task_contract.supports(num_classes=num_classes):
        if model.surface.task_contract.max_classes == 2:
            raise RuntimeError(f"stage={model.stage!r} is binary-only and requires num_classes=2")
        if (
            model.surface.task_contract.max_classes is not None
            and num_classes > model.surface.task_contract.max_classes
        ):
            limit = model.surface.task_contract.max_classes
            if limit == model.many_class_base:
                raise RuntimeError(
                    f"stage={model.stage!r} only supports num_classes <= many_class_base={model.many_class_base}, "
                    f"got {num_classes}"
                )
            raise RuntimeError(
                f"stage={model.stage!r} only supports num_classes <= {limit}, got {num_classes}"
            )
        raise RuntimeError(f"stage={model.stage!r} does not support num_classes={num_classes}")
    if model.surface.head != "many_class" and num_classes > model.many_class_base:
        raise RuntimeError(
            f"stage={model.stage!r} only supports num_classes <= many_class_base={model.many_class_base}, "
            f"got {num_classes}"
        )
