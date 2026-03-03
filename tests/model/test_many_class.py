from __future__ import annotations

import pytest
import torch

from tab_foundry.model.many_class import (
    balanced_bases,
    build_balanced_class_tree,
    decode_mixed_radix,
    encode_mixed_radix,
    map_labels_to_child_groups,
)


def test_mixed_radix_roundtrip() -> None:
    labels = torch.arange(0, 32, dtype=torch.int64)
    bases = balanced_bases(32, max_base=10)
    digits = encode_mixed_radix(labels, bases)
    decoded = decode_mixed_radix(digits, bases)
    assert torch.equal(labels, decoded)


def test_tree_branching_factor() -> None:
    tree = build_balanced_class_tree(list(range(32)), max_branch=10)
    stack = [tree]
    while stack:
        node = stack.pop()
        assert len(node.children) <= 10
        stack.extend(node.children)


def test_map_labels_to_child_groups_for_internal_and_leaf_nodes() -> None:
    tree = build_balanced_class_tree(list(range(12)), max_branch=10)
    root_groups = map_labels_to_child_groups(torch.tensor([0, 5, 6, 11]), tree)
    assert torch.equal(root_groups, torch.tensor([0, 0, 1, 1]))

    right_leaf = tree.children[1]
    leaf_groups = map_labels_to_child_groups(torch.tensor([6, 8, 11]), right_leaf)
    assert torch.equal(leaf_groups, torch.tensor([0, 2, 5]))


def test_map_labels_to_child_groups_rejects_unknown_labels() -> None:
    tree = build_balanced_class_tree(list(range(12)), max_branch=10)
    with pytest.raises(KeyError, match="labels not present in node classes"):
        _ = map_labels_to_child_groups(torch.tensor([12]), tree)


def test_map_labels_to_child_groups_populates_node_cache() -> None:
    tree = build_balanced_class_tree(list(range(12)), max_branch=10)
    _ = map_labels_to_child_groups(torch.tensor([0, 5, 6, 11]), tree)
    _ = map_labels_to_child_groups(torch.tensor([1, 2, 9]), tree)
    assert len(tree._group_classes_cache) == 1
    assert len(tree._group_indices_cache) == 1
