"""Many-class helpers: mixed-radix and hierarchical class trees."""

from __future__ import annotations

from dataclasses import dataclass, field
import functools
import math

import torch


@dataclass(slots=True, frozen=True)
class HierNode:
    """One node in a hierarchical class tree."""

    classes: list[int]
    children: list["HierNode"] = field(default_factory=list)
    _node_classes_cache: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)
    _group_classes_cache: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)
    _group_indices_cache: dict[str, torch.Tensor] = field(default_factory=dict, repr=False)

    @property
    def is_leaf(self) -> bool:
        return len(self.children) == 0

    @staticmethod
    def _cache_key(device: torch.device) -> str:
        index = device.index if device.index is not None else -1
        return f"{device.type}:{index}"

    def node_classes_tensor(self, device: torch.device) -> torch.Tensor:
        key = self._cache_key(device)
        cached = self._node_classes_cache.get(key)
        if cached is None:
            cached = torch.tensor(self.classes, device=device, dtype=torch.int64)
            self._node_classes_cache[key] = cached
        return cached

    def group_mapping_tensors(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        key = self._cache_key(device)
        classes = self._group_classes_cache.get(key)
        groups = self._group_indices_cache.get(key)
        if classes is not None and groups is not None:
            return classes, groups

        if self.is_leaf:
            classes = self.node_classes_tensor(device)
            groups = torch.arange(classes.shape[0], device=device, dtype=torch.int64)
        else:
            class_list: list[int] = []
            group_list: list[int] = []
            for child_idx, child in enumerate(self.children):
                class_list.extend(child.classes)
                group_list.extend([child_idx] * len(child.classes))
            classes = torch.tensor(class_list, device=device, dtype=torch.int64)
            groups = torch.tensor(group_list, device=device, dtype=torch.int64)
            order = torch.argsort(classes)
            classes = classes[order]
            groups = groups[order]

        self._group_classes_cache[key] = classes
        self._group_indices_cache[key] = groups
        return classes, groups


def balanced_bases(num_classes: int, max_base: int = 10) -> list[int]:
    """Return balanced mixed-radix bases with each base <= max_base."""

    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    if num_classes <= max_base:
        return [num_classes]

    dims = int(math.ceil(math.log(num_classes, max_base)))
    bases = [max_base] * dims
    product = max_base**dims
    if product == num_classes:
        return bases

    # Relax first bases to improve balance while keeping coverage.
    for i in range(dims):
        tail_product = 1
        for j in range(i + 1, dims):
            tail_product *= bases[j]
        best = bases[i]
        for candidate in range(2, max_base + 1):
            if candidate * tail_product >= math.ceil(num_classes / (max_base ** max(0, dims - i - 1))):
                best = candidate
                break
        bases[i] = best

    # Guarantee coverage.
    product = 1
    for base in bases:
        product *= base
    idx = len(bases) - 1
    while product < num_classes and idx >= 0:
        if bases[idx] < max_base:
            bases[idx] += 1
            product = 1
            for base in bases:
                product *= base
        else:
            idx -= 1

    if product < num_classes:
        bases.append(max_base)
    return bases


def encode_mixed_radix(labels: torch.Tensor, bases: list[int]) -> torch.Tensor:
    """Encode labels to mixed-radix digits.

    Returns tensor with shape [D, N].
    """

    if labels.ndim != 1:
        raise ValueError("labels must be 1D")
    if not bases:
        raise ValueError("bases must be non-empty")

    y = labels.to(torch.int64)
    digits: list[torch.Tensor] = []
    for i, base in enumerate(bases):
        divisor = 1
        for j in range(i + 1, len(bases)):
            divisor *= bases[j]
        digits.append(torch.div(y, divisor, rounding_mode="floor") % base)
    return torch.stack(digits, dim=0)


def decode_mixed_radix(digits: torch.Tensor, bases: list[int]) -> torch.Tensor:
    """Decode mixed-radix digits back to class IDs."""

    if digits.ndim != 2:
        raise ValueError("digits must have shape [D, N]")
    if digits.shape[0] != len(bases):
        raise ValueError("digit depth mismatch")

    out = torch.zeros(digits.shape[1], dtype=torch.int64, device=digits.device)
    for i in range(len(bases)):
        factor = 1
        for j in range(i + 1, len(bases)):
            factor *= bases[j]
        out = out + digits[i].to(torch.int64) * factor
    return out


def build_balanced_class_tree(classes: list[int], max_branch: int = 10) -> HierNode:
    """Recursively partition classes into near-equal groups."""

    classes = sorted(classes)
    if len(classes) <= max_branch:
        return HierNode(classes=classes)

    n_groups = int(math.ceil(len(classes) / max_branch))
    chunk_size = int(math.ceil(len(classes) / n_groups))
    children: list[HierNode] = []
    for start in range(0, len(classes), chunk_size):
        chunk = classes[start : start + chunk_size]
        children.append(build_balanced_class_tree(chunk, max_branch=max_branch))
    return HierNode(classes=classes, children=children)


@functools.lru_cache(maxsize=32)
def cached_build_balanced_class_tree(num_classes: int, max_branch: int = 10) -> HierNode:
    return build_balanced_class_tree(list(range(num_classes)), max_branch=max_branch)


def map_labels_to_child_groups(labels: torch.Tensor, node: HierNode) -> torch.Tensor:
    """Map absolute labels to child group indices for one node."""
    if labels.ndim != 1:
        raise ValueError("labels must be 1D")
    labels_i64 = labels.to(torch.int64)
    if labels_i64.numel() == 0:
        return labels_i64.clone()

    classes, groups = node.group_mapping_tensors(labels.device)

    if classes.numel() == 0:
        raise ValueError("node has no classes")

    positions = torch.searchsorted(classes, labels_i64)
    clamped = positions.clamp(max=int(classes.shape[0]) - 1)
    in_bounds = positions < int(classes.shape[0])
    matched = in_bounds & (classes[clamped] == labels_i64)
    if not bool(torch.all(matched)):
        unknown = torch.unique(labels_i64[~matched]).cpu().tolist()
        unseen = ", ".join(str(int(v)) for v in unknown)
        raise KeyError(f"labels not present in node classes: {unseen}")

    return groups[clamped]


def class_paths(node: HierNode) -> dict[int, list[int]]:
    """Return class -> child-index path mapping."""

    out: dict[int, list[int]] = {}

    def _walk(cur: HierNode, path: list[int]) -> None:
        if cur.is_leaf:
            for idx, cls in enumerate(cur.classes):
                out[cls] = path + [idx]
            return
        for child_idx, child in enumerate(cur.children):
            _walk(child, path + [child_idx])

    _walk(node, [])
    return out
