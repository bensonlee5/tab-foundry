"""Tabfoundry classifier and regressor."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import cast

import torch
import torch.nn.functional as F
from torch import nn

from tab_foundry.input_normalization import (
    InputNormalizationMode,
    SUPPORTED_INPUT_NORMALIZATION_MODES,
    normalize_train_test_tensors,
    prepare_train_test_tensors_with_missing,
)
from tab_foundry.model.missingness import normalize_missingness_mode
from tab_foundry.types import TaskBatch

from ..components.blocks import TFColEncoder, TFRowEncoder
from ..components.normalization import SUPPORTED_NORM_TYPES
from ..components.many_class import (
    HierNode,
    balanced_bases,
    cached_build_balanced_class_tree,
    encode_mixed_radix,
    map_labels_to_child_groups,
)
from ..components.qass import QASSTransformerEncoder


DEFAULT_MANY_CLASS_BASE = 10
DEFAULT_HEAD_HIDDEN_DIM = 1024
DEFAULT_REGRESSION_QUANTILES = 999


@dataclass(slots=True)
class ClassificationOutput:
    """Classifier forward output."""

    logits: torch.Tensor | None
    num_classes: int
    class_probs: torch.Tensor | None = None
    path_logits: list[torch.Tensor] | None = None
    path_targets: list[torch.Tensor] | None = None
    path_sample_counts: list[int] | None = None
    aux_metrics: dict[str, float] | None = None


@dataclass(slots=True)
class RegressionOutput:
    """Regressor forward output."""

    quantiles: torch.Tensor
    quantile_levels: torch.Tensor | None = None


class _TabFoundryBackbone(nn.Module):
    """Shared Tabfoundry backbone."""

    def __init__(
        self,
        *,
        d_col: int = 128,
        d_icl: int = 512,
        input_normalization: str = "none",
        missingness_mode: str = "none",
        feature_group_size: int = 1,
        norm_type: str = "layernorm",
        tfcol_n_heads: int = 8,
        tfcol_n_layers: int = 3,
        tfcol_n_inducing: int = 128,
        tfrow_n_heads: int = 8,
        tfrow_n_layers: int = 3,
        tfrow_cls_tokens: int = 4,
        tfrow_norm: str = "layernorm",
        tficl_n_heads: int = 8,
        tficl_n_layers: int = 12,
        tficl_ff_expansion: int = 2,
    ) -> None:
        super().__init__()
        self.d_col = d_col
        self.d_icl = d_icl
        self.input_normalization = str(input_normalization).strip().lower()
        if self.input_normalization not in SUPPORTED_INPUT_NORMALIZATION_MODES:
            raise ValueError(
                "input_normalization must be "
                f"{SUPPORTED_INPUT_NORMALIZATION_MODES}, "
                f"got {input_normalization!r}"
            )
        self.missingness_mode = normalize_missingness_mode(
            missingness_mode,
            context="missingness_mode",
        )
        if self.missingness_mode == "explicit_token":
            raise ValueError(
                "tabfoundry only supports missingness_mode='none' or 'feature_mask'; "
                "explicit_token is only valid on scalar-token paths"
            )
        self.group_shifts = (0, 1, 3)
        self.feature_group_size = int(feature_group_size)
        if self.feature_group_size <= 0:
            raise ValueError(
                f"feature_group_size must be positive, got {self.feature_group_size}"
            )
        self.norm_type = str(norm_type).strip().lower()
        self.tfcol_n_heads = int(tfcol_n_heads)
        self.tfcol_n_layers = int(tfcol_n_layers)
        self.tfcol_n_inducing = int(tfcol_n_inducing)
        self.tfrow_n_heads = int(tfrow_n_heads)
        self.tfrow_n_layers = int(tfrow_n_layers)
        self.tfrow_cls_tokens = int(tfrow_cls_tokens)
        self.tfrow_norm = str(tfrow_norm).strip().lower()
        self.tficl_n_heads = int(tficl_n_heads)
        self.tficl_n_layers = int(tficl_n_layers)
        self.tficl_ff_expansion = int(tficl_ff_expansion)
        for name, value in (
            ("tfcol_n_heads", self.tfcol_n_heads),
            ("tfcol_n_layers", self.tfcol_n_layers),
            ("tfcol_n_inducing", self.tfcol_n_inducing),
            ("tfrow_n_heads", self.tfrow_n_heads),
            ("tfrow_n_layers", self.tfrow_n_layers),
            ("tfrow_cls_tokens", self.tfrow_cls_tokens),
            ("tficl_n_heads", self.tficl_n_heads),
            ("tficl_n_layers", self.tficl_n_layers),
            ("tficl_ff_expansion", self.tficl_ff_expansion),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.norm_type not in SUPPORTED_NORM_TYPES:
            raise ValueError(
                f"norm_type must be one of {SUPPORTED_NORM_TYPES}, got {self.norm_type!r}"
            )
        if self.tfrow_norm not in SUPPORTED_NORM_TYPES:
            raise ValueError(
                f"tfrow_norm must be one of {SUPPORTED_NORM_TYPES}, got {self.tfrow_norm!r}"
            )

        missingness_multiplier = 2 if self.missingness_mode == "feature_mask" else 1
        group_in_dim = len(self.group_shifts) * self.feature_group_size * missingness_multiplier
        self.group_linear = nn.Linear(group_in_dim, d_col)
        self.tfcol = TFColEncoder(
            d_model=d_col,
            n_heads=self.tfcol_n_heads,
            n_layers=self.tfcol_n_layers,
            n_inducing=self.tfcol_n_inducing,
            norm_type=self.norm_type,
        )
        self.tfrow = TFRowEncoder(
            d_model=d_col,
            n_heads=self.tfrow_n_heads,
            n_layers=self.tfrow_n_layers,
            cls_tokens=self.tfrow_cls_tokens,
            d_out=d_icl,
            norm_type=self.tfrow_norm,
        )
        self.tficl = QASSTransformerEncoder(
            d_model=d_icl,
            n_heads=self.tficl_n_heads,
            n_layers=self.tficl_n_layers,
            ff_expansion=self.tficl_ff_expansion,
            use_qass=True,
            norm_type=self.norm_type,
        )

    def _group_features(
        self,
        x: torch.Tensor,
        *,
        missing_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # x: [N, M] -> grouped [N, G, group_size * len(shifts)], mask [G]
        n_rows, n_features = x.shape
        n_groups = int(math.ceil(float(n_features) / float(self.feature_group_size)))
        total_features = n_groups * self.feature_group_size
        pad = total_features - n_features
        x_padded = F.pad(x, (0, pad), mode="constant", value=0.0) if pad > 0 else x

        valid = torch.ones((total_features,), dtype=torch.bool, device=x.device)
        if pad > 0:
            valid[n_features:] = False
        token_padding_mask = ~valid.view(n_groups, self.feature_group_size).any(dim=-1)

        cols: list[torch.Tensor] = []
        base_idx = torch.arange(total_features, device=x.device)
        for shift in self.group_shifts:
            shifted = x_padded[:, base_idx.roll(-shift)]
            cols.append(shifted.view(n_rows, n_groups, self.feature_group_size))
        grouped = torch.cat(cols, dim=-1)
        if missing_mask is not None:
            missing_mask_f32 = missing_mask.to(dtype=x.dtype)
            mask_padded = (
                F.pad(missing_mask_f32, (0, pad), mode="constant", value=0.0)
                if pad > 0
                else missing_mask_f32
            )
            mask_cols: list[torch.Tensor] = []
            for shift in self.group_shifts:
                shifted_mask = mask_padded[:, base_idx.roll(-shift)]
                mask_cols.append(shifted_mask.view(n_rows, n_groups, self.feature_group_size))
            grouped = torch.cat([grouped, torch.cat(mask_cols, dim=-1)], dim=-1)
        return grouped, token_padding_mask

    def _build_e1(
        self,
        x_all: torch.Tensor,
        *,
        missing_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grouped, token_padding_mask = self._group_features(x_all, missing_mask=missing_mask)
        return self.group_linear(grouped), token_padding_mask

    def _column_encode_from_e1(
        self,
        e1: torch.Tensor,
    ) -> torch.Tensor:
        # Encode each column as one set element over rows.
        col_in = e1.permute(1, 0, 2)
        col_out = self.tfcol(col_in)
        return col_out.permute(1, 0, 2)

    def _row_encode(
        self,
        per_feature_embeddings: torch.Tensor,
        *,
        token_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.tfrow(per_feature_embeddings, token_padding_mask=token_padding_mask)

    def _icl_encode(
        self,
        row_embeddings: torch.Tensor,
        train_target_embed: torch.Tensor,
        *,
        n_train: int,
    ) -> torch.Tensor:
        seq = row_embeddings.clone()
        seq[:n_train] = seq[:n_train] + train_target_embed
        seq = seq.unsqueeze(0)
        n_tokens = seq.shape[1]
        allowed_keys = torch.zeros(
            (1, 1, n_tokens, n_tokens), dtype=torch.bool, device=seq.device
        )
        allowed_keys[:, :, :, :n_train] = True
        diag = torch.arange(n_tokens, device=seq.device)
        allowed_keys[:, :, diag, diag] = True
        encoded = self.tficl(seq, allowed_mask=allowed_keys, n_context=n_train)
        return encoded[0]

    def _feature_row_embeddings_from_e1(
        self,
        e1: torch.Tensor,
        *,
        token_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        col_out = self._column_encode_from_e1(e1)
        return self._row_encode(col_out, token_padding_mask=token_padding_mask)

    def _lift_digit_position_embed_to_icl(self, digit_position_embed: torch.Tensor) -> torch.Tensor:
        if int(digit_position_embed.shape[-1]) >= int(self.d_icl):
            return digit_position_embed[..., : self.d_icl]
        return F.pad(digit_position_embed, (0, self.d_icl - int(digit_position_embed.shape[-1])))

    def _prepare_inputs(
        self, batch: TaskBatch
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        n_train = batch.x_train.shape[0]
        normalization_mode = cast(InputNormalizationMode, self.input_normalization)
        missing_mask: torch.Tensor | None = None
        if self.missingness_mode == "feature_mask":
            x_train, x_test, train_missing, test_missing = prepare_train_test_tensors_with_missing(
                batch.x_train,
                batch.x_test,
                mode=normalization_mode,
            )
            missing_mask = torch.cat([train_missing, test_missing], dim=0)
        else:
            x_train, x_test = normalize_train_test_tensors(
                batch.x_train,
                batch.x_test,
                mode=normalization_mode,
            )
        x_all = torch.cat([x_train, x_test], dim=0)
        e1, token_padding_mask = self._build_e1(x_all, missing_mask=missing_mask)
        return e1, token_padding_mask, n_train

    def _encode_from_e1(
        self,
        e1: torch.Tensor,
        token_padding_mask: torch.Tensor,
        train_icl: torch.Tensor,
        n_train: int,
    ) -> torch.Tensor:
        row_embed = self._feature_row_embeddings_from_e1(
            e1,
            token_padding_mask=token_padding_mask,
        )
        icl_out = self._icl_encode(row_embed, train_icl, n_train=n_train)
        return icl_out[n_train:]


class TabFoundryClassifier(_TabFoundryBackbone):
    """Tabfoundry classification model."""

    def __init__(
        self,
        *,
        d_col: int = 128,
        d_icl: int = 512,
        input_normalization: str = "none",
        missingness_mode: str = "none",
        feature_group_size: int = 1,
        norm_type: str = "layernorm",
        many_class_train_mode: str = "path_nll",
        max_mixed_radix_digits: int = 64,
        tfcol_n_heads: int = 8,
        tfcol_n_layers: int = 3,
        tfcol_n_inducing: int = 128,
        tfrow_n_heads: int = 8,
        tfrow_n_layers: int = 3,
        tfrow_cls_tokens: int = 4,
        tfrow_norm: str = "layernorm",
        tficl_n_heads: int = 8,
        tficl_n_layers: int = 12,
        tficl_ff_expansion: int = 2,
        many_class_base: int = DEFAULT_MANY_CLASS_BASE,
        head_hidden_dim: int = DEFAULT_HEAD_HIDDEN_DIM,
        use_digit_position_embed: bool = True,
    ) -> None:
        super().__init__(
            d_col=d_col,
            d_icl=d_icl,
            input_normalization=input_normalization,
            missingness_mode=missingness_mode,
            feature_group_size=feature_group_size,
            norm_type=norm_type,
            tfcol_n_heads=tfcol_n_heads,
            tfcol_n_layers=tfcol_n_layers,
            tfcol_n_inducing=tfcol_n_inducing,
            tfrow_n_heads=tfrow_n_heads,
            tfrow_n_layers=tfrow_n_layers,
            tfrow_cls_tokens=tfrow_cls_tokens,
            tfrow_norm=tfrow_norm,
            tficl_n_heads=tficl_n_heads,
            tficl_n_layers=tficl_n_layers,
            tficl_ff_expansion=tficl_ff_expansion,
        )
        mode = many_class_train_mode.strip().lower()
        if mode not in {"path_nll", "full_probs"}:
            raise ValueError(
                f"many_class_train_mode must be 'path_nll' or 'full_probs', got {many_class_train_mode!r}"
            )
        self.many_class_train_mode = mode
        self.max_mixed_radix_digits = int(max_mixed_radix_digits)
        if self.max_mixed_radix_digits <= 0:
            raise ValueError(
                f"max_mixed_radix_digits must be positive, got {self.max_mixed_radix_digits}"
            )
        self.many_class_base = int(many_class_base)
        if self.many_class_base <= 1:
            raise ValueError(
                f"many_class_base must be >= 2, got {self.many_class_base}"
            )
        self.head_hidden_dim = int(head_hidden_dim)
        if self.head_hidden_dim <= 0:
            raise ValueError(
                f"head_hidden_dim must be positive, got {self.head_hidden_dim}"
            )
        self.use_digit_position_embed = bool(use_digit_position_embed)
        self.embed_tae = nn.Embedding(self.many_class_base, d_col)
        self.embed_icl = nn.Embedding(self.many_class_base, d_icl)
        self.digit_position_embed: nn.Embedding | None
        if self.use_digit_position_embed:
            self.digit_position_embed = nn.Embedding(self.max_mixed_radix_digits, d_col)
        else:
            self.digit_position_embed = None
        self.head = nn.Sequential(
            nn.Linear(d_icl, self.head_hidden_dim),
            nn.GELU(),
            nn.Linear(self.head_hidden_dim, self.many_class_base),
        )

    def _node_train_indices(
        self,
        *,
        node: HierNode,
        y_train: torch.Tensor,
    ) -> torch.Tensor:
        node_classes = node.node_classes_tensor(y_train.device)
        train_mask = torch.isin(y_train.to(torch.int64), node_classes)
        return torch.nonzero(train_mask, as_tuple=False).squeeze(-1)

    def _hierarchical_probs(
        self,
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
            idx = self._node_train_indices(node=node, y_train=y_train)
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
            node_train_labels = y_train[idx]
            mapped = (
                map_labels_to_child_groups(node_train_labels, node)
                .clamp(max=self.many_class_base - 1)
                .to(torch.int64)
            )

            seq = torch.cat([node_train_embed, test_embeddings], dim=0)
            node_train_target_embed = self.embed_icl(mapped)
            encoded = self._icl_encode(
                seq,
                node_train_target_embed,
                n_train=node_train_embed.shape[0],
            )
            test_out = encoded[node_train_embed.shape[0] :]

            if node.is_leaf:
                logits = self.head(test_out)[:, : len(node.classes)]
                probs = torch.softmax(logits, dim=-1)
                for local_idx, cls in enumerate(node.classes):
                    class_probs[:, cls] = (
                        class_probs[:, cls] + parent_prob * probs[:, local_idx]
                    )
                return

            logits = self.head(test_out)[:, : len(node.children)]
            probs = torch.softmax(logits, dim=-1)
            for child_idx, child in enumerate(node.children):
                _recurse(child, parent_prob * probs[:, child_idx])

        _recurse(tree, torch.ones((n_test,), device=row_embeddings.device))
        # Normalize in case of numerical drift.
        denom = class_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        return class_probs / denom, nodes_visited, empty_nodes

    def _hierarchical_path_terms(
        self,
        row_embeddings: torch.Tensor,
        y_train: torch.Tensor,
        y_test: torch.Tensor,
        tree: HierNode,
        *,
        n_train: int,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor], list[int], dict[str, float]]:
        n_test = row_embeddings.shape[0] - n_train
        if int(y_test.shape[0]) != int(n_test):
            raise ValueError("y_test shape does not match test rows")
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

            target_classes = y_test[sample_idx].to(torch.int64)
            mapped_test = (
                map_labels_to_child_groups(target_classes, node)
                .clamp(max=self.many_class_base - 1)
                .to(torch.int64)
            )
            n_choices = len(node.classes) if node.is_leaf else len(node.children)
            idx = self._node_train_indices(node=node, y_train=y_train)
            if idx.numel() == 0:
                empty_nodes += 1
                logits = torch.zeros(
                    (int(sample_idx.numel()), n_choices),
                    device=row_embeddings.device,
                    dtype=row_embeddings.dtype,
                )
            else:
                node_train_embed = row_embeddings[idx]
                node_train_labels = y_train[idx]
                mapped = (
                    map_labels_to_child_groups(node_train_labels, node)
                    .clamp(max=self.many_class_base - 1)
                    .to(torch.int64)
                )
                local_test_embed = test_embeddings[sample_idx]
                seq = torch.cat([node_train_embed, local_test_embed], dim=0)
                node_train_target_embed = self.embed_icl(mapped)
                encoded = self._icl_encode(
                    seq,
                    node_train_target_embed,
                    n_train=node_train_embed.shape[0],
                )
                test_out = encoded[node_train_embed.shape[0] :]
                logits = self.head(test_out)[:, :n_choices]
            logits_terms.append(logits)
            target_terms.append(mapped_test)
            sample_counts.append(int(sample_idx.numel()))

            if node.is_leaf:
                return
            for child_idx, child in enumerate(node.children):
                child_mask = mapped_test == int(child_idx)
                child_samples = sample_idx[child_mask]
                _recurse(child, child_samples)

        root_samples = torch.arange(n_test, device=row_embeddings.device)
        _recurse(tree, root_samples)
        avg_path_depth = float(total_path_steps) / float(max(1, n_test))
        metrics = {
            "many_class_nodes_visited": float(nodes_visited),
            "many_class_avg_path_depth": float(avg_path_depth),
            "many_class_empty_nodes": float(empty_nodes),
        }
        return logits_terms, target_terms, sample_counts, metrics

    def _forward_many_class(
        self,
        e1: torch.Tensor,
        y_train: torch.Tensor,
        y_test: torch.Tensor,
        *,
        num_classes: int,
        n_train: int,
        token_padding_mask: torch.Tensor | None,
    ) -> ClassificationOutput:
        bases = balanced_bases(num_classes=num_classes, max_base=self.many_class_base)
        digits = encode_mixed_radix(y_train, bases=bases)
        if int(digits.shape[0]) > self.max_mixed_radix_digits:
            raise RuntimeError(
                "mixed-radix depth exceeds model.max_mixed_radix_digits; "
                f"got {int(digits.shape[0])} > {self.max_mixed_radix_digits}"
            )

        row_embed = self._feature_row_embeddings_from_e1(
            e1,
            token_padding_mask=token_padding_mask,
        )
        icl_accum: torch.Tensor | None = None
        digit_pos_embed: torch.Tensor | None = None
        if self.use_digit_position_embed:
            digit_positions = torch.arange(
                digits.shape[0], device=row_embed.device, dtype=torch.int64
            )
            assert self.digit_position_embed is not None
            digit_pos_embed = self._lift_digit_position_embed_to_icl(
                self.digit_position_embed(digit_positions)
            )
        for view in range(digits.shape[0]):
            icl_embed = self.embed_icl(
                digits[view].clamp(max=self.many_class_base - 1).to(torch.int64)
            )
            if digit_pos_embed is not None:
                icl_embed = icl_embed + digit_pos_embed[view][None, :]
            conditioned = self._icl_encode(row_embed, icl_embed, n_train=n_train)
            icl_accum = conditioned if icl_accum is None else icl_accum + conditioned
        assert icl_accum is not None
        row_embed = icl_accum / float(digits.shape[0])
        tree = cached_build_balanced_class_tree(
            num_classes, max_branch=self.many_class_base
        )
        if self.training and self.many_class_train_mode == "path_nll":
            path_logits, path_targets, path_sample_counts, path_metrics = (
                self._hierarchical_path_terms(
                    row_embed,
                    y_train,
                    y_test.to(torch.int64),
                    tree,
                    n_train=n_train,
                )
            )
            return ClassificationOutput(
                logits=None,
                num_classes=num_classes,
                class_probs=None,
                path_logits=path_logits,
                path_targets=path_targets,
                path_sample_counts=path_sample_counts,
                aux_metrics=path_metrics,
            )

        class_probs, nodes_visited, empty_nodes = self._hierarchical_probs(
            row_embed,
            y_train,
            tree,
            n_train=n_train,
            num_classes=num_classes,
        )
        return ClassificationOutput(
            logits=None,
            num_classes=num_classes,
            class_probs=class_probs,
            aux_metrics={
                "many_class_nodes_visited": float(nodes_visited),
                "many_class_empty_nodes": float(empty_nodes),
            },
        )

    def forward(self, batch: TaskBatch) -> ClassificationOutput:
        y_train = batch.y_train.to(torch.int64)
        e1, token_padding_mask, n_train = self._prepare_inputs(batch)

        num_classes = int(batch.num_classes or int(y_train.max().item()) + 1)
        if num_classes > 10:
            return self._forward_many_class(
                e1,
                y_train,
                batch.y_test.to(torch.int64),
                num_classes=num_classes,
                n_train=n_train,
                token_padding_mask=token_padding_mask,
            )

        train_icl = self.embed_icl(y_train.clamp(max=9))
        test_out = self._encode_from_e1(
            e1,
            token_padding_mask,
            train_icl,
            n_train,
        )
        logits = self.head(test_out)
        return ClassificationOutput(
            logits=logits, num_classes=num_classes, class_probs=None
        )


class TabFoundryRegressor(_TabFoundryBackbone):
    """Tabfoundry regression model with 999 quantile outputs."""

    def __init__(
        self,
        *,
        d_col: int = 128,
        d_icl: int = 512,
        input_normalization: str = "none",
        missingness_mode: str = "none",
        feature_group_size: int = 1,
        norm_type: str = "layernorm",
        tfcol_n_heads: int = 8,
        tfcol_n_layers: int = 3,
        tfcol_n_inducing: int = 128,
        tfrow_n_heads: int = 8,
        tfrow_n_layers: int = 3,
        tfrow_cls_tokens: int = 4,
        tfrow_norm: str = "layernorm",
        tficl_n_heads: int = 8,
        tficl_n_layers: int = 12,
        tficl_ff_expansion: int = 2,
        head_hidden_dim: int = DEFAULT_HEAD_HIDDEN_DIM,
    ) -> None:
        super().__init__(
            d_col=d_col,
            d_icl=d_icl,
            input_normalization=input_normalization,
            missingness_mode=missingness_mode,
            feature_group_size=feature_group_size,
            norm_type=norm_type,
            tfcol_n_heads=tfcol_n_heads,
            tfcol_n_layers=tfcol_n_layers,
            tfcol_n_inducing=tfcol_n_inducing,
            tfrow_n_heads=tfrow_n_heads,
            tfrow_n_layers=tfrow_n_layers,
            tfrow_cls_tokens=tfrow_cls_tokens,
            tfrow_norm=tfrow_norm,
            tficl_n_heads=tficl_n_heads,
            tficl_n_layers=tficl_n_layers,
            tficl_ff_expansion=tficl_ff_expansion,
        )
        self.head_hidden_dim = int(head_hidden_dim)
        if self.head_hidden_dim <= 0:
            raise ValueError(
                f"head_hidden_dim must be positive, got {self.head_hidden_dim}"
            )
        self.embed_tae = nn.Linear(1, d_col)
        self.embed_icl = nn.Linear(1, d_icl)
        self.head = nn.Sequential(
            nn.Linear(d_icl, self.head_hidden_dim),
            nn.GELU(),
            nn.Linear(self.head_hidden_dim, DEFAULT_REGRESSION_QUANTILES),
        )
        q = torch.arange(
            1, DEFAULT_REGRESSION_QUANTILES + 1, dtype=torch.float32
        ) / float(DEFAULT_REGRESSION_QUANTILES + 1)
        self.register_buffer("quantile_levels", q, persistent=False)

    def forward(self, batch: TaskBatch) -> RegressionOutput:
        y_train = batch.y_train.to(torch.float32)
        e1, token_padding_mask, n_train = self._prepare_inputs(batch)

        icl = self.embed_icl(y_train[:, None])
        test_out = self._encode_from_e1(e1, token_padding_mask, icl, n_train)
        quantiles = self.head(test_out)

        return RegressionOutput(
            quantiles=quantiles,
            quantile_levels=cast(torch.Tensor, self.quantile_levels),
        )
