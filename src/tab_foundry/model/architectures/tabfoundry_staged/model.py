"""Resolved-surface staged tabfoundry architectures."""

from __future__ import annotations

from typing import cast

import torch
from torch import nn

from tab_foundry.input_normalization import (
    InputNormalizationMode,
    SUPPORTED_INPUT_NORMALIZATION_MODES,
    normalize_train_test_tensors,
)
from tab_foundry.model.architectures.tabfoundry import (
    ClassificationOutput,
    DEFAULT_REGRESSION_QUANTILES,
    RegressionOutput,
)
from tab_foundry.model.components.many_class import (
    HierNode,
    balanced_bases,
    cached_build_balanced_class_tree,
    encode_mixed_radix,
    map_labels_to_child_groups,
)
from tab_foundry.model.spec import (
    ModelBuildSpec,
    ModelStage,
    SUPPORTED_MANY_CLASS_TRAIN_MODES,
)
from tab_foundry.types import TaskBatch

from .builders import (
    build_column_encoder,
    build_context_encoder,
    build_context_label_embed,
    build_digit_position_embed,
    build_direct_head,
    build_feature_encoder,
    build_post_encoder_norm,
    build_row_pool,
    build_table_block,
    build_target_conditioner,
    build_tokenizer,
)
from .recipes import recipe_for_stage
from .resolved import resolve_staged_surface
from .states import CellTableState, HeadOutputState, RawInputState, RowState


class _TabFoundryStagedBase(nn.Module):
    """Shared staged family implementation used by classifier and regressor."""

    def __init__(
        self,
        *,
        task: str,
        stage: str | None = None,
        stage_label: str | None = None,
        module_overrides: dict[str, object] | None = None,
        d_col: int = 128,
        d_icl: int = 512,
        input_normalization: str = "none",
        feature_group_size: int = 1,
        many_class_train_mode: str = "path_nll",
        max_mixed_radix_digits: int = 64,
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
        many_class_base: int = 10,
        head_hidden_dim: int = 1024,
        use_digit_position_embed: bool = True,
    ) -> None:
        super().__init__()
        self.model_spec = ModelBuildSpec(
            task=task,
            arch="tabfoundry_staged",
            stage=stage,
            stage_label=stage_label,
            module_overrides=module_overrides,
            d_col=d_col,
            d_icl=d_icl,
            input_normalization=input_normalization,
            feature_group_size=feature_group_size,
            many_class_train_mode=many_class_train_mode,
            max_mixed_radix_digits=max_mixed_radix_digits,
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
            many_class_base=many_class_base,
            head_hidden_dim=head_hidden_dim,
            use_digit_position_embed=use_digit_position_embed,
        )
        self.surface = resolve_staged_surface(self.model_spec)
        if self.model_spec.task == "regression" and self.surface.head == "many_class":
            raise ValueError(
                "stage='many_class' is classification-only and is not supported for "
                "task='regression'"
            )
        self.recipe = recipe_for_stage(ModelStage(self.surface.stage))
        self.stage = self.surface.stage
        self.stage_label = self.surface.stage_label
        self.arch = "tabfoundry_staged"
        self.benchmark_profile = self.surface.benchmark_profile
        self.module_selection = self.surface.module_selection()
        self.module_hyperparameters = self.surface.component_hyperparameters()

        self.d_icl = int(self.model_spec.d_icl)
        self.input_normalization = str(self.model_spec.input_normalization).strip().lower()
        if self.input_normalization not in SUPPORTED_INPUT_NORMALIZATION_MODES:
            raise ValueError(
                "input_normalization must be "
                f"{SUPPORTED_INPUT_NORMALIZATION_MODES}, got {self.input_normalization!r}"
            )
        self.tficl_n_heads = int(self.model_spec.tficl_n_heads)
        self.tficl_n_layers = int(self.model_spec.tficl_n_layers)
        self.tficl_ff_expansion = int(self.model_spec.tficl_ff_expansion)
        self.max_mixed_radix_digits = int(self.model_spec.max_mixed_radix_digits)
        self.many_class_base = int(self.model_spec.many_class_base)
        self.head_hidden_dim = int(self.model_spec.head_hidden_dim)
        self.many_class_train_mode = str(self.model_spec.many_class_train_mode).strip().lower()
        if self.many_class_train_mode not in SUPPORTED_MANY_CLASS_TRAIN_MODES:
            raise ValueError(
                "many_class_train_mode must be "
                f"{SUPPORTED_MANY_CLASS_TRAIN_MODES}, got {self.many_class_train_mode!r}"
            )
        self.use_digit_position_embed = bool(self.model_spec.use_digit_position_embed)
        for name, value in (
            ("d_icl", self.d_icl),
            ("tficl_n_heads", self.tficl_n_heads),
            ("tficl_n_layers", self.tficl_n_layers),
            ("max_mixed_radix_digits", self.max_mixed_radix_digits),
            ("many_class_base", self.many_class_base),
            ("head_hidden_dim", self.head_hidden_dim),
        ):
            if value <= 0:
                raise ValueError(f"{name} must be positive, got {value}")
        if self.model_spec.task == "classification" and self.surface.head != "many_class":
            if self.many_class_train_mode != "path_nll":
                raise ValueError(
                    f"stage={self.stage!r} only supports many_class_train_mode='path_nll', "
                    f"got {self.many_class_train_mode!r}"
                )

        self.tokenizer = build_tokenizer(self.surface)
        self.feature_encoder = build_feature_encoder(
            self.surface,
            tokenizer=self.tokenizer,
            d_icl=self.d_icl,
        )
        self.target_conditioner = build_target_conditioner(
            self.surface,
            task=self.model_spec.task,
            d_icl=self.d_icl,
            many_class_base=self.many_class_base,
        )
        self.transformer_blocks = nn.ModuleList(
            [build_table_block(self.surface, d_icl=self.d_icl) for _ in range(self.tficl_n_layers)]
        )
        self.column_encoder = build_column_encoder(self.surface, d_icl=self.d_icl)
        self.row_pool = build_row_pool(self.surface, d_icl=self.d_icl)
        self.context_encoder = build_context_encoder(self.surface, d_icl=self.d_icl)
        self.context_label_embed = build_context_label_embed(
            self.surface,
            task=self.model_spec.task,
            d_icl=self.d_icl,
            many_class_base=self.many_class_base,
        )
        self.post_encoder_norm = build_post_encoder_norm(self.surface, d_icl=self.d_icl)
        self.digit_position_embed = build_digit_position_embed(
            self.surface,
            d_icl=self.d_icl,
            max_mixed_radix_digits=self.max_mixed_radix_digits,
            use_digit_position_embed=self.use_digit_position_embed,
        )
        self.direct_head = build_direct_head(
            self.surface,
            task=self.model_spec.task,
            d_icl=self.d_icl,
            head_hidden_dim=self.head_hidden_dim,
            many_class_base=self.many_class_base,
        )
        self._activation_trace: dict[str, list[float]] | None = None

    def enable_activation_trace(self) -> None:
        self._activation_trace = {}

    def disable_activation_trace(self) -> None:
        self._activation_trace = None

    def trace_activation(self, name: str, tensor: torch.Tensor) -> None:
        if self._activation_trace is None:
            return
        trace_value = float(tensor.detach().to(torch.float32).square().mean().sqrt().item())
        self._activation_trace.setdefault(name, []).append(trace_value)

    def flush_activation_trace(self) -> dict[str, float] | None:
        if self._activation_trace is None:
            return None
        snapshot = {
            name: values[-1] for name, values in self._activation_trace.items() if values
        }
        self._activation_trace = {}
        return snapshot

    def _prepare_task_inputs(
        self,
        batch: TaskBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        train_test_split_index = int(batch.x_train.shape[0])
        if train_test_split_index <= 0:
            raise RuntimeError("tabfoundry_staged requires at least one training row")
        x_all = torch.cat([batch.x_train, batch.x_test], dim=0).to(torch.float32).unsqueeze(0)
        if self.model_spec.task == "classification":
            y_train = batch.y_train.to(torch.int64).unsqueeze(0)
            y_test = batch.y_test.to(torch.int64).unsqueeze(0)
        else:
            y_train = batch.y_train.to(torch.float32).unsqueeze(0)
            y_test = batch.y_test.to(torch.float32).unsqueeze(0)
        return x_all, y_train, y_test, train_test_split_index

    @staticmethod
    def _validate_batched_inputs(
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        train_test_split_index: int,
    ) -> None:
        if x_all.ndim != 3:
            raise ValueError(f"x_all must have shape [B, R, C], got {tuple(x_all.shape)}")
        if y_train.ndim != 2:
            raise ValueError(f"y_train must have shape [B, R_train], got {tuple(y_train.shape)}")
        if int(x_all.shape[0]) != int(y_train.shape[0]):
            raise ValueError("x_all and y_train must have matching batch dimensions")
        if train_test_split_index <= 0 or train_test_split_index >= int(x_all.shape[1]):
            raise ValueError(
                "train_test_split_index must satisfy 0 < split < num_rows, got "
                f"split={train_test_split_index}, num_rows={x_all.shape[1]}"
            )
        if int(y_train.shape[1]) != train_test_split_index:
            raise ValueError("y_train length must match train_test_split_index")

    def _normalize_x_all(self, x_all: torch.Tensor, *, train_test_split_index: int) -> torch.Tensor:
        if self.surface.normalization_mode != "shared":
            return x_all
        x_train = x_all[:, :train_test_split_index, :]
        x_test = x_all[:, train_test_split_index:, :]
        train_parts: list[torch.Tensor] = []
        test_parts: list[torch.Tensor] = []
        for batch_idx in range(int(x_all.shape[0])):
            train_norm, test_norm = normalize_train_test_tensors(
                x_train[batch_idx],
                x_test[batch_idx],
                mode=cast(
                    InputNormalizationMode,
                    self.input_normalization,
                ),
            )
            train_parts.append(train_norm)
            test_parts.append(test_norm)
        return torch.cat(
            [
                torch.stack(train_parts, dim=0),
                torch.stack(test_parts, dim=0),
            ],
            dim=1,
        )

    def _batched_state_num_classes(self, y_train: torch.Tensor) -> int:
        if self.model_spec.task == "classification":
            return max(2, int(y_train.max().item()) + 1)
        return 1

    def _build_raw_input_state(
        self,
        *,
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        y_test: torch.Tensor | None,
        train_test_split_index: int,
        num_classes: int,
    ) -> RawInputState:
        self._validate_batched_inputs(x_all, y_train, train_test_split_index)
        return RawInputState(
            x_all=self._normalize_x_all(x_all, train_test_split_index=train_test_split_index),
            y_train=y_train,
            y_test=y_test,
            train_test_split_index=train_test_split_index,
            num_classes=num_classes,
        )

    def _feature_cells(self, raw_state: RawInputState) -> torch.Tensor:
        tokenized_x, _token_padding_mask = self.tokenizer(raw_state.x_all)
        if self.surface.feature_encoder == "nano":
            feature_cells = self.feature_encoder(raw_state.x_all, raw_state.train_test_split_index)
        else:
            feature_cells = self.feature_encoder(tokenized_x)
        self.trace_activation("post_feature_encoder", feature_cells)
        return feature_cells

    def _build_table_tokens_from_raw(self, raw_state: RawInputState) -> torch.Tensor:
        feature_cells = self._feature_cells(raw_state)
        target_cells = self.target_conditioner(
            raw_state.y_train,
            num_rows=int(raw_state.x_all.shape[1]),
        )
        self.trace_activation("post_target_conditioner", target_cells)
        return torch.cat([feature_cells, target_cells], dim=2)

    def _build_table_tokens_batched(
        self,
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        *,
        train_test_split_index: int,
    ) -> torch.Tensor:
        raw_state = self._build_raw_input_state(
            x_all=x_all,
            y_train=y_train,
            y_test=None,
            train_test_split_index=train_test_split_index,
            num_classes=self._batched_state_num_classes(y_train),
        )
        return self._build_table_tokens_from_raw(raw_state)

    def _encode_table_batched(
        self,
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        *,
        train_test_split_index: int,
    ) -> torch.Tensor:
        cells = self._build_table_tokens_batched(
            x_all,
            y_train,
            train_test_split_index=train_test_split_index,
        )
        if self.post_encoder_norm is not None:
            cells = self.post_encoder_norm(cells)
        self.trace_activation("pre_transformer", cells)
        for index, block in enumerate(self.transformer_blocks):
            cells = block(cells, train_test_split_index=train_test_split_index)
            self.trace_activation(f"post_transformer_block_{index}", cells)
        return cells

    def _build_table_tokens(self, batch: TaskBatch) -> tuple[torch.Tensor, int]:
        x_all, y_train, _y_test, train_test_split_index = self._prepare_task_inputs(batch)
        table = self._build_table_tokens_batched(
            x_all,
            y_train,
            train_test_split_index=train_test_split_index,
        )
        return table.squeeze(0), train_test_split_index

    def _encode_table(self, batch: TaskBatch) -> tuple[torch.Tensor, int]:
        x_all, y_train, _y_test, train_test_split_index = self._prepare_task_inputs(batch)
        encoded = self._encode_table_batched(
            x_all,
            y_train,
            train_test_split_index=train_test_split_index,
        )
        return encoded.squeeze(0), train_test_split_index

    def _encode_to_cell_state(self, raw_state: RawInputState) -> CellTableState:
        cells = self._build_table_tokens_from_raw(raw_state)
        if self.post_encoder_norm is not None:
            cells = self.post_encoder_norm(cells)
        self.trace_activation("pre_transformer", cells)
        for index, block in enumerate(self.transformer_blocks):
            cells = block(cells, train_test_split_index=raw_state.train_test_split_index)
            self.trace_activation(f"post_transformer_block_{index}", cells)
        return CellTableState(
            cells=cells,
            train_test_split_index=raw_state.train_test_split_index,
            num_classes=raw_state.num_classes,
        )

    def _pool_rows(self, cell_state: CellTableState) -> RowState:
        encoded_cells = self.column_encoder(cell_state.cells)
        self.trace_activation("post_column_encoder", encoded_cells)
        rows = self.row_pool(encoded_cells, token_padding_mask=None)
        self.trace_activation("post_row_pool", rows)
        return RowState(
            rows=rows,
            train_test_split_index=cell_state.train_test_split_index,
            num_classes=cell_state.num_classes,
        )

    def _condition_rows(
        self,
        row_state: RowState,
        *,
        train_target_embeddings: torch.Tensor | None = None,
    ) -> HeadOutputState:
        rows = row_state.rows
        if self.context_encoder is not None:
            if train_target_embeddings is None:
                raise RuntimeError("train_target_embeddings must be provided for context encoding")
            rows = self.context_encoder(
                rows,
                train_target_embeddings=train_target_embeddings,
                train_test_split_index=row_state.train_test_split_index,
            )
        return HeadOutputState(
            rows=rows,
            train_test_split_index=row_state.train_test_split_index,
            num_classes=row_state.num_classes,
        )

    def _context_train_embeddings(self, y_train: torch.Tensor) -> torch.Tensor:
        assert self.context_label_embed is not None
        if self.model_spec.task == "classification":
            return self.context_label_embed(y_train.clamp(max=self.many_class_base - 1))
        return self.context_label_embed(y_train.to(torch.float32))

    def _build_direct_head_state(self, raw_state: RawInputState) -> HeadOutputState:
        cell_state = self._encode_to_cell_state(raw_state)
        row_state = self._pool_rows(cell_state)
        if self.context_encoder is None:
            return HeadOutputState(
                rows=row_state.rows,
                train_test_split_index=row_state.train_test_split_index,
                num_classes=row_state.num_classes,
            )
        return self._condition_rows(
            row_state,
            train_target_embeddings=self._context_train_embeddings(raw_state.y_train),
        )

    def _direct_head_outputs(self, raw_state: RawInputState) -> torch.Tensor:
        head_state = self._build_direct_head_state(raw_state)
        return self.direct_head(head_state.rows[:, raw_state.train_test_split_index :, :])


class TabFoundryStagedClassifier(_TabFoundryStagedBase):
    """Staged classification architecture that resolves to an explicit surface."""

    def __init__(
        self,
        *,
        stage: str | None = None,
        stage_label: str | None = None,
        module_overrides: dict[str, object] | None = None,
        d_col: int = 128,
        d_icl: int = 512,
        input_normalization: str = "none",
        feature_group_size: int = 1,
        many_class_train_mode: str = "path_nll",
        max_mixed_radix_digits: int = 64,
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
        many_class_base: int = 10,
        head_hidden_dim: int = 1024,
        use_digit_position_embed: bool = True,
    ) -> None:
        super().__init__(
            task="classification",
            stage=stage,
            stage_label=stage_label,
            module_overrides=module_overrides,
            d_col=d_col,
            d_icl=d_icl,
            input_normalization=input_normalization,
            feature_group_size=feature_group_size,
            many_class_train_mode=many_class_train_mode,
            max_mixed_radix_digits=max_mixed_radix_digits,
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
            many_class_base=many_class_base,
            head_hidden_dim=head_hidden_dim,
            use_digit_position_embed=use_digit_position_embed,
        )

    @staticmethod
    def _task_num_classes(batch: TaskBatch) -> int:
        if batch.num_classes is not None:
            return int(batch.num_classes)
        if batch.y_train.numel() == 0:
            raise RuntimeError("tabfoundry_staged requires at least one training label")
        return int(batch.y_train.max().item()) + 1

    def forward_batched(
        self,
        *,
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        train_test_split_index: int,
    ) -> torch.Tensor:
        if self.surface.head == "many_class":
            raise RuntimeError("forward_batched() is only supported for direct-head staged recipes")
        raw_state = self._build_raw_input_state(
            x_all=x_all,
            y_train=y_train.to(torch.int64),
            y_test=None,
            train_test_split_index=train_test_split_index,
            num_classes=self._batched_state_num_classes(y_train.to(torch.int64)),
        )
        return self._direct_head_outputs(raw_state)

    def _node_train_indices(self, *, node: HierNode, y_train: torch.Tensor) -> torch.Tensor:
        node_classes = node.node_classes_tensor(y_train.device)
        train_mask = torch.isin(y_train.to(torch.int64), node_classes)
        return torch.nonzero(train_mask, as_tuple=False).squeeze(-1)

    def _encode_rows_with_targets(
        self,
        rows: torch.Tensor,
        *,
        train_targets: torch.Tensor,
        train_test_split_index: int,
    ) -> torch.Tensor:
        assert self.context_encoder is not None
        return self.context_encoder(
            rows.unsqueeze(0),
            train_target_embeddings=train_targets.unsqueeze(0),
            train_test_split_index=train_test_split_index,
        )[0]

    def _digit_conditioned_rows(self, row_state: RowState, y_train: torch.Tensor) -> torch.Tensor:
        digits = encode_mixed_radix(
            y_train,
            bases=balanced_bases(
                num_classes=row_state.num_classes,
                max_base=self.many_class_base,
            ),
        )
        if int(digits.shape[0]) > self.max_mixed_radix_digits:
            raise RuntimeError(
                "mixed-radix depth exceeds model.max_mixed_radix_digits; "
                f"got {int(digits.shape[0])} > {self.max_mixed_radix_digits}"
            )
        accum: torch.Tensor | None = None
        for view in range(int(digits.shape[0])):
            train_targets = self._context_train_embeddings(digits[view].unsqueeze(0))[0]
            if self.digit_position_embed is not None:
                pos = self.digit_position_embed(
                    torch.tensor([view], device=row_state.rows.device, dtype=torch.int64)
                )[0]
                train_targets = train_targets + pos[None, :]
            conditioned = self._encode_rows_with_targets(
                row_state.rows[0],
                train_targets=train_targets,
                train_test_split_index=row_state.train_test_split_index,
            )
            accum = conditioned if accum is None else accum + conditioned
        assert accum is not None
        return accum / float(digits.shape[0])

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
            mapped = (
                map_labels_to_child_groups(y_train[idx], node)
                .clamp(max=self.many_class_base - 1)
                .to(torch.int64)
            )
            seq = torch.cat([node_train_embed, test_embeddings], dim=0)
            conditioned = self._encode_rows_with_targets(
                seq,
                train_targets=self._context_train_embeddings(mapped.unsqueeze(0))[0],
                train_test_split_index=int(node_train_embed.shape[0]),
            )
            test_out = conditioned[int(node_train_embed.shape[0]) :]
            if node.is_leaf:
                logits = self.direct_head(test_out)[:, : len(node.classes)]
                probs = torch.softmax(logits, dim=-1)
                for local_idx, cls in enumerate(node.classes):
                    class_probs[:, cls] = class_probs[:, cls] + parent_prob * probs[:, local_idx]
                return

            logits = self.direct_head(test_out)[:, : len(node.children)]
            probs = torch.softmax(logits, dim=-1)
            for child_idx, child in enumerate(node.children):
                _recurse(child, parent_prob * probs[:, child_idx])

        _recurse(tree, torch.ones((n_test,), device=row_embeddings.device))
        denom = class_probs.sum(dim=-1, keepdim=True).clamp_min(1.0e-12)
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
                mapped_train = (
                    map_labels_to_child_groups(y_train[idx], node)
                    .clamp(max=self.many_class_base - 1)
                    .to(torch.int64)
                )
                seq = torch.cat([node_train_embed, test_embeddings[sample_idx]], dim=0)
                conditioned = self._encode_rows_with_targets(
                    seq,
                    train_targets=self._context_train_embeddings(mapped_train.unsqueeze(0))[0],
                    train_test_split_index=int(node_train_embed.shape[0]),
                )
                logits = self.direct_head(conditioned[int(node_train_embed.shape[0]) :])[:, :n_choices]
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
        return logits_terms, target_terms, sample_counts, {
            "many_class_nodes_visited": float(nodes_visited),
            "many_class_avg_path_depth": float(avg_path_depth),
            "many_class_empty_nodes": float(empty_nodes),
        }

    def _forward_many_class(self, raw_state: RawInputState) -> ClassificationOutput:
        assert raw_state.y_test is not None
        cell_state = self._encode_to_cell_state(raw_state)
        row_state = self._pool_rows(cell_state)
        digit_conditioned = self._digit_conditioned_rows(row_state, raw_state.y_train[0])
        tree = cached_build_balanced_class_tree(
            raw_state.num_classes,
            max_branch=self.many_class_base,
        )
        if self.training and self.many_class_train_mode == "path_nll":
            path_logits, path_targets, path_sample_counts, path_metrics = (
                self._hierarchical_path_terms(
                    digit_conditioned,
                    raw_state.y_train[0],
                    raw_state.y_test[0],
                    tree,
                    n_train=row_state.train_test_split_index,
                )
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

        class_probs, nodes_visited, empty_nodes = self._hierarchical_probs(
            digit_conditioned,
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

    def _validate_num_classes(self, num_classes: int) -> None:
        if num_classes < 2:
            raise RuntimeError("tabfoundry_staged requires at least 2 classes")
        if not self.surface.task_contract.supports(num_classes=num_classes):
            if self.surface.task_contract.max_classes == 2:
                raise RuntimeError(f"stage={self.stage!r} is binary-only and requires num_classes=2")
            if (
                self.surface.task_contract.max_classes is not None
                and num_classes > self.surface.task_contract.max_classes
            ):
                limit = self.surface.task_contract.max_classes
                if limit == self.many_class_base:
                    raise RuntimeError(
                        f"stage={self.stage!r} only supports num_classes <= "
                        f"many_class_base={self.many_class_base}, got {num_classes}"
                    )
                raise RuntimeError(
                    f"stage={self.stage!r} only supports num_classes <= "
                    f"{limit}, got {num_classes}"
                )
            raise RuntimeError(f"stage={self.stage!r} does not support num_classes={num_classes}")
        if self.surface.head != "many_class" and num_classes > self.many_class_base:
            raise RuntimeError(
                f"stage={self.stage!r} only supports num_classes <= "
                f"many_class_base={self.many_class_base}, got {num_classes}"
            )

    def forward(self, batch: TaskBatch) -> ClassificationOutput:
        num_classes = self._task_num_classes(batch)
        self._validate_num_classes(num_classes)
        x_all, y_train, y_test, train_test_split_index = self._prepare_task_inputs(batch)
        raw_state = self._build_raw_input_state(
            x_all=x_all,
            y_train=y_train,
            y_test=y_test,
            train_test_split_index=train_test_split_index,
            num_classes=num_classes,
        )
        if self.surface.head == "many_class" and num_classes > self.many_class_base:
            return self._forward_many_class(raw_state)

        logits = self._direct_head_outputs(raw_state).squeeze(0)
        return ClassificationOutput(
            logits=logits,
            num_classes=num_classes,
            class_probs=None,
        )


class TabFoundryStagedRegressor(_TabFoundryStagedBase):
    """Staged regression architecture with support-set target standardization."""

    _target_scale_eps = 1.0e-6

    def __init__(
        self,
        *,
        stage: str | None = None,
        stage_label: str | None = None,
        module_overrides: dict[str, object] | None = None,
        d_col: int = 128,
        d_icl: int = 512,
        input_normalization: str = "none",
        feature_group_size: int = 1,
        many_class_train_mode: str = "path_nll",
        max_mixed_radix_digits: int = 64,
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
        many_class_base: int = 10,
        head_hidden_dim: int = 1024,
        use_digit_position_embed: bool = True,
    ) -> None:
        super().__init__(
            task="regression",
            stage=stage,
            stage_label=stage_label,
            module_overrides=module_overrides,
            d_col=d_col,
            d_icl=d_icl,
            input_normalization=input_normalization,
            feature_group_size=feature_group_size,
            many_class_train_mode=many_class_train_mode,
            max_mixed_radix_digits=max_mixed_radix_digits,
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
            many_class_base=many_class_base,
            head_hidden_dim=head_hidden_dim,
            use_digit_position_embed=use_digit_position_embed,
        )
        q = torch.arange(
            1, DEFAULT_REGRESSION_QUANTILES + 1, dtype=torch.float32
        ) / float(DEFAULT_REGRESSION_QUANTILES + 1)
        self.register_buffer("quantile_levels", q, persistent=False)

    def _normalize_train_targets(
        self,
        y_train: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean = y_train.mean(dim=1, keepdim=True)
        raw_std = y_train.std(dim=1, keepdim=True, unbiased=False)
        std = torch.where(
            raw_std < self._target_scale_eps,
            torch.ones_like(raw_std),
            raw_std,
        )
        return (y_train - mean) / std, mean, std

    @staticmethod
    def _inverse_transform_quantiles(
        normalized_quantiles: torch.Tensor,
        *,
        target_mean: torch.Tensor,
        target_std: torch.Tensor,
    ) -> torch.Tensor:
        return normalized_quantiles * target_std.unsqueeze(-1) + target_mean.unsqueeze(-1)

    def forward_batched(
        self,
        *,
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        train_test_split_index: int,
    ) -> torch.Tensor:
        normalized_y_train, target_mean, target_std = self._normalize_train_targets(
            y_train.to(torch.float32)
        )
        raw_state = self._build_raw_input_state(
            x_all=x_all,
            y_train=normalized_y_train,
            y_test=None,
            train_test_split_index=train_test_split_index,
            num_classes=1,
        )
        normalized_quantiles = self._direct_head_outputs(raw_state)
        return self._inverse_transform_quantiles(
            normalized_quantiles,
            target_mean=target_mean,
            target_std=target_std,
        )

    def forward(self, batch: TaskBatch) -> RegressionOutput:
        x_all, y_train, y_test, train_test_split_index = self._prepare_task_inputs(batch)
        normalized_y_train, target_mean, target_std = self._normalize_train_targets(y_train)
        raw_state = self._build_raw_input_state(
            x_all=x_all,
            y_train=normalized_y_train,
            y_test=y_test,
            train_test_split_index=train_test_split_index,
            num_classes=1,
        )
        normalized_quantiles = self._direct_head_outputs(raw_state)
        quantiles = self._inverse_transform_quantiles(
            normalized_quantiles,
            target_mean=target_mean,
            target_std=target_std,
        )
        return RegressionOutput(
            quantiles=quantiles.squeeze(0),
            quantile_levels=cast(torch.Tensor, self.quantile_levels),
            normalized_quantiles=normalized_quantiles.squeeze(0),
            target_mean=target_mean.squeeze(0),
            target_std=target_std.squeeze(0),
        )
