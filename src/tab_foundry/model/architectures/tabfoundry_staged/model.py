"""Resolved-surface staged tabfoundry classifier."""

from __future__ import annotations

import math

import torch
from torch import nn

from tab_foundry.input_normalization import SUPPORTED_INPUT_NORMALIZATION_MODES
from tab_foundry.model.outputs import ClassificationOutput
from tab_foundry.model.spec import ModelBuildSpec, ModelStage, SUPPORTED_MANY_CLASS_TRAIN_MODES
from tab_foundry.types import TaskBatch

from . import direct_head as _direct_head
from . import forward_common as _forward_common
from . import many_class as _many_class
from .builders import (
    build_column_encoder,
    build_context_encoder,
    build_context_label_embed,
    build_digit_position_embed,
    build_direct_head,
    build_feature_encoder,
    build_post_encoder_norm,
    build_post_stack_norm,
    build_row_pool,
    build_table_block,
    build_target_conditioner,
    build_tokenizer,
)
from .recipes import recipe_for_stage
from .resolved import resolve_staged_surface


class TabFoundryStagedClassifier(nn.Module):
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
        staged_dropout: float = 0.0,
        pre_encoder_clip: float | None = None,
    ) -> None:
        super().__init__()
        self.model_spec = ModelBuildSpec(
            task="classification",
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
            staged_dropout=staged_dropout,
            pre_encoder_clip=pre_encoder_clip,
        )
        self.surface = resolve_staged_surface(self.model_spec)
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
        self.pre_encoder_clip = self.model_spec.pre_encoder_clip
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
        if self.surface.head != "many_class" and self.many_class_train_mode != "path_nll":
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
            d_icl=self.d_icl,
            many_class_base=self.many_class_base,
        )
        self.post_encoder_norm = build_post_encoder_norm(self.surface, d_icl=self.d_icl)
        self.post_stack_norm = build_post_stack_norm(self.surface, d_icl=self.d_icl)
        self.digit_position_embed = build_digit_position_embed(
            self.surface,
            d_icl=self.d_icl,
            max_mixed_radix_digits=self.max_mixed_radix_digits,
            use_digit_position_embed=self.use_digit_position_embed,
        )
        self.direct_head = build_direct_head(
            self.surface,
            d_icl=self.d_icl,
            head_hidden_dim=self.head_hidden_dim,
            many_class_base=self.many_class_base,
        )
        self._activation_trace: dict[str, tuple[float, int]] | None = None

    def enable_activation_trace(self) -> None:
        self._activation_trace = {}

    def disable_activation_trace(self) -> None:
        self._activation_trace = None

    def trace_activation(self, name: str, tensor: torch.Tensor) -> None:
        if self._activation_trace is None:
            return
        trace_tensor = tensor.detach().to(torch.float32)
        trace_sum_sq = float(trace_tensor.square().sum().item())
        trace_count = int(trace_tensor.numel())
        total_sum_sq, total_count = self._activation_trace.get(name, (0.0, 0))
        self._activation_trace[name] = (
            total_sum_sq + trace_sum_sq,
            total_count + trace_count,
        )

    def flush_activation_trace_stats(self) -> dict[str, tuple[float, int]] | None:
        if self._activation_trace is None:
            return None
        snapshot = {
            name: (float(total_sum_sq), int(total_count))
            for name, (total_sum_sq, total_count) in self._activation_trace.items()
            if total_count > 0
        }
        self._activation_trace = {}
        return snapshot

    def flush_activation_trace(self) -> dict[str, float] | None:
        snapshot = self.flush_activation_trace_stats()
        if snapshot is None:
            return None
        return {
            name: float(math.sqrt(total_sum_sq / float(total_count)))
            for name, (total_sum_sq, total_count) in snapshot.items()
            if total_count > 0
        }

    @staticmethod
    def _task_num_classes(batch: TaskBatch) -> int:
        if batch.num_classes is not None:
            return int(batch.num_classes)
        if batch.y_train.numel() == 0:
            raise RuntimeError("tabfoundry_staged requires at least one training label")
        return int(batch.y_train.max().item()) + 1

    @staticmethod
    def _prepare_task_inputs(batch: TaskBatch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return _forward_common.prepare_task_inputs(batch)

    @staticmethod
    def _validate_batched_inputs(
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        train_test_split_index: int,
    ) -> None:
        _forward_common.validate_batched_inputs(x_all, y_train, train_test_split_index)

    def _normalize_x_all(self, x_all: torch.Tensor, *, train_test_split_index: int) -> torch.Tensor:
        return _forward_common.normalize_x_all(self, x_all, train_test_split_index=train_test_split_index)

    def _build_raw_input_state(
        self,
        *,
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        y_test: torch.Tensor | None,
        train_test_split_index: int,
        num_classes: int,
    ):
        return _forward_common.build_raw_input_state(
            self,
            x_all=x_all,
            y_train=y_train,
            y_test=y_test,
            train_test_split_index=train_test_split_index,
            num_classes=num_classes,
        )

    def _feature_cells(self, raw_state):
        return _forward_common.feature_cells(self, raw_state)

    def _build_table_tokens_from_raw(self, raw_state):
        return _forward_common.build_table_tokens_from_raw(self, raw_state)

    def _build_table_tokens_batched(
        self,
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        *,
        train_test_split_index: int,
    ) -> torch.Tensor:
        return _forward_common.build_table_tokens_batched(
            self,
            x_all,
            y_train,
            train_test_split_index=train_test_split_index,
        )

    def _encode_table_batched(
        self,
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        *,
        train_test_split_index: int,
    ) -> torch.Tensor:
        return _forward_common.encode_table_batched(
            self,
            x_all,
            y_train,
            train_test_split_index=train_test_split_index,
        )

    def _build_table_tokens(self, batch: TaskBatch) -> tuple[torch.Tensor, int]:
        return _forward_common.build_table_tokens(self, batch)

    def _encode_table(self, batch: TaskBatch) -> tuple[torch.Tensor, int]:
        return _forward_common.encode_table(self, batch)

    def _encode_to_cell_state(self, raw_state):
        return _forward_common.encode_to_cell_state(self, raw_state)

    def _pool_rows(self, cell_state):
        return _forward_common.pool_rows(self, cell_state)

    def _condition_rows(self, row_state, *, train_target_embeddings: torch.Tensor | None = None):
        return _forward_common.condition_rows(
            self,
            row_state,
            train_target_embeddings=train_target_embeddings,
        )

    def _context_train_embeddings(self, y_train: torch.Tensor) -> torch.Tensor:
        return _forward_common.context_train_embeddings(self, y_train)

    def _build_direct_head_state(self, raw_state):
        return _direct_head.build_direct_head_state(self, raw_state)

    def forward_batched(
        self,
        *,
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        train_test_split_index: int,
    ) -> torch.Tensor:
        return _direct_head.forward_batched(
            self,
            x_all=x_all,
            y_train=y_train,
            train_test_split_index=train_test_split_index,
        )

    def _node_train_indices(self, *, node, y_train: torch.Tensor) -> torch.Tensor:
        return _many_class.node_train_indices(self, node=node, y_train=y_train)

    def _encode_rows_with_targets(
        self,
        rows: torch.Tensor,
        *,
        train_targets: torch.Tensor,
        train_test_split_index: int,
    ) -> torch.Tensor:
        return _many_class.encode_rows_with_targets(
            self,
            rows,
            train_targets=train_targets,
            train_test_split_index=train_test_split_index,
        )

    def _digit_conditioned_rows(self, row_state, y_train: torch.Tensor) -> torch.Tensor:
        return _many_class.digit_conditioned_rows(self, row_state, y_train)

    def _hierarchical_probs(
        self,
        row_embeddings: torch.Tensor,
        y_train: torch.Tensor,
        tree,
        *,
        n_train: int,
        num_classes: int,
    ):
        return _many_class.hierarchical_probs(
            self,
            row_embeddings,
            y_train,
            tree,
            n_train=n_train,
            num_classes=num_classes,
        )

    def _hierarchical_path_terms(
        self,
        row_embeddings: torch.Tensor,
        y_train: torch.Tensor,
        y_test: torch.Tensor,
        tree,
        *,
        n_train: int,
    ):
        return _many_class.hierarchical_path_terms(
            self,
            row_embeddings,
            y_train,
            y_test,
            tree,
            n_train=n_train,
        )

    def _forward_many_class(self, raw_state) -> ClassificationOutput:
        return _many_class.forward_many_class(self, raw_state)

    def _validate_num_classes(self, num_classes: int) -> None:
        _many_class.validate_num_classes(self, num_classes)

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

        head_state = self._build_direct_head_state(raw_state)
        logits = self.direct_head(head_state.rows[:, train_test_split_index:, :]).squeeze(0)
        return ClassificationOutput(
            logits=logits,
            num_classes=num_classes,
            class_probs=None,
        )
