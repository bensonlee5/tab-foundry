"""Grid-preserving sandwich pilot classifier."""

from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from tab_foundry.feature_types import (
    FEATURE_TYPE_VOCAB,
    feature_type_ids_from_resolved,
    feature_type_ids_from_task_metadata,
    normalize_feature_types,
)
from tab_foundry.model.components.tabular_primitives import (
    DirectMulticlassHead,
    FeatureTypeFiLM,
    LabelTokenTargetConditioner,
    ScalarPerFeatureMissingnessTokenizer,
    SharedLinearFeatureEncoder,
)
from tab_foundry.model.outputs import ClassificationOutput, flatten_classification_output_rows
from tab_foundry.model.spec import GRID_SANDWICH_DEFAULTS as _D, ModelBuildSpec
from tab_foundry.types import TaskBatch

from .. import shared_forward as _shared_forward
from .. import shared_hooks as _shared_hooks
from ..tabfoundry_sandwich import feature_flow as _feature_flow
from ..tabfoundry_sandwich.blocks import (
    _CrossAttentionBlock,
    _InducedSetAttentionBlock,
    _SelfAttentionBlock,
)
from ..tabfoundry_sandwich.states import SandwichFeatureState, SandwichRawInputState


_CLASSIFICATION_LOSS_SURFACE = "classification"
_MIN_CLASS_COUNT = 2


class _GridMixerLayer(nn.Module):
    """Alternate row-wise and column-wise mixing while preserving the cell grid."""

    def __init__(
        self,
        *,
        embedding_size: int,
        n_heads: int,
        ff_expansion: int,
        activation: str,
        block_norm: str,
        num_inducing: int,
        packed_attention: bool = False,
    ) -> None:
        super().__init__()
        self.row_mixer = _SelfAttentionBlock(
            embedding_size=embedding_size,
            n_heads=n_heads,
            ff_expansion=ff_expansion,
            activation=activation,
            block_norm=block_norm,
            packed_attention=packed_attention,
        )
        self.column_mixer = _InducedSetAttentionBlock(
            embedding_size=embedding_size,
            n_heads=n_heads,
            ff_expansion=ff_expansion,
            activation=activation,
            block_norm=block_norm,
            num_inducing=num_inducing,
            packed_attention=packed_attention,
        )


class GridSandwichClassifier(nn.Module):
    """Classification-only grid-preserving sandwich pilot."""

    def __init__(
        self,
        *,
        d_icl: int = _D["d_icl"],
        input_normalization: str = _D["input_normalization"],
        many_class_base: int = _D["many_class_base"],
        norm_type: str = _D["norm_type"],
        head_hidden_dim: int = _D["head_hidden_dim"],
        pre_encoder_clip: float | None = _D["pre_encoder_clip"],
        sandwich_layers: int = _D["sandwich_layers"],
        sandwich_heads: int = _D["sandwich_heads"],
        sandwich_ff_expansion: int = _D["sandwich_ff_expansion"],
        sandwich_activation: str = _D["sandwich_activation"],
        sandwich_block_norm: str = _D["sandwich_block_norm"],
        sandwich_pre_column_inducing_tokens: int = _D["sandwich_pre_column_inducing_tokens"],
        sandwich_packed_attention: bool = _D["sandwich_packed_attention"],
        feature_type_conditioning: str = _D["feature_type_conditioning"],
    ) -> None:
        super().__init__()
        self.model_spec = ModelBuildSpec(
            task="classification",
            arch="grid_sandwich",
            d_icl=d_icl,
            input_normalization=input_normalization,
            many_class_base=many_class_base,
            norm_type=norm_type,
            head_hidden_dim=head_hidden_dim,
            pre_encoder_clip=pre_encoder_clip,
            sandwich_layers=sandwich_layers,
            sandwich_heads=sandwich_heads,
            sandwich_ff_expansion=sandwich_ff_expansion,
            sandwich_activation=sandwich_activation,
            sandwich_block_norm=sandwich_block_norm,
            sandwich_pre_column_inducing_tokens=sandwich_pre_column_inducing_tokens,
            sandwich_packed_attention=sandwich_packed_attention,
            feature_type_conditioning=feature_type_conditioning,
        )
        self.arch = "grid_sandwich"
        self.loss_surface = _CLASSIFICATION_LOSS_SURFACE
        self.d_icl = int(self.model_spec.d_icl)
        self.input_normalization = str(self.model_spec.input_normalization).strip().lower()
        self.many_class_base = int(self.model_spec.many_class_base)
        self.norm_type = str(self.model_spec.norm_type).strip().lower()
        self.head_hidden_dim = int(self.model_spec.head_hidden_dim)
        self.pre_encoder_clip = self.model_spec.pre_encoder_clip
        self.sandwich_layers = int(self.model_spec.sandwich_layers)
        self.sandwich_heads = int(self.model_spec.sandwich_heads)
        self.sandwich_ff_expansion = int(self.model_spec.sandwich_ff_expansion)
        self.sandwich_activation = str(self.model_spec.sandwich_activation).strip().lower()
        self.sandwich_block_norm = str(self.model_spec.sandwich_block_norm).strip().lower()
        self.pre_column_inducing_tokens = int(self.model_spec.sandwich_pre_column_inducing_tokens)
        self.sandwich_packed_attention = bool(self.model_spec.sandwich_packed_attention)
        self.feature_type_conditioning = (
            str(self.model_spec.feature_type_conditioning).strip().lower()
        )
        if self.norm_type != "layernorm":
            raise ValueError(
                "grid_sandwich currently requires norm_type='layernorm', "
                f"got {self.norm_type!r}"
            )

        self.tokenizer = ScalarPerFeatureMissingnessTokenizer()
        self.feature_encoder = SharedLinearFeatureEncoder(
            token_dim=int(self.tokenizer.token_dim),
            embedding_size=self.d_icl,
        )
        self.feature_type_film: FeatureTypeFiLM | None
        self.feature_type_embedding: nn.Embedding | None
        if self.feature_type_conditioning == "film":
            self.feature_type_film = FeatureTypeFiLM(len(FEATURE_TYPE_VOCAB), self.d_icl)
            self.feature_type_embedding = None
        else:
            self.feature_type_film = None
            self.feature_type_embedding = nn.Embedding(len(FEATURE_TYPE_VOCAB), self.d_icl)
        self.pre_row_attention_blocks = nn.ModuleList()
        self.pre_column_attention_blocks = nn.ModuleList()
        self.grid_layers = nn.ModuleList(
            [
                _GridMixerLayer(
                    embedding_size=self.d_icl,
                    n_heads=self.sandwich_heads,
                    ff_expansion=self.sandwich_ff_expansion,
                    activation=self.sandwich_activation,
                    block_norm=self.sandwich_block_norm,
                    num_inducing=self.pre_column_inducing_tokens,
                    packed_attention=self.sandwich_packed_attention,
                )
                for _ in range(self.sandwich_layers)
            ]
        )
        self.y_conditioner = LabelTokenTargetConditioner(self.many_class_base, self.d_icl)
        self.y_role_embedding = nn.Embedding(2, self.d_icl)
        self.row_pool_query = nn.Parameter(torch.randn(1, 1, self.d_icl) * 0.02)
        self.row_pool = _CrossAttentionBlock(
            embedding_size=self.d_icl,
            n_heads=self.sandwich_heads,
            ff_expansion=self.sandwich_ff_expansion,
            activation=self.sandwich_activation,
            block_norm=self.sandwich_block_norm,
            packed_attention=self.sandwich_packed_attention,
        )
        self.direct_head = DirectMulticlassHead(
            self.d_icl,
            self.head_hidden_dim,
            self.many_class_base,
        )

        self._activation_checkpointing_enabled = False
        self._activation_trace: dict[str, tuple[float, int]] | None = None
        self._fourier_position_cache: dict[
            tuple[int, int, torch.device, torch.dtype],
            torch.Tensor,
        ] = {}

    def enable_activation_checkpointing(self) -> None:
        _shared_hooks.enable_activation_checkpointing(self)

    def disable_activation_checkpointing(self) -> None:
        _shared_hooks.disable_activation_checkpointing(self)

    def set_loss_surface(self, loss_surface: str) -> None:
        normalized = str(loss_surface).strip().lower()
        if normalized != _CLASSIFICATION_LOSS_SURFACE:
            raise ValueError(
                "grid_sandwich only supports loss_surface='classification', "
                f"got {loss_surface!r}"
            )
        self.loss_surface = normalized

    def _apply_activation_checkpoint(
        self,
        function,
        *args: torch.Tensor,
    ) -> torch.Tensor:
        return _shared_hooks.apply_activation_checkpoint(self, function, *args)

    def enable_activation_trace(self) -> None:
        _shared_hooks.enable_activation_trace(self)

    def disable_activation_trace(self) -> None:
        _shared_hooks.disable_activation_trace(self)

    def trace_activation(self, name: str, tensor: torch.Tensor) -> None:
        _shared_hooks.trace_activation(self, name, tensor)

    def flush_activation_trace_stats(self) -> dict[str, tuple[float, int]] | None:
        return _shared_hooks.flush_activation_trace_stats(self)

    def flush_activation_trace(self) -> dict[str, float] | None:
        return _shared_hooks.flush_activation_trace(self)

    @staticmethod
    def _task_num_classes(batch: TaskBatch) -> int:
        return _shared_forward.task_num_classes(batch, arch_name="grid_sandwich")

    @staticmethod
    def _prepare_task_inputs(
        batch: TaskBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return _shared_forward.prepare_task_inputs(batch, arch_name="grid_sandwich")

    @staticmethod
    def _validate_batched_inputs(
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        train_test_split_index: int,
    ) -> None:
        _shared_forward.validate_batched_inputs(x_all, y_train, train_test_split_index)

    def _normalize_x_all(self, x_all: torch.Tensor, *, train_test_split_index: int) -> torch.Tensor:
        return _shared_forward.normalize_x_all(
            x_all,
            train_test_split_index=train_test_split_index,
            input_normalization=self.input_normalization,
            preserve_non_finite=True,
        )

    def _build_raw_input_state(
        self,
        *,
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        y_test: torch.Tensor | None,
        train_test_split_index: int,
        num_classes: int,
        feature_type_ids: torch.Tensor,
    ) -> SandwichRawInputState:
        return _feature_flow.build_raw_input_state(
            x_all=x_all,
            y_train=y_train,
            y_test=y_test,
            train_test_split_index=train_test_split_index,
            num_classes=num_classes,
            feature_type_ids=feature_type_ids,
        )

    def _build_feature_state(
        self,
        raw_state: SandwichRawInputState,
        *,
        apply_input_normalization: bool = True,
    ) -> SandwichFeatureState:
        return _feature_flow.build_feature_state(
            self,
            raw_state,
            apply_input_normalization=apply_input_normalization,
        )

    def _fourier_positions(
        self,
        *,
        num_positions: int,
        embedding_size: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        key = (int(num_positions), int(embedding_size), torch.device(device), dtype)
        cached = self._fourier_position_cache.get(key)
        if cached is not None:
            return cached
        positions = _feature_flow.fourier_positions(
            num_positions=num_positions,
            embedding_size=embedding_size,
            device=device,
            dtype=dtype,
        )
        self._fourier_position_cache[key] = positions
        return positions

    @staticmethod
    def _feature_type_ids_from_resolved(
        resolved_types_by_task: list[list[str]],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        return feature_type_ids_from_resolved(
            resolved_types_by_task,
            device=device,
        )

    @staticmethod
    def _normalize_required_feature_types(
        feature_types: Any,
        *,
        expected_count: int,
        context: str,
    ) -> list[str]:
        if feature_types is None:
            raise ValueError(f"{context} is required for grid_sandwich")
        return normalize_feature_types(
            feature_types,
            expected_count=expected_count,
            context=context,
        )

    def _feature_type_ids_from_forward_batched(
        self,
        feature_types: list[str] | list[list[str]] | None,
        *,
        batch_size: int,
        num_features: int,
        device: torch.device,
    ) -> torch.Tensor:
        if feature_types is None:
            raise ValueError("grid_sandwich forward_batched() requires explicit feature_types")
        if not feature_types or isinstance(feature_types[0], str):
            if batch_size != 1:
                raise ValueError(
                    "grid_sandwich forward_batched() requires one feature_types list per task "
                    f"when batch_size={batch_size}"
                )
            resolved = [
                self._normalize_required_feature_types(
                    feature_types,
                    expected_count=num_features,
                    context="forward_batched.feature_types",
                )
            ]
            return self._feature_type_ids_from_resolved(resolved, device=device)
        if not isinstance(feature_types, list) or len(feature_types) != batch_size:
            raise ValueError(
                "grid_sandwich forward_batched() requires one feature_types list per task "
                f"when batch_size={batch_size}, got {type(feature_types).__name__}"
            )
        resolved_types_by_task = [
            self._normalize_required_feature_types(
                value,
                expected_count=num_features,
                context=f"forward_batched.feature_types[{index}]",
            )
            for index, value in enumerate(feature_types)
        ]
        return self._feature_type_ids_from_resolved(
            resolved_types_by_task,
            device=device,
        )

    def _feature_type_ids_from_metadata(
        self,
        metadata: dict[str, Any],
        *,
        batch_size: int,
        num_features: int,
        device: torch.device,
    ) -> torch.Tensor:
        return feature_type_ids_from_task_metadata(
            metadata,
            batch_size=batch_size,
            num_features=num_features,
            device=device,
        )

    def _feature_cells(
        self,
        x_all: torch.Tensor,
        *,
        train_test_split_index: int,
        feature_type_ids: torch.Tensor,
        apply_input_normalization: bool = True,
    ) -> torch.Tensor:
        return _feature_flow.feature_cells(
            self,
            x_all,
            train_test_split_index=train_test_split_index,
            feature_type_ids=feature_type_ids,
            apply_input_normalization=apply_input_normalization,
        )

    def _cross_block(
        self,
        block: _CrossAttentionBlock,
        query: torch.Tensor,
        key_value: torch.Tensor,
    ) -> torch.Tensor:
        def _apply(current_query: torch.Tensor, current_kv: torch.Tensor) -> torch.Tensor:
            return block(current_query, key_value=current_kv)

        return self._apply_activation_checkpoint(_apply, query, key_value)

    def _self_block(
        self,
        block: _SelfAttentionBlock,
        hidden: torch.Tensor,
        *,
        attn_bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        def _apply(current_hidden: torch.Tensor) -> torch.Tensor:
            return block(current_hidden, attn_bias=attn_bias)

        return self._apply_activation_checkpoint(_apply, hidden)

    def _row_feature_self_attention(
        self,
        block: _SelfAttentionBlock,
        feature_cells: torch.Tensor,
    ) -> torch.Tensor:
        return _feature_flow.row_feature_self_attention(self, block, feature_cells)

    def _column_row_isab(
        self,
        block: _InducedSetAttentionBlock,
        feature_cells: torch.Tensor,
    ) -> torch.Tensor:
        return _feature_flow.column_row_isab(self, block, feature_cells)

    def _pre_perceiver_cell_mixer(self, feature_cells: torch.Tensor) -> torch.Tensor:
        return _feature_flow.pre_perceiver_cell_mixer(self, feature_cells)

    def _validate_num_classes(self, num_classes: int) -> None:
        if num_classes < _MIN_CLASS_COUNT:
            raise RuntimeError(f"grid_sandwich requires at least 2 classes, got {num_classes}")
        if num_classes > self.many_class_base:
            raise RuntimeError(
                "grid_sandwich uses a direct multiclass head and requires "
                f"num_classes <= many_class_base={self.many_class_base}, got {num_classes}"
            )

    def _label_conditioned_cells(
        self,
        feature_cells: torch.Tensor,
        *,
        y_train: torch.Tensor,
    ) -> torch.Tensor:
        num_rows = int(feature_cells.shape[1])
        conditioned = self.y_conditioner(y_train, num_rows=num_rows).squeeze(2).to(
            dtype=feature_cells.dtype
        )
        conditioned[:, int(y_train.shape[1]) :, :] = 0.0
        role_ids = _feature_flow.role_ids(
            batch_size=int(feature_cells.shape[0]),
            num_rows=num_rows,
            num_train_rows=int(y_train.shape[1]),
            device=feature_cells.device,
        )
        role_embed = self.y_role_embedding(role_ids).to(dtype=feature_cells.dtype)
        conditioned_cells = feature_cells + conditioned.unsqueeze(2) + role_embed.unsqueeze(2)
        self.trace_activation("post_label_conditioned_cells", conditioned_cells)
        return conditioned_cells

    def _pool_test_rows(self, feature_cells: torch.Tensor, *, train_test_split_index: int) -> torch.Tensor:
        test_rows = feature_cells[:, train_test_split_index:, :, :]
        batch_size, num_test_rows, num_features, embedding_size = (
            int(test_rows.shape[0]),
            int(test_rows.shape[1]),
            int(test_rows.shape[2]),
            int(test_rows.shape[3]),
        )
        flat_rows = test_rows.reshape(batch_size * num_test_rows, num_features, embedding_size)
        row_query = self.row_pool_query.expand(batch_size * num_test_rows, -1, -1).to(
            device=test_rows.device,
            dtype=test_rows.dtype,
        )
        pooled = self._cross_block(self.row_pool, row_query, flat_rows)
        pooled = pooled.reshape(batch_size, num_test_rows, embedding_size)
        self.trace_activation("post_test_row_pool", pooled)
        return pooled

    def _forward_logits_batched(
        self,
        *,
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        train_test_split_index: int,
        feature_type_ids: torch.Tensor,
        num_classes: int | None = None,
    ) -> torch.Tensor:
        self._validate_batched_inputs(x_all, y_train, train_test_split_index)
        resolved_num_classes = int(num_classes) if num_classes is not None else 2
        if num_classes is None:
            resolved_num_classes = max(2, int(y_train.max().item()) + 1)
        self._validate_num_classes(resolved_num_classes)
        raw_state = self._build_raw_input_state(
            x_all=x_all,
            y_train=y_train,
            y_test=None,
            train_test_split_index=train_test_split_index,
            num_classes=resolved_num_classes,
            feature_type_ids=feature_type_ids,
        )
        feature_state = self._build_feature_state(raw_state)
        hidden = self._label_conditioned_cells(feature_state.feature_cells, y_train=y_train)
        for index, layer in enumerate(self.grid_layers):
            layer = cast(_GridMixerLayer, layer)
            hidden = self._row_feature_self_attention(layer.row_mixer, hidden)
            self.trace_activation(f"post_grid_row_mixer_{index}", hidden)
            hidden = self._column_row_isab(layer.column_mixer, hidden)
            self.trace_activation(f"post_grid_column_mixer_{index}", hidden)
        pooled_test_rows = self._pool_test_rows(
            hidden,
            train_test_split_index=train_test_split_index,
        )
        return self.direct_head(pooled_test_rows)

    def forward_batched(
        self,
        *,
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        train_test_split_index: int,
        feature_types: list[str] | list[list[str]],
    ) -> torch.Tensor:
        feature_type_ids = self._feature_type_ids_from_forward_batched(
            feature_types,
            batch_size=int(x_all.shape[0]),
            num_features=int(x_all.shape[2]),
            device=x_all.device,
        )
        return self._forward_logits_batched(
            x_all=x_all,
            y_train=y_train,
            train_test_split_index=train_test_split_index,
            feature_type_ids=feature_type_ids,
        )

    def forward_classification(self, batch: TaskBatch) -> ClassificationOutput:
        num_classes = self._task_num_classes(batch)
        self._validate_num_classes(num_classes)
        x_all, y_train, _y_test, train_test_split_index = self._prepare_task_inputs(batch)
        feature_type_ids = batch.feature_type_ids
        if feature_type_ids is None:
            feature_type_ids = self._feature_type_ids_from_metadata(
                batch.metadata,
                batch_size=int(x_all.shape[0]),
                num_features=int(x_all.shape[2]),
                device=x_all.device,
            )
        logits = self._forward_logits_batched(
            x_all=x_all,
            y_train=y_train,
            train_test_split_index=train_test_split_index,
            feature_type_ids=feature_type_ids,
            num_classes=num_classes,
        )
        return ClassificationOutput(
            logits=flatten_classification_output_rows(logits),
            class_probs=None,
            num_classes=num_classes,
        )

    def forward(self, batch: TaskBatch) -> ClassificationOutput:
        return self.forward_classification(batch)
