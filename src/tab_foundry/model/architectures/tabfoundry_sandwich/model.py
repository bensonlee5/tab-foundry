"""Hybrid full-cell / summary-stream Perceiver-style sandwich classifier."""

from __future__ import annotations

from typing import Any

import torch
from torch import nn

from tab_foundry.feature_types import FEATURE_TYPE_VOCAB
from tab_foundry.model.components.tabular_primitives import (
    DirectMulticlassHead,
    FeatureTypeFiLM,
    LabelTokenTargetConditioner,
    ScalarPerFeatureMissingnessTokenizer,
    SharedLinearFeatureEncoder,
    SharedMlpFeatureEncoder,
)
from tab_foundry.model.outputs import CellLikelihoodOutput, ClassificationOutput
from tab_foundry.model.spec import SANDWICH_DEFAULTS as _D, ModelBuildSpec
from tab_foundry.types import TaskBatch

from .. import shared_forward as _shared_forward
from .. import shared_hooks as _shared_hooks
from . import cell_likelihood as _cell_likelihood
from . import classification_flow as _classification_flow
from . import feature_flow as _feature_flow
from .blocks import (
    _CrossAttentionBlock,
    _InducedSetAttentionBlock,
    _PerceiverStage,
    _SelfAttentionBlock,
    _init_truncated_normal_,
)
from .states import SandwichFeatureState, SandwichRawInputState


_CLASSIFICATION_LOSS_SURFACE = "classification"
_CELL_BPC_LOSS_SURFACE = "cell_bpc"


class TabFoundrySandwichClassifier(nn.Module):
    """Small-class hybrid full-cell / summary-stream Perceiver-style classifier."""

    def __init__(
        self,
        *,
        d_icl: int = _D["d_icl"],
        input_normalization: str = _D["input_normalization"],
        many_class_base: int = _D["many_class_base"],
        norm_type: str = _D["norm_type"],
        head_hidden_dim: int = _D["head_hidden_dim"],
        pre_encoder_clip: float | None = _D["pre_encoder_clip"],
        sandwich_latents: int = _D["sandwich_latents"],
        sandwich_layers: int = _D["sandwich_layers"],
        sandwich_heads: int = _D["sandwich_heads"],
        sandwich_ff_expansion: int = _D["sandwich_ff_expansion"],
        sandwich_activation: str = _D["sandwich_activation"],
        sandwich_block_norm: str = _D["sandwich_block_norm"],
        sandwich_summary_tokens_per_axis: int = _D["sandwich_summary_tokens_per_axis"],
        sandwich_self_attention_per_cross: int = _D["sandwich_self_attention_per_cross"],
        sandwich_pre_row_attention_layers: int = _D["sandwich_pre_row_attention_layers"],
        sandwich_pre_column_attention_layers: int = _D["sandwich_pre_column_attention_layers"],
        sandwich_pre_column_inducing_tokens: int = _D["sandwich_pre_column_inducing_tokens"],
        sandwich_packed_attention: bool = _D["sandwich_packed_attention"],
        sandwich_last_stage_full_cell_refresh: bool = _D["sandwich_last_stage_full_cell_refresh"],
        sandwich_column_summary_mode: str = _D["sandwich_column_summary_mode"],
        sandwich_feature_encoder_kind: str = _D["sandwich_feature_encoder_kind"],
        sandwich_use_class_memory: bool = _D["sandwich_use_class_memory"],
        feature_type_conditioning: str = _D["feature_type_conditioning"],
        floating_likelihood: str = _D["floating_likelihood"],
        integer_likelihood: str = _D["integer_likelihood"],
    ) -> None:
        super().__init__()
        self.model_spec = ModelBuildSpec(
            task="classification",
            arch="tabfoundry_sandwich",
            d_icl=d_icl,
            input_normalization=input_normalization,
            many_class_base=many_class_base,
            norm_type=norm_type,
            head_hidden_dim=head_hidden_dim,
            pre_encoder_clip=pre_encoder_clip,
            sandwich_latents=sandwich_latents,
            sandwich_layers=sandwich_layers,
            sandwich_heads=sandwich_heads,
            sandwich_ff_expansion=sandwich_ff_expansion,
            sandwich_activation=sandwich_activation,
            sandwich_block_norm=sandwich_block_norm,
            sandwich_summary_tokens_per_axis=sandwich_summary_tokens_per_axis,
            sandwich_self_attention_per_cross=sandwich_self_attention_per_cross,
            sandwich_pre_row_attention_layers=sandwich_pre_row_attention_layers,
            sandwich_pre_column_attention_layers=sandwich_pre_column_attention_layers,
            sandwich_pre_column_inducing_tokens=sandwich_pre_column_inducing_tokens,
            sandwich_packed_attention=sandwich_packed_attention,
            sandwich_last_stage_full_cell_refresh=sandwich_last_stage_full_cell_refresh,
            sandwich_column_summary_mode=sandwich_column_summary_mode,
            sandwich_feature_encoder_kind=sandwich_feature_encoder_kind,
            sandwich_use_class_memory=sandwich_use_class_memory,
            feature_type_conditioning=feature_type_conditioning,
            floating_likelihood=floating_likelihood,
            integer_likelihood=integer_likelihood,
        )
        self.arch = "tabfoundry_sandwich"
        self.loss_surface = _CLASSIFICATION_LOSS_SURFACE
        self.d_icl = int(self.model_spec.d_icl)
        self.input_normalization = str(self.model_spec.input_normalization).strip().lower()
        self.many_class_base = int(self.model_spec.many_class_base)
        self.norm_type = str(self.model_spec.norm_type).strip().lower()
        self.head_hidden_dim = int(self.model_spec.head_hidden_dim)
        self.pre_encoder_clip = self.model_spec.pre_encoder_clip
        self.sandwich_latents = int(self.model_spec.sandwich_latents)
        self.sandwich_layers = int(self.model_spec.sandwich_layers)
        self.sandwich_heads = int(self.model_spec.sandwich_heads)
        self.sandwich_ff_expansion = int(self.model_spec.sandwich_ff_expansion)
        self.sandwich_activation = str(self.model_spec.sandwich_activation).strip().lower()
        self.sandwich_block_norm = str(self.model_spec.sandwich_block_norm).strip().lower()
        self.summary_tokens_per_axis = int(self.model_spec.sandwich_summary_tokens_per_axis)
        self.self_attention_per_cross = int(self.model_spec.sandwich_self_attention_per_cross)
        self.pre_row_attention_layers = int(self.model_spec.sandwich_pre_row_attention_layers)
        self.pre_column_attention_layers = int(self.model_spec.sandwich_pre_column_attention_layers)
        self.pre_column_inducing_tokens = int(self.model_spec.sandwich_pre_column_inducing_tokens)
        self.sandwich_packed_attention = bool(self.model_spec.sandwich_packed_attention)
        self.sandwich_last_stage_full_cell_refresh = bool(
            self.model_spec.sandwich_last_stage_full_cell_refresh
        )
        self.sandwich_column_summary_mode = (
            str(self.model_spec.sandwich_column_summary_mode).strip().lower()
        )
        self.sandwich_feature_encoder_kind = (
            str(self.model_spec.sandwich_feature_encoder_kind).strip().lower()
        )
        self.sandwich_use_class_memory = bool(self.model_spec.sandwich_use_class_memory)
        self.feature_type_conditioning = (
            str(self.model_spec.feature_type_conditioning).strip().lower()
        )
        self.floating_likelihood = str(self.model_spec.floating_likelihood).strip().lower()
        self.integer_likelihood = str(self.model_spec.integer_likelihood).strip().lower()
        if self.norm_type != "layernorm":
            raise ValueError(
                "tabfoundry_sandwich currently requires norm_type='layernorm', "
                f"got {self.norm_type!r}"
            )
        self.tokenizer = ScalarPerFeatureMissingnessTokenizer()
        self.feature_encoder: nn.Module
        if self.sandwich_feature_encoder_kind == "linear":
            self.feature_encoder = SharedLinearFeatureEncoder(
                token_dim=int(self.tokenizer.token_dim),
                embedding_size=self.d_icl,
            )
        else:
            self.feature_encoder = SharedMlpFeatureEncoder(
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
        self.row_summary_query = nn.Parameter(
            torch.randn(1, self.summary_tokens_per_axis, self.d_icl) * 0.02
        )
        self.column_summary_query = nn.Parameter(
            torch.randn(1, self.summary_tokens_per_axis, self.d_icl) * 0.02
        )
        self.test_row_pool_query = nn.Parameter(torch.randn(1, 1, self.d_icl) * 0.02)
        self.row_summary_builder = _CrossAttentionBlock(
            embedding_size=self.d_icl,
            n_heads=self.sandwich_heads,
            ff_expansion=self.sandwich_ff_expansion,
            activation=self.sandwich_activation,
            block_norm=self.sandwich_block_norm,
            packed_attention=self.sandwich_packed_attention,
        )
        self.column_summary_builder = _CrossAttentionBlock(
            embedding_size=self.d_icl,
            n_heads=self.sandwich_heads,
            ff_expansion=self.sandwich_ff_expansion,
            activation=self.sandwich_activation,
            block_norm=self.sandwich_block_norm,
            packed_attention=self.sandwich_packed_attention,
        )
        self.pre_row_attention_blocks = nn.ModuleList(
            [
                _SelfAttentionBlock(
                    embedding_size=self.d_icl,
                    n_heads=self.sandwich_heads,
                    ff_expansion=self.sandwich_ff_expansion,
                    activation=self.sandwich_activation,
                    block_norm=self.sandwich_block_norm,
                    packed_attention=self.sandwich_packed_attention,
                )
                for _ in range(self.pre_row_attention_layers)
            ]
        )
        self.pre_column_attention_blocks = nn.ModuleList(
            [
                _InducedSetAttentionBlock(
                    embedding_size=self.d_icl,
                    n_heads=self.sandwich_heads,
                    ff_expansion=self.sandwich_ff_expansion,
                    activation=self.sandwich_activation,
                    block_norm=self.sandwich_block_norm,
                    num_inducing=self.pre_column_inducing_tokens,
                    packed_attention=self.sandwich_packed_attention,
                )
                for _ in range(self.pre_column_attention_layers)
            ]
        )
        self.token_type_embedding = nn.Embedding(3, self.d_icl)
        self.y_conditioner = LabelTokenTargetConditioner(self.many_class_base, self.d_icl)
        self.y_role_embedding = nn.Embedding(2, self.d_icl)
        self.latent_seed = nn.Parameter(torch.empty(1, self.sandwich_latents, self.d_icl))
        _init_truncated_normal_(self.latent_seed, mean=0.0, std=0.02, a=-2.0, b=2.0)
        self.perceiver_stages = nn.ModuleList(
            [
                _PerceiverStage(
                    embedding_size=self.d_icl,
                    n_heads=self.sandwich_heads,
                    ff_expansion=self.sandwich_ff_expansion,
                    activation=self.sandwich_activation,
                    block_norm=self.sandwich_block_norm,
                    self_attention_per_cross=self.self_attention_per_cross,
                    packed_attention=self.sandwich_packed_attention,
                )
                for _ in range(self.sandwich_layers)
            ]
        )
        self.latent_readout = _CrossAttentionBlock(
            embedding_size=self.d_icl,
            n_heads=self.sandwich_heads,
            ff_expansion=self.sandwich_ff_expansion,
            activation=self.sandwich_activation,
            block_norm=self.sandwich_block_norm,
            packed_attention=self.sandwich_packed_attention,
        )
        self.cell_readout = _CrossAttentionBlock(
            embedding_size=self.d_icl,
            n_heads=self.sandwich_heads,
            ff_expansion=self.sandwich_ff_expansion,
            activation=self.sandwich_activation,
            block_norm=self.sandwich_block_norm,
            packed_attention=self.sandwich_packed_attention,
        )
        self.test_row_pool = _CrossAttentionBlock(
            embedding_size=self.d_icl,
            n_heads=self.sandwich_heads,
            ff_expansion=self.sandwich_ff_expansion,
            activation=self.sandwich_activation,
            block_norm=self.sandwich_block_norm,
            packed_attention=self.sandwich_packed_attention,
        )
        self.class_memory_query: nn.Parameter | None
        self.class_memory_builder: _CrossAttentionBlock | None
        self.class_memory_readout: _CrossAttentionBlock | None
        if self.sandwich_use_class_memory:
            self.class_memory_query = nn.Parameter(
                torch.randn(1, self.many_class_base, self.d_icl) * 0.02
            )
            self.class_memory_builder = _CrossAttentionBlock(
                embedding_size=self.d_icl,
                n_heads=self.sandwich_heads,
                ff_expansion=self.sandwich_ff_expansion,
                activation=self.sandwich_activation,
                block_norm=self.sandwich_block_norm,
                packed_attention=self.sandwich_packed_attention,
            )
            self.class_memory_readout = _CrossAttentionBlock(
                embedding_size=self.d_icl,
                n_heads=self.sandwich_heads,
                ff_expansion=self.sandwich_ff_expansion,
                activation=self.sandwich_activation,
                block_norm=self.sandwich_block_norm,
                packed_attention=self.sandwich_packed_attention,
            )
        else:
            self.class_memory_query = None
            self.class_memory_builder = None
            self.class_memory_readout = None
        self.direct_head = DirectMulticlassHead(
            self.d_icl,
            self.head_hidden_dim,
            self.many_class_base,
        )
        self.cell_bos = nn.Parameter(torch.randn(1, 1, self.d_icl) * 0.02)
        self.cell_decoder_blocks = nn.ModuleList(
            [
                _SelfAttentionBlock(
                    embedding_size=self.d_icl,
                    n_heads=self.sandwich_heads,
                    ff_expansion=self.sandwich_ff_expansion,
                    activation=self.sandwich_activation,
                    block_norm=self.sandwich_block_norm,
                    packed_attention=self.sandwich_packed_attention,
                )
                for _ in range(self.sandwich_layers)
            ]
        )
        self.gaussian_head = nn.Sequential(
            nn.Linear(self.d_icl, self.head_hidden_dim),
            nn.GELU(),
            nn.Linear(self.head_hidden_dim, 2),
        )
        self.discrete_query = nn.Linear(self.d_icl, self.d_icl)
        self.discrete_oov = nn.Linear(self.d_icl, 1)
        self.integer_gate = nn.Sequential(
            nn.Linear(self.d_icl, self.head_hidden_dim),
            nn.GELU(),
            nn.Linear(self.head_hidden_dim, 1),
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
        if normalized not in {_CLASSIFICATION_LOSS_SURFACE, _CELL_BPC_LOSS_SURFACE}:
            raise ValueError(f"unsupported sandwich loss_surface={loss_surface!r}")
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
        return _shared_forward.task_num_classes(batch, arch_name="tabfoundry_sandwich")

    @staticmethod
    def _prepare_task_inputs(
        batch: TaskBatch,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        return _shared_forward.prepare_task_inputs(batch, arch_name="tabfoundry_sandwich")

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
    def _normalize_required_feature_types(
        feature_types: Any,
        *,
        expected_count: int,
        context: str,
    ) -> list[str]:
        return _feature_flow.normalize_required_feature_types(
            feature_types,
            expected_count=expected_count,
            context=context,
        )

    @staticmethod
    def _feature_type_ids_from_resolved(
        resolved_types_by_task: list[list[str]],
        *,
        device: torch.device,
    ) -> torch.Tensor:
        return _feature_flow.feature_type_ids_from_resolved(
            resolved_types_by_task,
            device=device,
        )

    def _feature_type_ids_from_forward_batched(
        self,
        feature_types: list[str] | list[list[str]] | None,
        *,
        batch_size: int,
        num_features: int,
        device: torch.device,
    ) -> torch.Tensor:
        return _feature_flow.feature_type_ids_from_forward_batched(
            feature_types,
            batch_size=batch_size,
            num_features=num_features,
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
        return _feature_flow.feature_type_ids_from_metadata(
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

    @staticmethod
    def _role_ids(
        *,
        batch_size: int,
        num_rows: int,
        num_train_rows: int,
        device: torch.device,
    ) -> torch.Tensor:
        return _feature_flow.role_ids(
            batch_size=batch_size,
            num_rows=num_rows,
            num_train_rows=num_train_rows,
            device=device,
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

    def _summary_query_attention(
        self,
        block: _CrossAttentionBlock,
        *,
        query: torch.Tensor,
        key_value: torch.Tensor,
        outer_count: int,
    ) -> torch.Tensor:
        return _classification_flow.summary_query_attention(
            self,
            block,
            query=query,
            key_value=key_value,
            outer_count=outer_count,
        )

    def _pool_test_rows(
        self,
        *,
        latent_readout_rows: torch.Tensor,
        cell_readout_rows: torch.Tensor,
    ) -> torch.Tensor:
        return _classification_flow.pool_test_rows(
            self,
            latent_readout_rows=latent_readout_rows,
            cell_readout_rows=cell_readout_rows,
        )

    def _row_summary_bytes(self, feature_cells: torch.Tensor) -> torch.Tensor:
        return _classification_flow.row_summary_bytes(self, feature_cells)

    def _column_summary_bytes(self, feature_cells: torch.Tensor) -> torch.Tensor:
        return _classification_flow.column_summary_bytes(self, feature_cells)

    def _row_summary_tokens(
        self,
        *,
        feature_cells: torch.Tensor,
        y_train: torch.Tensor,
    ) -> torch.Tensor:
        return _classification_flow.row_summary_tokens(
            self,
            feature_cells=feature_cells,
            y_train=y_train,
        )

    def _column_summary_tokens(self, feature_cells: torch.Tensor) -> torch.Tensor:
        return _classification_flow.column_summary_tokens(self, feature_cells)

    def _full_cell_tokens(
        self,
        feature_cells: torch.Tensor,
        *,
        y_train: torch.Tensor,
    ) -> torch.Tensor:
        return _classification_flow.full_cell_tokens(
            self,
            feature_cells,
            y_train=y_train,
        )

    def _summary_input_tokens(
        self,
        feature_cells: torch.Tensor,
        *,
        y_train: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return _classification_flow.summary_input_tokens(
            self,
            feature_cells,
            y_train=y_train,
        )

    def _validate_num_classes(self, num_classes: int) -> None:
        _classification_flow.validate_num_classes(self, num_classes)

    def _cell_decoder_hidden(
        self,
        *,
        feature_cells: torch.Tensor,
        y_train: torch.Tensor,
    ) -> torch.Tensor:
        return _cell_likelihood.cell_decoder_hidden(
            self,
            feature_cells=feature_cells,
            y_train=y_train,
        )

    def _support_values_for_feature(
        self,
        train_values: torch.Tensor,
        *,
        feature_type: str,
    ) -> torch.Tensor:
        return _cell_likelihood.support_values_for_feature(
            train_values,
            feature_type=feature_type,
        )

    def _support_embeddings(
        self,
        support_values: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return _cell_likelihood.support_embeddings(
            self,
            support_values,
            device=device,
            dtype=dtype,
        )

    def _dynamic_discrete_logits(
        self,
        feature_hidden: torch.Tensor,
        *,
        support_embeddings: torch.Tensor,
    ) -> torch.Tensor:
        return _cell_likelihood.dynamic_discrete_logits(
            self,
            feature_hidden,
            support_embeddings=support_embeddings,
        )

    @staticmethod
    def _discrete_target_indices(
        values: torch.Tensor,
        *,
        support_values: torch.Tensor,
    ) -> torch.Tensor:
        return _cell_likelihood.discrete_target_indices(
            values,
            support_values=support_values,
        )

    def _integer_gate_logit(
        self,
        support_embeddings: torch.Tensor,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        return _cell_likelihood.integer_gate_logit(
            self,
            support_embeddings,
            device=device,
            dtype=dtype,
        )

    def _gaussian_params(self, feature_hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _cell_likelihood.gaussian_params(self, feature_hidden)

    @staticmethod
    def _feature_mean_bits(per_cell_bits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return _cell_likelihood.feature_mean_bits(per_cell_bits)

    def _forward_cell_likelihood_batched(
        self,
        *,
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        y_test: torch.Tensor | None,
        train_test_split_index: int,
        feature_type_ids: torch.Tensor,
        num_classes: int | None = None,
    ) -> CellLikelihoodOutput:
        self._validate_batched_inputs(x_all, y_train, train_test_split_index)
        resolved_num_classes = int(num_classes) if num_classes is not None else 2
        if num_classes is None:
            max_label = int(y_train.max().item()) if int(y_train.numel()) > 0 else 1
            if y_test is not None and int(y_test.numel()) > 0:
                max_label = max(max_label, int(y_test.max().item()))
            resolved_num_classes = max(2, max_label + 1)
        self._validate_num_classes(resolved_num_classes)
        raw_state = self._build_raw_input_state(
            x_all=x_all,
            y_train=y_train,
            y_test=y_test,
            train_test_split_index=train_test_split_index,
            num_classes=resolved_num_classes,
            feature_type_ids=feature_type_ids,
        )
        feature_state = self._build_feature_state(raw_state)
        return _cell_likelihood.forward_cell_likelihood(self, feature_state)

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
            if self.sandwich_use_class_memory:
                # forward_batched() does not carry batch.num_classes, so keep the
                # class-memory width aligned with the fixed direct-head surface
                # unless the caller explicitly provides a task class count.
                resolved_num_classes = max(resolved_num_classes, self.many_class_base)
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
        classification_state = _classification_flow.build_classification_state(self, feature_state)
        return _classification_flow.forward_logits(self, classification_state)

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

    def forward_batched_cell_likelihood(
        self,
        *,
        x_all: torch.Tensor,
        y_train: torch.Tensor,
        y_test: torch.Tensor | None = None,
        train_test_split_index: int,
        feature_types: list[str] | list[list[str]],
    ) -> CellLikelihoodOutput:
        feature_type_ids = self._feature_type_ids_from_forward_batched(
            feature_types,
            batch_size=int(x_all.shape[0]),
            num_features=int(x_all.shape[2]),
            device=x_all.device,
        )
        return self._forward_cell_likelihood_batched(
            x_all=x_all,
            y_train=y_train,
            y_test=y_test,
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
        return _classification_flow.build_classification_output(
            logits=logits,
            num_classes=num_classes,
        )

    def forward_cell_likelihood(self, batch: TaskBatch) -> CellLikelihoodOutput:
        num_classes = self._task_num_classes(batch)
        self._validate_num_classes(num_classes)
        x_all, y_train, y_test, train_test_split_index = self._prepare_task_inputs(batch)
        feature_type_ids = batch.feature_type_ids
        if feature_type_ids is None:
            feature_type_ids = self._feature_type_ids_from_metadata(
                batch.metadata,
                batch_size=int(x_all.shape[0]),
                num_features=int(x_all.shape[2]),
                device=x_all.device,
            )
        return self._forward_cell_likelihood_batched(
            x_all=x_all,
            y_train=y_train,
            y_test=y_test,
            train_test_split_index=train_test_split_index,
            feature_type_ids=feature_type_ids,
            num_classes=num_classes,
        )

    def forward(self, batch: TaskBatch) -> ClassificationOutput | CellLikelihoodOutput:
        if self.loss_surface == _CELL_BPC_LOSS_SURFACE:
            return self.forward_cell_likelihood(batch)
        return self.forward_classification(batch)
