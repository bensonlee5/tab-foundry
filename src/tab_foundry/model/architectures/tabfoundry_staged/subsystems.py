"""Family-local subsystem variants for the staged tabfoundry family."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.transformer import Linear, MultiheadAttention

from tab_foundry.model.architectures.tabfoundry_simple import (
    _Decoder as NanoDecoder,
    _FeatureEncoder as NanoFeatureEncoder,
    _TargetEncoder as NanoTargetEncoder,
)
from tab_foundry.model.components.normalization import build_norm
from tab_foundry.model.components.blocks import TFColEncoder, TFRowEncoder
from tab_foundry.model.components.qass import QASSTransformerEncoder


class ScalarPerFeatureTokenizer(nn.Module):
    """One scalar token per feature."""

    token_dim = 1

    def forward(self, x_all: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        return x_all.unsqueeze(-1), None


class ShiftedGroupedTokenizer(nn.Module):
    """Shifted feature tokenizer using the shared (0, 1, 3) offsets."""

    token_dim = 3

    def __init__(self) -> None:
        super().__init__()
        self.group_shifts = (0, 1, 3)

    def forward(self, x_all: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
        n_features = int(x_all.shape[-1])
        base_idx = torch.arange(n_features, device=x_all.device)
        shifted = [
            x_all.index_select(-1, base_idx.roll(-shift)) for shift in self.group_shifts
        ]
        return torch.stack(shifted, dim=-1), None


class SharedLinearFeatureEncoder(nn.Module):
    """Linear feature encoder for pre-tokenized feature vectors."""

    def __init__(self, token_dim: int, embedding_size: int) -> None:
        super().__init__()
        self.linear = nn.Linear(token_dim, embedding_size)

    def forward(self, tokenized_x: torch.Tensor) -> torch.Tensor:
        return self.linear(tokenized_x)


class PostEncoderNorm(nn.Module):
    """Optional normalization applied to the cell table before the transformer."""

    def __init__(self, embedding_size: int, norm_type: str = "layernorm") -> None:
        super().__init__()
        self.norm = build_norm(norm_type, embedding_size)

    def forward(self, cells: torch.Tensor) -> torch.Tensor:
        return self.norm(cells)


class MeanPaddedLinearTargetConditioner(nn.Module):
    """Exact nanoTabPFN mean-padded linear target path."""

    def __init__(self, embedding_size: int) -> None:
        super().__init__()
        self.encoder = NanoTargetEncoder(embedding_size)

    def forward(self, y_train: torch.Tensor, *, num_rows: int) -> torch.Tensor:
        return self.encoder(y_train.to(torch.float32), num_rows)


class LabelTokenTargetConditioner(nn.Module):
    """Train-label embeddings plus a learned test token."""

    def __init__(self, num_embeddings: int, embedding_size: int) -> None:
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_size)
        self.test_token = nn.Parameter(torch.randn(1, 1, embedding_size) * 0.02)

    def forward(self, y_train: torch.Tensor, *, num_rows: int) -> torch.Tensor:
        y_train_i64 = y_train.to(torch.int64).clamp(max=self.embedding.num_embeddings - 1)
        train_embed = self.embedding(y_train_i64)
        n_train = int(train_embed.shape[1])
        n_test = num_rows - n_train
        test_embed = self.test_token.expand(int(train_embed.shape[0]), n_test, -1)
        return torch.cat([train_embed, test_embed], dim=1).unsqueeze(2)


class ContinuousValueTargetConditioner(nn.Module):
    """Continuous-value train targets plus a learned test token."""

    def __init__(self, embedding_size: int) -> None:
        super().__init__()
        self.project = nn.Linear(1, embedding_size)
        self.test_token = nn.Parameter(torch.randn(1, 1, embedding_size) * 0.02)

    def forward(self, y_train: torch.Tensor, *, num_rows: int) -> torch.Tensor:
        if y_train.ndim == 2:
            y_train = y_train.unsqueeze(-1)
        train_embed = self.project(y_train.to(torch.float32))
        n_train = int(train_embed.shape[1])
        n_test = num_rows - n_train
        test_embed = self.test_token.expand(int(train_embed.shape[0]), n_test, -1)
        return torch.cat([train_embed, test_embed], dim=1).unsqueeze(2)


class ContinuousValueContextProjector(nn.Module):
    """Project continuous train targets into context embeddings."""

    def __init__(self, embedding_size: int) -> None:
        super().__init__()
        self.project = nn.Linear(1, embedding_size)

    def forward(self, y_train: torch.Tensor) -> torch.Tensor:
        if y_train.ndim == 1:
            y_train = y_train.unsqueeze(0)
        return self.project(y_train.to(torch.float32).unsqueeze(-1))


class NanoPostNormBlock(nn.Module):
    """Exact nanoTabPFN post-norm block."""

    def __init__(
        self,
        embedding_size: int,
        nhead: int,
        mlp_hidden_size: int,
        *,
        norm_type: str,
    ) -> None:
        super().__init__()
        from tab_foundry.model.architectures.tabfoundry_simple import (
            _TransformerEncoderLayer,
        )

        self.block = _TransformerEncoderLayer(
            embedding_size=embedding_size,
            nhead=nhead,
            mlp_hidden_size=mlp_hidden_size,
            norm_type=norm_type,
        )

    def forward(self, cells: torch.Tensor, *, train_test_split_index: int) -> torch.Tensor:
        return self.block(cells, train_test_split_index=train_test_split_index)


class PreNormCellBlock(nn.Module):
    """Pre-norm feature/row attention block over the cell tensor."""

    def __init__(
        self,
        embedding_size: int,
        nhead: int,
        mlp_hidden_size: int,
        *,
        allow_test_self_attention: bool,
        norm_type: str,
    ) -> None:
        super().__init__()
        self.allow_test_self_attention = allow_test_self_attention
        self.self_attention_between_features = MultiheadAttention(
            embedding_size,
            nhead,
            batch_first=True,
        )
        self.self_attention_between_datapoints = MultiheadAttention(
            embedding_size,
            nhead,
            batch_first=True,
        )
        self.feature_norm = build_norm(norm_type, embedding_size)
        self.row_norm = build_norm(norm_type, embedding_size)
        self.ff_norm = build_norm(norm_type, embedding_size)
        self.linear1 = Linear(embedding_size, mlp_hidden_size)
        self.linear2 = Linear(mlp_hidden_size, embedding_size)

    def _row_attention_mask(
        self,
        *,
        n_total: int,
        n_train: int,
        device: torch.device,
    ) -> torch.Tensor:
        mask = torch.zeros((n_total, n_total), device=device, dtype=torch.float32)
        if n_train >= n_total:
            return mask
        test_slice = slice(n_train, n_total)
        mask[test_slice, test_slice] = float("-inf")
        if self.allow_test_self_attention:
            diag = torch.arange(n_train, n_total, device=device)
            mask[diag, diag] = 0.0
        return mask

    def forward(self, cells: torch.Tensor, *, train_test_split_index: int) -> torch.Tensor:
        batch_size, rows_size, col_size, embedding_size = cells.shape
        feat_in = cells.reshape(batch_size * rows_size, col_size, embedding_size)
        feat_norm = self.feature_norm(feat_in)
        feat_out = self.self_attention_between_features(feat_norm, feat_norm, feat_norm)[0]
        cells = (feat_in + feat_out).reshape(batch_size, rows_size, col_size, embedding_size)

        row_in = cells.transpose(1, 2).reshape(batch_size * col_size, rows_size, embedding_size)
        row_norm = self.row_norm(row_in)
        row_mask = self._row_attention_mask(
            n_total=rows_size,
            n_train=train_test_split_index,
            device=cells.device,
        )
        row_out = self.self_attention_between_datapoints(
            row_norm,
            row_norm,
            row_norm,
            attn_mask=row_mask,
        )[0]
        row_residual = row_in + row_out
        cells = row_residual.reshape(batch_size, col_size, rows_size, embedding_size).transpose(2, 1)

        ff_norm = self.ff_norm(cells)
        return self.linear2(F.gelu(self.linear1(ff_norm))) + cells


class IdentityColumnEncoder(nn.Module):
    """No-op column encoder."""

    def forward(self, cells: torch.Tensor) -> torch.Tensor:
        return cells


class SetColumnEncoder(nn.Module):
    """Shared TFCol / ISAB set encoder over columns."""

    def __init__(
        self,
        *,
        embedding_size: int,
        n_heads: int,
        n_layers: int,
        n_inducing: int,
        norm_type: str,
    ) -> None:
        super().__init__()
        self.encoder = TFColEncoder(
            d_model=embedding_size,
            n_heads=n_heads,
            n_layers=n_layers,
            n_inducing=n_inducing,
            norm_type=norm_type,
        )

    def forward(self, cells: torch.Tensor) -> torch.Tensor:
        batch_size, rows_size, col_size, embedding_size = cells.shape
        flat = cells.permute(0, 2, 1, 3).reshape(batch_size * col_size, rows_size, embedding_size)
        encoded = self.encoder(flat)
        return encoded.reshape(batch_size, col_size, rows_size, embedding_size).permute(0, 2, 1, 3)


class TargetColumnPool(nn.Module):
    """Read row embeddings from the target column."""

    def forward(
        self,
        cells: torch.Tensor,
        *,
        token_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        _ = token_padding_mask
        return cells[:, :, -1, :]


class RowCLSPool(nn.Module):
    """Collapse per-row feature tokens into row embeddings with CLS tokens."""

    def __init__(
        self,
        *,
        embedding_size: int,
        n_heads: int,
        n_layers: int,
        cls_tokens: int,
        norm_type: str,
    ) -> None:
        super().__init__()
        self.encoder = TFRowEncoder(
            d_model=embedding_size,
            n_heads=n_heads,
            n_layers=n_layers,
            cls_tokens=cls_tokens,
            d_out=embedding_size,
            norm_type=norm_type,
        )

    def forward(
        self,
        cells: torch.Tensor,
        *,
        token_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size, rows_size, col_size, embedding_size = cells.shape
        flat = cells.reshape(batch_size * rows_size, col_size, embedding_size)
        row_embed = self.encoder(flat, token_padding_mask=token_padding_mask)
        return row_embed.reshape(batch_size, rows_size, embedding_size)


class SequenceContextEncoder(nn.Module):
    """Row-wise context encoder with shared masking semantics."""

    def __init__(
        self,
        *,
        embedding_size: int,
        n_heads: int,
        n_layers: int,
        ff_expansion: int,
        use_qass: bool,
        allow_test_self_attention: bool,
        norm_type: str,
    ) -> None:
        super().__init__()
        self.allow_test_self_attention = allow_test_self_attention
        self.encoder = QASSTransformerEncoder(
            d_model=embedding_size,
            n_heads=n_heads,
            n_layers=n_layers,
            ff_expansion=ff_expansion,
            use_qass=use_qass,
            norm_type=norm_type,
        )

    def _allowed_mask(
        self,
        *,
        batch_size: int,
        n_tokens: int,
        n_train: int,
        device: torch.device,
    ) -> torch.Tensor:
        allowed = torch.zeros(
            (batch_size, 1, n_tokens, n_tokens),
            device=device,
            dtype=torch.bool,
        )
        allowed[:, :, :, :n_train] = True
        if self.allow_test_self_attention:
            diag = torch.arange(n_tokens, device=device)
            allowed[:, :, diag, diag] = True
        return allowed

    def forward(
        self,
        rows: torch.Tensor,
        *,
        train_target_embeddings: torch.Tensor,
        train_test_split_index: int,
    ) -> torch.Tensor:
        seq = rows.clone()
        seq[:, :train_test_split_index, :] = (
            seq[:, :train_test_split_index, :] + train_target_embeddings
        )
        allowed_mask = self._allowed_mask(
            batch_size=int(seq.shape[0]),
            n_tokens=int(seq.shape[1]),
            n_train=train_test_split_index,
            device=seq.device,
        )
        return self.encoder(seq, allowed_mask=allowed_mask, n_context=train_test_split_index)


class NanoBinaryHead(nn.Module):
    """Exact nanoTabPFN binary decoder."""

    def __init__(self, embedding_size: int, hidden_size: int) -> None:
        super().__init__()
        self.decoder = NanoDecoder(embedding_size, hidden_size, 2)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.decoder(rows)


class NanoQuantileHead(nn.Module):
    """Nano-style decoder for quantile regression."""

    def __init__(self, embedding_size: int, hidden_size: int, num_outputs: int) -> None:
        super().__init__()
        self.decoder = NanoDecoder(embedding_size, hidden_size, num_outputs)

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.decoder(rows)


class DirectClassifierHead(nn.Module):
    """Generic small-class MLP head."""

    def __init__(self, embedding_size: int, hidden_size: int, output_width: int) -> None:
        super().__init__()
        self.output_width = output_width
        self.net = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_width),
        )

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.net(rows)


class DirectRegressionHead(nn.Module):
    """Generic small-class-style MLP for quantile regression."""

    def __init__(self, embedding_size: int, hidden_size: int, output_width: int) -> None:
        super().__init__()
        self.output_width = output_width
        self.net = nn.Sequential(
            nn.Linear(embedding_size, hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, output_width),
        )

    def forward(self, rows: torch.Tensor) -> torch.Tensor:
        return self.net(rows)


__all__ = [
    "ContinuousValueContextProjector",
    "ContinuousValueTargetConditioner",
    "DirectClassifierHead",
    "DirectRegressionHead",
    "IdentityColumnEncoder",
    "LabelTokenTargetConditioner",
    "MeanPaddedLinearTargetConditioner",
    "NanoBinaryHead",
    "NanoFeatureEncoder",
    "NanoQuantileHead",
    "NanoPostNormBlock",
    "PostEncoderNorm",
    "RowCLSPool",
    "ScalarPerFeatureTokenizer",
    "SequenceContextEncoder",
    "SetColumnEncoder",
    "SharedLinearFeatureEncoder",
    "ShiftedGroupedTokenizer",
    "TargetColumnPool",
]
