from __future__ import annotations

from functools import lru_cache
import importlib.util
from pathlib import Path

import pytest
import torch

from tab_foundry.model.architectures.tabfoundry_simple import (
    TabFoundrySimpleClassifier,
    _TransformerEncoderLayer,
)
from tab_foundry.types import TaskBatch


def _batch(*, x_test: torch.Tensor | None = None, num_classes: int = 2) -> TaskBatch:
    return TaskBatch(
        x_train=torch.tensor(
            [
                [1.0, 2.0, 3.0],
                [4.0, 5.0, 6.0],
                [7.0, 8.0, 9.0],
            ],
            dtype=torch.float32,
        ),
        y_train=torch.tensor([0, 1, 0], dtype=torch.int64),
        x_test=torch.tensor(
            [
                [0.5, 1.5, 2.5],
                [3.5, 4.5, 5.5],
            ],
            dtype=torch.float32,
        )
        if x_test is None
        else x_test,
        y_test=torch.zeros((2,), dtype=torch.int64),
        metadata={},
        num_classes=num_classes,
    )


def _model(**overrides: object) -> TabFoundrySimpleClassifier:
    kwargs = {
        "d_icl": 96,
        "input_normalization": "train_zscore_clip",
        "many_class_base": 2,
        "tficl_n_heads": 4,
        "tficl_n_layers": 3,
        "head_hidden_dim": 192,
    }
    kwargs.update(overrides)
    return TabFoundrySimpleClassifier(**kwargs)


@lru_cache(maxsize=1)
def _nanotabpfn_module():
    model_path = Path("~/dev/nanoTabPFN/model.py").expanduser()
    if not model_path.exists():
        pytest.skip("local nanoTabPFN reference is not available")
    spec = importlib.util.spec_from_file_location("local_nanotabpfn_model", model_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load nanoTabPFN model module from {model_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _synced_models(
    *,
    seed: int = 0,
    d_icl: int = 32,
    n_heads: int = 4,
    n_layers: int = 1,
    head_hidden_dim: int = 64,
):
    nano_module = _nanotabpfn_module()
    torch.manual_seed(seed)
    nano_model = nano_module.NanoTabPFNModel(
        embedding_size=d_icl,
        num_attention_heads=n_heads,
        mlp_hidden_size=head_hidden_dim,
        num_layers=n_layers,
        num_outputs=2,
    )
    model = TabFoundrySimpleClassifier(
        d_icl=d_icl,
        input_normalization="train_zscore_clip",
        many_class_base=2,
        tficl_n_heads=n_heads,
        tficl_n_layers=n_layers,
        head_hidden_dim=head_hidden_dim,
    )
    model.load_state_dict(nano_model.state_dict(), strict=True)
    return nano_model, model


def test_tabfoundry_simple_forward_shapes() -> None:
    out = _model()(_batch())

    assert out.logits is not None
    assert out.logits.shape == (2, 2)
    assert out.num_classes == 2


def test_tabfoundry_simple_requires_binary_num_classes() -> None:
    model = _model()

    with pytest.raises(RuntimeError, match="binary-only"):
        _ = model(_batch(num_classes=3))


def test_tabfoundry_simple_requires_exact_input_normalization() -> None:
    with pytest.raises(ValueError, match="input_normalization"):
        _ = TabFoundrySimpleClassifier(input_normalization="none", many_class_base=2)


def test_tabfoundry_simple_requires_many_class_base_two() -> None:
    with pytest.raises(ValueError, match="many_class_base=2"):
        _ = TabFoundrySimpleClassifier(
            input_normalization="train_zscore_clip",
            many_class_base=10,
        )


def test_tabfoundry_simple_table_tokens_match_features_plus_label() -> None:
    table, n_train = _model()._build_table_tokens(_batch())

    assert n_train == 3
    assert table.shape == (5, 4, 96)


def test_tabfoundry_simple_train_rows_do_not_depend_on_x_test() -> None:
    torch.manual_seed(0)
    model = TabFoundrySimpleClassifier(
        d_icl=32,
        input_normalization="train_zscore_clip",
        many_class_base=2,
        tficl_n_heads=4,
        tficl_n_layers=2,
        head_hidden_dim=64,
    )
    batch_a = _batch()
    batch_b = _batch(
        x_test=torch.tensor(
            [
                [30.0, 40.0, 50.0],
                [60.0, 70.0, 80.0],
            ],
            dtype=torch.float32,
        )
    )

    encoded_a, n_train_a = model._encode_table(batch_a)
    encoded_b, n_train_b = model._encode_table(batch_b)

    assert n_train_a == n_train_b == 3
    assert torch.allclose(encoded_a[:n_train_a], encoded_b[:n_train_b], atol=1.0e-6, rtol=1.0e-6)


def test_transformer_block_row_attention_matches_nanotabpfn_semantics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    block = _TransformerEncoderLayer(embedding_size=16, nhead=4, mlp_hidden_size=32)
    calls: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []

    def _capture_forward(
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        **_: object,
    ) -> tuple[torch.Tensor, None]:
        calls.append((tuple(query.shape), tuple(key.shape), tuple(value.shape)))
        return query, None

    monkeypatch.setattr(block.self_attention_between_datapoints, "forward", _capture_forward)

    src = torch.randn(1, 5, 4, 16)
    out = block(src, train_test_split_index=3)

    assert out.shape == src.shape
    assert calls == [
        ((4, 3, 16), (4, 3, 16), (4, 3, 16)),
        ((4, 2, 16), (4, 3, 16), (4, 3, 16)),
    ]


def test_tabfoundry_simple_rejects_nondefault_tabfoundry_only_knobs() -> None:
    with pytest.raises(ValueError, match="feature_group_size"):
        _ = TabFoundrySimpleClassifier(
            input_normalization="train_zscore_clip",
            many_class_base=2,
            feature_group_size=2,
        )


def test_tabfoundry_simple_feature_encoder_matches_nanotabpfn_reference() -> None:
    nano_model, model = _synced_models()
    x_all = torch.randn(2, 8, 3, dtype=torch.float32)

    observed = model.feature_encoder(x_all.clone(), 5)
    expected = nano_model.feature_encoder(x_all.clone(), 5)

    assert torch.allclose(observed, expected, atol=1.0e-6, rtol=1.0e-6)


def test_tabfoundry_simple_target_encoder_matches_nanotabpfn_reference() -> None:
    nano_model, model = _synced_models()
    y_train = torch.tensor(
        [
            [0.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).unsqueeze(-1)

    observed = model.target_encoder(y_train.clone(), 8)
    expected = nano_model.target_encoder(y_train.clone(), 8)

    assert torch.allclose(observed, expected, atol=1.0e-6, rtol=1.0e-6)


def test_tabfoundry_simple_transformer_block_matches_nanotabpfn_reference() -> None:
    nano_model, model = _synced_models()
    src = torch.randn(2, 8, 4, 32, dtype=torch.float32)

    observed = model.transformer_blocks[0](src.clone(), train_test_split_index=5)
    expected = nano_model.transformer_blocks[0](src.clone(), train_test_split_index=5)

    assert torch.allclose(observed, expected, atol=1.0e-6, rtol=1.0e-6)


def test_tabfoundry_simple_logits_match_nanotabpfn_reference() -> None:
    nano_model, model = _synced_models(seed=1, n_layers=2)
    x_all = torch.randn(2, 8, 3, dtype=torch.float32)
    y_train = torch.tensor(
        [
            [0.0, 1.0, 0.0, 1.0, 1.0],
            [1.0, 0.0, 1.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    observed = model.forward_batched(
        x_all=x_all.clone(),
        y_train=y_train.clone(),
        train_test_split_index=5,
    )
    expected = nano_model(
        (x_all.clone(), y_train.clone()),
        train_test_split_index=5,
    )

    assert torch.allclose(observed, expected, atol=1.0e-6, rtol=1.0e-6)

