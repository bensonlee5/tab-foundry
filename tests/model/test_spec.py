from __future__ import annotations

import pytest
import torch

from tab_foundry.model.factory import build_model_from_spec
from tab_foundry.model.architectures.grid_sandwich import GridSandwichClassifier
from tab_foundry.model.architectures.routed_sandwich import RoutedSandwichClassifier
from tab_foundry.model.architectures.tabfoundry_sandwich import TabFoundrySandwichClassifier
from tab_foundry.model.architectures.tabfoundry_simple import TabFoundrySimpleClassifier
from tab_foundry.model.spec import (
    GRID_SANDWICH_MODEL_ARCH,
    ROUTED_SANDWICH_MODEL_ARCH,
    SANDWICH_MODEL_ARCH,
    ModelBuildSpec,
    checkpoint_model_build_spec_from_mappings,
    model_build_spec_from_mappings,
)


def _build_model(*, task: str = "classification", **model_overrides: object) -> torch.nn.Module:
    spec = model_build_spec_from_mappings(task=task, primary=model_overrides)
    return build_model_from_spec(spec)


def test_model_build_spec_defaults_feature_group_size_to_one() -> None:
    spec = model_build_spec_from_mappings(
        task="classification",
        primary={},
    )

    assert spec.arch == SANDWICH_MODEL_ARCH
    assert spec.feature_group_size == 1


def test_build_model_defaults_feature_group_size_to_one() -> None:
    cls_model = _build_model(task="classification")

    assert int(cls_model.model_spec.feature_group_size) == 1


def test_build_model_rejects_regression() -> None:
    with pytest.raises(ValueError, match="Unsupported task"):
        _ = model_build_spec_from_mappings(task="regression", primary={})


def test_build_model_supports_tabfoundry_simple_classification() -> None:
    model = _build_model(
        task="classification",
        arch="tabfoundry_simple",
        d_icl=96,
        input_normalization="train_zscore_clip",
        many_class_base=2,
        tficl_n_heads=4,
        tficl_n_layers=3,
        head_hidden_dim=192,
    )

    assert isinstance(model, TabFoundrySimpleClassifier)


def test_build_model_supports_tabfoundry_sandwich_classification() -> None:
    model = _build_model(
        task="classification",
        arch="tabfoundry_sandwich",
        d_icl=96,
        input_normalization="train_zscore_clip",
        many_class_base=4,
        head_hidden_dim=128,
        sandwich_latents=16,
        sandwich_layers=2,
        sandwich_heads=4,
        sandwich_ff_expansion=2,
        sandwich_summary_tokens_per_axis=4,
        sandwich_self_attention_per_cross=4,
        sandwich_pre_column_inducing_tokens=8,
    )

    assert isinstance(model, TabFoundrySandwichClassifier)
    assert model.pre_column_inducing_tokens == 8


def test_build_model_supports_routed_sandwich_classification() -> None:
    model = _build_model(
        task="classification",
        arch="routed_sandwich",
        d_icl=96,
        input_normalization="train_zscore_clip",
        many_class_base=4,
        head_hidden_dim=128,
        sandwich_latents=16,
        sandwich_layers=2,
        sandwich_heads=4,
        sandwich_ff_expansion=2,
        routed_residual_streams=2,
        routed_row_summary_tokens=3,
        routed_column_summary_tokens=2,
        routed_evidence_tokens=5,
    )

    assert isinstance(model, RoutedSandwichClassifier)
    assert model.latent_seed.shape == (1, 16, 2, 96)
    assert model.evidence_query.shape == (1, 5, 96)


def test_build_model_supports_grid_sandwich_classification() -> None:
    model = _build_model(
        task="classification",
        arch="grid_sandwich",
        d_icl=96,
        input_normalization="train_zscore_clip",
        many_class_base=4,
        head_hidden_dim=128,
        sandwich_layers=2,
        sandwich_heads=4,
        sandwich_ff_expansion=2,
        sandwich_pre_row_attention_layers=2,
        sandwich_pre_column_attention_layers=1,
        sandwich_pre_column_inducing_tokens=8,
        grid_attention_mode="differential",
        grid_ffn_mode="swiglu",
    )

    assert isinstance(model, GridSandwichClassifier)
    assert len(model.grid_layers) == 2
    assert len(model.pre_row_attention_blocks) == 2
    assert len(model.pre_column_attention_blocks) == 1
    assert model.pre_column_inducing_tokens == 8
    assert model.grid_attention_mode == "differential"
    assert model.grid_ffn_mode == "swiglu"


def test_sandwich_constructor_defaults_match_factory_defaults() -> None:
    constructor_model = TabFoundrySandwichClassifier()
    factory_model = _build_model(task="classification", arch="tabfoundry_sandwich")

    constructor_params = sum(int(parameter.numel()) for parameter in constructor_model.parameters())
    factory_params = sum(int(parameter.numel()) for parameter in factory_model.parameters())

    assert constructor_model.model_spec == factory_model.model_spec
    assert constructor_params == factory_params
    assert 550_000 <= constructor_params <= 650_000


def test_sandwich_model_spec_defaults_to_small_v0_widths() -> None:
    spec = model_build_spec_from_mappings(
        task="classification",
        primary={"arch": "tabfoundry_sandwich"},
    )

    assert spec.d_icl == 60
    assert spec.head_hidden_dim == 96
    assert spec.sandwich_latents == 24
    assert spec.sandwich_summary_tokens_per_axis == 4
    assert spec.sandwich_self_attention_per_cross == 4
    assert spec.sandwich_pre_row_attention_layers == 1
    assert spec.sandwich_pre_column_attention_layers == 1
    assert spec.sandwich_pre_column_inducing_tokens == 16
    assert spec.sandwich_activation == "gelu"
    assert spec.sandwich_block_norm == "layernorm"
    assert spec.feature_type_conditioning == "film"
    assert spec.floating_likelihood == "single_gaussian"
    assert spec.integer_likelihood == "hybrid_mixture"


def test_routed_sandwich_model_spec_defaults_include_routed_fields() -> None:
    spec = model_build_spec_from_mappings(
        task="classification",
        primary={"arch": ROUTED_SANDWICH_MODEL_ARCH},
    )

    assert spec.arch == ROUTED_SANDWICH_MODEL_ARCH
    assert spec.routed_residual_mode == "dynamic_hyper"
    assert spec.routed_residual_streams == 2
    assert spec.routed_residual_scale == "deepnorm"
    assert spec.routed_row_summary_tokens == 4
    assert spec.routed_column_summary_tokens == 2
    assert spec.routed_evidence_tokens == 16
    assert spec.routed_direct_cell_bypass is False


def test_grid_sandwich_model_spec_defaults_reuse_sandwich_core_fields() -> None:
    spec = model_build_spec_from_mappings(
        task="classification",
        primary={"arch": GRID_SANDWICH_MODEL_ARCH},
    )

    assert spec.arch == GRID_SANDWICH_MODEL_ARCH
    assert spec.d_icl == 60
    assert spec.sandwich_layers == 2
    assert spec.sandwich_heads == 4
    assert spec.sandwich_pre_row_attention_layers == 1
    assert spec.sandwich_pre_column_attention_layers == 1
    assert spec.sandwich_pre_column_inducing_tokens == 16
    assert spec.grid_residual_mode == "prenorm"
    assert spec.grid_attention_mode == "standard"
    assert spec.grid_ffn_mode == "gelu"
    assert spec.grid_recurrence_steps is None


def test_grid_sandwich_model_spec_round_trips_grid_experiment_fields() -> None:
    spec = model_build_spec_from_mappings(
        task="classification",
        primary={
            "arch": GRID_SANDWICH_MODEL_ARCH,
            "sandwich_pre_row_attention_layers": 2,
            "sandwich_pre_column_attention_layers": 3,
            "grid_residual_mode": "hyper_connection_lite",
            "grid_attention_mode": "differential",
            "grid_ffn_mode": "swiglu",
            "grid_recurrence_steps": 8,
        },
    )

    assert spec.sandwich_pre_row_attention_layers == 2
    assert spec.sandwich_pre_column_attention_layers == 3
    assert spec.grid_residual_mode == "hyper_connection_lite"
    assert spec.grid_attention_mode == "differential"
    assert spec.grid_ffn_mode == "swiglu"
    assert spec.grid_recurrence_steps == 8
    assert spec.to_dict()["sandwich_pre_row_attention_layers"] == 2
    assert spec.to_dict()["sandwich_pre_column_attention_layers"] == 3
    assert spec.to_dict()["grid_residual_mode"] == "hyper_connection_lite"
    assert spec.to_dict()["grid_attention_mode"] == "differential"
    assert spec.to_dict()["grid_ffn_mode"] == "swiglu"
    assert spec.to_dict()["grid_recurrence_steps"] == 8


def test_sandwich_model_spec_to_dict_is_arch_scoped() -> None:
    spec = model_build_spec_from_mappings(
        task="classification",
        primary={
            "arch": "tabfoundry_sandwich",
            "sandwich_latents": 16,
            "sandwich_layers": 2,
            "sandwich_heads": 4,
            "sandwich_activation": "rational",
            "sandwich_block_norm": "none",
        },
        fallback={
            "tficl_n_heads": 4,
            "tficl_n_layers": 3,
            "tfrow_norm": "rmsnorm",
            "use_digit_position_embed": False,
        },
    )

    payload = spec.to_dict()

    assert payload["arch"] == "tabfoundry_sandwich"
    assert payload["sandwich_latents"] == 16
    assert payload["sandwich_layers"] == 2
    assert payload["sandwich_heads"] == 4
    assert payload["sandwich_activation"] == "rational"
    assert payload["sandwich_block_norm"] == "none"
    for unsupported_key in (
        "stage",
        "stage_label",
        "module_overrides",
        "tfcol_n_heads",
        "tfcol_n_layers",
        "tfcol_n_inducing",
        "tfrow_n_heads",
        "tfrow_n_layers",
        "tfrow_cls_tokens",
        "tfrow_norm",
        "tficl_n_heads",
        "tficl_n_layers",
        "tficl_ff_expansion",
        "use_digit_position_embed",
        "staged_dropout",
    ):
        assert unsupported_key not in payload


def test_routed_sandwich_model_spec_to_dict_includes_routed_fields() -> None:
    spec = model_build_spec_from_mappings(
        task="classification",
        primary={
            "arch": "routed_sandwich",
            "routed_residual_streams": 3,
            "routed_row_summary_tokens": 2,
            "routed_column_summary_tokens": 1,
            "routed_evidence_tokens": 7,
            "routed_direct_cell_bypass": True,
        },
    )

    payload = spec.to_dict()

    assert payload["arch"] == "routed_sandwich"
    assert payload["routed_residual_mode"] == "dynamic_hyper"
    assert payload["routed_residual_streams"] == 3
    assert payload["routed_residual_scale"] == "deepnorm"
    assert payload["routed_row_summary_tokens"] == 2
    assert payload["routed_column_summary_tokens"] == 1
    assert payload["routed_evidence_tokens"] == 7
    assert payload["routed_direct_cell_bypass"] is True
    assert "sandwich_summary_tokens_per_axis" not in payload
    for unsupported_key in ("stage", "stage_label", "module_overrides", "staged_dropout"):
        assert unsupported_key not in payload


def test_sandwich_checkpoint_spec_infers_legacy_additive_feature_type_conditioning() -> None:
    spec = checkpoint_model_build_spec_from_mappings(
        task="classification",
        primary={"arch": "tabfoundry_sandwich"},
        state_dict={"feature_type_embedding.weight": torch.zeros((5, 60))},
    )

    assert spec.feature_type_conditioning == "additive_embedding"


@pytest.mark.parametrize(
    "field_name",
    (
        "sandwich_self_attention_per_cross",
        "sandwich_pre_row_attention_layers",
        "sandwich_pre_column_attention_layers",
    ),
)
def test_sandwich_model_spec_allows_zero_repeat_count_fields(field_name: str) -> None:
    spec = model_build_spec_from_mappings(
        task="classification",
        primary={"arch": "tabfoundry_sandwich", field_name: 0},
    )

    assert getattr(spec, field_name) == 0


def test_sandwich_model_spec_accepts_packed_attention_flag() -> None:
    spec = model_build_spec_from_mappings(
        task="classification",
        primary={"arch": "tabfoundry_sandwich", "sandwich_packed_attention": True},
    )

    assert spec.sandwich_packed_attention is True
    assert spec.to_dict()["sandwich_packed_attention"] is True


def test_routed_sandwich_model_spec_rejects_unsupported_residual_mode() -> None:
    with pytest.raises(ValueError, match="routed_residual_mode"):
        _ = model_build_spec_from_mappings(
            task="classification",
            primary={"arch": "routed_sandwich", "routed_residual_mode": "static"},
        )


def test_routed_sandwich_model_spec_rejects_unsupported_residual_scale() -> None:
    with pytest.raises(ValueError, match="routed_residual_scale"):
        _ = model_build_spec_from_mappings(
            task="classification",
            primary={"arch": "routed_sandwich", "routed_residual_scale": "prenorm"},
        )


@pytest.mark.parametrize(
    ("field_name", "bad_value"),
    (
        ("grid_residual_mode", "dynamic_hyper"),
        ("grid_attention_mode", "flash"),
        ("grid_ffn_mode", "geglu"),
        ("grid_recurrence_steps", 0),
    ),
)
def test_grid_sandwich_model_spec_rejects_unsupported_experiment_fields(
    field_name: str,
    bad_value: object,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        _ = model_build_spec_from_mappings(
            task="classification",
            primary={"arch": "grid_sandwich", field_name: bad_value},
        )


def test_routed_sandwich_model_spec_sanitizes_merged_summary_tokens_field_on_arch_swap() -> None:
    spec = model_build_spec_from_mappings(
        task="classification",
        primary={
            "arch": ROUTED_SANDWICH_MODEL_ARCH,
            "sandwich_summary_tokens_per_axis": 3,
        },
    )

    assert spec.arch == ROUTED_SANDWICH_MODEL_ARCH
    assert "sandwich_summary_tokens_per_axis" not in spec.to_dict()
    assert spec.sandwich_summary_tokens_per_axis == 4


def test_grid_sandwich_model_spec_sanitizes_merged_dead_sandwich_fields_on_arch_swap() -> None:
    spec = model_build_spec_from_mappings(
        task="classification",
        primary={
            "arch": GRID_SANDWICH_MODEL_ARCH,
            "sandwich_latents": 32,
            "sandwich_self_attention_per_cross": 2,
            "sandwich_summary_tokens_per_axis": 3,
        },
    )

    payload = spec.to_dict()

    assert spec.arch == GRID_SANDWICH_MODEL_ARCH
    assert spec.sandwich_latents == 24
    assert spec.sandwich_self_attention_per_cross == 4
    assert spec.sandwich_summary_tokens_per_axis == 4
    for field_name in (
        "sandwich_latents",
        "sandwich_self_attention_per_cross",
        "sandwich_summary_tokens_per_axis",
    ):
        assert field_name not in payload


@pytest.mark.parametrize(
    ("primary", "fallback", "match"),
    (
        (
            {"sandwich_summary_tokens_per_axis": 3},
            {"arch": ROUTED_SANDWICH_MODEL_ARCH},
            "routed_row_summary_tokens",
        ),
        (
            {},
            {"arch": ROUTED_SANDWICH_MODEL_ARCH, "sandwich_summary_tokens_per_axis": 3},
            "routed_row_summary_tokens",
        ),
        (
            {"sandwich_latents": 32},
            {"arch": GRID_SANDWICH_MODEL_ARCH},
            "sandwich_latents",
        ),
        (
            {"sandwich_self_attention_per_cross": 2},
            {"arch": GRID_SANDWICH_MODEL_ARCH},
            "sandwich_self_attention_per_cross",
        ),
        (
            {},
            {"arch": GRID_SANDWICH_MODEL_ARCH, "sandwich_summary_tokens_per_axis": 3},
            "sandwich_summary_tokens_per_axis",
        ),
    ),
)
def test_followon_model_spec_rejects_explicit_layered_dead_sandwich_fields(
    primary: dict[str, object],
    fallback: dict[str, object],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _ = model_build_spec_from_mappings(
            task="classification",
            primary=primary,
            fallback=fallback,
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "sandwich_self_attention_per_cross",
        "sandwich_pre_row_attention_layers",
        "sandwich_pre_column_attention_layers",
    ),
)
def test_sandwich_model_spec_rejects_negative_repeat_count_fields(field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        _ = model_build_spec_from_mappings(
            task="classification",
            primary={"arch": "tabfoundry_sandwich", field_name: -1},
        )


@pytest.mark.parametrize(
    "field_name",
    (
        "sandwich_layers",
        "sandwich_summary_tokens_per_axis",
        "sandwich_pre_column_inducing_tokens",
    ),
)
def test_sandwich_model_spec_requires_positive_core_counts(field_name: str) -> None:
    with pytest.raises(ValueError, match=field_name):
        _ = model_build_spec_from_mappings(
            task="classification",
            primary={"arch": "tabfoundry_sandwich", field_name: 0},
        )


def test_sandwich_model_spec_rejects_legacy_dual_bank_fields() -> None:
    with pytest.raises(ValueError, match="sandwich_latents"):
        _ = model_build_spec_from_mappings(
            task="classification",
            primary={
                "arch": "tabfoundry_sandwich",
                "sandwich_row_latents": 32,
            },
        )

    with pytest.raises(ValueError, match="sandwich_latents"):
        _ = model_build_spec_from_mappings(
            task="classification",
            primary={
                "arch": "tabfoundry_sandwich",
                "sandwich_col_latents": 16,
            },
        )


@pytest.mark.parametrize(
    ("field_name", "field_value"),
    (
        ("sandwich_activation", "swish"),
        ("sandwich_block_norm", "rmsnorm"),
    ),
)
def test_sandwich_model_spec_rejects_unsupported_activation_and_block_norm_fields(
    field_name: str,
    field_value: str,
) -> None:
    with pytest.raises(ValueError, match=field_name):
        _ = model_build_spec_from_mappings(
            task="classification",
            primary={"arch": "tabfoundry_sandwich", field_name: field_value},
        )


def test_build_model_rejects_legacy_tabfoundry_arch() -> None:
    with pytest.raises(ValueError, match="Unsupported model arch"):
        _ = model_build_spec_from_mappings(task="classification", primary={"arch": "tabfoundry"})


def test_staged_model_defaults_stage_to_nano_exact() -> None:
    spec = model_build_spec_from_mappings(
        task="classification",
        primary={"arch": "tabfoundry_staged"},
    )

    assert spec.arch == "tabfoundry_staged"
    assert spec.stage == "nano_exact"


def test_staged_model_spec_accepts_stage_label_and_module_overrides() -> None:
    spec = model_build_spec_from_mappings(
        task="classification",
        primary={
            "arch": "tabfoundry_staged",
            "stage": "nano_exact",
            "stage_label": "delta_row_cls_pool",
            "module_overrides": {"row_pool": "row_cls"},
        },
    )

    assert spec.stage == "nano_exact"
    assert spec.stage_label == "delta_row_cls_pool"
    assert spec.module_overrides == {"row_pool": "row_cls"}


def test_non_staged_arch_rejects_stage() -> None:
    with pytest.raises(ValueError, match="model.stage"):
        _ = ModelBuildSpec(task="classification", arch="tabfoundry_simple", stage="nano_exact")


def test_non_staged_arch_rejects_stage_surface_fields() -> None:
    with pytest.raises(ValueError, match="stage_label"):
        _ = ModelBuildSpec(
            task="classification",
            arch="tabfoundry_simple",
            stage_label="delta_label_token",
        )
    with pytest.raises(ValueError, match="module_overrides"):
        _ = ModelBuildSpec(
            task="classification",
            arch="tabfoundry_simple",
            module_overrides={"row_pool": "row_cls"},
        )


def test_checkpoint_build_spec_round_trips_model_arch() -> None:
    spec = checkpoint_model_build_spec_from_mappings(
        task="classification",
        primary={"arch": "tabfoundry_simple"},
        state_dict={},
    )

    assert spec.arch == "tabfoundry_simple"


def test_model_build_spec_rejects_regression_task() -> None:
    with pytest.raises(ValueError, match="Unsupported task"):
        _ = ModelBuildSpec(task="regression")


def test_checkpoint_build_spec_rejects_legacy_tabfoundry_arch() -> None:
    with pytest.raises(ValueError, match="model.arch='tabfoundry'"):
        _ = checkpoint_model_build_spec_from_mappings(
            task="classification",
            primary={"arch": "tabfoundry"},
            state_dict={},
        )


def test_checkpoint_build_spec_rejects_legacy_tabfoundry_state_dict() -> None:
    with pytest.raises(ValueError, match="Legacy tabfoundry checkpoints"):
        _ = checkpoint_model_build_spec_from_mappings(
            task="classification",
            primary={},
            state_dict={"group_linear.weight": torch.zeros((128, 96))},
        )
