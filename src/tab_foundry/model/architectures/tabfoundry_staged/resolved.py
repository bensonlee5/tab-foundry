"""Resolved atomic surface definitions for staged tabfoundry models."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from tab_foundry.model.components.normalization import SUPPORTED_NORM_TYPES
from tab_foundry.model.spec import ModelBuildSpec, ModelStage

from .recipes import StageTaskContract, recipe_for_stage


SUPPORTED_FEATURE_ENCODERS = ("nano", "shared")
SUPPORTED_POST_ENCODER_NORMS = ("none", "layernorm", "rmsnorm")
SUPPORTED_TARGET_CONDITIONERS = ("mean_padded_linear", "label_token")
SUPPORTED_TOKENIZERS = ("scalar_per_feature", "scalar_per_feature_nan_mask", "shifted_grouped")
SUPPORTED_COLUMN_ENCODERS = ("none", "tfcol")
SUPPORTED_ROW_POOLS = ("target_column", "row_cls")
SUPPORTED_CONTEXT_ENCODERS = ("none", "plain", "qass")
SUPPORTED_HEADS = ("binary_direct", "small_class", "many_class")
SUPPORTED_TABLE_BLOCK_STYLES = ("nano_postnorm", "prenorm")
SUPPORTED_MODULE_OVERRIDE_KEYS = (
    "feature_encoder",
    "post_encoder_norm",
    "target_conditioner",
    "tokenizer",
    "column_encoder",
    "row_pool",
    "context_encoder",
    "head",
    "table_block_style",
    "allow_test_self_attention",
)

_LEGACY_TABLE_BLOCKS: dict[str, tuple[str, bool]] = {
    "nano_postnorm": ("nano_postnorm", False),
    "prenorm": ("prenorm", False),
    "prenorm_test_self": ("prenorm", True),
}


def _require_choice(*, value: str, supported: tuple[str, ...], context: str) -> str:
    normalized = str(value).strip().lower()
    if normalized not in supported:
        raise ValueError(f"{context} must be one of {supported}, got {value!r}")
    return normalized


def _coerce_override_bool(value: Any, *, context: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes", "on"}:
            return True
        if normalized in {"false", "0", "no", "off"}:
            return False
    raise ValueError(f"{context} must be boolean-compatible, got {value!r}")


def _override_mapping(spec: ModelBuildSpec) -> dict[str, Any]:
    raw = spec.module_overrides
    if raw is None:
        return {}
    unknown = sorted(set(raw) - set(SUPPORTED_MODULE_OVERRIDE_KEYS))
    if unknown:
        raise ValueError(
            "model.module_overrides contains unsupported keys: "
            f"{unknown}; supported={SUPPORTED_MODULE_OVERRIDE_KEYS}"
        )
    return dict(raw)


@dataclass(slots=True, frozen=True)
class TableBlockSurfaceSpec:
    style: str
    allow_test_self_attention: bool
    n_heads: int
    mlp_hidden_dim: int
    norm_type: str

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(slots=True, frozen=True)
class ColumnEncoderSurfaceSpec:
    name: str
    n_heads: int | None = None
    n_layers: int | None = None
    n_inducing: int | None = None
    norm_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(slots=True, frozen=True)
class RowPoolSurfaceSpec:
    name: str
    n_heads: int | None = None
    n_layers: int | None = None
    cls_tokens: int | None = None
    norm_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(slots=True, frozen=True)
class ContextEncoderSurfaceSpec:
    name: str
    n_heads: int | None = None
    n_layers: int | None = None
    ff_expansion: int | None = None
    use_qass: bool | None = None
    allow_test_self_attention: bool | None = None
    norm_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(slots=True, frozen=True)
class PostEncoderNormSurfaceSpec:
    name: str
    norm_type: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return dict(asdict(self))


@dataclass(slots=True, frozen=True)
class ResolvedStageSurface:
    stage: str
    stage_label: str
    benchmark_profile: str
    normalization_mode: str
    feature_encoder: str
    post_encoder_norm: str
    target_conditioner: str
    tokenizer: str
    column_encoder: str
    row_pool: str
    context_encoder: str
    head: str
    task_contract: StageTaskContract
    table_block: TableBlockSurfaceSpec
    column_encoder_config: ColumnEncoderSurfaceSpec
    row_pool_config: RowPoolSurfaceSpec
    context_encoder_config: ContextEncoderSurfaceSpec
    post_encoder_norm_config: PostEncoderNormSurfaceSpec
    staged_dropout: float = 0.0

    def module_selection(self) -> dict[str, Any]:
        return {
            "feature_encoder": self.feature_encoder,
            "post_encoder_norm": self.post_encoder_norm,
            "target_conditioner": self.target_conditioner,
            "tokenizer": self.tokenizer,
            "column_encoder": self.column_encoder,
            "row_pool": self.row_pool,
            "context_encoder": self.context_encoder,
            "head": self.head,
            "table_block_style": self.table_block.style,
            "allow_test_self_attention": bool(self.table_block.allow_test_self_attention),
        }

    def component_hyperparameters(self) -> dict[str, Any]:
        return {
            "table_block": self.table_block.to_dict(),
            "column_encoder": self.column_encoder_config.to_dict(),
            "row_pool": self.row_pool_config.to_dict(),
            "context_encoder": self.context_encoder_config.to_dict(),
            "post_encoder_norm": self.post_encoder_norm_config.to_dict(),
            "staged_dropout": self.staged_dropout,
        }


def _task_contract_for_head(head: str, *, many_class_base: int) -> StageTaskContract:
    if head == "binary_direct":
        return StageTaskContract(min_classes=2, max_classes=2)
    if head == "small_class":
        return StageTaskContract(min_classes=2, max_classes=many_class_base)
    return StageTaskContract(min_classes=2, max_classes=None, supports_many_class=True)


def _normalization_mode_for_feature_encoder(feature_encoder: str) -> str:
    if feature_encoder == "nano":
        return "internal"
    return "shared"


def _validate_effective_surface(
    *,
    feature_encoder: str,
    tokenizer: str,
    context_encoder: str,
    head: str,
) -> None:
    if feature_encoder == "nano" and tokenizer != "scalar_per_feature":
        raise ValueError(
            "model.module_overrides.tokenizer is ineffective when "
            "model.module_overrides.feature_encoder resolves to 'nano'; "
            "use tokenizer='scalar_per_feature' or switch the feature encoder"
        )
    if head == "many_class" and context_encoder == "none":
        raise ValueError(
            "model.module_overrides.head='many_class' requires a non-'none' "
            "context_encoder on the resolved staged surface"
        )


def resolve_staged_surface(spec: ModelBuildSpec) -> ResolvedStageSurface:
    """Resolve one staged model spec to an explicit atomic surface."""

    if spec.arch != "tabfoundry_staged":
        raise ValueError(f"resolve_staged_surface requires arch='tabfoundry_staged', got {spec.arch!r}")
    stage = ModelStage(spec.stage or ModelStage.NANO_EXACT.value)
    recipe = recipe_for_stage(stage)
    overrides = _override_mapping(spec)
    legacy_table_block, legacy_allow_test_self = _LEGACY_TABLE_BLOCKS[recipe.modules.table_block]

    feature_encoder = _require_choice(
        value=overrides.get("feature_encoder", recipe.modules.feature_encoder),
        supported=SUPPORTED_FEATURE_ENCODERS,
        context="model.module_overrides.feature_encoder",
    )
    post_encoder_norm = _require_choice(
        value=overrides.get("post_encoder_norm", "none"),
        supported=SUPPORTED_POST_ENCODER_NORMS,
        context="model.module_overrides.post_encoder_norm",
    )
    target_conditioner = _require_choice(
        value=overrides.get("target_conditioner", recipe.modules.target_conditioner),
        supported=SUPPORTED_TARGET_CONDITIONERS,
        context="model.module_overrides.target_conditioner",
    )
    tokenizer = _require_choice(
        value=overrides.get("tokenizer", recipe.modules.tokenizer),
        supported=SUPPORTED_TOKENIZERS,
        context="model.module_overrides.tokenizer",
    )
    column_encoder = _require_choice(
        value=overrides.get("column_encoder", recipe.modules.column_encoder),
        supported=SUPPORTED_COLUMN_ENCODERS,
        context="model.module_overrides.column_encoder",
    )
    row_pool = _require_choice(
        value=overrides.get("row_pool", recipe.modules.row_pool),
        supported=SUPPORTED_ROW_POOLS,
        context="model.module_overrides.row_pool",
    )
    norm_type = str(spec.norm_type).strip().lower()
    if norm_type not in SUPPORTED_NORM_TYPES:
        raise ValueError(
            f"model.norm_type must be one of {SUPPORTED_NORM_TYPES}, got {spec.norm_type!r}"
        )
    tfrow_norm = str(spec.tfrow_norm).strip().lower()
    if tfrow_norm not in SUPPORTED_NORM_TYPES:
        raise ValueError(
            f"model.tfrow_norm must be one of {SUPPORTED_NORM_TYPES}, got {spec.tfrow_norm!r}"
        )
    context_encoder = _require_choice(
        value=overrides.get("context_encoder", recipe.modules.context_encoder),
        supported=SUPPORTED_CONTEXT_ENCODERS,
        context="model.module_overrides.context_encoder",
    )
    head = _require_choice(
        value=overrides.get("head", recipe.modules.head),
        supported=SUPPORTED_HEADS,
        context="model.module_overrides.head",
    )
    table_block_style = _require_choice(
        value=overrides.get("table_block_style", legacy_table_block),
        supported=SUPPORTED_TABLE_BLOCK_STYLES,
        context="model.module_overrides.table_block_style",
    )
    allow_test_self_attention = (
        _coerce_override_bool(
            overrides["allow_test_self_attention"],
            context="model.module_overrides.allow_test_self_attention",
        )
        if "allow_test_self_attention" in overrides
        else legacy_allow_test_self
    )
    if table_block_style == "nano_postnorm" and allow_test_self_attention:
        raise ValueError(
            "model.module_overrides.allow_test_self_attention is only valid with "
            "table_block_style='prenorm'"
        )
    _validate_effective_surface(
        feature_encoder=feature_encoder,
        tokenizer=tokenizer,
        context_encoder=context_encoder,
        head=head,
    )

    return ResolvedStageSurface(
        stage=stage.value,
        stage_label=spec.stage_label or stage.value,
        benchmark_profile=spec.stage_label or recipe.benchmark_profile,
        normalization_mode=_normalization_mode_for_feature_encoder(feature_encoder),
        feature_encoder=feature_encoder,
        post_encoder_norm=post_encoder_norm,
        target_conditioner=target_conditioner,
        tokenizer=tokenizer,
        column_encoder=column_encoder,
        row_pool=row_pool,
        context_encoder=context_encoder,
        head=head,
        task_contract=_task_contract_for_head(head, many_class_base=int(spec.many_class_base)),
        staged_dropout=float(spec.staged_dropout),
        table_block=TableBlockSurfaceSpec(
            style=table_block_style,
            allow_test_self_attention=allow_test_self_attention,
            n_heads=int(spec.tficl_n_heads),
            mlp_hidden_dim=int(spec.head_hidden_dim),
            norm_type=norm_type,
        ),
        column_encoder_config=ColumnEncoderSurfaceSpec(
            name=column_encoder,
            n_heads=None if column_encoder == "none" else int(spec.tfcol_n_heads),
            n_layers=None if column_encoder == "none" else int(spec.tfcol_n_layers),
            n_inducing=None if column_encoder == "none" else int(spec.tfcol_n_inducing),
            norm_type=None if column_encoder == "none" else norm_type,
        ),
        row_pool_config=RowPoolSurfaceSpec(
            name=row_pool,
            n_heads=None if row_pool == "target_column" else int(spec.tfrow_n_heads),
            n_layers=None if row_pool == "target_column" else int(spec.tfrow_n_layers),
            cls_tokens=None if row_pool == "target_column" else int(spec.tfrow_cls_tokens),
            norm_type=None if row_pool == "target_column" else tfrow_norm,
        ),
        context_encoder_config=ContextEncoderSurfaceSpec(
            name=context_encoder,
            n_heads=None if context_encoder == "none" else int(spec.tficl_n_heads),
            n_layers=None if context_encoder == "none" else int(spec.tficl_n_layers),
            ff_expansion=None if context_encoder == "none" else int(spec.tficl_ff_expansion),
            use_qass=None if context_encoder == "none" else bool(context_encoder == "qass"),
            allow_test_self_attention=None
            if context_encoder == "none"
            else True,
            norm_type=None if context_encoder == "none" else norm_type,
        ),
        post_encoder_norm_config=PostEncoderNormSurfaceSpec(
            name=post_encoder_norm,
            norm_type=None if post_encoder_norm == "none" else post_encoder_norm,
        ),
    )


def staged_surface_uses_internal_benchmark_normalization(spec: ModelBuildSpec | Any) -> bool:
    """Whether a staged model keeps benchmark normalization internal."""

    if not isinstance(spec, ModelBuildSpec):
        raw_overrides = getattr(spec, "module_overrides", None)
        if isinstance(raw_overrides, Mapping):
            tokenizer = raw_overrides.get("tokenizer")
            if isinstance(tokenizer, str) and tokenizer.strip().lower() == "scalar_per_feature_nan_mask":
                return True
        stage_raw = getattr(spec, "stage", None)
        normalized_stage = ModelStage(
            str(stage_raw).strip().lower() if stage_raw is not None else ModelStage.NANO_EXACT.value
        )
        return recipe_for_stage(normalized_stage).constraints.normalization_mode == "internal"
    surface = resolve_staged_surface(spec)
    return surface.normalization_mode == "internal" or surface.tokenizer == "scalar_per_feature_nan_mask"
