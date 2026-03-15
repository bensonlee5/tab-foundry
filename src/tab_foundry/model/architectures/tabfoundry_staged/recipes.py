"""Recipe registry for the staged tabfoundry family."""

from __future__ import annotations

from dataclasses import dataclass

from tab_foundry.model.spec import ModelStage


@dataclass(slots=True, frozen=True)
class StageModuleSelection:
    """Concrete subsystem variants enabled for one stage."""

    feature_encoder: str
    target_conditioner: str
    table_block: str
    row_pool: str
    tokenizer: str
    column_encoder: str
    context_encoder: str
    head: str


@dataclass(slots=True, frozen=True)
class StageConfigConstraints:
    """Public config constraints for one stage."""

    normalization_mode: str


@dataclass(slots=True, frozen=True)
class StageTaskContract:
    """Task/class-count contract for one stage."""

    min_classes: int
    max_classes: int | None
    supports_many_class: bool = False

    def supports(self, *, num_classes: int) -> bool:
        if num_classes < self.min_classes:
            return False
        if self.max_classes is not None and num_classes > self.max_classes:
            return False
        return True


@dataclass(slots=True, frozen=True)
class StageRecipe:
    """Resolved recipe metadata for one public stage."""

    stage: ModelStage
    modules: StageModuleSelection
    constraints: StageConfigConstraints
    task_contract: StageTaskContract
    benchmark_profile: str


STAGE_RECIPE_REGISTRY: dict[ModelStage, StageRecipe] = {
    ModelStage.NANO_EXACT: StageRecipe(
        stage=ModelStage.NANO_EXACT,
        modules=StageModuleSelection(
            feature_encoder="nano",
            target_conditioner="mean_padded_linear",
            table_block="nano_postnorm",
            row_pool="target_column",
            tokenizer="scalar_per_feature",
            column_encoder="none",
            context_encoder="none",
            head="binary_direct",
        ),
        constraints=StageConfigConstraints(
            normalization_mode="internal",
        ),
        task_contract=StageTaskContract(min_classes=2, max_classes=2),
        benchmark_profile="nano_exact",
    ),
    ModelStage.LABEL_TOKEN: StageRecipe(
        stage=ModelStage.LABEL_TOKEN,
        modules=StageModuleSelection(
            feature_encoder="nano",
            target_conditioner="label_token",
            table_block="nano_postnorm",
            row_pool="target_column",
            tokenizer="scalar_per_feature",
            column_encoder="none",
            context_encoder="none",
            head="binary_direct",
        ),
        constraints=StageConfigConstraints(
            normalization_mode="internal",
        ),
        task_contract=StageTaskContract(min_classes=2, max_classes=2),
        benchmark_profile="label_token",
    ),
    ModelStage.SHARED_NORM: StageRecipe(
        stage=ModelStage.SHARED_NORM,
        modules=StageModuleSelection(
            feature_encoder="shared",
            target_conditioner="label_token",
            table_block="nano_postnorm",
            row_pool="target_column",
            tokenizer="scalar_per_feature",
            column_encoder="none",
            context_encoder="none",
            head="binary_direct",
        ),
        constraints=StageConfigConstraints(
            normalization_mode="shared",
        ),
        task_contract=StageTaskContract(min_classes=2, max_classes=2),
        benchmark_profile="shared_norm",
    ),
    ModelStage.PRENORM_BLOCK: StageRecipe(
        stage=ModelStage.PRENORM_BLOCK,
        modules=StageModuleSelection(
            feature_encoder="shared",
            target_conditioner="label_token",
            table_block="prenorm",
            row_pool="target_column",
            tokenizer="scalar_per_feature",
            column_encoder="none",
            context_encoder="none",
            head="binary_direct",
        ),
        constraints=StageConfigConstraints(
            normalization_mode="shared",
        ),
        task_contract=StageTaskContract(min_classes=2, max_classes=2),
        benchmark_profile="prenorm_block",
    ),
    ModelStage.SMALL_CLASS_HEAD: StageRecipe(
        stage=ModelStage.SMALL_CLASS_HEAD,
        modules=StageModuleSelection(
            feature_encoder="shared",
            target_conditioner="label_token",
            table_block="prenorm",
            row_pool="target_column",
            tokenizer="scalar_per_feature",
            column_encoder="none",
            context_encoder="none",
            head="small_class",
        ),
        constraints=StageConfigConstraints(
            normalization_mode="shared",
        ),
        task_contract=StageTaskContract(min_classes=2, max_classes=None),
        benchmark_profile="small_class_head",
    ),
    ModelStage.TEST_SELF: StageRecipe(
        stage=ModelStage.TEST_SELF,
        modules=StageModuleSelection(
            feature_encoder="shared",
            target_conditioner="label_token",
            table_block="prenorm_test_self",
            row_pool="target_column",
            tokenizer="scalar_per_feature",
            column_encoder="none",
            context_encoder="none",
            head="small_class",
        ),
        constraints=StageConfigConstraints(
            normalization_mode="shared",
        ),
        task_contract=StageTaskContract(min_classes=2, max_classes=None),
        benchmark_profile="test_self",
    ),
    ModelStage.GROUPED_TOKENS: StageRecipe(
        stage=ModelStage.GROUPED_TOKENS,
        modules=StageModuleSelection(
            feature_encoder="shared",
            target_conditioner="label_token",
            table_block="prenorm_test_self",
            row_pool="target_column",
            tokenizer="shifted_grouped",
            column_encoder="none",
            context_encoder="none",
            head="small_class",
        ),
        constraints=StageConfigConstraints(
            normalization_mode="shared",
        ),
        task_contract=StageTaskContract(min_classes=2, max_classes=None),
        benchmark_profile="grouped_tokens",
    ),
    ModelStage.ROW_CLS_POOL: StageRecipe(
        stage=ModelStage.ROW_CLS_POOL,
        modules=StageModuleSelection(
            feature_encoder="shared",
            target_conditioner="label_token",
            table_block="prenorm_test_self",
            row_pool="row_cls",
            tokenizer="shifted_grouped",
            column_encoder="none",
            context_encoder="plain",
            head="small_class",
        ),
        constraints=StageConfigConstraints(
            normalization_mode="shared",
        ),
        task_contract=StageTaskContract(min_classes=2, max_classes=None),
        benchmark_profile="row_cls_pool",
    ),
    ModelStage.COLUMN_SET: StageRecipe(
        stage=ModelStage.COLUMN_SET,
        modules=StageModuleSelection(
            feature_encoder="shared",
            target_conditioner="label_token",
            table_block="prenorm_test_self",
            row_pool="row_cls",
            tokenizer="shifted_grouped",
            column_encoder="tfcol",
            context_encoder="plain",
            head="small_class",
        ),
        constraints=StageConfigConstraints(
            normalization_mode="shared",
        ),
        task_contract=StageTaskContract(min_classes=2, max_classes=None),
        benchmark_profile="column_set",
    ),
    ModelStage.QASS_CONTEXT: StageRecipe(
        stage=ModelStage.QASS_CONTEXT,
        modules=StageModuleSelection(
            feature_encoder="shared",
            target_conditioner="label_token",
            table_block="prenorm_test_self",
            row_pool="row_cls",
            tokenizer="shifted_grouped",
            column_encoder="tfcol",
            context_encoder="qass",
            head="small_class",
        ),
        constraints=StageConfigConstraints(
            normalization_mode="shared",
        ),
        task_contract=StageTaskContract(min_classes=2, max_classes=None),
        benchmark_profile="qass_context",
    ),
    ModelStage.MANY_CLASS: StageRecipe(
        stage=ModelStage.MANY_CLASS,
        modules=StageModuleSelection(
            feature_encoder="shared",
            target_conditioner="label_token",
            table_block="prenorm_test_self",
            row_pool="row_cls",
            tokenizer="shifted_grouped",
            column_encoder="tfcol",
            context_encoder="qass",
            head="many_class",
        ),
        constraints=StageConfigConstraints(
            normalization_mode="shared",
        ),
        task_contract=StageTaskContract(min_classes=2, max_classes=None, supports_many_class=True),
        benchmark_profile="many_class",
    ),
}


def recipe_for_stage(stage: ModelStage) -> StageRecipe:
    """Return the declared recipe for one stage."""

    return STAGE_RECIPE_REGISTRY[stage]


def stage_uses_internal_benchmark_normalization(stage: ModelStage) -> bool:
    """Whether external benchmark wrappers must leave feature normalization internal."""

    return recipe_for_stage(stage).constraints.normalization_mode == "internal"
