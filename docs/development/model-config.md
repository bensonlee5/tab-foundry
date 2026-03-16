# Model Config Reference

This document describes the `tabfoundry` model configuration surface, the
default values used in the repo, and how those values are resolved across
training, evaluation, export, and bundle loading.

Related docs:

- architecture reference: `docs/development/model-architecture.md`
- inference contract: `docs/inference.md`
- canonical model spec: `src/tab_foundry/model/spec.py`
- config defaults: `configs/model/default.yaml`

## Source Of Truth

The model config surface is shared across several layers, but the roles differ:

- `configs/model/default.yaml`
  - Canonical repo-level default values for Hydra-composed train and eval runs.
  - This is the intended operator-facing default source.
- `src/tab_foundry/model/spec.py`
  - Canonical typed representation of model settings.
  - Provides typed normalization and direct-construction defaults for internal
    call sites that are not reconstructing persisted artifacts.
- Checkpoint payload `config.model`
  - Persists the model settings used for a training run.
  - Is the source of truth for checkpoint evaluation/export reconstruction.
- Export manifest `model`
  - Persists the resolved model settings that an inference/runtime loader needs
    to reconstruct the model.

Explicitness policy:

- Hydra-composed training config may rely on repo defaults from
  `configs/model/default.yaml`.
- Persisted artifact readers must not silently backfill omitted reconstruction
  fields from current repo defaults.
- Checkpoint/export/bundle loading accepts only:
  - persisted artifact metadata
  - explicit user override fields where the caller exposes them
- If required reconstruction metadata is missing or ambiguous, loading fails
  with a compatibility error instead of inferring or substituting behavior.

Current canonical default:

- `feature_group_size = 1`
- `missingness_mode = "none"`

That means the default model uses one token per feature. Larger values such as
`32` are non-default grouped-token experiments that reduce token count and
change the inductive bias.

## Resolution Order

### Training

Training composes Hydra config and resolves the model spec from `cfg.model`.

Resolution order:

1. Explicit Hydra override or experiment value in `cfg.model`
1. `configs/model/default.yaml`
1. `ModelBuildSpec` fallback defaults for any remaining omitted fields

In normal repo usage, `configs/model/default.yaml` should be treated as the
source of truth for default values.

### Evaluate Checkpoint

Checkpoint evaluation resolves settings from checkpoint metadata plus explicit
user overrides.

Resolution order:

1. `checkpoint["config"]["model"]`
1. explicit caller-provided overrides such as `eval.model_overrides`
1. hard failure if required reconstruction fields are missing or ambiguous

Current repo defaults do not implicitly override or fill checkpoint metadata.
Legacy checkpoints that omitted reconstructive fields such as
`feature_group_size` or `missingness_mode` must be regenerated or loaded with
explicit overrides that match the saved weights.

### Export Checkpoint

Export reconstruction resolves the model spec from the checkpoint payload.

Resolution order:

1. `checkpoint["config"]["model"]`
1. hard failure if required reconstruction fields are missing or ambiguous

The resolved spec is then written into `manifest.json`, including the embedded
`manifest.inference` section for v3 bundles.
`missingness_mode` is serialized into the manifest/inference payload for new
exports, and the exporter now requires the checkpoint to persist the fields
needed to reconstruct the model exactly.

### Load Export Bundle

Bundle loading reconstructs the model from the manifest.

Resolution order:

1. `manifest.model`
1. hard failure if required reconstruction fields are missing

Bundle validators and loaders require the manifest to persist the model fields
needed to reconstruct the model exactly. Older manifests that omitted those
fields must be regenerated.

## Parameter Reference

| Name | Type | Default | Applies To | Meaning |
| ---- | ---- | ---- | ---- | ---- |
| `arch` | `str` | `"tabfoundry"` | both | Model architecture. Supported values are `tabfoundry`, the frozen binary repro architecture `tabfoundry_simple`, and the staged classification research family `tabfoundry_staged`. |
| `stage` | `str \| null` | `null` | classification | Stage selector for `tabfoundry_staged`. `null` resolves to `nano_exact` when `arch=tabfoundry_staged`; non-null values are rejected for `tabfoundry` and `tabfoundry_simple`. |
| `stage_label` | `str \| null` | `null` | classification | Optional reporting label for staged runs. When present, benchmark/profile metadata uses this label while the underlying recipe still resolves from `stage`. |
| `module_overrides` | `mapping \| null` | `null` | classification | Additive atomic staged-surface overrides. Supported top-level keys are `feature_encoder`, `post_encoder_norm`, `target_conditioner`, `tokenizer`, `column_encoder`, `row_pool`, `context_encoder`, `head`, `table_block_style`, and `allow_test_self_attention`. |
| `d_col` | `int` | `128` | both | Width of grouped feature tokens and the column encoder. |
| `d_icl` | `int` | `512` | both | Width of row embeddings and the final in-context encoder. |
| `input_normalization` | `str` | `"none"` | both | Train/test feature normalization mode. Supported values are `none`, `train_zscore`, and `train_zscore_clip`. |
| `missingness_mode` | `str` | `"none"` | both | Model-native missing-value handling mode. Supported values are `none`, `explicit_token`, and `feature_mask`. Missing values are any non-finite feature entries (`NaN` or `Inf`). |
| `feature_group_size` | `int` | `1` | both | Number of raw features per grouped token before shifted concatenation. `1` is the paper-faithful per-feature default. |
| `many_class_train_mode` | `str` | `"path_nll"` | classification | Training branch for many-class classification. `path_nll` returns path terms; `full_probs` trains through full probabilities. |
| `max_mixed_radix_digits` | `int` | `64` | classification | Maximum allowed depth for mixed-radix many-class decomposition. |
| `tfcol_n_heads` | `int` | `8` | both | Attention heads in the column encoder. |
| `tfcol_n_layers` | `int` | `3` | both | Number of ISAB blocks in the column encoder. |
| `tfcol_n_inducing` | `int` | `128` | both | Inducing-token count used by the column encoder ISAB blocks. |
| `tfrow_n_heads` | `int` | `8` | both | Attention heads in the row encoder. |
| `tfrow_n_layers` | `int` | `3` | both | Transformer depth in the row encoder. |
| `tfrow_cls_tokens` | `int` | `4` | both | Learned CLS tokens used to aggregate one row embedding. |
| `tficl_n_heads` | `int` | `8` | both | Attention heads in the final in-context encoder. |
| `tficl_n_layers` | `int` | `12` | both | Transformer depth in the final in-context encoder. |
| `tficl_ff_expansion` | `int` | `2` | both | Feedforward expansion factor in the final in-context encoder. |
| `many_class_base` | `int` | `10` | classification | Small-class head width and branching/base parameter for the many-class path. |
| `head_hidden_dim` | `int` | `1024` | both | Hidden width of the task head MLP. |
| `use_digit_position_embed` | `bool` | `true` | classification | Whether many-class mixed-radix views get a learned digit-position embedding. |

## Configuration Groups

### Core Widths And Depth

These parameters set the overall model size:

- `arch`
- `stage`
- `d_col`
- `d_icl`
- `tfcol_n_layers`
- `tfrow_n_layers`
- `tficl_n_layers`
- `head_hidden_dim`

`d_col` controls the token width before row aggregation. `d_icl` controls the
row representation width and the final QASS transformer width.

### Tokenization And Preprocessing

- `input_normalization`
- `missingness_mode`
- `feature_group_size`

`feature_group_size` is the highest-leverage token-count knob:

- `1`: one token per feature, paper-faithful default
- `>1`: grouped-token mode, fewer tokens and lower attention cost

`missingness_mode` controls whether missingness is ignored by the architecture
(`none`), encoded through a dedicated learned replacement token
(`explicit_token`), or exposed explicitly as one extra 0/1 channel per feature
(`feature_mask`).

### Many-Class Classification

- `many_class_train_mode`
- `max_mixed_radix_digits`
- `many_class_base`
- `use_digit_position_embed`

These parameters matter only for classification and mostly only when the model
enters the many-class path.

### Task Heads

- `head_hidden_dim`

Regression also has a fixed, non-configurable `999`-quantile output grid.

## Interaction Notes

- `tabfoundry_simple` is frozen as the exact nanoTabPFN-style binary repro and
  benchmark anchor.
  It requires:
  - `task=classification`
  - `num_classes=2`
  - `input_normalization=train_zscore_clip`
  - `many_class_base=2`
    It reuses `d_icl`, `tficl_n_heads`, `tficl_n_layers`, and `head_hidden_dim`,
    and rejects non-default tabfoundry-only knobs such as grouped-token,
    row/column encoder, and many-class-path settings.
  - `missingness_mode=none` is the exact parity path.
  - `missingness_mode=feature_mask` and `missingness_mode=explicit_token` are
    supported opt-in deviations.
- `tabfoundry` supports `missingness_mode=none` and `feature_mask`.
  `missingness_mode=explicit_token` is rejected because the architecture always
  uses shifted grouped tokens.
- `tabfoundry_staged` is the classification-only staged research family.
  `model.stage` defaults to `nano_exact`, and non-null `model.stage` is rejected
  for `tabfoundry` and `tabfoundry_simple`.
- `model.stage` remains the legacy recipe-selector and compatibility surface.
  Supported recipe names are:
  - `nano_exact`
  - `label_token`
  - `shared_norm`
  - `prenorm_block`
  - `small_class_head`
  - `test_self`
  - `grouped_tokens`
  - `row_cls_pool`
  - `column_set`
  - `qass_context`
  - `many_class`
- New isolated staged experiments should prefer queue-managed
  `stage_label + module_overrides` on top of the base recipe rather than
  treating the legacy stage list as a promotion ladder.
- On the resolved staged surface, normalization mode depends on the effective
  feature encoder:
  - `feature_encoder=nano` keeps internal benchmark normalization
  - `feature_encoder=shared` uses the shared repo normalization pipeline and
    honors `input_normalization`
- Runtime missingness requirements:
  - `missingness_mode=none` works with the usual preprocessing settings
  - `missingness_mode!=none` requires `data.allow_missing_values=true`
  - `missingness_mode!=none` also requires `preprocessing.impute_missing=false`
- Architecture compatibility for missingness:
  - `feature_mask` is supported across `tabfoundry`, `tabfoundry_simple`, and
    `tabfoundry_staged`
  - `explicit_token` is only valid on scalar-token paths
  - staged `tokenizer=shifted_grouped` rejects `explicit_token`
- `module_overrides` supports these atomic change families:
  - `feature_encoder`
  - `post_encoder_norm`
  - `target_conditioner`
  - `tokenizer`
  - `column_encoder`
  - `row_pool`
  - `context_encoder`
  - `head`
  - `table_block_style`
  - `allow_test_self_attention`
- Important staged override constraints:
  - `tokenizer` overrides are ineffective while the effective feature encoder is
    `nano`
  - `post_encoder_norm` defaults to `none` and applies to the full cell table
    immediately before the transformer stack when set to `layernorm` or
    `rmsnorm`
  - `allow_test_self_attention=true` is only valid with
    `table_block_style=prenorm`
  - `head=many_class` requires a non-`none` `context_encoder`
- The low-level numeric tuning surface for `tabfoundry_staged` is still mainly
  `d_icl`, `tficl_n_heads`, `tficl_n_layers`, `head_hidden_dim`, and
  `input_normalization`. The sweep system adds `stage_label` and
  `module_overrides` so isolated structural changes are explicit and
  attributable.
- `feature_group_size` changes both compute and inductive bias. Larger groups
  reduce token count but make each token represent a wider local feature bundle.
- `many_class_base` affects both the small-class classifier head width and the
  many-class decomposition tree/radix. It does not currently control the branch
  threshold directly.
- The current small-class vs many-class split is still fixed in code at
  `num_classes > 10`. That matches the current default `many_class_base`, but
  it is not yet a separately configurable threshold.
- `d_col` and `d_icl` belong to different stages. Increasing only `d_col`
  changes tokenization/row-aggregation width without changing the final ICL
  encoder width.
- Exported bundles persist resolved model settings, so changing defaults only
  affects runs where the field is omitted.

## Minimal Override Examples

Paper-faithful tokenization is now the default:

```bash
uv run tab-foundry train experiment=cls_smoke
```

Grouped-token experiment:

```bash
uv run tab-foundry train experiment=cls_smoke model.feature_group_size=32
```

Per-feature missingness mask experiment:

```bash
uv run tab-foundry train \
  experiment=cls_smoke \
  data.allow_missing_values=true \
  preprocessing.overrides.impute_missing=false \
  model.missingness_mode=feature_mask
```

Frozen nanoTabPFN repro benchmark:

```bash
uv run tab-foundry train \
  experiment=cls_benchmark_linear_simple \
  data.manifest_path=<binary_manifest.parquet>
```

Staged benchmark family from the exact repro starting point:

```bash
uv run tab-foundry train \
  experiment=cls_benchmark_staged \
  data.manifest_path=<binary_manifest.parquet>
```

Queue-driven isolated staged delta surface:

```yaml
model:
  stage: nano_exact
  stage_label: delta_row_cls_pool
  module_overrides:
    row_pool: row_cls
  d_icl: 96
  tficl_n_heads: 4
  tficl_n_layers: 3
  head_hidden_dim: 192
```

Prefer emitting staged deltas through `reference/system_delta_campaign_template.md`
and the active queue row rather than relying on ad hoc CLI mapping syntax.

Many-class evaluation through full probabilities:

```bash
uv run tab-foundry eval \
  --checkpoint outputs/cls_smoke/checkpoints/best.pt \
  experiment=cls_smoke \
  model.many_class_train_mode=full_probs \
  model.many_class_base=12
```

Wider final ICL encoder:

```bash
uv run tab-foundry train \
  experiment=cls_smoke \
  model.d_icl=768 \
  model.tficl_n_layers=16
```

## Files To Update When The Config Surface Changes

If you add, remove, or rename a model config field, update all of these:

- `configs/model/default.yaml`
- `src/tab_foundry/model/spec.py`
- `src/tab_foundry/model/factory.py`
- `src/tab_foundry/model/architectures/tabfoundry.py` if constructor defaults change
- `docs/development/model-architecture.md`
- `docs/inference.md` if the field is serialized into export bundles
- tests that validate config resolution, export manifests, or checkpoint loading
