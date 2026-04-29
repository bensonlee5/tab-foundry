# Model Config Reference

Use this reference when you need to know which model settings matter, where
they come from, and how they resolve across training, evaluation, export, and
bundle loading.

Use these alongside this reference:

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
  - Provides fallback defaults for non-Hydra paths such as checkpoint-based
    evaluation, export reconstruction, and bundle loading.
- Checkpoint payload `config.model`
  - Persists the model settings used for a training run.
  - Takes precedence over fallback config when evaluating or exporting a
    checkpoint.
- Export manifest `model`
  - Persists the resolved model settings that an inference/runtime loader needs
    to reconstruct the model.

Current canonical default:

- repo default experiment: `arch = grid_sandwich`
- raw `configs/model/default.yaml`: `arch = tabfoundry_sandwich`
- `feature_group_size = 1`

That means normal repo composition uses the grid-preserving classification
family with one token per feature. The raw model-default file remains a
generic compatibility surface and is overridden by the default experiment.
Larger `feature_group_size` values such as `32` are non-default grouped-token
experiments that reduce token count and change the inductive bias. The staged
family remains loadable as a historical/reference lane, `tabfoundry_sandwich`
remains the previous carried comparison family, and `tabfoundry_simple`
remains the frozen control.

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

Checkpoint evaluation resolves settings from both the checkpoint payload and the
current Hydra config.

Resolution order:

1. `checkpoint["config"]["model"]`
1. `cfg.model`
1. `ModelBuildSpec` fallback defaults

This lets evaluation preserve the checkpoint's original architecture settings
while still tolerating older checkpoints that omitted newer fields.
When `feature_group_size` is omitted, checkpoint-backed reconstruction now
resolves it to `1`. If the saved weights are incompatible with that per-feature
default, loading fails with a compatibility error instead of silently
reconstructing a grouped-token model. Legacy checkpoints that omitted
`feature_group_size` and were trained with grouped tokens must be regenerated
or loaded with an explicit `feature_group_size` override.

### Export Checkpoint

Export reconstruction resolves the model spec from the checkpoint payload.

Resolution order:

1. `checkpoint["config"]["model"]`
1. `ModelBuildSpec` fallback defaults

The resolved spec is then written into `manifest.json`, including the embedded
`manifest.inference` section for v3 bundles.
As with checkpoint evaluation, omitted `feature_group_size` values now resolve
to `1`, and incompatible grouped-token legacy checkpoints fail fast with a
compatibility error.

### Load Export Bundle

Bundle loading reconstructs the model from the manifest.

Resolution order:

1. `manifest.model`
1. `ModelBuildSpec` fallback defaults for omitted optional manifest fields

This exists mainly so validators and loaders can tolerate older manifests that
did not yet serialize every reconstruction field.

## Parameter Reference

| Name | Type | Default | Applies To | Meaning |
| ---- | ---- | ---- | ---- | ---- |
| `arch` | `str` | `"tabfoundry_sandwich"` in `model/default.yaml`; repo default experiment overrides to `"grid_sandwich"` | classification | Model architecture. Supported values are `tabfoundry_simple` (frozen binary repro), `tabfoundry_staged` (historical reference family), `tabfoundry_sandwich` (previous carried fixed-latent family), `routed_sandwich` (routed sidecar), and `grid_sandwich` (active carried grid-preserving family). |
| `stage` | `str \| null` | `null` | classification | Stage selector for `tabfoundry_staged`. `null` resolves to `nano_exact` when `arch=tabfoundry_staged`; non-null values are rejected for non-staged families. |
| `stage_label` | `str \| null` | `null` | classification | Optional reporting label for staged runs. When present, benchmark/profile metadata uses this label while the underlying recipe still resolves from `stage`. |
| `module_overrides` | `mapping \| null` | `null` | classification | Additive atomic staged-surface overrides. Supported top-level keys are `feature_encoder`, `post_encoder_norm`, `post_stack_norm`, `target_conditioner`, `tokenizer`, `column_encoder`, `row_pool`, `context_encoder`, `head`, `table_block_style`, `table_block_residual_scale`, and `allow_test_self_attention`. Rejected for non-staged families. |
| `d_col` | `int` | `128` | both | Width of grouped feature tokens and the column encoder. |
| `d_icl` | `int` | `512` | both | Width of row embeddings and the final in-context encoder. |
| `input_normalization` | `str` | `"none"` | both | Train/test feature normalization mode. Supported values are `none`, `train_zscore`, and `train_zscore_clip`. |
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
| `staged_dropout` | `float` | `0.0` | staged | Dropout used by the staged family. |
| `pre_encoder_clip` | `float \| null` | `null` | staged, sandwich, grid | Optional finite-value clip applied before feature encoding. |
| `sandwich_latents` | `int` | `24` | sandwich | Fixed latent-array size for `tabfoundry_sandwich`; rejected for `grid_sandwich` when explicitly supplied. |
| `sandwich_layers` | `int` | `2` | sandwich, grid | Number of repeated stages. For `grid_sandwich`, this is the number of row/column grid-mixer layers. |
| `sandwich_heads` | `int` | `4` | sandwich, grid | Attention heads used by the shared attention blocks. For `grid_sandwich`, this covers row mixers, column ISAB mixers, and row pooling. |
| `sandwich_ff_expansion` | `int` | `2` | sandwich, grid | Feedforward expansion factor used inside shared attention blocks. |
| `sandwich_activation` | `str` | `"gelu"` | sandwich, grid | Feedforward activation used inside shared attention blocks. `rational` selects the local version-A `5/4` GELU-initialized rational activation. |
| `sandwich_block_norm` | `str` | `"layernorm"` | sandwich, grid | Internal pre-norm module used inside shared attention blocks. `none` disables block-local norms; the global `norm_type` contract remains `layernorm`. |
| `sandwich_packed_attention` | `bool` | `false` | sandwich, grid | Opt-in speedrun path that keeps the same attention weights but fuses self-attention QKV and cross-attention KV projections before SDPA. |
| `grid_residual_mode` | `str` | `"prenorm"` | grid | Grid-core residual topology. `hyper_connection_lite` uses two cell-token residual streams with width/depth mixing around each grid row/column mixer. |
| `grid_attention_mode` | `str` | `"standard"` | grid | Grid-core attention family. `differential` computes `softmax(Q1K1^T)V - lambda * softmax(Q2K2^T)V` with one learned scalar initialized to `0.1` per attention block. |
| `grid_ffn_mode` | `str` | `"gelu"` | grid | Grid-core FFN family. `swiglu` and `geglu` use hidden width `round_up(ceil((2/3) * sandwich_ff_expansion * d_icl), 8)` to stay near the GELU FFN parameter budget. |
| `grid_recurrence_steps` | `int \| null` | `null` | grid | When null, the grid core uses `sandwich_layers` distinct layers. When positive, the grid core runs for this many recurrent refinement steps. |
| `grid_recurrence_unique_layers` | `int \| null` | `null` | grid | Optional recurrent-core cycle size. When null with `grid_recurrence_steps` set, one `_GridMixerLayer` is shared; when positive, that many distinct grid layers are cycled through the recurrent steps. |
| `grid_moe_scope` | `str` | `"none"` | grid | Optional sparse MoE scope. `none` keeps dense FFNs; `grid_core_ffn` replaces only grid-core SwiGLU FFNs with sparse experts. |
| `grid_moe_num_experts` | `int` | `1` | grid | Number of MoE experts when `grid_moe_scope` is enabled. Enabled MoE requires a value greater than `1`. |
| `grid_moe_top_k` | `int` | `1` | grid | Number of experts selected per token. Must be no larger than `grid_moe_num_experts`; v1 uses dynamic sparse dispatch with no token dropping. |
| `grid_moe_router_init_std` | `float` | `0.01` | grid | Normal initialization standard deviation for MoE router weights. |
| `grid_moe_normalize_top_k` | `bool` | `false` | grid | When true, selected top-k router probabilities are renormalized per token before expert outputs are combined. Defaults false to preserve the original raw-probability weighting. |
| `classification_logit_softcap` | `float \| null` | `null` | grid | Optional classification-logit softcap. When set, grid logits are transformed as `cap * tanh(logits / cap)` before loss/eval/export consumers see them. |
| `attention_qk_norm` | `bool` | `false` | grid | Optional QK normalization for grid-sandwich attention sites. Query/key heads are L2-normalized with a learnable per-head scale initialized to `sqrt(head_dim)`. |
| `feature_type_conditioning` | `str` | `"film"` | sandwich, grid | Feature-type conditioning path for cell states. `film` modulates encoded cells after the shared feature encoder; `additive_embedding` is retained only for legacy checkpoint reconstruction. |
| `floating_likelihood` | `str` | `"single_gaussian"` | sandwich | Floating-cell likelihood family for the legacy sandwich `cell_bpc` generative lane. Active classification benchmarks use `training.loss_surface=classification` instead. |
| `integer_likelihood` | `str` | `"hybrid_mixture"` | sandwich | Integer-cell likelihood family for the legacy sandwich `cell_bpc` generative lane. `hybrid_mixture` combines dynamic-support discrete likelihood with a single-Gaussian branch. Active classification benchmarks use `training.loss_surface=classification` instead. |

Training-side classification stability also exposes
`training.classification_z_loss_coeff`, defaulting to `0.0`. When positive, the
active training objective adds `coeff * mean(logsumexp(logits)^2)` for
classification-logit outputs; evaluation and benchmark metrics remain plain
classification loss/metrics.

Training-side grid MoE losses expose `training.moe_load_balance_loss_coeff` and
`training.moe_router_z_loss_coeff`, both defaulting to `0.0`. When positive,
classification training adds the model-emitted MoE auxiliary losses to the
task loss; evaluation keeps the task objective unchanged while still reporting
detached route-health metrics when the model emits them.

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
- `sandwich_layers`
- `head_hidden_dim`

`d_col` controls the token width before row aggregation on staged surfaces.
`d_icl` controls the active grid cell-state width and the shared attention
block width.

### Grid Core And Shared Attention

- `sandwich_layers`
- `sandwich_heads`
- `sandwich_ff_expansion`
- `sandwich_activation`
- `sandwich_block_norm`
- `sandwich_pre_row_attention_layers`
- `sandwich_pre_column_attention_layers`
- `sandwich_pre_column_inducing_tokens`
- `grid_residual_mode`
- `grid_attention_mode`
- `grid_ffn_mode`
- `grid_recurrence_steps`
- `grid_recurrence_unique_layers`
- `grid_moe_scope`
- `grid_moe_num_experts`
- `grid_moe_top_k`
- `grid_moe_router_init_std`
- `grid_moe_normalize_top_k`
- `classification_logit_softcap`
- `attention_qk_norm`

These parameters control the active `grid_sandwich` core and the previous
`tabfoundry_sandwich` comparison family. In `grid_sandwich`, `sandwich_layers`
counts alternating row-wise feature self-attention and column-wise row ISAB
layers while preserving the `[row, feature]` grid. The pre-grid mixer uses the
same row/column pattern before label conditioning. `sandwich_activation` only
changes the grid/shared attention FF blocks; the auxiliary heads remain on GELU.
`sandwich_block_norm` only changes the internal block pre-norm
modules; `norm_type` still stays globally fixed to `layernorm` for grid and
sandwich families.

The `grid_*` knobs are grid-only experiment gates. Defaults preserve the current
anchor topology and parameter count. Non-default values change only the
row/column grid core: pre-grid mixers and test-row pooling keep the shared
sandwich blocks.

`grid_moe_scope=grid_core_ffn` is classification-only and currently requires
`grid_ffn_mode=swiglu`. It replaces only recurrent grid-core row/column FFNs
with sparse SwiGLU experts. Router probabilities remain differentiable, the
selected experts are evaluated dynamically, and v1 does not drop tokens or
apply a capacity factor.

### Tokenization And Preprocessing

- `input_normalization`
- `feature_group_size`

`feature_group_size` is the highest-leverage token-count knob:

- `1`: one token per feature, paper-faithful default
- `>1`: grouped-token mode, fewer tokens and lower attention cost

### Many-Class Classification

- `many_class_train_mode`
- `max_mixed_radix_digits`
- `many_class_base`
- `use_digit_position_embed`

These parameters matter only for classification and mostly only when the model
enters the many-class path.

### Task Heads And Candidate-Specific Capacity

- `head_hidden_dim`
- `many_class_base`
- `pre_encoder_clip`

The repo is currently classification-only. Regression will be rebuilt later on
top of the promoted post-staged architecture line rather than restored from the
removed legacy family.

## Interaction Notes

- `tabfoundry_simple` is frozen as the exact nanoTabPFN-style binary repro and
  benchmark anchor.
  It requires:
  - `task=classification`
  - `num_classes=2`
  - `input_normalization=train_zscore_clip`
  - `many_class_base=2`
    It reuses `d_icl`, `tficl_n_heads`, `tficl_n_layers`, and `head_hidden_dim`,
    and rejects staged-only knobs such as grouped-token, row/column encoder,
    and many-class-path settings.
- `tabfoundry_staged` is the classification-only incumbent reference family.
  `model.stage` defaults to `nano_exact`, and non-null `model.stage` is
  rejected for non-staged families.
- `grid_sandwich` is the active classification-only grid-preserving candidate
  family. The live forward-path walkthrough, runtime contract, feature-type
  metadata contract, and technical diagram now live in
  `docs/development/model-architecture.md`. It rejects `stage`,
  `stage_label`, `module_overrides`, and inherited latent/summary-only
  sandwich fields such as `sandwich_latents`,
  `sandwich_summary_tokens_per_axis`, and
  `sandwich_self_attention_per_cross`.
  It currently requires:
  - `task=classification`
  - `norm_type=layernorm`
  - `2 <= num_classes <= many_class_base`
    Its main public tuning knobs are `sandwich_layers`, `sandwich_heads`,
    `sandwich_ff_expansion`, `sandwich_activation`, `sandwich_block_norm`,
    `sandwich_pre_row_attention_layers`,
    `sandwich_pre_column_attention_layers`,
    `sandwich_pre_column_inducing_tokens`, `d_icl`, `head_hidden_dim`,
    `grid_residual_mode`, `grid_attention_mode`, `grid_ffn_mode`,
    `grid_recurrence_steps`, `grid_recurrence_unique_layers`,
    `grid_moe_scope`, `grid_moe_num_experts`, `grid_moe_top_k`,
    `grid_moe_router_init_std`, `grid_moe_normalize_top_k`,
    `input_normalization`, and `pre_encoder_clip`.
    `sandwich_layers` counts alternating row/column grid-mixer layers unless
    `grid_recurrence_steps` is positive. `feature_types` are required at
    runtime and on `forward_batched(..., feature_types=...)`. Export-bundle `preprocessor`
    payloads stay policy-only and do not serialize this list.
- `tabfoundry_sandwich` is the previous carried fixed-latent comparison family.
  Its latent, summary, and dual-readout details remain in historical evidence
  and checkpoint compatibility surfaces rather than the active architecture
  walkthrough.
- `model.stage` remains the stable public recipe selector and compatibility
  surface for the staged family.
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
- Treat the public `stage` surface as the canonical migration ladder for live
  staged-family attribution work.
- Use queue-managed `stage_label + module_overrides` to make isolated
  attribution rows explicit, especially when a public stage bundles more than
  one mechanism or when a control row needs to hold the ladder steady.
- On the resolved staged surface, normalization mode depends on the effective
  feature encoder:
  - `feature_encoder=nano` keeps internal benchmark normalization
  - `feature_encoder=shared` uses the shared repo normalization pipeline and
    honors `input_normalization`
- `module_overrides` supports these atomic change families:
  - `feature_encoder`
  - `post_encoder_norm`
  - `post_stack_norm`
  - `target_conditioner`
  - `tokenizer`
  - `column_encoder`
  - `row_pool`
  - `context_encoder`
  - `head`
  - `table_block_style`
  - `table_block_residual_scale`
  - `allow_test_self_attention`
- Important staged override constraints:
  - `tokenizer` overrides are ineffective while the effective feature encoder is
    `nano`
  - `post_encoder_norm` defaults to `none` and applies to the cell table
    immediately before the transformer stack when set to `layernorm` or
    `rmsnorm`
  - `post_stack_norm` defaults to `none` and applies after the full
    transformer-block stack but before row pooling
  - `table_block_residual_scale` defaults to `none`; `depth_scaled` multiplies
    each prenorm residual branch by `1 / sqrt(3 * tficl_n_layers)`
  - `allow_test_self_attention=true` is only valid with
    `table_block_style=prenorm`
  - `head=many_class` requires a non-`none` `context_encoder`
- The low-level numeric tuning surface for `tabfoundry_staged` is still mainly
  `d_icl`, `tficl_n_heads`, `tficl_n_layers`, `head_hidden_dim`, and
  `input_normalization`. The sweep system adds `stage_label` and
  `module_overrides` so isolated structural changes are explicit and
  attributable without replacing the public stage ladder.
- The low-level numeric tuning surface for `grid_sandwich` is mainly
  `sandwich_layers`, `sandwich_heads`, `sandwich_ff_expansion`,
  `sandwich_activation`, `sandwich_block_norm`,
  `d_icl`, `head_hidden_dim`, `input_normalization`, and `pre_encoder_clip`.
- `feature_group_size` changes both compute and inductive bias. Larger groups
  reduce token count but make each token represent a wider local feature bundle.
  This knob does not apply to the current grid architecture.
- `many_class_base` affects both the small-class classifier head width and the
  many-class decomposition tree/radix on staged models, and it also sets the
  direct-head width and maximum supported class count on grid and sandwich models.
  It does not currently control the branch threshold directly.
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
tab-foundry train run experiment=cls_smoke
```

Grouped-token experiment:

```bash
tab-foundry train run experiment=cls_smoke model.feature_group_size=32
```

Frozen nanoTabPFN repro benchmark:

```bash
tab-foundry train run \
  experiment=cls_benchmark_linear_simple \
  data.manifest_path=<binary_manifest.parquet>
```

Staged benchmark family from the exact repro starting point:

```bash
tab-foundry train run \
  experiment=cls_benchmark_staged_corpus \
  data.corpus_ref=tf_rd_013_current_corpus_default_v1
```

Current grid-anchor workstation run. This resolves to the TF-RD-026 row `10`
two-layer recurrent SwiGLU grid core:

```bash
tab-foundry train run \
  experiment=cls_workstation_grid_sandwich
```

Core-only grid MoE screen row:

```bash
tab-foundry train run \
  experiment=cls_workstation_grid_sandwich \
  model.grid_moe_scope=grid_core_ffn \
  model.grid_moe_num_experts=4 \
  training.moe_load_balance_loss_coeff=1e-2 \
  training.moe_router_z_loss_coeff=1e-4
```

Use `experiment=cls_workstation_sandwich` when you need the previous
shared-latent sandwich comparison surface.

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
tab-foundry eval checkpoint \
  --checkpoint outputs/cls_smoke/checkpoints/best.pt \
  experiment=cls_smoke \
  model.many_class_train_mode=full_probs \
  model.many_class_base=12
```

Wider final ICL encoder:

```bash
tab-foundry train run \
  experiment=cls_smoke \
  model.d_icl=768 \
  model.tficl_n_layers=16
```

## Files To Update When The Config Surface Changes

If you add, remove, or rename a model config field, update all of these:

- `configs/model/default.yaml`
- `src/tab_foundry/model/spec.py`
- `src/tab_foundry/model/factory.py`
- `src/tab_foundry/model/architectures/tabfoundry_sandwich/model.py`
- `src/tab_foundry/model/architectures/tabfoundry_staged/model.py`
- `src/tab_foundry/model/architectures/tabfoundry_simple.py`
- `docs/development/model-architecture.md`
- `docs/inference.md` if the field is serialized into export bundles
- tests that validate config resolution, export manifests, or checkpoint loading
