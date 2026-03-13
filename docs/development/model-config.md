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

- `feature_group_size = 1`

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

The resolved spec is then written into `manifest.json` and
`inference_config.json`.
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

## Configuration Groups

### Core Widths And Depth

These parameters set the overall model size:

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

### Task Heads

- `head_hidden_dim`

Regression also has a fixed, non-configurable `999`-quantile output grid.

## Interaction Notes

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
