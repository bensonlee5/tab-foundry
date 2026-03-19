# Model Architecture

This document describes the current model surface in `tab-foundry`.

The repo now has one active architecture-development surface:

- `tabfoundry_staged`: the staged classification family used for new model work

It also keeps one frozen anchor:

- `tabfoundry_simple`: the exact nanoTabPFN-style binary compatibility path

The legacy `tabfoundry` family has been removed. Regression is also removed for
now and will be rebuilt later on top of `tabfoundry_staged`.

Related docs:

- `docs/development/model-config.md`
- `docs/inference.md`

Related code paths:

- `src/tab_foundry/model/architectures/tabfoundry_staged/`
- `src/tab_foundry/model/architectures/tabfoundry_simple.py`
- `src/tab_foundry/model/components/`
- `src/tab_foundry/model/spec.py`
- `src/tab_foundry/model/factory.py`

## High-Level Structure

`tabfoundry_staged` is a resolved-surface classifier. Construction starts from
`ModelBuildSpec`, resolves a public `stage` plus optional
`module_overrides`, then builds a concrete subsystem mix.

The forward path is organized as:

1. input preparation and train/test normalization
1. feature tokenization / feature encoding
1. target conditioning
1. table blocks over row-major cell tokens
1. optional column encoder
1. row pooling
1. optional context encoder
1. direct classification head or many-class head

The implementation is split across:

- `recipes.py`: the public staged recipe registry
- `resolved.py`: surface resolution from `stage` and `module_overrides`
- `builders.py`: subsystem construction
- `forward_common.py`: shared input prep, normalization, token building, and
  row/context helpers
- `direct_head.py`: small-class direct-head flow
- `many_class.py`: many-class hierarchical flow
- `subsystems.py`: reusable staged subsystem implementations
- `model.py`: the public `TabFoundryStagedClassifier` facade

## Public Stage Surface

The supported public stages are:

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

`model.stage` is still the stable public selector, but new research work should
prefer queue-managed `stage_label + module_overrides` so changes stay isolated
and attributable.

At a high level, the staged family evolves along these axes:

- feature encoder: `nano` vs `shared`
- target conditioner: mean-padded linear vs label token
- table block: nano-style postnorm vs prenorm
- tokenizer: scalar-per-feature vs shifted-grouped variants
- row pool: target-column vs row-CLS pooling
- column encoder: none vs `tfcol`
- context encoder: none vs plain vs QASS
- head: binary direct, small-class, or many-class

## Normalization And Tokenization

The staged family has two normalization regimes:

- `internal`: the nano-compatible path used by early benchmark-anchor stages
- `shared`: the repo-wide normalization path used by later staged surfaces

The normalization owner is resolved from the stage recipe rather than inferred
from CLI defaults.

Tokenization choices include:

- `scalar_per_feature`
- `scalar_per_feature_nan_mask`
- `shifted_grouped`

The non-finite-aware tokenizer keeps separate pre-embedding channels for:

- `NaN`
- `+inf`
- `-inf`

Finite-only clipping is applied before encoding when `pre_encoder_clip` is set.

## Heads And Class Coverage

The repo is classification-only today.

The staged family has two main head modes:

- direct heads for binary and ordinary small-class classification
- hierarchical many-class routing for larger class counts

The many-class path uses:

- mixed-radix conditioning
- optional digit-position embeddings
- hierarchical class routing
- `many_class_train_mode` of `path_nll` or `full_probs`

The staged many-class implementation is intentionally still single-task at the
`TaskBatch` level. Batched tensor fast paths exist for direct-head execution,
but `_forward_many_class()` now errors clearly if asked to process `B > 1`.

## `tabfoundry_simple`

`tabfoundry_simple` remains as the frozen exact binary anchor.

Its role is deliberately narrow:

- exact benchmark-anchor reproduction
- binary-only classification
- compatibility baseline for comparisons against `tabfoundry_staged`

It is not the place for new architecture work.

## Output Surface

Shared model outputs now live in `src/tab_foundry/model/outputs.py`.

Active output type:

- `ClassificationOutput`

There is no longer a repo-supported `RegressionOutput`.

## Code Navigation Map

- `src/tab_foundry/model/spec.py`
  - canonical build spec, supported arch/task/stage values, and checkpoint
    compatibility rules
- `src/tab_foundry/model/factory.py`
  - model construction for `tabfoundry_simple` and `tabfoundry_staged`
- `src/tab_foundry/model/architectures/tabfoundry_staged/model.py`
  - public staged classifier facade
- `src/tab_foundry/model/architectures/tabfoundry_staged/recipes.py`
  - public stage registry
- `src/tab_foundry/model/architectures/tabfoundry_staged/resolved.py`
  - resolved staged surface metadata
- `src/tab_foundry/model/architectures/tabfoundry_staged/subsystems.py`
  - staged subsystem implementations
- `src/tab_foundry/model/architectures/tabfoundry_staged/forward_common.py`
  - shared forward-path helpers
- `src/tab_foundry/model/architectures/tabfoundry_staged/direct_head.py`
  - direct-head execution path
- `src/tab_foundry/model/architectures/tabfoundry_staged/many_class.py`
  - many-class execution path
- `src/tab_foundry/model/architectures/tabfoundry_simple.py`
  - frozen exact binary anchor
- `src/tab_foundry/model/components/`
  - reusable QASS, many-class, non-finite, and normalization components

## Maintenance Notes

- `tabfoundry_staged` is the only model family that should absorb new feature
  work.
- `tabfoundry_simple` should change only for bug fixes or compatibility
  maintenance.
- Future regression support should be introduced as a staged-family extension,
  not by reviving the removed legacy `tabfoundry` family.
