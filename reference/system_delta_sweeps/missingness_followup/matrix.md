# Missingness Follow-Up Comparison Matrix

| Row | Delta Ref | Benchmark Bundle | Tokenizer | Prior Missingness | Status |
|-----|-----------|------------------|-----------|-------------------|--------|
| A | nan_token_no_prior_missingness | binary_large_allow_missing | scalar_per_feature_nan_mask | disabled | ready |
| B | nan_token_prior_missingness_005 | binary_large_allow_missing | scalar_per_feature_nan_mask | fixed 0.05 | ready |

## Design

This sweep is about explicit missingness handling on the stabilized prenorm
surface. It is not an input-normalization sweep.

- Both rows keep `input_normalization` fixed at the promoted default
  `train_zscore_clip`.
- Both rows swap in the fixed-width `scalar_per_feature_nan_mask` tokenizer so
  each feature token becomes `value + missingness bit` without changing token
  count.
- **A** tests whether the explicit missingness-aware tokenizer is enough on its
  own.
- **B** adds fixed `5%` synthetic missingness during prior training so the
  missingness path is explicitly trained before evaluation on the large bundle.

## Acceptance

- Rank rows by benchmark ROC AUC on the large bundle first, then divergence or
  non-finite incidence, then drift/stability diagnostics.
- Treat row A underperformance as evidence that explicit missingness
  representation still needs missingness-aware prior exposure.
- Promote row B only if it materially improves over row A without introducing a
  new instability regime.

## Anchor Surface

Both rows keep the stabilized prenorm foundation fixed aside from the tokenizer
and the optional synthetic prior-missingness setting.

- `arch: tabfoundry_staged`, `stage: prenorm_block`
- `feature_encoder: shared`, `post_encoder_norm: layernorm`
- `table_block_style: prenorm`, `head: binary_direct`
- `row_pool: target_column`, `context_encoder: none`, `column_encoder: none`
- `training.surface_label: prior_cosine_warmup`
- Benchmark bundle: `src/tab_foundry/bench/nanotabpfn_openml_binary_large_v1.json`
- `data.surface_overrides.allow_missing_values: true`
