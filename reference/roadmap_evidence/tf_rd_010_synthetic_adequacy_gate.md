# TF-RD-010: Synthetic Adequacy Gate On Factorization-Correct Dagzoo Data

This note records the immediate TF-RD-010 gate that now blocks any new medium
or large rerun evidence.

## Why This Gate Exists

- `dagzoo` factorization changed to the equation `(1)` semantics, so prior
  TF-RD-010 training dynamics are operational context only.
- The next trustworthy question is narrower than "what training regime wins?":
  - is the refreshed synthetic data generated under the intended factorization?
  - can the current sandwich model learn on that refreshed data at all?
  - how do variance and teacher-relative bias proxies move with `n`?

## Active Gate

- Diagnostic id: `tf_rd_010_synthetic_adequacy_v1`
- Refreshed corpus family:
  - `tf_rd_010_dagzoo_medium_control_v3`
  - `tf_rd_010_missingness_mcar_v3`
  - `tf_rd_010_missingness_mar_v3`
  - `tf_rd_010_missingness_mnar_v3`
  - `tf_rd_010_factorized_canary_v1`
- Refreshed target semantics:
  - the `v3` adequacy corpora pin `dagzoo`'s near-dense target-parent prior
  - `target_parent_prior=near_max_mixture`
  - modal target-parent count remains at the maximum allowed feature count
  - only a small tail of datasets falls below `sqrt(m)` target sparsity
- Canonical metric key remains
  `final_log_loss_at_matched_regime_budget`, interpreted explicitly here as
  label-target log loss per test cell.

## Adequacy Outputs

The adequacy readout should end in one of three buckets:

- `generator_problem`
  - simple baselines also fail, or teacher-conditional diagnostics look
    inconsistent with the emitted factorization-correct synthetic data
- `training_regime_problem`
  - simple baselines learn, the sandwich model underperforms, and
    variance/bias trends still indicate a learnable target family
- `architecture_capacity_problem`
  - simple baselines learn, sandwich learns weakly, and the error-vs-`n`
    pattern suggests bias remains dominant even as variance falls

## Blocked Packages

- `tf_rd_010_classification_evolution_medium_v4`
- `tf_rd_010_classification_evolution_large_v2`

These packages are historical operational drafts only until the adequacy gate is
interpreted under issue [#205](https://github.com/bensonlee5/tab-foundry/issues/205).
