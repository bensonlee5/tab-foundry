# Stability Follow-Up Comparison Matrix

Source files:

- `reference/system_delta_sweeps/stability_followup/sweep.yaml`
- `reference/system_delta_sweeps/stability_followup/queue.yaml`
- `reference/system_delta_catalog.yaml`

This follow-up is a `delta_prenorm_block` bridge sweep, not a continuation of the stronger post-norm `binary_md_v3` surface.
The queue is intentionally expanded to 12 rows so hyperparameters can be calibrated on the prenorm bridge surface and one promotion-gate confirmation row can be run before any later larger-model work.

## Anchor

- Stage: `nano_exact`
- Locked bridge overrides: `table_block_style=prenorm`, `allow_test_self_attention=false`
- Feature encoder: `nano`
- Target conditioning: `mean_padded_linear`
- Default readout for rows 1-10: `target_column`
- Baseline training recipe: `prior_cosine_warmup`, `grad_clip=1.0`, `max_steps=2500`, `trace_activations=true`
- Prior-dump policy for the whole queue: `training.prior_dump_non_finite_policy=skip`

## Diagnostics

Every row must report the same training-dynamics diagnostics in both `telemetry.json` and wandb:

- clipped-step fraction
- encoder/head gradient ratio windows: `early_1_25`, `post_warmup_100`, `final_10pct`
- activation windows for `post_feature_encoder` and `pre_transformer`

## Rows

| Order | Row | Purpose | Key Change |
| --- | --- | --- | --- |
| 1 | `dpnb_baseline_cosine_warmup_2500` | Bridge baseline | Cosine warmup on `delta_prenorm_block` |
| 2 | `dpnb_linear_decay_lr4e3` | Decay-only check | `prior_linear_decay`, no warmup |
| 3 | `dpnb_linear_warmup_decay_lr4e3_warm5` | Center schedule candidate | `prior_linear_warmup_decay`, `lr_max=0.004`, `warmup_ratio=0.05` |
| 4 | `dpnb_linear_warmup_decay_lr3e3_warm5` | Low-LR neighbor | `lr_max=0.003` |
| 5 | `dpnb_linear_warmup_decay_lr5e3_warm5` | High-LR neighbor | `lr_max=0.005` |
| 6 | `dpnb_linear_warmup_decay_lr4e3_warm2` | Short warmup | `warmup_ratio=0.02` |
| 7 | `dpnb_linear_warmup_decay_lr4e3_warm10` | Long warmup | `warmup_ratio=0.10` |
| 8 | `dpnb_linear_warmup_decay_lr4e3_warm5_clip05` | Tight clip probe | `grad_clip=0.5` |
| 9 | `dpnb_linear_warmup_decay_lr4e3_warm5_wd5e4` | Small weight decay | `weight_decay=5e-4` |
| 10 | `dpnb_linear_warmup_decay_lr4e3_warm5_adamw` | Optimizer-family comparator | `optimizer.name=adamw` |
| 11 | `dpnb_row_cls_cls2_linear_warmup_decay` | Bounded RowPool revisit | `row_pool=row_cls`, `tfrow_cls_tokens=2` |
| 12 | `dpnb_row_cls_cls2_linear_warmup_decay_warm10` | Promotion confirmation row | Row 11 plus `warmup_ratio=0.10` |

## Interpretation

- Rows 1-10 are training-dynamics calibration rows. Rank by best ROC AUC first, final ROC AUC second, drift third.
- Row 3 is the center of the local calibration neighborhood. Rows 4-10 should be interpreted relative to it.
- Row 11 is the bounded RowPool reopen. Row 12 is the single promotion-gate confirmation row that combines the winning RowPool seed with the best target-column schedule.
- Input normalization and long-horizon claims are intentionally out of scope for this queue.

## Outcome

- Promoted `dpnb_row_cls_cls2_linear_warmup_decay` as the new bridge-surface default for `cls_benchmark_staged_prior`.
- Row 11 was the clean winner for promotion: best/final ROC AUC `0.7650`, zero best-checkpoint drift, and much better encoder-head balance than the target-column rows.
- Row 12 (`dpnb_row_cls_cls2_linear_warmup_decay_warm10`) confirmed that `row_cls` is genuinely reopened on the bridge surface, but it did not improve final ROC AUC materially over row 11 and reintroduced small drift plus a higher clipped-step fraction.
