# Stability Ladder Comparison Matrix

| Row | Delta Ref | Table Block | Warmup | Grad Clip | Dropout | Pre-Encoder Clip | Status |
|-----|-----------|-------------|--------|-----------|---------|------------------|--------|
| A | prenorm_baseline | prenorm | 0.0 | 1.0 | 0.0 | null | done |
| B | warmup_grad_clip | prenorm | 0.05 | 1.0 | 0.0 | null | done |
| C | shared_dropout | prenorm | 0.05 | 1.0 | 0.1 | null | done |
| D | pre_encoder_guard | prenorm | 0.05 | 1.0 | 0.1 | 10.0 | done |

## Design

Each row adds exactly one stability mechanism to the previous row, forming a
cumulative ladder. This design isolates the marginal contribution of each mechanism:

- **A -> B**: Effect of cosine warmup on early-step stability
- **B -> C**: Effect of unified dropout on generalization stability
- **C -> D**: Effect of input clipping on extreme-value robustness

Note: All rows share `grad_clip=1.0` from the base config. The real isolation
between A and B is warmup alone, not grad clip (which is always on).

## Results (2500 steps, H100, prior dump 300k_150x5_2.h5)

| Metric | A (baseline) | B (+warmup) | C (+dropout) | D (+clip) |
|--------|-------------|-------------|--------------|-----------|
| Final loss | 0.4729 | 0.4727 | 0.4742 | 0.4725 |
| Max grad norm | **13.161** | 2.754 | 2.725 | 2.759 |
| Mean grad norm | 0.288 | 0.257 | 0.285 | 0.284 |
| Max loss delta | **1.394** | 0.220 | 0.222 | 0.222 |
| Max train loss | **2.226** | 0.762 | 0.760 | 0.760 |
| Loss variance | **0.00306** | 0.00153 | 0.00153 | 0.00153 |
| Min train loss | 0.365 | 0.365 | 0.362 | 0.363 |
| Wall time (s) | 55.1 | 55.2 | 55.5 | 56.4 |

## Conclusions

1. **Warmup is the dominant stabilizer (A->B).** Cosine warmup with 5%
   warm-up ratio eliminates the early-step loss spike entirely. Max gradient
   norm drops 4.8x (13.16 -> 2.75), max step-to-step loss delta drops 6.3x
   (1.39 -> 0.22), and loss variance halves (0.0031 -> 0.0015).

1. **Dropout has negligible marginal stability effect (B->C).** At 2500
   steps on clean prior-dump data, `staged_dropout=0.1` does not measurably
   change gradient norms, loss variance, or loss delta. Its regularization
   benefit likely requires longer training or downstream fine-tuning to
   manifest.

1. **Input clip has negligible marginal effect (C->D).** `pre_encoder_clip=10.0`
   shows no measurable change because the prior-dump data is clean synthetic
   data without extreme input values. This mechanism is defense-in-depth for
   real-world data with outliers.

1. **No NaN/Inf in any rung.** All four runs completed cleanly with zero
   non-finite feature or label values detected.

1. **Recommendation:** Warmup should be the default for prenorm block training.
   Dropout and input clip are low-cost additions worth keeping as defaults for
   robustness on non-synthetic data, but they are not stability-critical on
   this prior surface.

## Anchor Surface

All rows share the same base architecture:

- `arch: tabfoundry_staged`, `stage: prenorm_block`
- `feature_encoder: shared`, `post_encoder_norm: layernorm`
- `table_block_style: prenorm`, `head: binary_direct`
- `tokenizer: scalar_per_feature`, `row_pool: target_column`
- `context_encoder: none`, `column_encoder: none`
