# Stability Ladder Comparison Matrix

| Row | Delta Ref | Table Block | Warmup | Grad Clip | Dropout | Pre-Encoder Clip | Status |
|-----|-----------|-------------|--------|-----------|---------|------------------|--------|
| A | prenorm_baseline | prenorm | 0.0 | 0.0 | 0.0 | null | ready |
| B | warmup_grad_clip | prenorm | 0.05 | 1.0 | 0.0 | null | ready |
| C | shared_dropout | prenorm | 0.05 | 1.0 | 0.1 | null | ready |
| D | pre_encoder_guard | prenorm | 0.05 | 1.0 | 0.1 | 10.0 | ready |

## Design

Each row adds exactly one stability mechanism to the previous row, forming a
cumulative ladder. This design isolates the marginal contribution of each mechanism:

- **A -> B**: Effect of warmup + grad clip on early-step stability
- **B -> C**: Effect of unified dropout on generalization stability
- **C -> D**: Effect of input clipping on extreme-value robustness

## Anchor Surface

All rows share the same base architecture:

- `arch: tabfoundry_staged`, `stage: prenorm_block`
- `feature_encoder: shared`, `post_encoder_norm: layernorm`
- `table_block_style: prenorm`, `head: binary_direct`
- `tokenizer: scalar_per_feature`, `row_pool: target_column`
- `context_encoder: none`, `column_encoder: none`

## Metrics to Track

- Best/final validation loss
- Early-step loss stability (loss delta, loss EMA)
- Gradient norm statistics (mean, max, clip frequency)
- NaN guard trigger count
- Training throughput (steps/second)
