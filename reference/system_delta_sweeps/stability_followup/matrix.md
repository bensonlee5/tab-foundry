# Stability Follow-Up Comparison Matrix

| Row | Delta Ref | Training Surface | Max Steps | Warmup | Grad Clip | Row Pool | Status |
|-----|-----------|------------------|-----------|--------|-----------|----------|--------|
| A | horizon_10000_baseline | prior_cosine_warmup | 10000 | 0.05 | 1.0 | target_column | ready |
| B | schedule_linear_decay_prenorm | prior_linear_decay | 2500 | 0.00 | 1.0 | target_column | ready |
| C | schedule_linear_warmup_decay_prenorm | prior_linear_warmup_decay | 2500 | 0.05 | 1.0 | target_column | ready |
| D | rowpool_row_cls_cls2 | prior_cosine_warmup | 2500 | 0.05 | 1.0 | row_cls | ready |

## Design

This follow-up stays on the stabilized prenorm foundation and asks only the
remaining high-signal questions.

- **A** confirms the frozen baseline at `10000` steps and forces explicit
  encoder/head gradient-ratio reporting over steps `1-25`, the first `100`
  logged steps after warmup ends, and the final `10%` of the run.
- **B** tests plain `linear_decay` on prenorm to see whether decay alone fixes
  best-checkpoint drift.
- **C** tests `linear_warmup_decay` on prenorm to see whether decay can help
  without surrendering the warmup stability win.
- **D** revisits `row_cls` with the old cleanest `cls_tokens=2` seed and ranks
  it benchmark-first on the stabilized surface.

Dropout and dagzoo are retired from this sweep. The remaining training-dynamics
work is schedule closure plus long-horizon adequacy, and the only architecture
revisit is one controlled RowPool row.

## Concern #3 Telemetry

Use the existing `gradient_history.jsonl` module telemetry for
`feature_encoder` and `direct_head`.

- Report the encoder/head gradient ratio over steps `1-25`.
- Report the same ratio for the first `100` logged steps after warmup ends.
- Report the same ratio for the final `10%` of each run.

Treat this as diagnostic evidence. Benchmark quality and checkpoint drift remain
higher priority than module-gradient balance by itself.

## Acceptance

- **Long-horizon baseline**
  - Accept the frozen baseline as horizon-adequate only if the run stays finite
    and does not enter a new late-run variance or gradient regime.
  - If this row fails, reopen horizon stabilization as a separate follow-up
    rather than broadening this sweep.
- **Linear schedules**
  - Rank schedule rows by benchmark best ROC AUC first, final ROC AUC second,
    and drift third.
  - Promote a linear row only if final ROC AUC stays within `0.002` of the
    cosine-warmup anchor and drift improves materially.
  - Use post-warmup loss variance and encoder/head gradient ratio as supporting
    diagnostics, not as promotion criteria by themselves.
- **RowPool revisit**
  - Rank the `row_cls` row by benchmark ROC AUC first, drift second, and
    stability diagnostics third.
  - Treat a clear benchmark miss with no drift win as renewed negative evidence
    for RowPool on the stabilized surface.

## Anchor Surface

All rows share the same frozen staged baseline unless the row explicitly changes
schedule or row pooling.

- `arch: tabfoundry_staged`, `stage: prenorm_block`
- `feature_encoder: shared`, `post_encoder_norm: layernorm`
- `table_block_style: prenorm`, `head: binary_direct`
- `tokenizer: scalar_per_feature`, `row_pool: target_column`
- `context_encoder: none`, `column_encoder: none`
- `training.surface_label: prior_cosine_warmup`
