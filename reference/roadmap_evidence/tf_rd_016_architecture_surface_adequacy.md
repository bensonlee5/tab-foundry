# TF-RD-016: Architecture Surface Adequacy And Selective Expansion

This is the canonical long-form evidence note for
[TF-RD-016](../../docs/development/roadmap.md#tf-rd-016-architecture-surface-adequacy-and-selective-expansion).

- Status: `planned`
- Milestone: `Next`
- Dependency position: follows
  [TF-RD-014](tf_rd_014_missingness_robustness.md) and
  [TF-RD-017](tf_rd_017_class_imbalance_robustness.md), then gates
  [TF-RD-010](tf_rd_010_many_class_promotion.md),
  [TF-RD-015](tf_rd_015_regression_rebuild.md),
  [TF-RD-012](tf_rd_012_inference_handoff_and_later_modalities.md), and
  [TF-RD-009](tf_rd_009_scaling_law_measurement.md)

## External Evidence

- Shared bibliography: [reference/papers.md](../papers.md)
- Current curated context is broad rather than knob-specific: `TabICLv2`,
  `FT-Transformer`, `Deep Sets`, `Set Transformer`, and compact-transformer
  recipe references from `nanochat`
- Dedicated micro-architecture adequacy literature is not yet curated as its
  own repo section
- External evidence to curate next: norm-placement, initialization, activation,
  and compact-capacity papers only if the existing-surface ladder remains
  low-signal

## Repo-Local Evidence

- the staged surface already exposes tokenization choice, `feature_group_size`,
  norm family and placement, widths, depths, row CLS count, TFCol inducing
  count, context FF expansion, dropout, and clipping
- learned special-token and inducing-token initialization scale remains
  hardcoded
- optimizer adequacy work, including `muon`, is already scoped out of this epic
  and belongs to [TF-RD-018](tf_rd_018_training_surface_adequacy.md)

## Current Interpretation

- Phase 1 should read the existing surface on harder post-008 ladders before
  adding new config fields
- tokenizer and norm-family or placement choices are the first explicit
  subtracks because they are already exposed and could matter across multiple
  future regimes
- selective surface expansion belongs only in Phase 2, and only if Phase 1
  remains low-signal

## Open Evidence Gaps

- there is no explicit keep or defer decision yet on whether the current staged
  surface is sufficient on harder post-008 regimes
- the repo has not yet shown whether the remaining hardcoded choices are
  decision-relevant enough to expose
- the knob-specific literature set is intentionally deferred until Phase 1 says
  it is needed

## Exit Signals

- the repo has an explicit keep or defer decision on whether the current staged
  architecture surface is sufficient on harder post-008 regimes
- any newly exposed architecture knobs are bounded, justified, and tied to one
  coherent staged comparison surface
