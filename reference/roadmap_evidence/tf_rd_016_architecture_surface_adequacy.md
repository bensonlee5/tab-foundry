# TF-RD-016: Architecture Surface Adequacy, Sandwich Simplification, And Selective Expansion

This is the canonical long-form evidence note for
[TF-RD-016](../../docs/development/roadmap.md#tf-rd-016-architecture-surface-adequacy-sandwich-simplification-and-selective-expansion).

- Status: `planned`
- Milestone: `Next`
- Dependency position: now includes an earlier simplified-parent phase before
  the main harder-surface ladders, then gates
  [TF-RD-010](tf_rd_010_many_class_promotion.md) as the first carried
  many-class plus missingness slice before follow-on robustness lanes such as
  [TF-RD-014](tf_rd_014_missingness_robustness.md) and
  [TF-RD-017](tf_rd_017_class_imbalance_robustness.md), plus
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
- TF-RD-021B now makes sandwich-parent simplification the first explicit
  architecture task before broader harder-surface adequacy work
- TF-RD-010 is now the first carried harder regime after that simplification:
  many-class plus missingness on a dagzoo-backed slice
- learned special-token and inducing-token initialization scale remains
  hardcoded
- optimizer adequacy work, including `muon`, is already scoped out of this epic
  and belongs to [TF-RD-018](tf_rd_018_training_surface_adequacy.md)

## Current Interpretation

- Phase 0 should choose and freeze a simplified sandwich parent before broader
  harder-surface adequacy work
- Phase 1 should then read the frozen parent on one dagzoo-backed many-class
  plus missingness slice before adding new config fields
- Phase 2 should use TF-RD-014 and TF-RD-017 as follow-on robustness lanes
  rather than as blockers to the first scaling target
- tokenizer and norm-family or placement choices are the first explicit
  subtracks because they are already exposed and could matter across multiple
  future regimes
- selective surface expansion belongs only in Phase 3, and only if the carried
  harder-surface reads remain low-signal

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
