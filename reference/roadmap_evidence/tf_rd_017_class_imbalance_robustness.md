# TF-RD-017: Class-Imbalance Robustness On The Classification-First Sandwich Target

This is the canonical long-form evidence note for
[TF-RD-017](../../docs/development/roadmap.md#tf-rd-017-class-imbalance-robustness-on-the-promoted-anchor).

- Status: `planned`
- Milestone: `Next`
- Dependency position: follows the first carried sandwich many-class plus
  missingness dagzoo gate under
  [TF-RD-010](tf_rd_010_many_class_promotion.md) as a side robustness lane,
  sits adjacent to the synthetic-data-only front
  [TF-RD-021](tf_rd_021_steering_derived_dagzoo_corpus_fronts.md), and does
  not block [TF-RD-009](tf_rd_009_scaling_law_measurement.md)

## External Evidence

- Shared bibliography: [reference/papers.md](../papers.md)
- The current curated evidence set is still general tabular-model context
  rather than an imbalance-specific bibliography
- Dedicated imbalance literature is not yet curated in this repo
- External evidence to curate next: tabular-classification work on imbalance
  metrics, weighted-loss and focal-loss tradeoffs, and calibration under skewed
  priors

## Repo-Local Evidence

- current benchmark bundles only enforce `min_minority_class_pct = 2.5`, so the
  repo still lacks a dedicated imbalance-focused bundle ladder
- benchmark-facing reporting remains centered on ROC AUC, log loss, and Brier
  score
- the roadmap explicitly requires PR AUC, average precision, and balanced
  accuracy before imbalance conclusions are treated as benchmark-ready
- TF-RD-020 now occupies the adjacent synthetic harder-dagzoo slot and does
  not replace this benchmark-front imbalance program

## Current Interpretation

- define the canonical imbalance-focused binary ladder on the carried sandwich
  family before changing losses
- measure the carried sandwich family first without class reweighting or
  focal-style loss changes
- only if the baseline read is weak should weighted-loss or focal-loss follow-up
  work become decision-relevant

## Open Evidence Gaps

- there is no benchmark-backed keep or defer decision on the carried sandwich
  family under materially skewed priors
- the imbalance-specific literature set still needs to be curated
- the canonical imbalance reporting contract does not yet exist in benchmark
  outputs

## Exit Signals

- the repo has a benchmark-backed keep or defer decision on the promoted
  sandwich family under class imbalance
- benchmark-facing outputs include explicit imbalance metrics rather than
  relying only on the current general binary bundle surface
