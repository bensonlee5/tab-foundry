# TF-RD-017: Class-Imbalance Robustness On The Promoted Anchor

This is the canonical long-form evidence note for
[TF-RD-017](../../docs/development/roadmap.md#tf-rd-017-class-imbalance-robustness-on-the-promoted-anchor).

- Status: `planned`
- Milestone: `Next`
- Dependency position: follows [TF-RD-018](tf_rd_018_training_surface_adequacy.md)
  as one of the first harder benchmark-backed ladders and feeds into
  [TF-RD-016](tf_rd_016_architecture_surface_adequacy.md)

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

## Current Interpretation

- define the canonical imbalance-focused binary ladder on the promoted row-first
  anchor before changing losses
- measure the promoted anchor first without class reweighting or focal-style
  loss changes
- only if the baseline read is weak should weighted-loss or focal-loss follow-up
  work become decision-relevant

## Open Evidence Gaps

- there is no benchmark-backed keep or defer decision on the promoted row-first
  line under materially skewed priors
- the imbalance-specific literature set still needs to be curated
- the canonical imbalance reporting contract does not yet exist in benchmark
  outputs

## Exit Signals

- the repo has a benchmark-backed keep or defer decision on the promoted
  row-first line under class imbalance
- benchmark-facing outputs include explicit imbalance metrics rather than
  relying only on the current general binary bundle surface
