# TF-RD-015: Regression Rebuild On The Promoted Row-First Base

This is the canonical long-form evidence note for
[TF-RD-015](../../docs/development/roadmap.md#tf-rd-015-regression-rebuild-on-the-promoted-row-first-base).

- Status: `planned`
- Milestone: `Next`
- Dependency position: follows
  [TF-RD-016](tf_rd_016_architecture_surface_adequacy.md) and rebuilds
  regression on the same staged family rather than on a separate architecture
  lane

## External Evidence

- Shared bibliography: [reference/papers.md](../papers.md)
- Current curated context is broad rather than regression-specific: general
  tabular foundation-model references and the repo's existing data and contract
  notes
- Dedicated regression literature is not yet curated in this repo
- External evidence to curate next: tabular-regression benchmark references,
  regression-head and loss-function guidance, and normalization or calibration
  work that fits a staged row-first model family

## Repo-Local Evidence

- regression support is intentionally removed from the active repo surface
- regression metrics and benchmark-bundle normalization support still exist in
  parts of the repo
- there is no active staged regression program, canonical regression bundle, or
  staged regression head and loss contract

## Current Interpretation

- rebuild regression as a staged-family extension on top of the promoted
  row-first base
- keep one OpenML baseline where possible and use license-cleared
  manifest-backed external datasets only as bounded augmentations
- stay on the same staged family instead of reopening a second model lane

## Open Evidence Gaps

- the repo does not yet have a canonical regression benchmark surface
- the regression-specific literature set still needs to be curated
- the staged regression head and loss contract are still undefined

## Exit Signals

- regression has a benchmark-facing staged baseline and a bounded roadmap for
  promotion or deferral
- the rebuilt regression lane remains an extension of the promoted row-first
  staged family
