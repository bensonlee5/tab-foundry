# TF-RD-015: Regression Rebuild Deferred From The Classification-First Scaling Plan

This is the canonical long-form evidence note for
[TF-RD-015](../../docs/development/roadmap.md#tf-rd-015-regression-rebuild-deferred-from-the-classification-first-scaling-plan).

- Status: `research`
- Milestone: `Later`
- Dependency position: intentionally deferred until after the first
  classification-first sandwich scaling program lands a stable runtime policy,
  harder-surface evidence, and a usable scaling contract

## External Evidence

- Current curated context is still broad rather than regression-specific:
  general tabular references, calibration references, and compact-model papers
  already exist in `reference/papers.md`
- Dedicated regression literature is still not curated in this repo
- External evidence to curate later:
  - tabular-regression benchmark references
  - regression-head and loss-function guidance
  - normalization or calibration references specific to regression

## Repo-Local Evidence

- regression support is intentionally removed from the active repo surface
- regression metrics and benchmark-bundle normalization support still exist in
  parts of the repo
- there is no active regression program, canonical regression bundle, or
  regression head/loss contract
- the roadmap now explicitly removes regression as a blocker for the first
  classification scaling plan

## Current Interpretation

- regression should not absorb roadmap attention before the repo settles the
  classification-first sandwich family
- the right time to rebuild regression is after the classification lane has:
  - one inherited runtime policy
  - one harder-surface classification contract
  - one initial scaling-law contract
- the eventual regression lane should be judged on its own benchmark-facing
  evidence rather than being bundled into the first classification scaling gate

## Open Evidence Gaps

- the repo does not yet have a canonical regression benchmark surface
- the regression-specific literature set still needs to be curated
- the eventual regression head and loss contract remain undefined

## Exit Signals

- regression resumes only after the first classification scaling program is no
  longer blocked on runtime, missingness, or scaling-law design work
- the rebuilt regression lane has a benchmark-facing baseline and a bounded
  roadmap for promotion or deferral
