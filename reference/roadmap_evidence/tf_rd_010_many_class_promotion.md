# TF-RD-010: Many-Class Promotion On The Row-First Base

This is the canonical long-form evidence note for
[TF-RD-010](../../docs/development/roadmap.md#tf-rd-010-many-class-promotion-on-the-row-first-base).

- Status: `completed`
- Milestone: `Next`
- Dependency position: follows
  [TF-RD-016](tf_rd_016_architecture_surface_adequacy.md) in the current
  roadmap ordering and extends the same staged family rather than opening a
  separate architecture lane

## External Evidence

- Shared bibliography: [reference/papers.md](../papers.md)
- Current curated context is primarily `EquiTabPFN` plus the broader tabular
  foundation-model references that keep label conditioning modular
- Dedicated many-class literature is not yet curated beyond that shared context
- External evidence to curate next: multiclass calibration, hierarchical
  prediction, and many-class efficiency references if later expansion reopens
  this lane

## Repo-Local Evidence

- the staged family already contains `many_class`
- the hierarchical many-class machinery already exists
- `nanotabpfn_openml_classification_small_v1.json` already exists as a
  benchmark-facing multiclass bundle
- the roadmap now records TF-RD-010 as completed rather than leaving it as
  unvalidated scaffolding

## Current Interpretation

- many-class remains an extension of the promoted row-first staged family
- the promoted row-first backbone should remain the parent for future many-class
  work unless new evidence justifies reopening the lane
- future many-class work should stay benchmark-first rather than opening a new
  model family

## Open Evidence Gaps

- there is no dedicated many-class literature subset yet beyond the shared
  foundation-model references
- any future expansion would still need clearer evidence on multiclass
  calibration and hierarchy-specific tradeoffs
- the current note is mainly a record of the accepted extension path rather than
  an active research queue

## Exit Signals

- preserved historical signal: many-class uses the promoted row-first backbone,
  has benchmark-facing evidence, and is no longer only untested scaffolding
