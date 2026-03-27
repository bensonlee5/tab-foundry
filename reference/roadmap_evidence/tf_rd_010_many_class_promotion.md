# TF-RD-010: First Many-Class + Missingness Dagzoo Gate On The Row-First Base

This is the canonical long-form evidence note for
[TF-RD-010](../../docs/development/roadmap.md).

- Status: `planned`
- Milestone: `Next`
- Dependency position: follows
  [TF-RD-016](tf_rd_016_architecture_surface_adequacy.md) in the current
  roadmap ordering, establishes the first carried many-class plus missingness
  dagzoo slice, and feeds
  [TF-RD-009](tf_rd_009_scaling_law_measurement.md) rather than opening a
  separate architecture lane

## External Evidence

- Shared bibliography: [reference/papers.md](../papers.md)
- Current curated context is primarily `EquiTabPFN` plus the broader tabular
  foundation-model references that keep label conditioning modular
- Dedicated many-class literature is not yet curated beyond that shared context
- External evidence to curate next: multiclass calibration, hierarchical
  prediction, many-class efficiency, and missingness-aware multiclass
  evaluation references for the first carried dagzoo slice

## Repo-Local Evidence

- the staged family already contains `many_class`
- the hierarchical many-class machinery already exists
- `nanotabpfn_openml_classification_small_v1.json` already exists as a
  benchmark-facing multiclass bundle
- issue [#52](https://github.com/bensonlee5/tab-foundry/issues/52) is the epic
  for this lane, and issue
  [#99](https://github.com/bensonlee5/tab-foundry/issues/99) is the first
  execution issue
- the roadmap now treats a dagzoo-backed many-class plus missingness slice as
  the first anti-saturation classification gate before the first scaling fit

## Current Interpretation

- keep this lane on the promoted classification family rather than opening a
  separate multiclass model track
- use one dagzoo-backed many-class plus missingness slice as the first carried
  harder classification regime
- interpret the lane by multiclass log loss first, with runtime, stability,
  and calibration-oriented metrics as guardrails

## Open Evidence Gaps

- there is no dedicated many-class literature subset yet beyond the shared
  foundation-model references
- the repo does not yet have one explicit carried dagzoo many-class plus
  missingness slice on the frozen sandwich parent
- the first benchmark-facing many-class plus missingness sweep and decision
  package do not exist yet

## Exit Signals

- the repo has one explicit carried dagzoo many-class plus missingness slice on
  the promoted backbone
- multiclass is no longer only untested scaffolding on the first scaling path
- TF-RD-009 can inherit a fixed anti-saturation classification target instead
  of reopening regime selection
