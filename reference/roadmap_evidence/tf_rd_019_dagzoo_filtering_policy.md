# TF-RD-019: Predictable Dagzoo Filtering Policy For Training Corpora

This is the canonical long-form evidence note for
[TF-RD-019](../../docs/development/roadmap.md#tf-rd-019-predictable-dagzoo-filtering-policy-for-training-corpora).

- Status: `planned`
- Milestone: `Next`
- Dependency position: follows TF-RD-013 as a training-data policy question and
  sits adjacent to [TF-RD-018](tf_rd_018_training_surface_adequacy.md) rather
  than replacing it

## External Evidence

- Shared bibliography: [reference/papers.md](../papers.md)
- Nearby curated context today is still broad rather than filtering-specific:
  `TabICLv2` for cheap predictive filtering as a donor idea and `TabDPT` for
  broader data-source flexibility
- Dedicated filtering-policy literature is not yet curated in this repo
- External evidence to curate next: cheap predictive filtering, data-selection
  heuristics, and corpus predictability papers that fit small-transformer or
  tabular pretraining settings

## Repo-Local Evidence

- [#120](https://github.com/bensonlee5/tab-foundry/issues/120) recorded the
  first runnable unfiltered dagzoo generated-source surface and support
  artifacts for the promoted anchor
- TF-RD-013 settled an unfiltered shape-aware medium rung as the representative
  post-008 synthetic training-data surface
- the roadmap still treats filtering as optional and cost-sensitive rather than
  as part of the default training-data lane

## Current Interpretation

- do not assume a filtering stage belongs in the default dagzoo lane
- define the failure modes filtering is meant to solve before comparing
  candidate implementations
- evaluate predictive filtering, lighter heuristics, and no-filter baselines
  under an explicit throughput budget

## Open Evidence Gaps

- the repo does not yet define what "predictable training corpora" means in
  operational terms
- there is no accepted throughput budget or provenance contract for filtered
  dagzoo surfaces
- the literature set for filtering-specific decisions still needs to be curated

## Exit Signals

- the repo has an explicit recommendation on whether dagzoo filtering belongs
  in the default training-data pipeline
- if filtering is kept, the acceptable implementation and throughput budget are
  documented strongly enough that TF-RD-013 surfaces can be re-enabled under a
  clear contract
