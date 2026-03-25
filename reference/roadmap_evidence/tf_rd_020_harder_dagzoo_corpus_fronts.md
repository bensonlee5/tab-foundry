# TF-RD-020: Harder Dagzoo Corpus Fronts On The Promoted Anchor

This is the canonical long-form evidence note for
[TF-RD-020](../../docs/development/roadmap.md#tf-rd-020-harder-dagzoo-corpus-fronts-on-the-promoted-anchor).

- Status: `planned`
- Milestone: `Next`
- Dependency position: follows the initial TF-RD-018 batch-ladder closure under
  [#109](https://github.com/bensonlee5/tab-foundry/issues/109), blocks the
  remaining optimizer or LR or clipping continuation under
  [TF-RD-018](tf_rd_018_training_surface_adequacy.md), and sits adjacent to the
  benchmark-front harder-surface epics
  [TF-RD-014](tf_rd_014_missingness_robustness.md) and
  [TF-RD-017](tf_rd_017_class_imbalance_robustness.md); the active blocker is
  the uncapped execution and interpretation of
  `tf_rd_020_harder_dagzoo_ladder_v1`

## External Evidence

- Shared bibliography: [reference/papers.md](../papers.md)
- The current curated evidence is still broad tabular pretraining and
  robustness context rather than a dedicated “harder synthetic dagzoo corpus”
  bibliography
- Dedicated literature for this epic is not yet curated in this repo
- External evidence to curate next: synthetic tabular robustness papers on
  missingness, shift or drift, mechanism diversity, heavy-tail noise, and
  regime-separation criteria for compact tabular transformers

## Repo-Local Evidence

- TF-RD-013 settled `tf_rd_013_dagzoo_shape_aware_size_medium_v1` as the
  representative post-008 synthetic training-data surface
- `row_first_training_adequacy_v1` completed the first TF-RD-018 dataset-batch
  ladder and closed [#109](https://github.com/bensonlee5/tab-foundry/issues/109)
- issue [#147](https://github.com/bensonlee5/tab-foundry/issues/147) now records
  the canonical pre-filter harder-front ladder in
  [`tf_rd_020_harder_dagzoo_ladder_v1`](../system_delta_sweeps/tf_rd_020_harder_dagzoo_ladder_v1/matrix.md)
  plus the matching `tf_rd_020_*_v1` corpus recipes
- Dagzoo already exposes explicit surfaces for missingness, shift or drift,
  mechanism diversity, and noise on the same synthetic-data lane
- Dagzoo now also ships a small-shot ease filter contract rather than the
  removed threshold-era filter contract, but TF-RD-020 now stays pre-filter and
  leaves later filtering policy to TF-RD-019
- The benchmark-front missingness and imbalance epics already exist as
  [#97](https://github.com/bensonlee5/tab-foundry/issues/97) and
  [#106](https://github.com/bensonlee5/tab-foundry/issues/106), so TF-RD-020
  should remain synthetic-data-only rather than absorbing those programs
- The larger-corpus and winner-mix follow-up directions under closed issues
  [#154](https://github.com/bensonlee5/tab-foundry/issues/154),
  [#155](https://github.com/bensonlee5/tab-foundry/issues/155), and
  [#156](https://github.com/bensonlee5/tab-foundry/issues/156) are deferred
  pending results from the uncapped v1 ladder

## Current Interpretation

- reuse the settled row-first recipe rather than reopening TF-RD-018 recipe
  choice inside this epic
- use `tf_rd_020_harder_dagzoo_ladder_v1` as the canonical pre-filter ladder
  before the remaining optimizer-family, LR-shape, clipping, and budget work
  resumes under TF-RD-018
- treat missingness, shift or drift, and mechanism-diversity or noise as the
  bounded first candidate fronts rather than opening a broad new corpus program
- keep those first harder-front comparisons pre-filter and close TF-RD-020 on
  the uncapped v1 ladder rather than handing them to a separate filter issue
- keep TF-RD-020 distinct from benchmark-front missingness or imbalance
  conclusions even when the underlying regimes overlap conceptually
- keep the broader default-pipeline filtering-policy question under
  [TF-RD-019](tf_rd_019_dagzoo_filtering_policy.md) rather than absorbing it

## Open Evidence Gaps

- there are no completed uncapped v1 results yet for the missingness,
  shift/drift, and mechanism or noise families
- there is no recorded set of family winners from the uncapped ladder
- the handoff contract from TF-RD-020 back into TF-RD-018 optimizer or LR or
  clipping continuation is not yet documented in one place

## Exit Signals

- the repo has explicit keep, defer, or reject decisions across the harder
  dagzoo corpus fronts, including exactly one kept row in each TF-RD-020 family
- issue [#147](https://github.com/bensonlee5/tab-foundry/issues/147) is closed
  because the canonical pre-filter ladder and handoff are now recorded
- TF-RD-018 continuation resumes only after the uncapped v1 ladder records the
  family winners and the carry-forward interpretation
- the relationship between TF-RD-020 and the benchmark-front epics TF-RD-014
  and TF-RD-017 plus the later filtering-policy lane TF-RD-019 remains
  explicit and non-overlapping
