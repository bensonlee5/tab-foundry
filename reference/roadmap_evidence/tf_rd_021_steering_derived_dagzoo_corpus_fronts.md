# TF-RD-021: Steering-Derived Dagzoo Corpus Fronts On The Classification-First Sandwich Target

This is the canonical long-form evidence note for
[TF-RD-021](../../docs/development/roadmap.md#tf-rd-021-steering-derived-dagzoo-corpus-fronts-on-the-promoted-anchor).

- Status: `planned`
- Milestone: `Next`
- Dependency position: follows the first carried many-class plus missingness
  dagzoo slice under [TF-RD-010](tf_rd_010_many_class_promotion.md), depends on
  dagzoo RD-002 and RD-005 expansion landing under
  [bensonlee5/dagzoo#249](https://github.com/bensonlee5/dagzoo/issues/249)
  and [bensonlee5/dagzoo#247](https://github.com/bensonlee5/dagzoo/issues/247),
  and runs before later kernel/runtime tuning under
  [TF-RD-022](tf_rd_022_training_runtime_vram_efficiency.md)

## External Evidence

- Shared bibliography: [reference/papers.md](../papers.md)
- Dedicated literature for coverage-steered synthetic harder fronts is not yet
  curated in this repo
- The closest current directional external evidence is still
  [A Closer Look at TabPFN v2](../papers.md), which supports testing
  meta-feature sensitivity structurally rather than assuming one fixed surface
  is sufficient
- External evidence to curate next: synthetic curriculum or task-difficulty
  steering papers, meta-feature coverage steering references, and any evidence
  about optimizer sensitivity to corpus-front mismatch

## Repo-Local Evidence

- TF-RD-020 closed under
  [#146](https://github.com/bensonlee5/tab-foundry/issues/146) with explicit
  staged-control harder-front winners that now serve as historical dagzoo
  context rather than the active sandwich carried slice
- TF-RD-010 now owns the first explicit carried sandwich dagzoo many-class plus
  missingness slice that TF-RD-021 will attempt to improve
- Completed dagzoo issue
  [bensonlee5/dagzoo#246](https://github.com/bensonlee5/dagzoo/issues/246)
  remains historical steering context
- Dagzoo issues
  [bensonlee5/dagzoo#249](https://github.com/bensonlee5/dagzoo/issues/249)
  and [bensonlee5/dagzoo#247](https://github.com/bensonlee5/dagzoo/issues/247)
  now define the upstream expansion of admissible synthetic surfaces and
  metadata that TF-RD-021 phase 1 must freeze for evaluation
- Tab-foundry now tracks the local steering-derived continuation under
  [#165](https://github.com/bensonlee5/tab-foundry/issues/165), with one phase
  1 candidate-freeze child and phase 2 sweep contract issue
  [#167](https://github.com/bensonlee5/tab-foundry/issues/167)

## Current Interpretation

- Treat TF-RD-020 as settled historical harder-front evidence rather than the
  place to reopen curriculum-steered corpora
- Treat TF-RD-021 as a two-phase lane:
  - phase 1 freezes the admissible post-RD-002/RD-005 dagzoo candidate surface
    and metadata contract for matched-regime-budget evaluation
  - phase 2 runs one bounded carry-forward read: one control row on the
    carried sandwich dagzoo slice plus a small set of named post-RD-002/RD-005
    corpus rows
- Interpret rows by multiclass log loss first, with runtime, clipped-step
  fraction, and stability telemetry as guardrails
- Keep exactly one carry-forward surface only if it clearly beats the
  incumbent control; otherwise retain the original carried slice
- Feed the kept carry-forward decision into TF-RD-022 and TF-RD-009 rather
  than reopening TF-RD-010 or TF-RD-018

## Open Evidence Gaps

- The repo does not yet have curated external literature specific to
  meta-feature coverage steering as a synthetic harder-front continuation
- There is no phase 1 candidate-freeze package or phase 2 carry-forward sweep
  result package yet in tab-foundry
- The repo still needs a practical stop rule for how much post-RD-002/RD-005
  surface movement and metric gain counts as a genuinely new carry-forward
  surface rather than a noisy variant of the incumbent carried control slice

## Exit Signals

- The repo has one explicit keep/defer decision on whether any frozen
  post-RD-002/RD-005 corpus front replaces the incumbent carried sandwich
  control slice
- TF-RD-022 can inherit a documented synthetic carry-forward decision without
  reopening TF-RD-010 or blurring the completed TF-RD-020 ladder
