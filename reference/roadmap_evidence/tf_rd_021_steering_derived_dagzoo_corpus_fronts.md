# TF-RD-021: Steering-Derived Dagzoo Corpus Fronts On The Promoted Anchor

This is the canonical long-form evidence note for
[TF-RD-021](../../docs/development/roadmap.md#tf-rd-021-steering-derived-dagzoo-corpus-fronts-on-the-promoted-anchor).

- Status: `planned`
- Milestone: `Next`
- Dependency position: follows the first carried many-class plus missingness
  dagzoo slice under [TF-RD-010](tf_rd_010_many_class_promotion.md), depends on
  dagzoo RD-008 steering landing under
  [bensonlee5/dagzoo#246](https://github.com/bensonlee5/dagzoo/issues/246),
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
  missingness slice that steering will attempt to improve
- Dagzoo issue
  [bensonlee5/dagzoo#246](https://github.com/bensonlee5/dagzoo/issues/246)
  plus its child chain now define the upstream steering implementation,
  deterministic policy metadata, and coverage-movement diagnostics
- Tab-foundry now tracks the local steering-derived continuation under
  [#165](https://github.com/bensonlee5/tab-foundry/issues/165), with first
  sweep contract issue
  [#167](https://github.com/bensonlee5/tab-foundry/issues/167)

## Current Interpretation

- Treat TF-RD-020 as settled historical harder-front evidence rather than the
  place to reopen curriculum-steered corpora
- Keep the first steering-derived read small and explicit: one control row on
  the carried sandwich dagzoo slice plus `3-4` steering-derived corpus rows
  from named steering policies or presets
- Interpret rows by multiclass log loss first, with runtime, clipped-step
  fraction, and stability telemetry as guardrails
- Keep exactly one steering-derived carry-forward surface only if it clearly
  beats the incumbent control; otherwise retain the original carried slice
- Feed the kept steering decision into TF-RD-022 and TF-RD-009 rather than
  reopening TF-RD-010 or TF-RD-018

## Open Evidence Gaps

- The repo does not yet have curated external literature specific to
  meta-feature coverage steering as a synthetic harder-front continuation
- There is no first steering-derived sweep matrix or result package yet in
  tab-foundry
- The repo still needs a practical stop rule for how much steering movement and
  metric gain counts as a genuinely new carry-forward surface rather than a
  noisy variant of TF-RD-020 row `06`

## Exit Signals

- The repo has one explicit keep/defer decision on whether any steering-derived
  corpus front replaces `tf_rd_020_shift_noise_drift_v1`
- TF-RD-022 can inherit a documented synthetic carry-forward decision without
  reopening TF-RD-010 or blurring the completed TF-RD-020 ladder
