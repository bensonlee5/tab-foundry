# TF-RD-021: Steering-Derived Dagzoo Corpus Fronts On The Promoted Anchor

This is the canonical long-form evidence note for
[TF-RD-021](../../docs/development/roadmap.md#tf-rd-021-steering-derived-dagzoo-corpus-fronts-on-the-promoted-anchor).

- Status: `planned`
- Milestone: `Next`
- Dependency position: follows completed
  [TF-RD-020](tf_rd_020_harder_dagzoo_corpus_fronts.md) and the remaining
  TF-RD-018 recipe-closure work under
  [#138](https://github.com/bensonlee5/tab-foundry/issues/138) and
  [#139](https://github.com/bensonlee5/tab-foundry/issues/139); depends on
  dagzoo RD-008 steering landing under
  [bensonlee5/dagzoo#246](https://github.com/bensonlee5/dagzoo/issues/246)
  before the benchmark-front epics
  [TF-RD-014](tf_rd_014_missingness_robustness.md) and
  [TF-RD-017](tf_rd_017_class_imbalance_robustness.md)

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
  kept rows for missingness, shift or drift, and mechanism or noise, and the
  current default carried surface is `tf_rd_020_shift_noise_drift_v1`
- TF-RD-018 issue
  [#137](https://github.com/bensonlee5/tab-foundry/issues/137) compared
  `schedulefree_adamw`, `adamw`, and `muon` on that inherited control and left
  Muon deferred rather than kept as an optimizer fallback
- TF-RD-018 still needs issues
  [#138](https://github.com/bensonlee5/tab-foundry/issues/138) and
  [#139](https://github.com/bensonlee5/tab-foundry/issues/139) to finish one
  explicit default LR, clipping, and step-budget recipe
- Dagzoo issue
  [bensonlee5/dagzoo#246](https://github.com/bensonlee5/dagzoo/issues/246)
  plus its child chain now define the upstream steering implementation,
  deterministic policy metadata, and coverage-movement diagnostics
- Tab-foundry now tracks the local steering-derived continuation under
  [#165](https://github.com/bensonlee5/tab-foundry/issues/165), with first
  sweep contract issue
  [#167](https://github.com/bensonlee5/tab-foundry/issues/167) and conditional
  optimizer retry issue
  [#166](https://github.com/bensonlee5/tab-foundry/issues/166)

## Current Interpretation

- Treat TF-RD-020 as settled v1 harder-front evidence rather than the place to
  reopen curriculum-steered corpora
- Finish TF-RD-018 recipe closure first so any later corpus comparison reads as
  a surface change rather than unresolved LR, clipping, or budget noise
- Use TF-RD-021 to test whether the earlier Muon miss was partly a
  corpus-front mismatch instead of retrying Muon inside TF-RD-018
- Keep the first steering-derived read small and explicit: one control row on
  `tf_rd_020_shift_noise_drift_v1` plus `3-4` steering-derived corpus rows from
  named steering policies or presets
- Interpret rows by final log loss first, then final Brier score, with runtime,
  clipped-step fraction, and stability telemetry as guardrails
- Keep exactly one steering-derived carry-forward surface only if it clearly
  beats the incumbent control; otherwise retain `tf_rd_020_shift_noise_drift_v1`
- Only if a steering-derived front wins should tab-foundry run the bounded
  `schedulefree_adamw` versus `muon` retry tracked by issue `#166`

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
- If a steering-derived front wins, the repo has one bounded optimizer-family
  follow-up on that new front; otherwise the retry is explicitly skipped
- TF-RD-014 and TF-RD-017 can inherit a documented synthetic carry-forward
  decision without reopening TF-RD-018 or blurring the completed TF-RD-020
  ladder
