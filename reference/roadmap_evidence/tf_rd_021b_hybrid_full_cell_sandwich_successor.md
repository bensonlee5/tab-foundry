# TF-RD-021B: Hybrid Full-Cell Sandwich Successor, Simplification, And Classification-First Scaling Prep

This is the long-form evidence note for the hybrid full-cell / summary-stream
successor to the original `tabfoundry_sandwich` summary-bottleneck replay.
The lane lives under the broader
[TF-RD-016](../../docs/development/roadmap.md#tf-rd-016-architecture-surface-adequacy-sandwich-simplification-and-selective-expansion)
architecture-adequacy workstream.

- Status: `partial`
- Milestone: `Next`
- Dependency position: successor architecture sub-lane under TF-RD-016, opened
  after TF-RD-021A closed as negative evidence for the summary-bottleneck
  replay

## External Evidence

- Relevant references already called out in `reference/papers.md`:
  - Perceiver for latent bottlenecks and repeated latent reads
  - PerceiverIO for output-query style readout over the latent state
  - SAINT and Set Transformer for tabular set-style aggregation
  - PFN-style tabular references for train-conditioned ICL semantics
- The scaling-side literature relevant to the next phase now matters more than
  the original replay literature:
  - μP / muTransfer as the first width-transfer prior
  - depth-aware μP follow-ups as guidance for width-depth fits
  - synthetic-data curriculum references as guidance for later dagzoo slices

## Repo-Local Evidence

- predecessor issue [#179](https://github.com/bensonlee5/tab-foundry/issues/179)
  closed TF-RD-021A after the locked prior replay trained stably but
  underperformed badly on the benchmark surface
- umbrella issue [#178](https://github.com/bensonlee5/tab-foundry/issues/178)
  still owns long-running sandwich stabilization and iteration
- successor replay issue [#181](https://github.com/bensonlee5/tab-foundry/issues/181)
  records the first bounded replay and interpretation pass for the hybrid
  full-cell sandwich
- child issue [#182](https://github.com/bensonlee5/tab-foundry/issues/182)
  records the completed architecture-only sandwich knob-sensitivity screen
- child issue [#183](https://github.com/bensonlee5/tab-foundry/issues/183)
  records the completed bounded width and head-capacity follow-up
- child issue [#184](https://github.com/bensonlee5/tab-foundry/issues/184)
  now owns the post-screen simplified-parent and classification-scaling-prep
  follow-up
- the queued TF-RD-021B removal-first follow-up is
  `tf_rd_021b_sandwich_feature_removal_v1`, which recasts the earlier
  self-attention shrink idea as full removal and keeps the follow-up package
  bounded
- `tabfoundry_sandwich` now uses:
  - one fixed learned latent bank
  - a stage-`0` hybrid input stream of `full cells + row summaries + column summaries`
  - later repeated Perceiver stages over the compact `R + C` summary stream
  - train-label or test-query conditioning fused into both row summaries and
    full cell tokens
  - dual-source readout: test-row queries over final latents and then over the
    full cell stream
  - the same explicit `feature_types` runtime contract as before
- `research sweep inspect` / `diff` now resolve the materialized
  `default_effective_surface`, and benchmark-registry records now preserve the
  full resolved sandwich build spec plus regime-budget/runtime metadata
- the compact hybrid control `tf_rd_021b_hybrid_full_cell_compact_prior_v1`
  remains the local benchmarked control on the pinned medium binary bundle:
  - final ROC AUC `0.7370`
  - final log loss `0.4672`
  - final Brier `0.3072`
  - best checkpoint = final checkpoint at `step_002500`

## Current Interpretation

- TF-RD-021A already answered the first important question: the summary-only
  bottleneck can train, but it loses too much signal to justify more tuning on
  that topology
- the hybrid full-cell successor answered the next question: letting stage `0`
  and the final readout see the full cell stream recovers enough signal to keep
  sandwich alive as the primary classification architecture target
- the completed bounded sensitivity screens argue for simplification first, not
  for keeping many free architecture knobs alive during scaling
- the next bounded read is therefore a removal-first simplification package
  centered on:
  - `sandwich_self_attention_per_cross=0`
  - `sandwich_ff_expansion=1`
  - the combined row
  - the combined row plus `sandwich_summary_tokens_per_axis=1`
- after that choice, the non-shape sandwich knobs should be frozen while the
  simplified parent is carried onto dagzoo classification and then missingness
- any later width-depth scaling fits belong under TF-RD-009 and should inherit
  the simplified parent, not reopen the local sandwich knob screen

## Open Evidence Gaps

- the confirmatory simplification package is not yet executed
- the repo does not yet have one explicit kept simplified sandwich parent
- the simplified parent has not yet been carried onto one curriculum-backed
  dagzoo slice or onto missingness
- there is still no justification for a public single-knob scaling surface

## Exit Signals

- the repo records and interprets the bounded simplification package
- one explicit simplified sandwich parent is chosen and its non-shape knobs are
  frozen for follow-on classification work
- the simplified parent is carried onto one dagzoo curriculum slice and then
  onto missingness before the first main scaling fit
- any later scale-control interface is explicitly derived from the later
  TF-RD-009 law fits rather than from ad hoc sandwich-local knob guesses
