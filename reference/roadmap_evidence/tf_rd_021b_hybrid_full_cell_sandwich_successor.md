# TF-RD-021B: Hybrid Full-Cell Sandwich Successor, Sensitivity Screens, And Power-Curve Preparation

This is the long-form evidence note for the hybrid full-cell / summary-stream
successor to the original `tabfoundry_sandwich` summary-bottleneck replay.
The lane lives under the broader
[TF-RD-016](../../docs/development/roadmap.md#tf-rd-016-architecture-surface-adequacy-and-selective-expansion)
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
- Dedicated long-form external synthesis for the hybrid stage-0 full-cell read
  versus later summary-only reads is still missing.
- Sources to curate next:
  - PerceiverIO ablations that separate encoder-context fidelity from readout
    capacity
  - tabular or set-model references that preserve fine-grained feature tokens
    deeper into the architecture before pooling

## Repo-Local Evidence

- predecessor issue [#179](https://github.com/bensonlee5/tab-foundry/issues/179)
  closed TF-RD-021A after the locked prior replay trained stably but
  underperformed badly on the benchmark surface
- umbrella issue [#178](https://github.com/bensonlee5/tab-foundry/issues/178)
  still owns long-running sandwich stabilization and iteration
- successor replay issue [#181](https://github.com/bensonlee5/tab-foundry/issues/181)
  now records the first bounded replay and interpretation pass for the hybrid
  full-cell sandwich
- child issue [#182](https://github.com/bensonlee5/tab-foundry/issues/182)
  now owns the 9-run architecture-only sandwich knob-sensitivity screen
- child issue [#183](https://github.com/bensonlee5/tab-foundry/issues/183)
  now owns the bounded width and head-capacity follow-up
- child issue [#184](https://github.com/bensonlee5/tab-foundry/issues/184)
  now owns the later sandwich-local empirical power-curve phase before any
  single-toggle scaling recipe is authored
- `tabfoundry_sandwich` now uses:
  - one fixed learned latent bank
  - a stage-`0` hybrid input stream of `full cells + row summaries + column summaries`
  - later repeated Perceiver stages over the compact `R + C` summary stream
  - train-label or test-query conditioning fused into both row summaries and
    full cell tokens
  - dual-source readout: test-row queries over final latents and then over the
    full cell stream
  - the same explicit `feature_types` runtime contract as before
- a successor benchmark preset
  `configs/experiment/cls_benchmark_sandwich_hybrid_prior.yaml`
  now exists for the first locked prior replay
- the compact hybrid control `tf_rd_021b_hybrid_full_cell_compact_prior_v1`
  is now canonically benchmarked on the pinned medium binary bundle with no
  external comparator:
  - final ROC AUC `0.7370`
  - final log loss `0.4672`
  - final Brier `0.3072`
  - best checkpoint = final checkpoint at `step_002500`

## Current Interpretation

- TF-RD-021A answered the first important question already: the summary-only
  bottleneck can train, but it loses too much signal to justify more latent or
  width tuning on the same topology
- the hybrid full-cell successor already answers the next question at a coarse
  level: letting stage `0` and the final readout see the full cell stream does
  recover enough signal to justify more architecture work
- the immediate target is no longer “does one replay exist”; it is “which
  sandwich knobs are actually sensitive enough to survive into a later compound
  scaling recipe”
- the next bounded read is the 9-run local-only knob screen, then the width or
  head-capacity follow-up, both on the same locked prior surface and fixed
  `2500`-step budget
- after those bounded passes, the repo should fit sandwich-local empirical
  power curves before collapsing the family into any single-toggle scaling
  interface
- these sandwich-local curve fits are internal architecture-family evidence;
  they do not close TF-RD-009 on the promoted row-first anchor

## Open Evidence Gaps

- the 9-run sandwich knob-sensitivity screen is authored but not yet executed
- the bounded width or head-capacity follow-up is authored but intentionally
  blocked on the knob screen interpretation
- no sandwich-local empirical power-curve artifacts exist yet
- no harder-surface or longer-budget read exists yet for the successor
  architecture beyond the compact control replay
- the later one-toggle scaling recipe is still unjustified until the bounded
  sensitivity passes and sandwich-local power-curve fits exist

## Exit Signals

- the repo records and interprets the 9-run sandwich knob-sensitivity screen
- the immediate width or head-capacity follow-up lands on the same locked local
  control contract
- sandwich-local empirical power curves exist for the hybrid family before any
  single-toggle scaling recipe is authored
- any later one-toggle scaling surface is explicitly derived from those bounded
  sensitivity reads and curve fits rather than from ad hoc knob guesses
