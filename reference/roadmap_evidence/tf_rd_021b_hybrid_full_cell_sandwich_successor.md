# TF-RD-021B: Hybrid Full-Cell Sandwich Successor And Locked Prior Replay

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
- successor issue [#181](https://github.com/bensonlee5/tab-foundry/issues/181)
  now owns the first bounded replay and interpretation pass for the hybrid
  full-cell sandwich
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

## Current Interpretation

- TF-RD-021A answered the first important question already: the summary-only
  bottleneck can train, but it loses too much signal to justify more latent or
  width tuning on the same topology
- the next useful question is whether letting stage `0` and the final readout
  see the full cell stream recovers enough fidelity to reopen a capacity ladder
- the immediate target is still the locked nanoTabPFN prior surface because it
  gives the fastest feedback on whether the topology move matters at all
- latent-count and width sweeps should stay closed until one successor replay
  exists on this same locked surface

## Open Evidence Gaps

- no completed benchmark replay is recorded yet for the successor preset
- no explicit comparison is recorded yet between the successor replay and the
  closed TF-RD-021A replay row
- no harder-surface or longer-budget read exists yet for the successor
  architecture
- the fit floor that would justify reopening latent or width sweeps remains
  implicit until the first successor replay lands

## Exit Signals

- the repo records one explicit benchmark replay for the hybrid successor on
  the locked nanoTabPFN prior surface
- that replay is clearly interpreted against the closed TF-RD-021A replay row
- the next sandwich sweep, if any, is authored from the successor replay rather
  than from the abandoned TF-RD-021A latent or width ladder
